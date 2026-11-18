"""单机 RDMA/RXE server+client 启动器；以监听端口就绪为准，不抢占握手连接。"""
# argparse 用来解析工具类型、设备、端口和 GPU 等选项。
import argparse
# Path 用来定位程序、读取 Linux 状态文件和管理日志。
from pathlib import Path
# shutil.which 用来检查系统里是否找得到 ib_write_bw 命令。
import shutil
# subprocess 启动并等待 RDMA 的 server/client 两个进程。
import subprocess
# time 控制等待就绪的超时和轮询间隔。
import time


# 判断本机是否已经有程序监听指定 TCP 端口，返回 True 或 False。
def listening(port):
    # Linux /proc，仅观察，不连接 perftest 的单次控制通道。
    # 分别读取 Linux 的 IPv4、IPv6 TCP 状态表；只观察，不建立连接。
    for name in ('/proc/net/tcp', '/proc/net/tcp6'):
        # 把状态表文件名转换为 Path 对象。
        path = Path(name)
        # 某些环境可能没有其中一张表，存在才读取。
        if path.exists():
            # 读取并按行拆分，跳过第一行表头。
            for line in path.read_text().splitlines()[1:]:
                # 按空白拆出各列，以便检查本地地址、端口和 TCP 状态。
                cells = line.split()
                # 0A 表示 LISTEN；将本地地址最后的十六进制端口转为整数再比较。
                if cells[3] == '0A' and int(cells[1].rsplit(':', 1)[1], 16) == port:
                    # 发现匹配的监听端口，就立即返回“已监听”。
                    return True
    # 两张表里都没有匹配项，说明尚未监听。
    return False


# 定义自动启动 RDMA 测试双方的主入口。
def main():
    # 创建命令行参数解析器。
    parser = argparse.ArgumentParser()
    # verbs 选择自己编译的九步示例；perftest 选择系统 ib_write_bw 工具。
    parser.add_argument('--tool', choices=('verbs', 'perftest'), default='verbs')
    # RDMA 设备名默认为空，交给实际程序选择；可填 mlx5_0、rxe0 等。
    parser.add_argument('--device', default='')
    # TCP 控制端口默认 29877；它用来交换信息，数值数据由 RDMA 路径传输。
    parser.add_argument('--port', type=int, default=29877)
    # HCA 物理/逻辑端口号默认 1，不是上面的 TCP 端口。
    parser.add_argument('--ib-port', type=int, default=1)
    # GID 是 RDMA 网络地址表里的条目；索引需按实际 RoCE/IB 环境选择。
    parser.add_argument('--gid-index', type=int, default=0)
    # -1 表示用主存；非负编号表示选择对应 GPU 的显存。
    parser.add_argument('--cuda', type=int, default=-1)
    # 解析命令行，将所有选项存入 args。
    args = parser.parse_args()
    # 从当前脚本位置找到当前实验目录。
    root = Path(__file__).resolve().parent
    # 如果选择自己的 Verbs 示例，就组合它的启动参数。
    if args.tool == 'verbs':
        # 根据是否选择 GPU，使用主存版或带 CUDA 支持的可执行文件。
        executable = root / 'build' / ('rdma_write_cuda' if args.cuda >= 0 else 'rdma_write')
        # 构造 server 命令列表，传入角色、TCP 端口和 HCA 端口；列表项各是一个参数。
        server = [str(executable), '--server', '--port', str(args.port), '--ib-port', str(args.ib_port),
                  # 接着把 GID 索引参数加进同一个命令列表。
                  '--gid-index', str(args.gid_index)]
        # 用户显式填了 RDMA 设备名时才添加设备选项。
        if args.device:
            # 把指定的 HCA 名称传给 Verbs 程序。
            server += ['--device', args.device]
        # 非负 GPU 编号表示请求直接注册显存。
        if args.cuda >= 0:
            # 把显存所在的 GPU 编号传给 CUDA 版本程序。
            server += ['--cuda', str(args.cuda)]
        # 复制 server 参数，去掉服务端角色，改成连接本机；不会修改原 server 列表。
        client = [x for x in server if x != '--server'] + ['--connect', '127.0.0.1']
    # 如果选择 perftest，就构造系统 ib_write_bw 的参数。
    else:
        # 先查 PATH 中有没有这个程序，避免启动时只得到难懂的文件不存在错误。
        if not shutil.which('ib_write_bw'):
            # 缺少 perftest 时显示安装提示并退出。
            parser.error('找不到 ib_write_bw；请安装 perftest，参见 README.md')
        # 开始拼接带宽测试命令，指定 TCP 控制端口和 HCA 端口。
        server = ['ib_write_bw', '-p', str(args.port), '-i', str(args.ib_port),
                  # -F 忽略 CPU 频率检查，带宽用 Gbit/s 显示；传 64 KiB 数据、做 100 次迭代。
                  '-F', '--report_gbits', '-n', '100', '-s', '65536']
        # 非负 GID 索引表示显式选择对应条目；-1 时不传 -x。
        if args.gid_index >= 0:
            # -x 是 perftest 选择 GID 表条目的选项。
            server += ['-x', str(args.gid_index)]
        # 如果指定了设备，就给 perftest 加上设备参数。
        if args.device:
            # -d 指定 RDMA HCA 名称，与普通以太网接口名并不一定相同。
            server += ['-d', args.device]
        # GPU 编号非负时请求 perftest 的 CUDA buffer 模式。
        if args.cuda >= 0:
            # --use_cuda 要求 perftest 自身编译了 CUDA 支持，并且硬件驱动支持显存注册。
            server += [f'--use_cuda={args.cuda}']
        # perftest 最后多一个服务端 IP 就表示 client；这里测试同一主机的回环。
        client = server + ['127.0.0.1']
    # 启动之前检查端口，避免误把其他服务当成本例 server。
    if listening(args.port):
        # 端口已占用就报错，提示通过 Makefile 的 PORT 变量更换端口。
        parser.error(f'端口 {args.port} 已被占用，请设置 PORT')
    # 指定本实验的本地输出目录。
    build = root / 'build'
    # 确保输出目录存在；已有目录也可继续使用。
    build.mkdir(exist_ok=True)
    # 为不同工具生成不同的 server 日志文件名。
    log_path = build / f'{args.tool}-server.log'
    # 先把实际将要执行的 server 命令打印出来，便于手动复查。
    print('server:', ' '.join(server), flush=True)
    # 覆盖本次 server 日志，关闭文件对象后子进程仍持有自己的输出句柄。
    with log_path.open('w') as log:
        # 异步启动 server，将输出和错误合并到日志中；主脚本接着等就绪。
        process = subprocess.Popen(server, stdout=log, stderr=subprocess.STDOUT)
    # 从这里起无论哪里失败，都由 finally 回收 server。
    try:
        # 设置 15 秒的服务端启动等待上限。
        deadline = time.monotonic() + 15
        # 反复查看监听状态，直到 server 真的开始接收控制连接。
        while not listening(args.port):
            # 如果 server 已经退出，说明启动失败或异常结束，不能继续等。
            if process.poll() is not None:
                # 先显示 server 日志中的原始错误。
                print(log_path.read_text(), end='')
                # 返回失败状态；即使 server 意外以 0 提前退出，也不能算本次测试通过。
                return process.returncode or 1
            # 还未监听且超过启动时限时，停止等待。
            if time.monotonic() > deadline:
                # 抛出启动超时；后面的 finally 仍会负责清理。
                raise TimeoutError('server 未在 15 秒内监听控制端口')
            # 每次检测间隔 50 毫秒，避免连续读取状态表占用 CPU。
            time.sleep(0.05)
        # 显示真正将要运行的 client 命令。
        print('client:', ' '.join(client), flush=True)
        # 同步执行 client，最多允许 45 秒；它完成后才继续运行下面代码。
        result = subprocess.run(client, timeout=45)
        # client 非零退出表示测试失败。
        if result.returncode:
            # 将 client 的状态码交给 Makefile，禁止把失败隐藏成成功。
            return result.returncode
        # client 成功后，再给 server 15 秒输出结果并正常退出。
        code = process.wait(timeout=15)
        # 显示 server 完整日志，便于同时看到双方结果。
        print(log_path.read_text(), end='')
        # 返回 server 状态码，只有双方都成功才是 0。
        return code
    # 成功、失败、超时或中途异常，都会进入这里回收 server。
    finally:
        # poll() 为 None 表示 server 仍在运行。
        if process.poll() is None:
            # 先尝试正常终止它，避免留下占用端口的后台进程。
            process.terminate()
            # 尝试等待 server 响应终止信号。
            try:
                # 最多等待 3 秒，控制清理阶段的总时长。
                process.wait(timeout=3)
            # 3 秒后仍未退出，就进入强制结束流程。
            except subprocess.TimeoutExpired:
                # 强制杀死 server 子进程。
                process.kill()
                # 回收退出状态，避免残留僵尸进程。
                process.wait()


# 只有直接执行脚本时才开始启动双方。
if __name__ == '__main__':
    # 把 main 的结果转为脚本的系统退出码，便于 Makefile 判断是否成功。
    raise SystemExit(main())
