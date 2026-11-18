"""启动 holder，等待 READY 后再启动 writer；统一超时和回收子进程。"""
# argparse 解析命令行的 --port 选项。
import argparse
# Path 负责寻找可执行文件和读取 holder 日志。
from pathlib import Path
# subprocess 用来启动并管理两个独立的操作系统进程。
import subprocess
# time 提供单调计时和短暂休眠，用来等待服务就绪。
import time


# 定义自动运行 IPC 双进程实验的入口。
def main():
    # 创建命令行参数解析器。
    parser = argparse.ArgumentParser()
    # 允许指定 TCP 控制端口；这里用整数，默认端口为 29777。
    parser.add_argument('--port', type=int, default=29777)
    # 把实际传入的命令行选项解析到 args。
    args = parser.parse_args()
    # 定位当前脚本旁的 build 目录，里面应已有编译好的 CUDA 程序。
    build = Path(__file__).resolve().parent / 'build'
    # 把 exp6_ipc 可执行文件的路径转为子进程启动接口使用的字符串。
    binary = str(build / 'exp6_ipc')
    # 指定 holder 的日志文件，方便启动器判断它是否已就绪。
    log_path = build / 'holder.log'
    # 以写入方式打开日志，每次运行会覆盖上次内容；退出 with 时关闭当前文件对象。
    with log_path.open('w') as log:
        # 异步启动 holder，把标准输出和错误输出合并写入日志；Popen 不等它结束。
        holder = subprocess.Popen([binary, 'holder', str(args.port)], stdout=log, stderr=subprocess.STDOUT)
    # 从这里开始保证，即使 writer 失败，也会清理仍在运行的 holder。
    try:
        # 以单调时钟计算 30 秒后的截止时刻，系统时钟调整不会影响这次等待。
        deadline = time.monotonic() + 30
        # 轮询日志中的 READY 标记；它意味着 holder 已经开始监听控制端口。
        while ' READY ' not in log_path.read_text():
            # poll() 非空表示 holder 已退出，此时不能继续傻等 READY。
            if holder.poll() is not None:
                # 先显示 holder 的实际错误信息，方便定位 CUDA 或端口问题。
                print(log_path.read_text(), end='')
                # 把 holder 的退出码原样交给调用者；例如 77 表示缺少 P2P 条件。
                return holder.returncode
            # 如果超过截止时间仍未就绪，就终止本次等待。
            if time.monotonic() > deadline:
                # 抛出超时异常，随后 finally 会负责回收 holder。
                raise TimeoutError('等待 holder READY 超时')
            # 每次轮询间隔 50 毫秒，避免忙循环占满 CPU。
            time.sleep(0.05)
        # 同步运行 writer，最多等 30 秒；它会连接 holder、导入句柄并执行 kernel。
        writer = subprocess.run([binary, 'writer', str(args.port)], timeout=30)
        # 子进程返回非零值表示失败，不能把该次实验报告为成功。
        if writer.returncode:
            # 把 writer 的失败状态返回给 Makefile，随后仍会执行 finally。
            return writer.returncode
        # writer 正常完成后，最多再等 holder 30 秒，让它回读并校验显存。
        result = holder.wait(timeout=30)
        # 把 holder 的完整日志打印到终端，显示最终读到了什么。
        print(log_path.read_text(), end='')
        # 返回 holder 的状态码；为 0 才表示整个配合流程成功。
        return result
    # 无论正常返回还是抛异常，都执行子进程清理。
    finally:
        # 如果 holder 还没有结束，就需要主动停止它。
        if holder.poll() is None:
            # 先发正常终止信号，给 holder 退出机会。
            holder.terminate()
            # 尝试等待终止后的进程被回收。
            try:
                # 最多等 3 秒，避免清理阶段再次无限等待。
                holder.wait(timeout=3)
            # 若温和终止没有生效，就进入强制清理分支。
            except subprocess.TimeoutExpired:
                # 发送强制结束信号，防止留下后台进程。
                holder.kill()
                # 最后 wait 回收进程，避免留下僵尸进程。
                holder.wait()


# 只有直接运行这个脚本时才自动启动双进程。
if __name__ == '__main__':
    # 把 main 返回值作为脚本退出码，让 Makefile 能判断成功、失败或跳过。
    raise SystemExit(main())
