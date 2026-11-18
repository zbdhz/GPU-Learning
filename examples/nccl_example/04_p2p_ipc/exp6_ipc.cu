// holder 在 GPU 1 分配显存；writer 在 GPU 0 导入 IPC 句柄并直写。
// 导入 CUDA 错误检查和 GPU 0→1 的 P2P 能力查询。
#include "cuda_check.h"
// 引入 IP 地址结构和 htons/htonl 等主机端序与网络端序转换函数。
#include <arpa/inet.h>
// 引入 errno、EINTR 等系统错误信息，用于判断 socket 调用失败原因。
#include <cerrno>
// 引入 printf、fprintf 和 fflush 等输出函数。
#include <cstdio>
// 引入 strcmp、strerror，分别用于比较角色字符串和解释错误码。
#include <cstring>
// 引入创建、监听、连接和收发 TCP socket 的接口。
#include <sys/socket.h>
// 引入 close、getpid 等 Linux 进程和文件描述符接口。
#include <unistd.h>

// 用一个 C++ 结构包装 socket；对象离开作用域时自动关 fd，这种管理方式叫 RAII。
struct Socket {
  // fd 是操作系统给连接或监听 socket 的整数编号，不是内存地址。
  int fd;
  // 构造时接管传入的 fd；explicit 禁止把普通整数悄悄当作 Socket 对象。
  explicit Socket(int value) : fd(value) {
    // fd 小于 0 表示创建或接受连接失败，此时抛出系统错误。
    if (fd < 0) throw std::runtime_error(std::strerror(errno));
  }
  // 析构函数在对象销毁时执行，自动关闭这个 socket，包含异常退出的情况。
  ~Socket() { close(fd); }
  // 禁止复制构造，避免两个对象同时认为自己负责关闭同一个 fd。
  Socket(const Socket&) = delete;
  // 禁止复制赋值，原因同样是防止一个 fd 被重复关闭。
  Socket& operator=(const Socket&) = delete;
};

// send 和 recv 均可能短读写；对断开和 EINTR 显式处理。
// 完整收发 size 字节；sending=true 为发送，false 为接收；static 表示函数只在本源文件使用。
static void transfer(int fd, void* data, size_t size, bool sending) {
  // 把无类型的 void* 转为字节指针 char*，便于按已传字节数移动地址。
  auto* p = static_cast<char*>(data);
  // 只要还有字节没处理完，就继续收发；TCP 一次调用不保证处理完整条消息。
  while (size) {
    // 按方向调用 send 或 recv；MSG_NOSIGNAL 防止连接断开时由 SIGPIPE 直接终止进程。
    ssize_t n = sending ? send(fd, p, size, MSG_NOSIGNAL) : recv(fd, p, size, 0);
    // 若调用被信号打断，则重试；EINTR 不代表对方已经断开。
    if (n < 0 && errno == EINTR) continue;
    // 返回 0 通常表示对方已关闭，负值表示错误；这里统一抛异常停止收发。
    if (n <= 0) throw std::runtime_error("IPC 控制连接中断或超时");
    // 把指针前移 n 字节，下次从尚未处理的位置继续。
    p += n;
    // 从剩余字节数扣掉本次成功收发的部分；cast 将有符号计数转成 size_t。
    size -= static_cast<size_t>(n);
  }
}
// 给一个已建立的连接设置收发等待上限。
static void timeout(int fd) {
  // timeval 的两个字段是秒和微秒，这里表示 20 秒。
  timeval tv{20, 0};
  // 先给 recv 设置超时；setsockopt 非零表示设置失败。
  if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) ||
      // 再给 send 设置超时；这里只限制单次收发等待，不是整个实验总时长。
      setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)))
    // 任意一项设置失败就报错，不能假设超时机制已经生效。
    throw std::runtime_error(std::strerror(errno));
}
// 在 GPU 上执行的 kernel；一个线程就能把导入的对端显存第一个 int 改成 42。
__global__ void write_peer(int* peer) { peer[0] = 42; }

// 定义显存持有方；它负责在 GPU 1 分配、导出句柄、接收完成通知和验证结果。
static void holder(int port) {
  // 先初始化空指针，异常路径可据此判断是否需要释放 allocation。
  int* buffer = nullptr;
  // 保护分配后的显存，保证出现异常时仍能进入清理。
  try {
    // 持有方使用 GPU 1，此后的分配和清零都在该卡进行。
    CUDA_CHECK(cudaSetDevice(1));
    // 申请一个 int 的显存；buffer 得到的是本进程能够使用的 GPU 地址。
    CUDA_CHECK(cudaMalloc(&buffer, sizeof(int)));
    // 将显存初始化为零，后面读到 42 才能说明 writer 真正修改过它。
    CUDA_CHECK(cudaMemset(buffer, 0, sizeof(int)));
    // 等待清零完成，再让另一进程导入，避免异步初始化与直写发生竞争。
    CUDA_CHECK(cudaDeviceSynchronize());
    // 创建 IPC handle 结构；句柄是 CUDA 的不透明凭据，不能当作数据指针解引用。
    cudaIpcMemHandle_t handle{};
    // 从 allocation 生成可传给同机其他进程的句柄；不是直接把 buffer 指针发过去。
    CUDA_CHECK(cudaIpcGetMemHandle(&handle, buffer));
    // 创建 IPv4 TCP socket，由 Socket 包装来自动关闭；当前只是监听端的准备阶段。
    Socket server(socket(AF_INET, SOCK_STREAM, 0));
    // 选项值 1 表示启用下面设置的 SO_REUSEADDR。
    int yes = 1;
    // 允许重启时复用合适的旧地址状态；不代表可以抢占其他仍在监听的服务。
    if (setsockopt(server.fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)))
      // 如果选项设置失败，将系统错误抛出。
      throw std::runtime_error(std::strerror(errno));
    // 创建并清零 IPv4 地址结构，准备填写监听地址。
    sockaddr_in addr{};
    // AF_INET 表示这是 IPv4 地址。
    addr.sin_family = AF_INET;
    // 把主机格式的端口号转为网络使用的大端字节序。
    addr.sin_port = htons(port);
    // 只绑定 127.0.0.1 回环地址；CUDA IPC 本来就只在同一主机可用。
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    // 把 socket 绑定到指定地址并开始监听，待接收队列长度设为 1。
    if (bind(server.fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) || listen(server.fd, 1))
      // bind 或 listen 失败就报错，常见原因是端口已被其他进程占用。
      throw std::runtime_error(std::strerror(errno));
    // 输出进程号、端口和句柄字节数；READY 供 Python 启动器判断监听已就绪。
    std::printf("holder pid=%d READY port=%d handle_bytes=%zu\n", getpid(), port, sizeof(handle));
    // 立即刷新输出，确保重定向到日志时启动器也能马上读到 READY。
    std::fflush(stdout);
    // 阻塞等待 writer 接入，得到独立的连接 fd；监听 socket 本身仍由 server 管理。
    Socket conn(accept(server.fd, nullptr, nullptr));
    // 为这条连接设置 20 秒收发超时。
    timeout(conn.fd);
    // 通过 TCP 发出整个 IPC 句柄；发送的是访问凭据，不是显存内的值 42。
    transfer(conn.fd, &handle, sizeof(handle), true);
    // 预留一个字节保存 writer 的确认消息。
    char ack = 0;
    // 等待确认；协议约定 writer 必须先同步 kernel 并关闭导入映射，再发这个字节。
    transfer(conn.fd, &ack, 1, false); // writer 已同步 kernel 并关闭导入映射。
    // 只接受约定的 K 确认，不把任意一个字节都视为写入完成。
    if (ack != 'K') throw std::runtime_error("writer 未确认完成");
    // 定义 CPU 回读变量，初值 -1 方便排查未写入的情况。
    int value = -1;
    // 把自己 GPU 1 上的 buffer 回读到 CPU；检查真实显存内容而不是 TCP 消息。
    CUDA_CHECK(cudaMemcpy(&value, buffer, sizeof(value), cudaMemcpyDeviceToHost));
    // 验证远程 kernel 写入的数是否为 42。
    if (value != 42) throw std::runtime_error("IPC 写入结果不等于 42");
    // writer 已确认关闭映射，holder 才能安全释放原始 allocation。
    CUDA_CHECK(cudaFree(buffer));
    // 清空指针，防止成功释放后异常路径再次释放它。
    buffer = nullptr;
    // 输出持有方通过标记及实际观察到的数。
    std::puts("holder PASS: 导出显存回读 value=42");
  // 接住任意类型的异常，在传播错误前先处理显存。
  } catch (...) {
    // 如果 allocation 还活着，就在当前 GPU 1 上尽力释放。
    if (buffer) cudaFree(buffer);
    // 重新抛出原异常，交给 main 统一报告失败。
    throw;
  }
}

// 定义写入方；它从 holder 接收句柄，并在 GPU 0 上执行跨卡 kernel。
static void writer(int port) {
  // 创建主动连接用的 IPv4 TCP socket，同样由 RAII 自动关闭。
  Socket conn(socket(AF_INET, SOCK_STREAM, 0));
  // 为控制通道设置收发超时，避免对端失联后一直等。
  timeout(conn.fd);
  // 创建空的服务端地址结构。
  sockaddr_in addr{};
  // 服务端地址使用 IPv4 格式。
  addr.sin_family = AF_INET;
  // 把服务端监听端口转为网络字节序。
  addr.sin_port = htons(port);
  // 服务端固定为本机回环地址，与 holder 的监听地址一致。
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  // 主动连接 holder；返回非零表示失败。
  if (connect(conn.fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)))
    // 连接失败时给出启动顺序提示，应该先启动 holder。
    throw std::runtime_error("connect 失败，请先启动 holder");
  // 分配句柄结构，准备保存从另一进程传过来的原始句柄字节。
  cudaIpcMemHandle_t handle{};
  // 接收完整句柄；TCP 短读已经由 transfer 的循环处理。
  transfer(conn.fd, &handle, sizeof(handle), false);
  // 写入方选择 GPU 0，后面的 kernel 将在这张卡执行。
  CUDA_CHECK(cudaSetDevice(0));
  // 初始化导入后的本地指针；它可能与 holder 里的 buffer 数值不同。
  int* peer = nullptr;
  // 导入句柄得到本进程可用的地址；LazyEnablePeerAccess 允许 CUDA 按需要建立 peer 映射。
  CUDA_CHECK(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&peer), handle, cudaIpcMemLazyEnablePeerAccess));
  // 导入成功后进入保护区，保证后续失败也会关闭导入映射。
  try {
    // 在 GPU 0 上启动 1 block×1 thread，写的是 holder 的 GPU 1 allocation。
    write_peer<<<1, 1>>>(peer);
    // 检查 kernel 的提交错误。
    CUDA_CHECK(cudaGetLastError());
    // 等待 GPU 0 的写入真正结束，再通知对方；仅提交成功不足以证明数据写完。
    CUDA_CHECK(cudaDeviceSynchronize());
    // 关闭当前进程的导入映射；这不是释放 holder 的原始显存。
    CUDA_CHECK(cudaIpcCloseMemHandle(peer)); // 必须先关闭，再让 holder 释放 allocation。
    // 清空指针，记录本进程已不再持有该映射。
    peer = nullptr;
    // 构造协议约定的单字节完成确认 K。
    char ack = 'K';
    // 把确认发给 holder，允许它回读并释放原始 allocation。
    transfer(conn.fd, &ack, 1, true);
    // 输出 writer 完成标记；holder 还会独立检查显存中的真实数值。
    std::puts("writer PASS: GPU 0 kernel 已写入 holder 的 GPU 1 显存");
  // 如果 kernel、同步或控制通信失败，进入映射清理分支。
  } catch (...) {
    // 若导入映射还没有关闭，就尽力关闭，避免本进程遗漏映射资源。
    if (peer) cudaIpcCloseMemHandle(peer);
    // 把原始异常继续交给 main 打印。
    throw;
  }
}

// 程序入口；argc 是参数个数，argv 是各参数字符串，argv[0] 是程序名。
int main(int argc, char** argv) {
  // 在最外层统一处理参数、CUDA 和网络操作抛出的异常。
  try {
    // 只接受 holder 或 writer 角色，外加可选端口；strcmp 返回 0 才表示字符串相等。
    if (argc < 2 || argc > 3 || (std::strcmp(argv[1], "holder") && std::strcmp(argv[1], "writer")))
      // 参数格式不对就抛出可直接照做的用法说明。
      throw std::runtime_error("用法: exp6_ipc holder|writer [port=29777]");
    // 有第三个参数就将其解析成整数端口，否则使用 29777。
    int port = argc == 3 ? std::stoi(argv[2]) : 29777;
    // 检查端口范围；0 不能作为本例两方约定的固定连接端口。
    if (port < 1 || port > 65535) throw std::runtime_error("端口范围为 1..65535");
    // 两种角色都先检查同机双卡和 GPU 0→1 的 P2P 能力。
    if (!has_peer_path()) {
      // 无 P2P 时明确说明跨卡 IPC 不能创造硬件或驱动原本没有的直写能力。
      std::puts("SKIP: GPU 0 -> GPU 1 无 P2P 路径，不能跨卡导入并直写；IPC 不会创造 P2P 能力。");
      // 按约定返回 77，自动启动脚本和 Makefile 会据此显示 SKIP。
      return 77;
    }
    // 根据角色进入对应函数；holder 与 writer 是两个独立进程运行的同一程序。
    if (!std::strcmp(argv[1], "holder")) holder(port); else writer(port);
    // 对应角色执行并校验成功后，以 0 退出。
    return 0;
  // 捕获标准异常，统一转换为可读错误和非零退出码。
  } catch (const std::exception& e) {
    // e.what() 提取具体错误文字，输出到标准错误流。
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    // 以 1 退出，告知 Makefile 本次实验失败。
    return 1;
  }
}
