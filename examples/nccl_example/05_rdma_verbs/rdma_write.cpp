// 将文中的 verbs 九步骨架补成可运行的 RC RDMA WRITE。
// 引入 libibverbs 的设备、保护域、队列、内存注册等 RDMA 接口。
#include <infiniband/verbs.h>
// 引入网络字节序转换以及 IP 地址文本和二进制之间的转换函数。
#include <arpa/inet.h>
// 引入 getaddrinfo，把服务端名称或 IP 解析成可连接的地址。
#include <netdb.h>
// 引入 poll，等待监听 socket 就绪时可以设置毫秒级超时。
#include <poll.h>
// 引入 TCP socket 的创建、绑定、连接、发送和接收接口。
#include <sys/socket.h>
// 引入 close，用来关闭 Linux 文件描述符。
#include <unistd.h>
// 引入 std::min，后面协商双方都能支持的最大包长。
#include <algorithm>
// 引入 errno 和 EINTR，便于解释系统错误并处理被信号打断的调用。
#include <cerrno>
// 引入单调时钟和时间长度，限制等待 RDMA 完成回执的时长。
#include <chrono>
// 引入 uint32_t、uint64_t 等固定宽度整数，便于描述协议字段和地址。
#include <cstdint>
// 引入 printf、fprintf、puts 等终端输出接口。
#include <cstdio>
// 引入主存分配与释放函数，例如 posix_memalign 和 free。
#include <cstdlib>
// 引入 memcpy、memset、strerror，分别处理字节复制、清零和错误描述。
#include <cstring>
// 引入字符串流，将交换信息编码成文本，再从文本解析回来。
#include <sstream>
// 引入标准异常，遇到错误时终止当前实验并进入清理流程。
#include <stdexcept>
// 引入 std::string，管理参数、主机名和错误消息。
#include <string>
// 引入线程调度接口；轮询 CQ 时可用 yield 暂时让出 CPU。
#include <thread>
// 只有编译时定义 USE_CUDA，才编译下面这段 CUDA 支持代码。
#ifdef USE_CUDA
// 引入 CUDA 显存分配和拷贝接口；主存版本不会依赖这份头文件。
#include <cuda_runtime.h>
// 定义 CUDA 返回码检查函数，避免每次调用都手写错误判断。
static void cu(cudaError_t e) {
  // CUDA 调用失败时把错误码转成可读说明并抛出异常。
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
// 结束 USE_CUDA 条件编译区，后面的通用 RDMA 代码两个版本都会使用。
#endif
// 定义通用条件检查器；ok 为真才继续，what 说明失败的是哪个步骤。
static void require(bool ok, const char* what) {
  // 条件不满足时抛异常；若 errno 非零，再附上操作系统给出的错误描述。
  if (!ok) throw std::runtime_error(std::string(what) + (errno ? std::string(": ") + std::strerror(errno) : ""));
}
// 用结构体把命令行配置放在一起，方便传递给实验函数。
struct Options {
  // 默认启动为 server；通过 --connect 才改成 client。
  bool server = true;
  // host 是 server 的 IP/名称，device 是 RDMA HCA 名称，例如 mlx5_0 或 rxe0。
  std::string host, device;
  // port 是 TCP 控制端口；ib_port 是 HCA 端口；gid_index 选择地址条目；gpu=-1 表示主存。
  int port = 29877, ib_port = 1, gid_index = 0, gpu = -1;
};
// 集中保存实验申请的资源；对象离开作用域时由析构函数统一释放。
struct Resources {
  // ctx 表示已经打开的 RDMA 设备上下文；nullptr 表示尚未成功打开。
  ibv_context* ctx = nullptr;
  // PD 是保护域，给队列与注册内存划定共同的资源归属范围。
  ibv_pd* pd = nullptr;
  // CQ 是完成队列，保存网卡执行请求后返回的完成记录。
  ibv_cq* cq = nullptr;
  // QP 是队列对，包含发送队列和接收队列；本例使用可靠连接 RC 类型。
  ibv_qp* qp = nullptr;
  // MR 是注册内存的描述对象，包含 lkey、rkey 等访问所需信息。
  ibv_mr* mr = nullptr;
  // buf 指向实际数据空间，可能在 CPU 主存，也可能在 GPU 显存。
  void* buf = nullptr;
  // fd 保存 TCP 控制连接的编号；-1 表示还没有连接可关闭。
  int fd = -1;
  // 记录 buf 是否来自 cudaMalloc，释放时必须选择匹配的接口。
  bool gpu = false;
  // 析构函数自动运行，包括抛异常退出 experiment 的情况；这是 C++ 的 RAII 管理。
  ~Resources() {
    // 有 TCP 连接时先关闭它，结束控制信息交换。
    if (fd >= 0) close(fd);
    // 先销毁 QP，避免它继续引用后面要释放的 CQ、MR 或 PD。
    if (qp) ibv_destroy_qp(qp);
    // 注销 MR，让网卡不再把这块地址当作已登记的可访问内存；此时还没释放 buf。
    if (mr) ibv_dereg_mr(mr);
    // 销毁 CQ，释放完成记录队列。
    if (cq) ibv_destroy_cq(cq);
    // 其他资源处理完后释放保护域。
    if (pd) ibv_dealloc_pd(pd);
    // 最后关闭 RDMA 设备上下文。
    if (ctx) ibv_close_device(ctx);
// CUDA 版本需要为显存选择专门的释放方式。
#ifdef USE_CUDA
    // 若 buf 来自 GPU，就调用 cudaFree；else 与下方 std::free 配对处理主存。
    if (gpu && buf) cudaFree(buf); else
// 结束条件编译；主存版本只保留下面的 free 语句。
#endif
    // 释放 CPU 主存；free(nullptr) 也是安全操作，不会要求实际分配成功。
    std::free(buf);
  }
};
// 完整收发 bytes 字节；sending 选择 send 或 recv，fd 是已建立的 TCP 连接。
static void io(int fd, void* data, size_t bytes, bool sending) {
  // 把无类型地址转成 char*，使每次移动指针的单位为一个字节。
  auto* p = static_cast<char*>(data);
  // 只要尚有字节未传完就继续；TCP 不保证一次 send/recv 就处理全部内容。
  while (bytes) {
    // 按方向执行收发；MSG_NOSIGNAL 防止写入断开的连接时触发 SIGPIPE 直接杀死进程。
    ssize_t n = sending ? send(fd, p, bytes, MSG_NOSIGNAL) : recv(fd, p, bytes, 0);
    // EINTR 表示操作被信号打断；重试即可，不应立刻当作断开。
    if (n < 0 && errno == EINTR) continue;
    // 非正数表示连接关闭、出错或等待超时，抛出异常停止传输。
    require(n > 0, "TCP 控制通道断开或超时");
    // 指针跳过本次已成功处理的 n 字节。
    p += n;
    // 剩余长度相应减少；size_t 是用于长度的无符号整数类型。
    bytes -= static_cast<size_t>(n);
  }
}
// 建立交换配置用的 TCP 慢通道；const& 表示只读引用传参，不复制 Options。
static int control(const Options& o) {
  // 创建 IPv4、流式 TCP socket；此时它还没有绑定或连接。
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  // socket 返回负值表示失败，立即停止。
  require(fd >= 0, "socket");
  // 准备 15 秒、0 微秒的收发超时配置。
  timeval timeout{15, 0};
  // 尝试限制 recv 的等待时间；注意原示例在这里没有检查 setsockopt 的返回码。
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  // 同样尝试限制 send 的等待；server 的 accept 等待另由后面的 poll 控制。
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
  // 保护已创建的 fd，确保后续步骤抛异常时会关闭它。
  try {
    // server 负责监听，等待 client 主动连接进来。
    if (o.server) {
      // 准备选项值 1，表示打开地址复用选项。
      int yes = 1;
      // 设置 SO_REUSEADDR，便于服务结束后重启；不能用它抢占别的监听服务。
      require(!setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)), "setsockopt");
      // 声明并清零一个 IPv4 地址结构。
      sockaddr_in addr{};
      // 设置地址族为 IPv4。
      addr.sin_family = AF_INET;
      // 把用户填写的 TCP 端口转成网络字节序。
      addr.sin_port = htons(o.port);
      // 监听所有本机 IPv4 地址，使另一台主机也可以接入；不是只监听回环。
      addr.sin_addr.s_addr = htonl(INADDR_ANY);
      // 将 socket 绑定到指定地址和端口，失败通常与权限或端口占用有关。
      require(!bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), "bind");
      // 开始监听连接，等待队列长度设为 1。
      require(!listen(fd, 1), "listen");
      // 输出 READY 标记和端口，方便人或自动脚本判断 server 已就绪。
      std::printf("READY TCP port=%d\n", o.port);
      // 立即刷新输出，避免重定向到文件时 READY 一直留在输出缓冲区。
      std::fflush(stdout);
      // 构造 poll 的监视项：监听 fd 是否有可接受的新连接。
      pollfd pfd{fd, POLLIN, 0};
      // 最多等 30000 毫秒，也就是 30 秒；未就绪则报告等待失败。
      require(poll(&pfd, 1, 30000) > 0, "等待 client 超时");
      // 接受 client，得到专用于双方收发的 conn；它不同于原来的监听 fd。
      int conn = accept(fd, nullptr, nullptr);
      // 检查 accept 是否成功。
      require(conn >= 0, "accept");
      // 本例只服务一个 client，所以接受连接后关闭监听 socket。
      close(fd);
      // 将 fd 改为已建立的连接编号，之后函数将它交给 experiment 使用。
      fd = conn;
      // 给新连接也设置接收超时；这里同样只是尝试设置，未检查返回码。
      setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
      // 给新连接设置发送超时，使后续控制信息收发不会无条件永远等。
      setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    // 不是 server 时，进入 client 的主动连接路径。
    } else {
      // hints 用来限制地址解析类型；addresses 接收系统分配的地址结果。
      addrinfo hints{}, *addresses = nullptr;
      // 只解析 IPv4 地址，与 server 的监听类型一致。
      hints.ai_family = AF_INET;
      // 要求返回适用于 TCP 流式 socket 的地址。
      hints.ai_socktype = SOCK_STREAM;
      // 把 host 和十进制端口字符串交给系统解析，结果写入 addresses。
      int e = getaddrinfo(o.host.c_str(), std::to_string(o.port).c_str(), &hints, &addresses);
      // getaddrinfo 有独立错误码；用 gai_strerror 生成可读说明。
      if (e) throw std::runtime_error(gai_strerror(e));
      // 向解析结果中的第一个地址主动发起连接；这里没有遍历备用地址。
      int result = connect(fd, addresses->ai_addr, addresses->ai_addrlen);
      // 连接尝试结束后释放地址解析结果，防止主存泄漏。
      freeaddrinfo(addresses);
      // connect 返回 0 才表示连接成功，失败时抛出错误。
      require(!result, "connect");
    }
    // 返回已建立的 TCP 连接 fd，外层 Resources 会负责最终关闭。
    return fd;
  // 任何阶段失败都关闭当前 fd，再把原始异常继续向上传递。
  } catch (...) { close(fd); throw; }
}
// 使用文本编码交换地址，避免结构体 padding、主机端序和指针宽度的差异。
// Peer 保存对端执行 RDMA 所需的地址和“钥匙”；同一结构也可描述自己。
struct Peer {
  // addr 是已注册 buffer 的起始地址，以 64 位整数传输；它不是要通过 TCP 发送的数据内容。
  uint64_t addr = 0;
  // rkey 是远端访问这块 MR 所需的凭据；qpn 是对端 QP 的编号。
  uint32_t rkey = 0, qpn = 0;
  // lid 是 IB 本地标识；mtu 是最大包长的枚举编码，并非直接用字节数表示。
  int lid = 0, mtu = 0;
  // gid 是 128 位全局标识，RoCE 路径需要用它描述网络目的地址。
  ibv_gid gid{};
};
// 通过 TCP 交换双方元信息；self 是本端配置，返回值是解析好的对端配置。
static Peer exchange(int fd, const Peer& self) {
  // 为 GID 的 IPv6 文本表示预留足够字符，并初始化为空。
  char gid[INET6_ADDRSTRLEN]{};
  // 将二进制 GID 编码成标准文本，仅借用 IPv6 表示格式，不表示数据走 TCP/IPv6。
  require(inet_ntop(AF_INET6, &self.gid, gid, sizeof(gid)) != nullptr, "inet_ntop");
  // 创建输出字符串流，下面将把各字段依次写成一行文本。
  std::ostringstream os;
  // 先编码本端 buffer 地址、rkey 和 QP 编号，字段间用空格隔开。
  os << self.addr << ' ' << self.rkey << ' ' << self.qpn << ' '
     // 再编码 LID、MTU 和 GID，使对端能配置自己的到本端连接。
     << self.lid << ' ' << self.mtu << ' ' << gid;
  // 双方协议固定交换 256 字节；两个数组分别存本端发送文本和对端接收文本。
  char local[256]{}, remote[256]{};
  // 先确认编码后文本放得下，避免越过数组边界。
  require(os.str().size() < sizeof(local), "metadata 长度");
  // 将字符串内容拷入本端固定长度数组，剩余位置保持为 0。
  std::memcpy(local, os.str().data(), os.str().size());
  // 先发送完整的 256 字节元信息；这里是小消息，双方随后都会接收。
  io(fd, local, sizeof(local), true);
  // 再接收对端固定长度的 256 字节元信息。
  io(fd, remote, sizeof(remote), false);
  // 强制最后一个字节为字符串终止符，防止解析时读到数组外。
  remote[sizeof(remote) - 1] = 0;
  // 创建存放对端地址和凭据的结果对象。
  Peer peer;
  // 先用普通字符串保存对端 GID 的文本表示。
  std::string remote_gid;
  // 创建输入字符串流，准备按原先字段顺序解析对端消息。
  std::istringstream is(remote);
  // 依次解析地址、rkey、qpn、lid、mtu、gid；任意字段解析失败就报错。
  require(bool(is >> peer.addr >> peer.rkey >> peer.qpn >> peer.lid >> peer.mtu >> remote_gid), "metadata 解析");
  // 将 GID 文本转回二进制，放入 peer.gid，供 QP 路由配置使用。
  require(inet_pton(AF_INET6, remote_gid.c_str(), &peer.gid) == 1, "GID 解析");
  // 检查 MTU 枚举是否处于接口允许范围，避免将任意对端数值当成配置。
  require(peer.mtu >= IBV_MTU_256 && peer.mtu <= IBV_MTU_4096, "对端 MTU");
  // 把已解析的对端元信息返回给调用者。
  return peer;
}
// 通过 TCP 做一次轻量双方就绪握手，确保两边 QP 都已完成配置。
static void barrier(int fd) {
  // 本端准备发字符 R，同时预留一个字节接收对端字符。
  char ready = 'R', peer = 0;
  // 告诉对方“我的 QP 已准备好”。
  io(fd, &ready, 1, true);
  // 等待对方也确认准备好，再进入实际 RDMA 数据操作。
  io(fd, &peer, 1, false);
  // 检查收到的是协议约定的 R，而不是其他错误或杂乱消息。
  require(peer == 'R', "QP ready 握手");
}
// 完整实验入口：先申请资源和换钥匙，再配置 QP、写远端内存、检查结果。
static void experiment(const Options& o) {
  // 把资源对象放在栈上，函数结束或抛异常时都会触发统一清理。
  Resources r;
  // 预留变量存系统发现的 RDMA 设备数量。
  int count = 0;
  // 步骤①：枚举 RDMA 设备，列表和数量由 libibverbs 返回；这里还未打开某张卡。
  ibv_device** devices = ibv_get_device_list(&count);                  // ① 打开网卡
  // 没有设备列表或设备数为零时，无法执行真实 Verbs 实验。
  if (!devices || !count) {
    // 即便数量为零，系统也可能返回了列表对象，有就先释放。
    if (devices) ibv_free_device_list(devices);
    // 给出硬件或 RXE 配置要求，不把普通 TCP 路径伪装成 RDMA 成功。
    throw std::runtime_error("没有可用 RDMA 设备（需要真实 HCA 或已配置的 RXE）");
  }
  // 遍历系统返回的每个 RDMA 设备。
  for (int i = 0; i < count; ++i)
    // 未指定名字时选第一个；指定名字时只选与用户输入相同的 HCA。
    if (o.device.empty() || o.device == ibv_get_device_name(devices[i])) {
      // 真正打开匹配的设备，得到后续资源申请所需的设备上下文。
      r.ctx = ibv_open_device(devices[i]);
      // 找到一次就停止寻找；打开失败也会在后面的 require 报错，不尝试其他设备。
      break;
    }
  // 释放枚举列表；已打开的设备上下文并不会因此被关闭。
  ibv_free_device_list(devices);
  // 确认设备成功打开；否则可能是设备名错误或设备权限不足。
  require(r.ctx != nullptr, "ibv_open_device / 设备名");
  // 步骤②：分配保护域，后续 QP 和 MR 都放在这个 PD 中。
  r.pd = ibv_alloc_pd(r.ctx);                                         // ② 保护域
  // 检查保护域是否分配成功。
  require(r.pd != nullptr, "ibv_alloc_pd");
  // 步骤③：创建可容纳 8 个完成条目的 CQ；不使用异步通知通道，而是主动轮询。
  r.cq = ibv_create_cq(r.ctx, 8, nullptr, nullptr, 0);                   // ③ 完成队列
  // 确认完成队列创建成功。
  require(r.cq != nullptr, "ibv_create_cq");
  // 准备 QP 创建参数，并将所有字段先初始化为 0。
  ibv_qp_init_attr init{};                                           // ④ RC 队列对
  // 选择 RC（Reliable Connected，可靠连接）类型，一条 QP 连接一个明确的对端。
  init.qp_type = IBV_QPT_RC;
  // 发送和接收完成都使用同一个 CQ；本例只投递一次 RDMA WRITE。
  init.send_cq = init.recv_cq = r.cq;
  // 给发送、接收队列各申请容纳 2 个工作请求的容量，够本例使用。
  init.cap.max_send_wr = init.cap.max_recv_wr = 2;
  // 每个请求只允许一个本地连续数据段；不需要把多个分散片段拼接。
  init.cap.max_send_sge = init.cap.max_recv_sge = 1;
  // 步骤④：按上述配置在保护域中创建 QP。
  r.qp = ibv_create_qp(r.pd, &init);
  // 检查 QP 是否创建成功。
  require(r.qp != nullptr, "ibv_create_qp");
// 只有 CUDA 构建才包含下面的显存分配分支。
#ifdef USE_CUDA
  // gpu 编号非负表示用户要求注册指定 GPU 的显存。
  if (o.gpu >= 0) {
    // 设置当前进程使用的 GPU 编号。
    cu(cudaSetDevice(o.gpu));
    // 分配 4096 字节显存，地址写入资源对象的 buf 指针。
    cu(cudaMalloc(&r.buf, 4096));
    // 记录 allocation 来自 GPU，析构时才知道应调用 cudaFree。
    r.gpu = true;
    // 将这块显存清零，防止后面校验读到未初始化内容。
    cu(cudaMemset(r.buf, 0, 4096));
    // 等待清零完成，避免后面的外部 DMA 写入与初始化操作发生竞争。
    cu(cudaDeviceSynchronize());
  // CUDA 版本中未选择 GPU 时进入下方主存分支。
  } else
// 结束条件编译；不带 CUDA 的构建会直接保留下面的主存代码块。
#endif
  // 开始主存分配代码块；它也接在上面的 else 后面。
  {
    // 分配 4 KiB 主存，并让起始地址按 4 KiB 对齐，地址通过 &r.buf 写回。
    require(!posix_memalign(&r.buf, 4096, 4096), "posix_memalign");
    // 用 CPU 将整个主存 buffer 初始化为 0。
    std::memset(r.buf, 0, 4096);
  }
  // 步骤⑤：把 buffer 地址及长度注册给网卡；MR 不等于另分配一份数据副本。
  r.mr = ibv_reg_mr(r.pd, r.buf, 4096,                               // ⑤ 注册主存或显存
                   // 允许本地设备写入以及对端 RDMA WRITE；权限按位或组合，buf 仍是原来那块内存。
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  // 注册失败就报告错误；主存要检查锁页额度，显存还要检查 GPUDirect 支持。
  require(r.mr != nullptr, "ibv_reg_mr（检查 memlock / GPUDirect 支持）");
  // 创建端口属性结构，等待驱动填写端口状态和地址信息。
  ibv_port_attr port{};
  // 读取指定 HCA 端口的属性，o.ib_port 不是 TCP 的端口号。
  require(!ibv_query_port(r.ctx, o.ib_port, &port), "ibv_query_port");
  // 只有端口处于 ACTIVE 状态才继续进行数据通信。
  require(port.state == IBV_PORT_ACTIVE, "RDMA 端口未 ACTIVE");
  // 创建本端元信息结构，准备把地址和凭据发给对方。
  Peer self;
  // 把 buffer 指针转成足够大的整数保存；只是记录地址，没有复制内存内容。
  self.addr = reinterpret_cast<uintptr_t>(r.buf);
  // 从 MR 取得远端访问用的 rkey，稍后交给对方填入其 RDMA 请求。
  self.rkey = r.mr->rkey;
  // 记录本端 QP 编号，对端 RTR 配置需要知道该连向哪个 QP。
  self.qpn = r.qp->qp_num;
  // 记录 IB 端口的本地地址标识 LID。
  self.lid = port.lid;
  // 记录当前端口实际使用的 MTU 枚举，以便两端选择共同支持的包长。
  self.mtu = port.active_mtu;
  // 非负 GID 索引表示需要用选定的 GID 表条目。
  if (o.gid_index >= 0)
    // 从端口的 GID 表读取该条目，写入本端元信息。
    require(!ibv_query_gid(r.ctx, o.ib_port, o.gid_index, &self.gid), "ibv_query_gid");
  // GID 索引为 -1 时，不使用上面的全局路由配置。
  else
    // 无 GID 路径仅用于这里的 InfiniBand/LID 模式；RoCE 必须配置有效的 GID。
    require(port.link_layer == IBV_LINK_LAYER_INFINIBAND, "RoCE 必须指定有效 GID_INDEX");
  // 建立 TCP 控制连接；这一步不走 RDMA，也不发送实验数值 42。
  r.fd = control(o);
  // 步骤⑥：通过 TCP 交换地址、rkey、qpn、gid 等信息，得到对端“地址与钥匙”。
  Peer peer = exchange(r.fd, self);                                  // ⑥ TCP 换钥匙
  // 准备 QP 修改属性结构，并清空所有字段。
  ibv_qp_attr attr{};
  // 步骤⑦的第一阶段：将刚创建时处于 RESET 的 QP 改为 INIT。
  attr.qp_state = IBV_QPS_INIT;                                      // ⑦ RESET -> INIT
  // 指定这个 QP 使用哪一个本地 HCA 端口。
  attr.port_num = o.ib_port;
  // 选择分区密钥表的第 0 项；实际网络需保证该分区允许双方通信。
  attr.pkey_index = 0;
  // 设置 QP 允许的内存访问类型，使后面能够接受远端写入。
  attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
  // 调用状态修改接口；属性掩码声明这次哪些字段有效，不能只填 attr 而不传掩码。
  require(!ibv_modify_qp(r.qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                         // 继续声明端口和访问权限字段有效；这两个物理行属于同一次 INIT 修改调用。
                         IBV_QP_PORT | IBV_QP_ACCESS_FLAGS), "QP INIT");
  // 清空上一阶段的属性，避免无关配置混入下一阶段。
  attr = {};
  // 第二阶段：进入 RTR（Ready To Receive，准备接收），填好到对端的路径。
  attr.qp_state = IBV_QPS_RTR;                                       // INIT -> RTR
  // 选择双方端口中较小的 MTU 编码，确保都能处理该大小的数据包。
  attr.path_mtu = static_cast<ibv_mtu>(std::min(self.mtu, peer.mtu));
  // 指定目标 QP 编号；这是从 TCP 元信息交换中得到的对端队列对。
  attr.dest_qp_num = peer.qpn;
  // 设置接收方向的起始包序号为 0；需要与对端发送方向的起始序号一致。
  attr.rq_psn = 0;
  // 配置允许对端发起的未完成 RDMA Read/Atomic 数量上限；本例只用 Write，不实际发这类请求。
  attr.max_dest_rd_atomic = 1;
  // 配置 RNR（接收方尚未准备好）的重试等待编码；这个数是协议编码，不是直接的秒数。
  attr.min_rnr_timer = 12;
  // 给地址向量填写目标 LID，InfiniBand 路径会使用它。
  attr.ah_attr.dlid = peer.lid;
  // 指定地址向量使用的本地端口。
  attr.ah_attr.port_num = o.ib_port;
  // 如果指定了 GID，就补充全局路由头 GRH 相关字段。
  if (o.gid_index >= 0) {
    // 启用全局路由地址信息，RoCE 通常需要这条路径。
    attr.ah_attr.is_global = 1;
    // 目标 GID 填对端交换过来的地址。
    attr.ah_attr.grh.dgid = peer.gid;
    // 源 GID 用本地用户选择的 GID 表索引，而不是对端的索引。
    attr.ah_attr.grh.sgid_index = o.gid_index;
    // 设置这一示例的 hop limit 为 1，适用于简单直连/本地网段，跨路由拓扑需另行配置。
    attr.ah_attr.grh.hop_limit = 1;
  }
  // 应用 RTR 修改，声明状态、地址向量和路径 MTU 等字段有效。
  require(!ibv_modify_qp(r.qp, &attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                         // 继续声明目标 QP、接收序号及 Read/Atomic 接收上限有效。
                         IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
                         // 最后包括 RNR 等待字段；这些物理行合起来是一条 QP RTR 修改语句。
                         IBV_QP_MIN_RNR_TIMER), "QP RTR（检查 GID / LID / 路由）");
  // 再次清空属性，为进入可发送状态做准备。
  attr = {};
  // 第三阶段：进入 RTS（Ready To Send，准备发送）。
  attr.qp_state = IBV_QPS_RTS;                                       // RTR -> RTS
  // 设置本地 ACK 超时的协议编码 14；它不是直接的 14 毫秒。
  attr.timeout = 14;
  // 设置传输失败时的重试次数编码。
  attr.retry_cnt = 7;
  // 设置接收方未准备好时的 RNR 重试编码；值 7 在该字段中表示无限重试。
  attr.rnr_retry = 7;
  // 发送起始包序号设为 0，与对端 RTR 中的 rq_psn 配对。
  attr.sq_psn = 0;
  // 设置本端未完成 RDMA Read/Atomic 请求上限；本例仍只实际执行 Write。
  attr.max_rd_atomic = 1;
  // 调用修改接口，声明状态、超时和重试字段有效。
  require(!ibv_modify_qp(r.qp, &attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                         // 接着声明 RNR 重试、发送序号和 Read/Atomic 上限；成功后 QP 可以发送。
                         IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC), "QP RTS");
  // 通过 TCP 互报 QP 就绪，避免 client 在 server 还未配置完成时下单。
  barrier(r.fd);
  // client 是真正投递 RDMA WRITE 的一方；server 等待远端写入。
  if (!o.server) {
    // 把数值 42 编为 32 位网络字节序，让不同端序的机器也能一致解释。
    uint32_t value = htonl(42);
// 如果编译了 CUDA 支持，接下来先判断 buffer 是否真的在 GPU 上。
#ifdef USE_CUDA
    // 显存模式用 cudaMemcpy 放入发送数据；else 对应下方 CPU memcpy。
    if (r.gpu) cu(cudaMemcpy(r.buf, &value, sizeof(value), cudaMemcpyHostToDevice)); else
// 结束条件编译；纯主存版会直接执行下面的 memcpy。
#endif
    // 主存模式把编码好的 4 字节数值复制进本地注册的发送 buffer。
    std::memcpy(r.buf, &value, sizeof(value));
    // 创建 SGE（Scatter/Gather Element），描述网卡该从哪一段本地内存读数据。
    ibv_sge sge{};
    // SGE 的 addr 填本地发送 buffer 的地址，不是远端地址。
    sge.addr = reinterpret_cast<uintptr_t>(r.buf);
    // 本次仅发送一个 uint32_t，也就是 4 字节，注册的 4 KiB 不会全发出去。
    sge.length = sizeof(value);
    // lkey 是网卡访问本地 MR 时需要的键；与远端访问用的 rkey 区分。
    sge.lkey = r.mr->lkey;
    // 创建工作请求 WR，并预留 bad 指针：提交失败时接口可指出哪个请求有问题。
    ibv_send_wr wr{}, *bad = nullptr;                                // ⑧ RDMA WRITE
    // 为这次请求标记 ID=1，后面从 CQ 收回执时核对，避免认错请求。
    wr.wr_id = 1;
    // 让请求引用刚配置好的本地数据段描述。
    wr.sg_list = &sge;
    // 本次请求只有一个数据段，所以 num_sge 为 1。
    wr.num_sge = 1;
    // 选择操作码 RDMA WRITE：将本地数据直接写进对端已注册内存。
    wr.opcode = IBV_WR_RDMA_WRITE;
    // 要求为该请求生成完成回执；否则成功写入也可能没有可轮询的 CQE。
    wr.send_flags = IBV_SEND_SIGNALED;
    // 填写对端的目标内存地址，来自此前 TCP 握手交换的 peer.addr。
    wr.wr.rdma.remote_addr = peer.addr;
    // 填写对端 MR 的 rkey，配合远端地址授权该次访问；不是填本地 lkey。
    wr.wr.rdma.rkey = peer.rkey;
    // 步骤⑧：把 WR 投递到 QP 的发送队列，让网卡执行请求；提交成功不等于传输已完成。
    require(!ibv_post_send(r.qp, &wr, &bad), "ibv_post_send");
    // 准备 WC（Work Completion）结构，用来存从 CQ 取到的一条完成记录。
    ibv_wc wc{};                                                     // ⑨ 查询完成回执
    // 设置轮询结束时限为当前单调时钟之后 10 秒，防止无限空转。
    auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    // 循环轮询，直到取到完成记录、查询失败或等待超时。
    while (true) {
      // 步骤⑨：每次最多从 CQ 取 1 条记录，返回 0 表示暂时没有，正数表示取到数量。
      int n = ibv_poll_cq(r.cq, 1, &wc);
      // 负返回值表示轮询调用本身失败，立即报告错误。
      require(n >= 0, "ibv_poll_cq");
      // 取到记录就跳出循环；记录是否成功还需要在下面检查 status。
      if (n) break;
      // 超过 10 秒没有结果就抛异常，不让程序一直卡住。
      if (std::chrono::steady_clock::now() > end) throw std::runtime_error("CQ 等待超时");
      // 暂时让出 CPU 调度机会；这不是阻塞式 CQ 通知，后面仍继续轮询。
      std::this_thread::yield();
    }
    // 拿到 CQE 不代表请求一定成功；检查状态，失败时打印 Verbs 的具体描述。
    if (wc.status != IBV_WC_SUCCESS) throw std::runtime_error(ibv_wc_status_str(wc.status));
    // 核对请求 ID 和完成操作类型，确认回执对应本例的那次 RDMA WRITE。
    require(wc.wr_id == 1 && wc.opcode == IBV_WC_RDMA_WRITE, "非预期 CQE");
    // 准备单字节完成通知 D，以及接收 server 校验结果的变量 ack。
    char done = 'D', ack = 0;
    // 通过 TCP 告知 server 写入已经完成；这里传的是通知字节，不是数值 42。
    io(r.fd, &done, 1, true); // 仅发送控制通知；数值 42 通过 RDMA 数据面写入。
    // 等待 server 真正读取其内存并返回验证结果。
    io(r.fd, &ack, 1, false);
    // 只有收到 K 才算双方确认成功，避免只根据本地 CQE 宣称结果正确。
    require(ack == 'K', "server 校验未通过");
    // 输出 client 成功标记，说明 RDMA 请求完成且 server 验证通过。
    std::puts("client PASS: RDMA WRITE 已完成，server 校验通过");
  // server 分支不投递发送请求，而是等待自己的注册内存被 client 写入。
  } else {
    // 为接收 client 的完成通知预留一个字节。
    char done = 0;
    // 等待 TCP 控制通知；普通 RDMA WRITE 不会自动给 server 生成接收 CQE。
    io(r.fd, &done, 1, false);
    // 检查对端通知必须为约定的 D，再读取数据内存。
    require(done == 'D', "client 完成通知");
    // 创建 CPU 上的 32 位变量，准备接收注册内存的最终值。
    uint32_t value = 0;
// CUDA 构建需要针对显存 buffer 使用不同的回读方式。
#ifdef USE_CUDA
    // 网络完成通知之后再调用 CUDA API，保证后续 CPU 回读观察到外部 DMA 写入。
    // 显存模式通过 CUDA API 将最终值回读到 CPU；else 对应下方主存 memcpy。
    if (r.gpu) cu(cudaMemcpy(&value, r.buf, sizeof(value), cudaMemcpyDeviceToHost)); else
// 结束 CUDA 条件编译；纯主存构建直接执行后面的复制。
#endif
    // 主存模式从本地已注册 buffer 中读出 4 字节；42 是通过 RDMA 到达这里的。
    std::memcpy(&value, r.buf, sizeof(value));
    // 把网络字节序转换回本机整数，再验证是否确实收到 42。
    require(ntohl(value) == 42, "远端内存不是 42");
    // 构造验证通过确认 K，表示 server 读到的真实内存内容正确。
    char ack = 'K';
    // 通过 TCP 回复 client，完成双方收尾握手。
    io(r.fd, &ack, 1, true);
    // 打印数据来自 GPU 还是主存，及最终校验值；之后 Resources 自动释放各项资源。
    std::printf("server PASS: %s 内存 value=42\n", r.gpu ? "GPU" : "host");
  }
}
// 程序入口；argc 是参数总数，argv 是每个参数的字符串指针。
int main(int argc, char** argv) {
  // 最外层 try/catch 将全部参数或执行错误转换为明确的失败退出码。
  try {
    // 创建默认配置：server、主存、TCP 端口 29877、HCA 端口 1。
    Options o;
    // 从 argv[1] 开始逐项读取；argv[0] 是程序名，不作为选项解析。
    for (int i = 1; i < argc; ++i) {
      // 把当前参数名复制为 std::string，方便与 --server 等文本比较。
      std::string arg = argv[i];
      // --server 不带后续值；设置角色后 continue，直接读取下一个选项。
      if (arg == "--server") { o.server = true; continue; }
      // --help 同样不带值，进入用法输出分支。
      if (arg == "--help") {
        // 打印所有参数及其默认值范围，帮助用户手动启动双方。
        std::puts("rdma_write --server|--connect HOST [--device HCA] [--port 29877] [--ib-port 1] [--gid-index 0|-1] [--cuda GPU]");
        // 显示帮助后正常退出，不真正打开 RDMA 设备。
        return 0;
      }
      // 其他选项都需要一个值；先移动到下一项，若没有了就报缺少参数。
      if (++i == argc) throw std::runtime_error("选项缺少值");
      // 读取当前选项后面的值，例如设备名或数字端口文本。
      std::string value = argv[i];
      // --connect HOST 表示 client 角色，同时保存 server 地址。
      if (arg == "--connect") { o.server = false; o.host = value; }
      // --device 设置要打开的 RDMA HCA 名称。
      else if (arg == "--device") o.device = value;
      // --port 将 TCP 控制端口从字符串解析成整数。
      else if (arg == "--port") o.port = std::stoi(value);
      // --ib-port 设置 RDMA 设备内部的端口编号，不是操作系统 TCP 端口。
      else if (arg == "--ib-port") o.ib_port = std::stoi(value);
      // --gid-index 选择设备 GID 表里的条目；-1 表示不启用此全局地址分支。
      else if (arg == "--gid-index") o.gid_index = std::stoi(value);
      // --cuda 指定 GPU 编号，非负表示申请并注册显存。
      else if (arg == "--cuda") o.gpu = std::stoi(value);
      // 遇到未识别选项就报错，避免用户拼错参数后仍以默认配置运行。
      else throw std::runtime_error("未知选项: " + arg);
    }
    // 检查 TCP 和 HCA 端口的合法范围，避免整数被截断成无意义的地址字段。
    require(o.port > 0 && o.port <= 65535 && o.ib_port > 0 && o.ib_port <= 255 &&
            // 继续检查 GID 索引和 GPU 编号范围；这是与上一行同一次参数检查。
            o.gid_index >= -1 && o.gid_index <= 255 && o.gpu >= -1, "参数范围");
// 只有没有定义 USE_CUDA 的主存构建才保留下面的检查。
#ifndef USE_CUDA
    // 如果主存版被要求用 GPU，就提示先编译 CUDA 版本，不悄悄回退。
    if (o.gpu >= 0) throw std::runtime_error("显存注册请先 make build-cuda 并使用 rdma_write_cuda");
// 结束主存构建专用的条件编译区。
#endif
    // 参数检查完成，进入真正的 RDMA 资源申请、握手和传输实验。
    experiment(o);
    // 实验成功则返回 0，供 Makefile 和终端判断。
    return 0;
  // 接住参数转换或实验过程中的标准异常。
  } catch (const std::exception& e) {
    // 把具体失败原因打印到标准错误输出。
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    // 以非零状态退出，Makefile 会停止而不是继续显示成功。
    return 1;
  }
}
