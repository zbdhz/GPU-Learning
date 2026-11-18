// GPU 0 的 kernel 直接写 GPU 1 的显存；传输可能走 NVLink，也可能走 PCIe。
// 包含本目录的 CUDA 检查函数和 P2P 能力查询函数。
#include "cuda_check.h"
// 引入 puts、fprintf 等终端输出函数。
#include <cstdio>

// __global__ 表示这是 GPU kernel；peer 指向另一张卡的显存，peer[0]=value 是实际写入动作。
__global__ void write_peer(int* peer, int value) { peer[0] = value; }

// C++ 程序从 main 开始；返回值 0 表示成功，非零表示失败或本例约定的跳过。
int main() {
  // 声明整型指针，初始不指向任何 allocation；后面它会保存 GPU 1 的显存地址。
  int* peer = nullptr;
  // 记录是否已开启 peer access，方便异常时只清理已开启的权限。
  bool enabled = false;
  // 捕获设备查询、分配和执行中的异常，统一输出错误。
  try {
    // 先查 GPU 0 能否直接访问 GPU 1；不具备条件就不运行 kernel。
    if (!has_peer_path()) {
      // 明确打印 SKIP，防止把“没运行”误认为“校验通过”。
      std::puts("SKIP: cudaDeviceCanAccessPeer(0,1)=0，无法执行显存直写。");
      // 77 是本例约定的跳过退出码，Makefile 会将其显示为 SKIP。
      return 77;
    }
    // 把当前线程的 CUDA 设备切到 GPU 1，后面的显存分配就属于这张卡。
    CUDA_CHECK(cudaSetDevice(1));
    // 在 GPU 1 分配一个 int 的空间；&peer 让 CUDA 把分配后的地址写回指针。
    CUDA_CHECK(cudaMalloc(&peer, sizeof(int)));
    // 将这几个显存字节清零，保证写入前的数不是 42。
    CUDA_CHECK(cudaMemset(peer, 0, sizeof(int)));
    // 等待 GPU 1 上的清零完成，再让 GPU 0 去写它，避免初始化覆盖写入结果。
    CUDA_CHECK(cudaDeviceSynchronize());  // 跨设备初始化在 writer 开始前完成。
    // 切回 GPU 0，接下来会从这张卡启动 kernel。
    CUDA_CHECK(cudaSetDevice(0));
    // 在 GPU 0 的地址空间中开启对 GPU 1 的访问；不是把数据拷到 GPU 0。
    CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0));
    // 标记该权限已经开启，错误清理时就知道需要关闭。
    enabled = true;
    // <<<1,1>>> 启动 1 个 block、每个 block 1 个线程；该线程把 GPU 1 显存写成 42。
    write_peer<<<1, 1>>>(peer, 42);
    // 检查 kernel 是否提交成功，例如启动参数是否合法；这不是等待 kernel 完成。
    CUDA_CHECK(cudaGetLastError());
    // 等待 GPU 0 的 kernel 执行完成，同时暴露执行期间可能发生的错误。
    CUDA_CHECK(cudaDeviceSynchronize());
    // 切换到 buffer 所属的 GPU 1，准备在持有方读取结果。
    CUDA_CHECK(cudaSetDevice(1));
    // 定义 CPU 上的普通 int，先放 -1，接下来接收回读结果。
    int value = -1;
    // 将 GPU 1 的一个 int 拷到 CPU 变量 value；这次拷贝仅用于观察最终结果。
    CUDA_CHECK(cudaMemcpy(&value, peer, sizeof(value), cudaMemcpyDeviceToHost));
    // 结果不是 42 就抛异常，保证错误的跨卡写入不会被显示为成功。
    if (value != 42) throw std::runtime_error("P2P 结果不等于 42");
    // 校验结束，释放 GPU 1 上分配的显存。
    CUDA_CHECK(cudaFree(peer));
    // 清空指针，表示不再持有 allocation，防止后面的异常清理重复释放。
    peer = nullptr;
    // 切到此前开启 peer access 的 GPU 0。
    CUDA_CHECK(cudaSetDevice(0));
    // 撤销 GPU 0 对 GPU 1 的 peer access，结束本次跨卡访问。
    CUDA_CHECK(cudaDeviceDisablePeerAccess(1));
    // 更新记录，表示权限已关闭，无需再清理。
    enabled = false;
    // 打印成功，并提醒实际物理链路要结合拓扑判断，不能只凭这段程序判断 NVLink。
    std::puts("PASS: GPU 0 kernel -> GPU 1 显存，value=42；链路类型请结合 topo -m 判断。");
    // 返回 0，告诉终端或 Makefile 本次运行通过。
    return 0;
  // 接住上面抛出的标准异常；e 保存具体错误描述。
  } catch (const std::exception& e) {
    // 把错误写到标准错误输出，便于终端和日志区分失败信息。
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    // 如果显存还没有释放，就切回所属 GPU 1 做尽力清理。
    if (peer) { cudaSetDevice(1); cudaFree(peer); }
    // 如果权限仍开着，就切回 GPU 0 关闭；避免正常路径没走完时遗漏清理。
    if (enabled) { cudaSetDevice(0); cudaDeviceDisablePeerAccess(1); }
    // 以失败状态退出，让 Makefile 停止并报告错误。
    return 1;
  }
}
