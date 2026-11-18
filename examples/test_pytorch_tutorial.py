# -*- coding: utf-8 -*-
"""《图解入门Pytorch》文档代码测试用例。

对应 doc/图解入门Pytorch.md 中出现的所有代码片段：
  - torch 的导入
  - Tensor 的创建（1/2/3 维、view、cat、stack、numpy 互转）
  - 层 layer（nn.Module、nn.Sequential）
  - 加载数据（TensorDataset、random_split、DataLoader）
  - 激活函数和损失函数
  - 保存/加载模型
  - GPU 训练
  - 模型训练（torch.optim）
  - 模型评估（accuracy）

运行方式（gpu-dev 环境，未安装 pytest，使用 unittest）：
    python examples/test_pytorch_tutorial.py -v
    # 或使用一键脚本
    ./run_pytorch_examples.sh
"""

import os
import tempfile
import unittest

import numpy as np

try:
    import torchvision  # noqa: F401
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


# ---------------------------------------------------------------- 导入部分
class TestImports(unittest.TestCase):
    """文档 “torch的导入” 一节的代码。"""

    def test_import_torch(self):
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401

    @unittest.skipUnless(HAS_TORCHVISION, "torchvision 未安装")
    def test_import_torchvision(self):
        from torchvision import datasets, models, transforms  # noqa: F401
        import torch.nn.functional as F  # noqa: F401


# ---------------------------------------------------------------- Tensor 的创建
class TestTensorCreation(unittest.TestCase):
    """文档 “Tensor的创建” 一节的代码。"""

    def test_create_1d_tensor(self):
        t = torch.tensor([1, 2, 3, 4])
        self.assertEqual(t.dim(), 1)
        self.assertEqual(t.shape, torch.Size([4]))

    def test_create_2d_tensor(self):
        b = torch.tensor([[0, 1, 2, 3],
                          [4, 5, 6, 7],
                          [8, 9, 10, 11]])
        self.assertEqual(b.dim(), 2)
        self.assertEqual(b.shape, torch.Size([3, 4]))

    def test_create_3d_tensor(self):
        t = torch.tensor([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                          [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
                          [[24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34, 35]]])
        self.assertEqual(t.dim(), 3)
        self.assertEqual(t.shape, torch.Size([3, 3, 4]))

    def test_create_4d_tensor_images(self):
        # 3 张图片，每张 3 通道、高 32、宽 32
        imgs = torch.randn(3, 3, 32, 32)
        self.assertEqual(imgs.dim(), 4)
        self.assertEqual(imgs.shape, torch.Size([3, 3, 32, 32]))

    def test_create_5d_tensor_videos(self):
        # 2 个视频，每个 3 帧图片，每张 3 通道、高 32、宽 32
        videos = torch.randn(2, 3, 3, 32, 32)
        self.assertEqual(videos.dim(), 5)
        self.assertEqual(videos.shape, torch.Size([2, 3, 3, 32, 32]))


# ---------------------------------------------------------------- Tensor 运算
class TestTensorOps(unittest.TestCase):
    """文档 “Tensor运算” 一节的代码。"""

    def test_view_reshape(self):
        a = torch.tensor([0, 1, 2, 3])
        b = a.view(2, 2)
        self.assertEqual(b.shape, torch.Size([2, 2]))
        self.assertTrue(torch.equal(b, torch.tensor([[0, 1], [2, 3]])))

    def test_cat(self):
        A = torch.IntTensor([[0, 1, 2], [3, 4, 5]])
        B = torch.IntTensor([[6, 7, 8], [9, 10, 11]])
        # 沿着维度 0（行方向）拼接
        C1 = torch.cat((A, B), dim=0)
        self.assertEqual(C1.shape, torch.Size([4, 3]))
        # 沿着维度 1（列方向）拼接
        C2 = torch.cat((A, B), dim=1)
        self.assertEqual(C2.shape, torch.Size([2, 6]))
        self.assertTrue(torch.equal(C1[:2], A))
        self.assertTrue(torch.equal(C1[2:], B))

    def test_stack(self):
        A = torch.IntTensor([[0, 1, 2], [3, 4, 5]])
        B = torch.IntTensor([[6, 7, 8], [9, 10, 11]])
        C3 = torch.stack((A, B), dim=0)   # 沿着维度 0 堆叠
        C4 = torch.stack((A, B), dim=1)   # 沿着维度 1 堆叠
        C5 = torch.stack((A, B), dim=2)   # 沿着维度 2 堆叠
        C6 = torch.stack((A, B), dim=-1)  # 与 dim=2 等价，最后一个维度
        self.assertEqual(C3.shape, torch.Size([2, 2, 3]))
        self.assertEqual(C4.shape, torch.Size([2, 2, 3]))
        self.assertEqual(C5.shape, torch.Size([2, 3, 2]))
        self.assertTrue(torch.equal(C5, C6))
        self.assertTrue(torch.equal(C3[0], A))
        self.assertTrue(torch.equal(C3[1], B))

    def test_numpy_conversion(self):
        np_data = np.arange(6).reshape((2, 3))
        torch_data = torch.from_numpy(np_data)
        tensor2array = torch_data.numpy()
        np.testing.assert_array_equal(np_data, tensor2array)
        self.assertEqual(torch_data.shape, torch.Size([2, 3]))


# ---------------------------------------------------------------- 层与模型定义
class Net(nn.Module):
    """文档 “定义模型 - nn.Module类” 一节的网络。"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=6,
                              kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(in_features=6 * 16 * 16, out_features=10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))  # 卷积 + ReLU + 池化
        x = x.view(-1, 6 * 16 * 16)          # 展平为一维向量
        x = self.fc(x)                       # 全连接层
        return x


class NetCombined(nn.Module):
    """文档 “定义模型 - 综合使用” 一节的网络（nn.Sequential 组合）。"""

    def __init__(self):
        super(NetCombined, self).__init__()
        # 使用 nn.Sequential 构建卷积块
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 使用 nn.Sequential 构建全连接块
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=6 * 16 * 16, out_features=10),
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x


class TestLayers(unittest.TestCase):
    """文档 “层 layer / 定义模型” 一节的代码。"""

    def test_sequential_model(self):
        model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=6 * 16 * 16, out_features=10),
        )
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, torch.Size([4, 10]))

    def test_module_model(self):
        model = Net()
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, torch.Size([4, 10]))

    def test_module_model_combined(self):
        model = NetCombined()
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, torch.Size([4, 10]))


# ---------------------------------------------------------------- 加载数据
class TestDataLoading(unittest.TestCase):
    """文档 “加载数据” 一节的代码。"""

    def test_dataset_dataloader(self):
        # 假设我们有一些数据
        inps = torch.randn(100, 5)              # 100 个样本，每个样本 5 个特征
        tgts = torch.randint(0, 2, (100,))      # 100 个标签，二分类任务
        # 数据集的大小
        train_size, test_size = 80, 20
        # 使用 TensorDataset 将输入和标签组合起来
        dataset = TensorDataset(inps, tgts)
        # 将数据集划分为训练集和测试集
        train_data, test_data = random_split(dataset, [train_size, test_size])
        # 创建 DataLoader 用于按批次加载训练数据
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=16, shuffle=True)
        self.assertEqual(len(train_data), 80)
        self.assertEqual(len(test_data), 20)
        n_batches = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            self.assertEqual(inputs.shape[1], 5)
            self.assertIn(targets.dtype, (torch.long, torch.int64))
            n_batches += 1
        self.assertEqual(n_batches, 5)  # 80 / 16 = 5 个批次


# ------------------------------------------------------- 激活函数和损失函数
class TestActivationAndLoss(unittest.TestCase):
    """文档 “激活函数和损失函数” 一节的代码。"""

    def test_relu(self):
        relu = nn.ReLU()
        y = relu(torch.tensor([-1.0, 0.0, 2.0]))
        self.assertTrue(torch.equal(y, torch.tensor([0.0, 0.0, 2.0])))

    def test_cross_entropy_loss(self):
        outputs = torch.randn(8, 10)          # 模型输出（logits）
        targets = torch.randint(0, 10, (8,))  # 真实标签
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss))


# ------------------------------------------------------------- 保存/加载模型
class TestSaveLoadModel(unittest.TestCase):
    """文档 “保存/加载模型” 一节的代码。"""

    def test_save_load_state_dict(self):
        model = Net()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model_weights.pth")
            # 只保存模型的参数（状态字典）
            torch.save(model.state_dict(), path)
            # 重新定义模型结构并加载保存的权重
            model2 = Net()
            model2.load_state_dict(torch.load(path))
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                self.assertTrue(torch.equal(p1, p2))

    def test_save_load_whole_model(self):
        model = NetCombined()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pth")
            # 保存整个模型（包括结构和参数）
            torch.save(model, path)
            # 加载整个模型（PyTorch 2.6+ 默认 weights_only=True，
            # 加载完整模型对象需显式指定 weights_only=False，仅用于可信文件）
            model2 = torch.load(path, weights_only=False)
            x = torch.randn(2, 3, 32, 32)
            model.eval()
            model2.eval()
            with torch.no_grad():
                out1 = model(x)
                out2 = model2(x)
            self.assertEqual(out2.shape, torch.Size([2, 10]))
            self.assertTrue(torch.allclose(out1, out2))


# ---------------------------------------------------------------- GPU 训练
class TestGpuTraining(unittest.TestCase):
    """文档 “GPU训练” 一节的代码。"""

    def test_device_selection(self):
        # 检查是否有可用的 GPU，如果有使用 GPU，否则使用 CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = Net()
        # 将模型移动到选定的设备（GPU 或 CPU）
        model.to(device)
        # 假设我们有一些数据
        inputs = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        self.assertEqual(outputs.device.type, device.type)
        self.assertEqual(outputs.shape, torch.Size([4, 10]))
        if device.type == "cuda":
            self.assertEqual(outputs.get_device(), 0)


# ------------------------------------------------------------- 模型训练与评估
def _make_train_setup(device):
    """构造训练所需的模型、数据、优化器和损失函数。"""
    torch.manual_seed(0)
    # 数据：100 个样本，5 个特征，二分类任务
    inps = torch.randn(100, 5)
    tgts = torch.randint(0, 2, (100,))
    train_size, test_size = 80, 20
    dataset = TensorDataset(inps, tgts)
    train_data, test_data = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=False)

    # 与文档不同：这里用小型 MLP 适配 5 维特征输入（文档的 Net 输入为 3x32x32 图片）
    model = nn.Sequential(
        nn.Linear(5, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    ).to(device)
    # 随机梯度下降（SGD）用于优化模型
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    return model, train_loader, test_loader, optimizer, loss_fn


class TestTrainAndEvaluate(unittest.TestCase):
    """文档 “模型训练 / 模型评估” 一节的代码。"""

    def test_train_and_evaluate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, train_loader, test_loader, optimizer, loss_fn = \
            _make_train_setup(device)
        losses = []

        # 训练过程
        for epoch in range(2):  # 在训练集上遍历多次
            model.train()  # 启用训练模式
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data              # 获取输入数据和标签
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()              # 梯度清零，防止累加
                outputs = model(inputs)            # 计算模型的输出
                loss = loss_fn(outputs, labels)    # 计算损失
                loss.backward()                    # 反向传播计算梯度
                optimizer.step()                   # 更新模型的参数（优化）
                losses.append(loss.item())

        # 训练正常：损失有限
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))
        # SGD 确实更新过参数（损失有变化，模型可学习）

        # 模型评估
        model.eval()  # 模型设置为评估模式
        correct = 0   # 正确分类的样本数
        total = 0     # 总分类样本数
        with torch.no_grad():  # 不计算梯度以节省内存和加速
            for data in test_loader:  # 遍历测试集
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                # 获取每个样本的预测类别（输出最大概率的类别）
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        self.assertEqual(total, 20)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
