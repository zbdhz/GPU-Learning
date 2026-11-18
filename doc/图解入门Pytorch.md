
> 🌞欢迎来到人工智能的世界  
> 🌈博客主页：[卿云阁](https://blog.csdn.net/zzqingyun?spm=1011.2415.3001.5343 "卿云阁")
> 
> 💌欢迎关注🎉点赞👍收藏⭐️留言📝
> 
> 🌟本文由卿云阁原创！
> 
> 📆首发时间：🌹2025年1月23日🌹
> 
> 🙏作者水平很有限，如果发现错误，请留言[轰炸](https://so.csdn.net/so/search?q=%E8%BD%B0%E7%82%B8&spm=1001.2101.3001.7020 "轰炸")哦！万分感谢！
> 
> * * *
> 
> **目录**
> 
> [前言](#t0)
> 
> [torch的导入](#t1)
> 
> [Tensor的创建](#t2)
> 
> [Tensor运算](#t3)
> 
> [层 layer](#t4)
> 
> [加载数据](#t5)
> 
> [激活函数和损失函数](#t6)
> 
> [保存/加载模型](#t7)
> 
> [GPU训练](#t8)
> 
> [定义模型](#t9)
> 
> [模型训练](#t10)
> 
> [模型评估](#t11)

![](https://i-blog.csdnimg.cn/blog_migrate/ea4c2773df63943b57b180c6fffafd69.gif)

> #### 前言

**什么是PyTorch？**

**PyTorch 就像是深度学习的“工具箱”**： 就像你做饭需要锅、铲子、刀和菜一样，做 深度学习

就需要工具。而 PyTorch 就是那个专门为做深度学习准备的“工具箱”。它提供了很多现成的工具和

操作，让你做 人工智能 的研究或开发变得更简单、快捷。

**优点：**

**能跑在 GPU 上，速度超级快，****自动求导帮你“自己学”**

**什么是 Tensor？**     

深度学习模型的核心是**数据的表示**，而 Tensor 是数据的基本表示形式。比如，图像、声音、视

频、文本等数据，最终都需要转化为 Tensor 来进行处理。如果你不了解如何创建和操作 Tensor，

就无法深入理解深度学习的核心操作。

**Tensor** 是 PyTorch 中的核心数据结构，可以理解为一个“**高维数组**”，用来存储数据。它类似

于  NumPy 数组，但是 **Tensor 支持 GPU** 加速，因此非常适合深度学习。

![](https://i-blog.csdnimg.cn/direct/bf656fbd6f4e4a62b9a89a36bd554db9.png)

**例子：**

一张图片是一个二维矩阵，颜色通道是第三个维度，总体来说，图片可以表示成一个三维的

Tensor（高度、宽度、颜色通道）。你要训练一个分类模型，数据输入的就是 Tensor，模型输出的也是 Tensor。

#### torch的导入

```python
# 导入基础工具库
import torch
import torch.nn as nn
```

```python
# pytorch中用于视觉的数据集、模型与变换
from torchvision import datasets, models, transforms
import torch.nn.functional as F  # Function Collection
```

* * *

> #### **Tensor的创建**

![](https://i-blog.csdnimg.cn/direct/835be11690164e4b87731b5ebe025213.png)   

**创建张量（一维、二维、三维）**

**一维、二维**

```python
import torch

# 创建一维 Tensor
t = torch.tensor([1, 2, 3, 4])

# 输出 Tensor 的维度
print("维度 (dim):", t.dim())

# 输出 Tensor 的形状
print("形状 (size):", t.shape)
```

```python
import torch

# 创建一个 3x4 的 Tensor
b = torch.tensor([[0, 1, 2, 3],
                  [4, 5, 6, 7],
                  [8, 9, 10, 11]])

# 输出 Tensor 的维度
print("维度 (dim):", b.dim())

# 输出 Tensor 的形状
print("形状 (shape):", b.shape)
```

![](https://i-blog.csdnimg.cn/direct/2c52fb8edeea4188899873d7f71d14c2.png)

**三维**

```python
import torch

# 创建一个 3x3x4 的三维 Tensor
t = torch.tensor([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                  [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
                  [[24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34, 35]]])

# 输出 Tensor 的维度
print("维度 (dim):", t.dim())

# 输出 Tensor 的形状
print("形状 (shape):", t.shape)
```

![](https://i-blog.csdnimg.cn/direct/ac661f5fcd5c4c03ab195efd4b5f0329.png)

**4维张量**

假设我们现在有3张图片，每张图片用一个3维的张量，表示。**第一个维度**表示照片的数量，**第二个**

**维度**表示图片的通道数，**第三个维度**和**第四个维度**表示图片的高度和宽度，

![](https://i-blog.csdnimg.cn/direct/412e741cd1ea418ab2c8ebeca117dde9.png)

**5维张量**

假设我们现在有2个视频，每个视频有3帧图片，每张图片用一个3维的张量表示。

**第二个维度**表示视频的数量，**第二个维度**表示照片的数量，**第三个维度**表示图片的通道数，**第四个**

**维度**和**第五个维度**表示图片的高度和宽度。

![](https://i-blog.csdnimg.cn/direct/f9f5589ab122468786d7961e433197e0.png)

* * *

> ####  Tensor运算

Tensor 运算是神经网络中进行数学计算的关键工具，所有神经网络的计算步骤（如前向传播、

反向传播）都离不开 Tensor 运算。加法、乘法、矩阵乘法、逐元素运算等基本的 Tensor 运算在模

型训练和推理中都频繁使用。

**例子：**

在神经网络的训练中，输入数据经过层与层之间的矩阵乘法（Tensor 运算）后，才会产生预测

结果。然后，计算损失（loss）并通过梯度下降来更新参数，这个过程也完全依赖于 Tensor 运

算。

![](https://i-blog.csdnimg.cn/direct/970f7445dc844e0490e406b14a64c1f4.png) 

##### 改变形状

```python
import torch

# 创建一个一维的 Tensor
a = torch.tensor([0, 1, 2, 3])

# 使用 view() 改变形状，变成 2x2 的二维 Tensor
b = a.view(2, 2)

print(b)
```

![](https://i-blog.csdnimg.cn/direct/a97c0d9dc6904a1c86afaa0410f0631a.png) 

##### 拼接

**`torch.cat`** 是沿着现有的维度进行拼接，它不会增加新的维度。

```python
import torch

# 创建两个 2x3 的 Tensor
A = torch.IntTensor([[0, 1, 2], [3, 4, 5]])
B = torch.IntTensor([[6, 7, 8], [9, 10, 11]])

# 沿着维度 0 (行方向) 拼接
C1 = torch.cat((A, B), dim=0)
print("C1 (dim=0):\n", C1)

# 沿着维度 1 (列方向) 拼接
C2 = torch.cat((A, B), dim=1)
print("C2 (dim=1):\n", C2)
```

![](https://i-blog.csdnimg.cn/direct/bab19708ed684706a817f39449d266b2.png)   

##### 堆叠

**`torch.stack`** 是沿着指定维度堆叠多个 Tensor，它会在堆叠的维度上增加一个新的维度。

```python
# 沿着维度 0 堆叠
C3 = torch.stack((A, B), dim=0)
print("C3 (dim=0):\n", C3)

# 沿着维度 1 堆叠
C4 = torch.stack((A, B), dim=1)
print("C4 (dim=1):\n", C4)

# 沿着维度 2 堆叠
C5 = torch.stack((A, B), dim=2)
print("C5 (dim=2):\n", C5)

# 沿着维度 -1 堆叠（与 dim=2 等价），最后一个维度
C6 = torch.stack((A, B), dim=-1)
print("C6 (dim=-1):\n", C6)
```

![](https://i-blog.csdnimg.cn/direct/e0e794b7efbc4aabbe693c30f5f77de5.png) 

![](https://i-blog.csdnimg.cn/direct/50bb4fa942ab4cc68c5b1a830333bca3.png) 

##### 格式转换

**转换的原因**：

**数据的预处理**：你可能会使用 **`NumPy`** 来处理数据，因为 **`NumPy`** 提供了丰富的数学和数组操作功能。

处理完数据后，再将其转换为 **`Torch`** 的 Tensor，便于送入深度学习模型进行训练和推理。

**与其它库兼容**：许多科学计算、数据分析库，比如 **`SciPy`、`Pandas`**，通常使用 **`NumPy`** 数组。如果

使用这些库中的函数，你需要将 **`Torch`** Tensor 转换成 **`NumPy`** 数组。

**GPU 加速**：PyTorch 的 Tensor 可以直接在 GPU 上计算，而 **`NumPy`** 数组只能在 CPU 上操作。

因此，转换格式可以帮助你在 GPU 上加加速计算。

```python
import torch
import numpy as np

np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()

print('\nnumpy array:', np_data)
print('\ntorch tensor:', torch_data)
print('\ntensor to array:', tensor2array)
```

![](https://i-blog.csdnimg.cn/direct/aedea276d9f544bda441ba9a5babf47e.png) 

* * *

> #### 层 layer

在神经网络中，层是构建模型的基本单元，它们帮助我们对数据进行特征提取、变换和预测。

![](https://i-blog.csdnimg.cn/direct/766acbce64604284992bff4710e07b26.png)

![](https://i-blog.csdnimg.cn/direct/ee9f7016ec9340dfbb5a757b7ffaea17.png)

层就是一个操作模块，它接收输入并通过某种计算产生输出。常见的层包括：

**线性层（Linear Layer）：** 用于实现全连接操作。

**卷积层（Convolutional Layer）：** 用于处理图像的局部特征。

**池化层（Pooling Layer）：** 用于降低特征图的维度。

**Dropout层：** 用于减少过拟合。

**归一化层（Normalization Layer）：** 用于对数据进行标准化处理。

![](https://i-blog.csdnimg.cn/direct/d3817afb3fbd47e38b0bac320ce17285.png)

神经网络之所以能实现复杂的任务，就是因为我们可以通过堆叠多个不同的层，让模型具备非

线性表达能力。每种层都有自己的功能，比如提取特征、减少噪声或者增强模型的泛化能力。在

PyTorch中，我们可以直接使用 **`torch.nn`** 模块来定义层。

![](https://i-blog.csdnimg.cn/direct/cf329f2a244e4bc9aa74c8c708ed354d.png)

* * *

> #### 加载数据

在深度学习中，数据是模型训练的基础，而PyTorch提供了非常高效且灵活的工具来帮助我们

处理数据。

在PyTorch中，加载数据的核心模块是：**`Dataset，`** 用于定义和封装数据，**`DataLoader，`**用于

批量加载数据。一条样本可以用从**Dataset**（(Features, Label)元组构建的序列）继承而来的类来

初始化表示。使用 **DataLoader**，可以分批完成加载。数据集通常分 为训练数据（通常80%）和测

试数据（通常20%）。

![](https://i-blog.csdnimg.cn/direct/003f32f307474a7f98c62cf7f3a6200a.png)

![](https://i-blog.csdnimg.cn/direct/20faab93e25e4027be605da7aa5f8b18.png) 

**导入模块**：先从 PyTorch 中导入了一些处理数据集的工具。

```python
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
```

**读入数据：**我们首先准备好数据，告诉程序数据是什么样子的。

```python
# 假设我们有一些数据
inps = torch.randn(100, 5)  # 100个样本，每个样本有5个特征
tgts = torch.randint(0, 2, (100,))  # 100个标签，二分类任务

# 数据集的大小
train_size = 80
test_size = 20

# 使用 TensorDataset 将输入和标签组合起来
dataset = TensorDataset(inps, tgts)
print(dataset)
```

**划分数据集**：将数据集按照指定的比例划分为训练集和测试集（例如，80% 用于训练，20% 用于测试）。

```python
# 将数据集划分为训练集和测试集
train_data, test_data = random_split(dataset, [train_size, test_size])
```

**创建 DataLoader**：通过 **`DataLoader`** 来批量加载训练数据，每次加载 16 个样本，并且每次加载

时打乱数据顺序。这就是代码的详细解释。

```python
# 创建 DataLoader 用于按批次加载训练数据
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
```

**综合：**

![](https://i-blog.csdnimg.cn/direct/004ccc3e34544957be55478a59cbaa3b.png)

```python
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

# 假设我们有一些数据
inps = torch.randn(100, 5)  # 100个样本，每个样本有5个特征
tgts = torch.randint(0, 2, (100,))  # 100个标签，二分类任务

# 数据集的大小
train_size = 80
test_size = 20

# 使用 TensorDataset 将输入和标签组合起来
dataset = TensorDataset(inps, tgts)

# 将数据集划分为训练集和测试集
train_data, test_data = random_split(dataset, [train_size, test_size])

# 创建 DataLoader 用于按批次加载训练数据
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)

# 打印加载的数据
for batch_idx, (inputs, targets) in enumerate(train_loader):
    print(f"Batch {batch_idx+1}:")
    print("Inputs:", inputs)
    print("Targets:", targets)
    break  # 这里只打印第一个批次的数据，实际中会遍历整个数据集
```

* * *

> #### 激活函数和损失函数

**激活函数** 和 **损失函数**，这两个神经网络中非常重要的概念。激活函数决定了每一层神经元的输

出，而损失函数则告诉我们网络预测的好坏。

![](https://i-blog.csdnimg.cn/direct/d28f9539debd45bd92c7a088138f9b25.png)

##### ![](https://i-blog.csdnimg.cn/direct/7135afa3ab4f467ba612fd89f6dd388d.png)

##### 激活函数

在神经网络中，激活函数的主要作用是引入 **非线性**，否则，整个网络的输出就只能是输入的线

性组合（即多个加权和），不能模拟复杂的函数关系。通过激活函数，神经网络能够逼近任何复杂

的非线性函数。

![](https://i-blog.csdnimg.cn/direct/b25191e2c0134a3c8ccf2d7d504a9c5e.png)

```python
nn.ReLU()
```

##### **损失函数**

损失函数用于衡量神经网络预测值与真实值之间的差距。在训练过程中，神经网络会通过优化损

失函数来更新权重。回归任务一般使用**均方误差**，分类任务常用**交叉熵损失**。

![](https://i-blog.csdnimg.cn/direct/9f330c9a47e54b5aa5a7ce85583de552.png)

```python
# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, targets)
print(f"交叉熵损失：{loss.item()}")
```

* * *

> #### 保存/加载模型

![](https://i-blog.csdnimg.cn/direct/26e457847a72450cbac81d74ac77456a.png)

##### ![](https://i-blog.csdnimg.cn/direct/35a3ff7b3ed041f6ab30a941fda5505b.png)

保存和加载模型是深度学习训练中的一个重要步骤，尤其是在你训练了一个良好的模型，并希

望以后能复用或继续训练时。PyTorch 提供了简单易用的接口来保存和加载模型。

**PyTorch 提供了两种方式来保存和加载模型：**

##### 保存和加载模型的参数（权重）

**保存模型的参数：** 保存模型的状态字典（**`state_dict`**）

![](https://i-blog.csdnimg.cn/direct/33e4aaf524cd4ee4b0166c88984d7910.png)

```python
import torch

# 假设你已经定义并训练好了一个模型
model = Net()  # 模型实例
torch.save(model.state_dict(), 'model_weights.pth')  # 只保存模型的参数
```

`state_dict` 是一个 Python 字典，包含了模型的所有参数和缓冲区。这是保存和加载模型最常用

的方式。

![](https://i-blog.csdnimg.cn/direct/a41204f850e34d8d8fcbfc6fcf7b5523.png)

**`model_weights.pth`**是保存的文件名，你可以根据需要修改路径和文件名 **。**

当你执行 `torch.save(model.state_dict(), 'model_weights.pth')` 时，PyTorch 会将

`state_dict` 中包含的所有参数保存到 `model_weights.pth` 文件中。

**加载模型的参数**

![](https://i-blog.csdnimg.cn/direct/429df23365464f8b9c077179f3e1e4ea.png)

在加载模型时，你需要创建一个新的模型实例，然后加载 `state_dict`：这时，模型的参数（例

如卷积层和全连接层的权重、偏置等）就会从 `model_weights.pth`中加载到模型中，恢复到之前保

存时的状态。

```python
# 假设你已经重新定义了模型架构
model = Net()  # 重新定义模型结构

# 加载保存的权重
model.load_state_dict(torch.load('model_weights.pth'))
```

**总结：**

*   只保存模型的“记忆”（参数），不包含模型的结构。需要重新写结构代码。
*   **优点**：灵活，不容易遇到版本问题，节省存储空间。
*   **缺点**：加载时稍微麻烦，需要手动定义模型结构。

##### **保存整个模型（包括结构和参数）**

**保存模型**

![](https://i-blog.csdnimg.cn/direct/96fdace372e849dda996c59c4ae1ae8c.png)

```python
import torch

# 假设你已经定义并训练好了一个模型
model = Net()  # 模型实例
torch.save(model, 'model.pth')  # 保存模型
```

`model` 就是一个包含了网络结构和当前参数的对象。

![](https://i-blog.csdnimg.cn/direct/0046da9030da44a7a28033a052174c6d.png)

**`model.pth`**  
`model.pth` 是保存的文件，它包含了整个模型的对象。使用 `torch.save(model,`

`'model.pth')` 进行保存后，`model.pth` 文件中保存了整个模型对象，包括模型的结构

（即神经网络的层次和各个层的定义）和参数（权重、偏置）。

**加载模型**

![](https://i-blog.csdnimg.cn/direct/3fb6baae2a7742d29e9c4d7efdce0522.png)

```python
# 加载整个模型
model = torch.load('model.pth', weights_only=False)
```

注：PyTorch 2.6+ 中 `torch.load` 默认 `weights_only=True`，加载完整模型对象需显式指定 `weights_only=False`（仅用于可信文件）。

`torch.load('model.pth')` 会将整个模型（包括模型的结构和参数）加载回来，你可以直

接使用加载后的模型进行推理或继续训练。

**保存整个模型（`torch.save(model, 'model.pth')`）**

*   就是把整个模型“打包”，包括了结构和参数，取出来直接用。
*   **优点**：简单方便，加载时不需要重新定义模型。
*   **缺点**：代码和版本依赖强，结构改变时容易出问题

* * *

> #### GPU训练

##### **CPU和GPU**

机器学习 和深度学习来说，使用GPU更适合一点。两者的区别可以这么理解，CPU相当于5-6

个大学教授，GPU相当于100个高中生，显然如果我们只是单纯的想做1000道简单的数学题的话，

使用GPU更合适。

![](https://i-blog.csdnimg.cn/direct/11860544e2284f9283f3fd1576242453.png)

今天我们来讲解一下如何在 PyTorch 中选择和使用 GPU 或 CPU 来训练模型

**part1**

首先，我们来看看这一行代码：

```python
import torch

# 检查是否有可用的 GPU，如果有使用 GPU，否则使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
```

**`torch.cuda.is_available()`：**

这是一个 PyTorch 提供的函数，它会返回一个布尔值（`True` 或 `False`）。如果你的电脑上安

装了 GPU 且配置正确，并且支持 CUDA ，它就会返回 `True`。如果没有 GPU，或者没有安装正确

的驱动和配置，它会返回 `False`。

**`torch.device()`：**

这是一个 PyTorch 的类，用来指定设备（比如 GPU 或 CPU）。

![](https://i-blog.csdnimg.cn/direct/24c1045db4bc40b8a9a27a1585c5d554.png)

这行代码的作用是：先通过 `torch.cuda.is_available()` 检查是否有可用的 GPU。如果有

GPU 可用，即使用 GPU 进行计算。如果没有 GPU，它会设置 `device` 为 `cpu`，即使用 CPU 进行计算。

**part2**

```python
model.to(device)
```

这行代码的含义是：

将我们的模型（**`model`**）移动到前面选定的设备上。

![](https://i-blog.csdnimg.cn/direct/74eb3f6d1dc34a9f9218a0edee993e14.png)

**part3**

```python
# 假设我们有一些数据
inputs, labels = data
inputs = inputs.to(device)
labels = labels.to(device)
```

**`inputs` 和 `labels` 是什么**

`inputs` 是模型的输入数据，通常是图像、文本、声音等类型的数据。在这段代码中，它可能是

一个张量（tensor），比如一批图片数据，形状可能是 **`(batch_size, channels, height,`**

**`width)`。**

`labels` 是每个输入数据对应的标签（例如，分类任务中的类别编号）。它的形状可能是

`(batch_size,)`，表示每个输入数据的标签。

![](https://i-blog.csdnimg.cn/direct/9e4cdfeafde1459eb49f7ee660607c1a.png)

**完整代码：**

```python
# 检查是否有可用的 GPU，如果有使用 GPU，否则使用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 将模型移动到选定的设备（GPU 或 CPU）
model.to(device)

# 假设我们有一些数据
inputs, labels = data
inputs = inputs.to(device)
labels = labels.to(device)
```

* * *

> #### 定义模型

##### ![](https://i-blog.csdnimg.cn/direct/b40c051d9a5240ff940f5b041228dfaa.png)

#####    nn.Module类

nn.Module是PyTorch提供的神经网络类，并在类中实现了网络各层的定义及前向计算与反向传

播机制。在实际使用时，如果想要实现某个神经网络，只需继承nn.Module，在初始化中定义模型

结构与参数，在函数**forward()**中编写网络前向过程即可。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)  # 输入通道为 3
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层
        self.fc = nn.Linear(in_features=6 * 16 * 16, out_features=10)  # 池化后尺寸为 16x16

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))  # 卷积层 + ReLU 激活函数 + 池化层
        x = x.view(-1, 6 * 16 * 16)  # 展平为一维向量
        x = self.fc(x)  # 全连接层
        return x


# 实例化模型
model = Net()

# 打印网络结构
from torchsummary import summary
summary(model, input_size=(3, 32, 32))  # 假设输入是 32x32 的 RGB 图像
```

##### ![](https://i-blog.csdnimg.cn/direct/44a88d6f0636425fb986f489c6c4e563.png)

![](https://i-blog.csdnimg.cn/direct/d9c4823ac1ce40efab08d9543f50f8f3.png)

##### torch.nn.Sequential

我们可以把它想象成一个装配流水线，它可以把多个神经网络的层（如卷积层、激活函数、全

连接层等）**按顺序串起来**，然后按顺序依次执行。可以理解为一个“管道”，每一层都把自己的输出

传给下一层。

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),  # 2D卷积层
    nn.ReLU(),  # 激活函数
    nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化层
    nn.Flatten(),  # 将多维输入展平为一维
    nn.Linear(in_features=6*16*16, out_features=10)  # 全连接层，输入调整为1536
)

# 打印网络结构
print(model)
```

##### ![](https://i-blog.csdnimg.cn/direct/10d384ef82fb411abdc8a78472977b58.png)

##### 综合使用

综合使用上述两种方式搭建上述网络

![](https://i-blog.csdnimg.cn/direct/033dd6c405e84b9094318da780599690.png)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # 使用 nn.Sequential 构建卷积块
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层
        )

        # 使用 nn.Sequential 构建全连接块
        self.fc_block = nn.Sequential(
            nn.Flatten(),  # 展平成一维向量
            nn.Linear(in_features=6 * 16 * 16, out_features=10)  # 全连接层
        )

    def forward(self, x):
        x = self.conv_block(x)  # 通过卷积块
        x = self.fc_block(x)  # 通过全连接块
        return x


# 实例化模型
model = Net()

# 打印模型结构
print(model)

# 打印测试数据流动结果
from torchsummary import summary
summary(model, input_size=(3, 32, 32))  # 输入尺寸为 32x32 的 RGB 图片
```

* * *

> #### 模型训练

##### torch.optim

torch.optim包提供非常多的可实现参数自动优化的类，如SGD、AdaGrad、RMSProp、Adam

等。优化器就是用来更新模型的权重，使得模型的预测更加准确。

![](https://i-blog.csdnimg.cn/direct/dc14d5997b47419ab06993ca857dc792.png) 

**使用示例：**

```python
# 随机梯度下降（SGD）用于优化模型
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

optimizer.zero_grad()  # 计算模型输出之前，将优化器的梯度清零，防止累加梯度
optimizer.step()  # 反向传播计算梯度之后，更新模型的参数（优化）
```

##### 训练过程

通过不断输入数据、计算损失、反向传播、更新参数，模型会逐步优化，以达到尽可能准确的

预测。每一轮（epoch）都在用新的数据训练模型，让它越来越适应任务目标。

![](https://i-blog.csdnimg.cn/direct/6c0404093ab3419385959f92cb4a8bb0.png)

```python
# 训练过程
for epoch in range(2):  # 在训练集上遍历多次
    model.train()  # 启用训练模式
    for i, data in enumerate(train_loader, 0):  # 遍历训练集的每一个 batch
        inputs, labels = data  # 获取输入数据和标签

        optimizer.zero_grad()  # 将优化器的梯度清零，防止累加梯度

        outputs = model(inputs)  # 计算模型的输出
        loss = loss_fn(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型的参数（优化）

        if i % 100 == 0:  # 每 100 个 batch 打印一次损失
            print(f"Epoch [{epoch+1}/2], Step [{i+1}], Loss: {loss.item():.4f}")
```

* * *

> #### 模型评估

根据数据指标评估模型的训练结果。评估目标不同，使用的数据指标也随之变化。常用的指

标包括：准确度（acurracy）、精确度（precision）、召回率（recall）、F1或BLEU。

```python
import torch

# 模型设置为评估模式
model.eval()

# 初始化变量
correct = 0  # 正确分类的样本数
total = 0  # 总分类样本数

# 不计算梯度以节省内存和加速
with torch.no_grad():
    for data in test_loader:  # 遍历测试集
        inputs, labels = data  # 获取输入和标签
        outputs = model(inputs)  # 模型预测

        # 获取每个样本的预测类别（即输出最大概率的类别）
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)  # 更新总样本数
        correct += (predicted == labels).sum().item()  # 更新正确分类的样本数

# 打印准确率
accuracy = correct / total
print(f'Accuracy: {accuracy * 100:.2f}%')
```

**`model.eval()`：**

设置模型为评估模式，这样在模型中某些层（如 **`Dropout`** 和 **`BatchNorm`**）在训练和测试时的行

为就会发生变化，确保测试时使用的是训练时学到的参数。

`**torch.no_grad()**`**：**

禁用梯度计算，在测试时我们不需要梯度，因此使用 **`torch.no_grad()`** 来节省内存和提高

计算效率。

**![](https://i-blog.csdnimg.cn/blog_migrate/f8e24d220835469478373bd65c7f6456.gif)**

###

本文转自 <https://blog.csdn.net/zzqingyun/article/details/145329268>，如有侵权，请联系删除。
