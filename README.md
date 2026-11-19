# 实验五：基于LeNet卷积神经网络的手写数字识别

## 一、实验目的

1. 熟悉 PyTorch 深度学习框架的基本使用方法，掌握相关模块的导入、数据加载以及神经网络模型的构建方法。
2. 掌握 MNIST 手写数字数据集的加载与预处理方法，理解训练集和测试集的作用，熟悉批量数据加载及张量转换操作。
3. 理解 LeNet 卷积神经网络的基本结构和工作原理，掌握卷积层、Sigmoid 激活函数、平均池化层及全连接层的基本功能。
4. 掌握卷积神经网络前向传播、损失计算、反向传播和参数优化的基本过程，理解利用优化算法训练神经网络的方法。
5. 通过 MNIST 手写数字识别实验，实现图像特征提取和数字分类，掌握利用卷积神经网络解决图像识别问题的基本方法。
6. 通过测试集验证模型的识别效果，分析模型训练损失和准确率的变化，加深对深度学习在图像分类领域应用的理解。

## 二、实验环境

- 操作系统：Windows
- 编程语言：Python
- 深度学习框架：PyTorch
- 数据集：MNIST 手写数字数据集
- 辅助库：Torchvision、Matplotlib
- 开发工具：PyCharm 或 Anaconda

安装实验所需依赖库：

```bash
pip install torch torchvision matplotlib
```

## 三、实验步骤与对应代码

### 步骤一：导入实验所需的库

首先导入 PyTorch 深度学习框架及相关模块。其中，torch.nn 用于构建神经网络，torch.optim 用于模型参数优化，torchvision 用于加载 MNIST 数据集，Matplotlib 用于绘制实验结果图像。

**对应代码：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
import matplotlib.pyplot as plt
```

### 步骤二：设置实验参数及运行设备

设置数据批次大小为512，并根据计算机运行环境自动选择 GPU 或 CPU。通过 DataLoader 实现数据的批量读取，提高模型训练效率。

**对应代码：**

```python
# 设置批次大小
batch_size = 512

# 设置学习率
learning_rate = 0.01

# 设置训练轮数
epochs = 10

# 自动选择运行设备
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

print("当前运行设备：", device)
```

### 步骤三：加载MNIST手写数字数据集

使用 torchvision.datasets.MNIST 加载 MNIST 数据集。MNIST 数据集包含0至9共10类手写数字图像，每张图像为28×28像素的灰度图像。

利用 transforms.ToTensor() 将图像转换为 PyTorch 张量，并分别构建训练数据加载器和测试数据加载器。

**对应代码：**

```python
# 加载训练集
trainloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        'data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    ),
    batch_size=batch_size,
    shuffle=True
)

# 加载测试集
testloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        'data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    ),
    batch_size=batch_size,
    shuffle=False
)

print("MNIST数据集加载成功！")
```

### 步骤四：构建LeNet卷积神经网络模型

根据实验参考代码补全 Net 类，构建 LeNet 卷积神经网络。

网络主要由两个卷积层、两个全连接层和一个分类层组成。

第一层卷积输入通道为1，输出通道为6，卷积核大小为5×5，padding设置为2；第二层卷积输入通道为6，输出通道为16，卷积核大小为5×5。

全连接层分别设置为400→120、120→84，最终分类层输出10个类别，对应数字0至9。

**对应代码：**

```python
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # 第一层卷积
        self.conv1 = nn.Conv2d(
            1, 6, kernel_size=5, padding=2
        )

        # 第二层卷积
        self.conv2 = nn.Conv2d(
            6, 16, kernel_size=5
        )

        # 第一层全连接
        self.fc1 = nn.Linear(
            5 * 5 * 16, 120
        )

        # 第二层全连接
        self.fc2 = nn.Linear(
            120, 84
        )

        # 分类层
        self.clf = nn.Linear(
            84, 10
        )

    def forward(self, x):

        # 第一层卷积
        x = self.conv1(x)

        # Sigmoid激活函数
        x = torch.sigmoid(x)

        # 平均池化
        x = F.avg_pool2d(
            x, kernel_size=2, stride=2
        )

        # 第二层卷积
        x = self.conv2(x)

        # Sigmoid激活函数
        x = torch.sigmoid(x)

        # 平均池化
        x = F.avg_pool2d(
            x, kernel_size=2, stride=2
        )

        # 展平操作
        x = torch.flatten(
            x, start_dim=1
        )

        # 第一层全连接
        x = self.fc1(x)
        x = torch.sigmoid(x)

        # 第二层全连接
        x = self.fc2(x)
        x = torch.sigmoid(x)

        # 分类输出
        x = self.clf(x)

        return x
```

### 步骤五：创建模型并设置损失函数和优化器

实例化 LeNet 网络模型，并将模型加载到指定计算设备上。

使用交叉熵损失函数计算预测结果与真实标签之间的误差，使用随机梯度下降（SGD）优化器更新网络参数。

**对应代码：**

```python
# 创建网络模型
model = Net().to(device)

# 设置交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 设置SGD优化器
optimizer = optim.SGD(
    model.parameters(),
    lr=learning_rate
)

# 输出网络结构
print(model)
```

### 步骤六：训练LeNet卷积神经网络

将 MNIST 训练集输入网络模型，通过前向传播得到预测结果，使用交叉熵损失函数计算误差。

随后执行反向传播，根据梯度更新网络参数。重复上述过程，使网络逐渐学习手写数字图像的特征。

实验设置训练轮数为10轮，并记录每轮的损失值和准确率。

**对应代码：**

```python
# 保存每轮训练损失和准确率
train_losses = []
train_accuracies = []

# 开始训练
for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in trainloader:

        images = images.to(device)
        labels = labels.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新网络参数
        optimizer.step()

        running_loss += loss.item()

        # 获取预测类别
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 计算平均损失
    epoch_loss = running_loss / len(trainloader)

    # 计算训练准确率
    epoch_accuracy = 100 * correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(
        f'Epoch [{epoch + 1}/{epochs}] '
        f'Loss: {epoch_loss:.4f} '
        f'Accuracy: {epoch_accuracy:.2f}%'
    )
```

### 步骤七：使用测试集进行模型测试

训练完成后，将模型切换为测试模式，使用 MNIST 测试集对模型进行验证。

通过 torch.no_grad() 关闭梯度计算，减少测试过程中的计算开销。统计测试集中识别正确的图像数量，计算最终分类准确率。

**对应代码：**

```python
# 设置模型为测试模式
model.eval()

correct = 0
total = 0

# 不计算梯度
with torch.no_grad():

    for images, labels in testloader:

        images = images.to(device)
        labels = labels.to(device)

        # 模型预测
        outputs = model(images)

        # 获取预测类别
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

# 计算测试准确率
test_accuracy = 100 * correct / total

print("测试集总数量：", total)
print("识别正确数量：", correct)

print(
    "测试集准确率：{:.2f}%".format(
        test_accuracy
    )
)
```

### 步骤八：绘制训练损失曲线

为了观察模型训练过程中损失值的变化情况，使用 Matplotlib 绘制训练损失曲线。

通过横坐标表示训练轮数，纵坐标表示对应的损失值，直观分析模型的训练情况。

**对应代码：**

```python
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, epochs + 1),
    train_losses,
    marker='o',
    label='Training Loss'
)

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.title('LeNet Training Loss')

plt.legend()
plt.grid()

# 保存训练损失图
plt.savefig('training_loss.png')

plt.show()
```

### 步骤九：绘制训练准确率曲线

通过绘制训练准确率曲线，观察模型随着训练轮数增加，其识别能力的变化情况。

横坐标表示训练轮数，纵坐标表示训练准确率。

**对应代码：**

```python
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, epochs + 1),
    train_accuracies,
    marker='o',
    label='Training Accuracy'
)

plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

plt.title('LeNet Training Accuracy')

plt.legend()
plt.grid()

# 保存训练准确率图
plt.savefig('training_accuracy.png')

plt.show()
```

### 步骤十：保存训练好的模型

训练和测试完成后，使用 torch.save() 保存模型参数，生成 lenet_mnist.pth 文件，便于后续加载和使用训练好的网络。

**对应代码：**

```python
# 保存模型参数
torch.save(
    model.state_dict(),
    'lenet_mnist.pth'
)

print("模型保存成功！")
```

## 四、实验结果与分析

### 1. 模型训练结果

模型训练过程中，程序会输出每一轮的损失值和训练准确率。通过观察训练过程，可以了解模型参数优化的变化情况。

随着训练不断进行，模型通常能够逐渐学习 MNIST 手写数字图像中的特征，训练损失一般呈下降趋势，训练准确率一般呈上升趋势。

### 2. 模型测试结果

使用 MNIST 测试集对训练完成的模型进行验证。测试集包含10000张手写数字图像，通过统计识别正确数量与测试集总数量，计算模型的测试准确率。

测试准确率计算公式：

\[
Accuracy=\frac{正确识别数量}{测试集总数量}\times100\%
\]

实际测试准确率以程序运行输出为准。

### 3. 训练曲线分析

通过训练损失曲线，可以观察模型在不同训练轮数下的误差变化情况。

通过训练准确率曲线，可以观察模型对训练数据的分类能力变化情况。

两条曲线能够帮助分析模型的训练效果，为后续调整学习率、训练轮数等参数提供参考。

## 五、实验小结

本次实验利用 PyTorch 深度学习框架完成了基于 LeNet 卷积神经网络的 MNIST 手写数字识别任务。通过本次实验，我进一步熟悉了 PyTorch 的基本使用方法，掌握了 MNIST 数据集的加载、预处理以及批量数据读取操作，并按照实验要求补全了 LeNet 网络模型。

在模型构建过程中，我深入了解了卷积层、Sigmoid 激活函数、平均池化层和全连接层的基本作用，理解了卷积神经网络通过多层卷积运算提取图像特征，并利用全连接层实现分类的基本原理。在模型训练过程中，通过前向传播计算预测结果，利用交叉熵损失函数计算误差，再通过反向传播和 SGD 优化器不断调整网络参数，使模型逐步学习手写数字的图像特征。

实验过程中，我还完成了模型测试、准确率计算以及训练损失曲线和准确率曲线的绘制，对神经网络的训练和测试流程有了更加直观的认识。

通过本次实验，我不仅提高了使用 Python 和 PyTorch 进行深度学习编程的能力，也加深了对卷积神经网络工作原理的理解。同时认识到，模型的训练效果会受到网络结构、学习率、训练轮数以及数据质量等因素的影响。在今后的学习中，还需要进一步尝试调整网络参数、优化模型结构，提高模型的识别性能和泛化能力。

## 六、实验文件说明

实验运行完成后，项目目录中主要包含以下文件：

```text
实验五/
│
├── lenet.py
│
├── data/
│   └── MNIST数据集
│
├── training_loss.png
│
├── training_accuracy.png
│
└── lenet_mnist.pth
```

- `lenet.py`：实验完整 Python 源代码。
- `data/`：MNIST 手写数字数据集。
- `training_loss.png`：模型训练损失曲线。
- `training_accuracy.png`：模型训练准确率曲线。
- `lenet_mnist.pth`：训练完成后保存的模型参数文件。
