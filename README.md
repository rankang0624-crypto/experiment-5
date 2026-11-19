我看到了你上传的实验结果图。这张图是 **LeNet 模型训练损失变化曲线（Training Loss）**，横轴为训练轮数 Epoch，纵轴为损失值 Loss。

因此，你的实验报告只需要保留这一张结果图，不需要再添加准确率曲线、测试图片或其他虚构的实验结果。

从图中可以看出：第 1 轮训练损失约为 2.3046，第 2 轮迅速下降至约 2.3013，之后基本维持在 2.301 左右，整体变化较小。这说明当前训练过程中损失下降不明显，模型可能尚未有效收敛。

下面整理成可以直接复制到 GitHub README.md 的格式。

## **一、实验目的**

1. 熟悉 PyTorch 深度学习框架，掌握神经网络模型的基本构建方法。
2. 了解 MNIST 手写数字数据集，掌握数据集的加载、预处理及批量读取方法。
3. 学习 LeNet 卷积神经网络的基本结构，理解卷积层、池化层和全连接层的作用。
4. 掌握模型训练的基本流程，能够使用损失函数和优化器完成模型参数更新。
5. 通过训练损失曲线观察模型训练过程，初步分析模型的训练效果。

## **二、实验步骤及对应代码**

### **1. 导入相关库**

导入 PyTorch、torchvision 和 Matplotlib 等相关库，为后续数据处理、模型构建、训练及结果可视化做准备。

```
import torchimport torch.nn as nnimport torch.nn.functional as Fimport torch.optim as optimfrom torchvision import datasets, transformsfrom torch.utils.data import DataLoaderimport matplotlib.pyplot as plt
```

### **2. 设置实验参数及运行设备**

设置批次大小、学习率和训练轮数，并根据计算机环境自动选择 GPU 或 CPU。

```
batch_size = 512learning_rate = 0.01epochs = 10device = torch.device(    "cuda" if torch.cuda.is_available() else "cpu")print("当前运行设备：", device)
```

### **3. 加载 MNIST 数据集**

使用 torchvision 加载 MNIST 手写数字数据集，并通过 `ToTensor()` 将图像转换为张量格式。

```
transform = transforms.ToTensor()train_dataset = datasets.MNIST(    root="./data",    train=True,    download=True,    transform=transform)test_dataset = datasets.MNIST(    root="./data",    train=False,    download=True,    transform=transform)train_loader = DataLoader(    train_dataset,    batch_size=batch_size,    shuffle=True)test_loader = DataLoader(    test_dataset,    batch_size=batch_size,    shuffle=False)
```

### **4. 构建 LeNet 神经网络模型**

根据实验参考代码构建 LeNet 网络，主要由两个卷积层和三个全连接层组成，通过卷积、池化和全连接运算提取图像特征。

```
class Net(nn.Module):    def __init__(self):        super(Net, self).__init__()        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)        self.conv2 = nn.Conv2d(6, 16, 5)        self.fc1 = nn.Linear(5 * 5 * 16, 120)        self.fc2 = nn.Linear(120, 84)        self.clf = nn.Linear(84, 10)    def forward(self, x):        x = torch.sigmoid(self.conv1(x))        x = F.avg_pool2d(x, 2, 2)        x = torch.sigmoid(self.conv2(x))        x = F.avg_pool2d(x, 2, 2)        x = torch.flatten(x, 1)        x = torch.sigmoid(self.fc1(x))        x = torch.sigmoid(self.fc2(x))        x = self.clf(x)        return xmodel = Net().to(device)
```

### **5. 设置损失函数和优化器**

使用交叉熵损失函数计算模型预测结果与真实标签之间的差异，采用 SGD 随机梯度下降优化器更新网络参数。

```
criterion = nn.CrossEntropyLoss()optimizer = optim.SGD(    model.parameters(),    lr=learning_rate)
```

### **6. 训练模型并记录损失值**

将训练数据输入网络，计算损失值并反向传播更新模型参数。每轮训练结束后记录平均损失值，为绘制损失曲线做准备。

```
train_losses = []for epoch in range(epochs):    model.train()    running_loss = 0.0    for images, labels in train_loader:        images = images.to(device)        labels = labels.to(device)        optimizer.zero_grad()        outputs = model(images)        loss = criterion(outputs, labels)        loss.backward()        optimizer.step()        running_loss += loss.item()    epoch_loss = running_loss / len(train_loader)    train_losses.append(epoch_loss)    print(        f"Epoch [{epoch + 1}/{epochs}], "        f"Loss: {epoch_loss:.4f}"    )
```

### **7. 绘制训练损失曲线**

使用 Matplotlib 将每轮训练得到的损失值绘制成折线图，观察损失随训练轮数的变化情况。

```
plt.figure(figsize=(10, 6))plt.plot(    range(1, epochs + 1),    train_losses,    marker='o',    label='Training Loss')plt.title("LeNet Training Loss")plt.xlabel("Epoch")plt.ylabel("Loss")plt.legend()plt.grid(True)plt.savefig("training_loss.png", dpi=300)plt.show()
```

**实验结果：**

将你上传的图片放在此处，图片文件上传到 GitHub 仓库后，可以使用以下 Markdown 语句显示：

```
![LeNet Training Loss](./training_loss.png)
```

实验结果图：LeNet Training Loss（使用你上传的实际截图）

[Building a Minimal Vision-Language Model from Scratch – Nipun Batra Blog](https://images.openai.com/static-rsc-4/iBH4UfcE4ZTkDBmqnkrdzeJWKqTWI-9LGb9dNuPXsWLwLOi5-aLS-kqhfPt79uLgluSe1RUuMkAvC4hO0jxZSnqo1GunDBbyyuE_Ks8-YCBA2xJ1v6DQe6fOXT2jIweSCrcmfAXox8faTZauDSTsJwhWztGXS7_hu4V9-84JR-U?purpose=inline)

## **三、实验结果分析**

通过实验得到 LeNet 模型训练损失变化曲线。从图中可以观察到，模型在第 1 轮训练时损失值约为 2.3046，第 2 轮下降至约 2.3013，随后在第 3 至第 10 轮之间基本保持稳定，损失值变化幅度较小。

这表明模型在训练初期损失有所下降，但后续优化效果不明显，模型尚未表现出明显的收敛趋势。可能与当前网络结构、参数设置、学习率以及训练过程有关。后续可以通过调整学习率、优化器和网络参数等方式进一步改善训练效果。

## **四、实验小结**

本次实验基于 PyTorch 深度学习框架，使用 MNIST 手写数字数据集完成了 LeNet 卷积神经网络的构建与训练。通过实验，我掌握了数据集加载、网络模型定义、损失函数设置、优化器配置以及模型训练的基本流程，并利用 Matplotlib 绘制了训练损失变化曲线。

从实验结果可以看出，训练损失在初始阶段有所下降，但后续变化较小，说明当前模型的训练效果仍有改进空间。通过本次实验，我进一步理解了卷积神经网络的基本工作原理，也熟悉了使用 PyTorch 完成深度学习模型训练和结果可视化的方法，为后续开展图像分类及其他计算机视觉实验打下了基础。

**注意：** 你上传的图是训练损失曲线，不是模型分类准确率图。报告中应如实描述训练损失变化，不要写成模型已经达到较高识别准确率。
