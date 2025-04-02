import torch.utils
import torch,torchvision
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

"""
import zipfile
with zipfile.ZipFile(f"./data.zip","r") as zip_ref:
    zip_ref.extractall("data")
"""

batch_size=32

# 定义数据增强流水线
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 加载验证数据集（不进行数据增强，只进行标准化）
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 加载数据集
trainset = datasets.ImageFolder('./data/train', transform=data_transforms)
testset = datasets.ImageFolder('./data/val', transform=val_transforms)

# 创建数据加载器，用于迭代训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# 创建数据加载器，用于迭代测试数据集
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

print(trainset.class_to_idx)
# 获取训练数据加载器的一个迭代器
dataiter = iter(trainloader)

# 从迭代器中获取下一个批次的数据
images, labels = next(dataiter)

# 打印图像数据的形状
print(images.shape)

# 打印标签数据的形状
print(labels.shape)

print(images[3].shape)
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255],
)
print(images[3].numpy().shape)
inv_images=inv_normalize(images[3]).numpy().transpose(1, 2, 0)
inv_images= np.clip(inv_images, 0, 1)
plt.imshow(inv_images)

device = (
    "cuda"
    if torch.cuda.is_available()
    #else "mps"
    #if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

input_size=224
num_classes=105

model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
for param in model.parameters():
    param.requires_grad = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

model = model.to(device)
print(model)

# Observe that only parameters of final layer are being optimized as
# opposed to before.
params_to_update = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(params_to_update, lr=0.05,momentum=0.9,weight_decay=0.00004)
def adjust_learning_rate_poly(optimizer, initial_lr, iteration, max_iter):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = initial_lr * ( 1 - (iteration / max_iter)) * ( 1 - (iteration / max_iter))
    if ( lr < 1.0e-7 ):
      lr = 1.0e-7

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # 获取整个数据集的大小
    model.train()  # 将模型设置为训练模式，启用训练过程中独有的层（如dropout, batchnorm等）
    running_loss = 0.0
    correct = 0

    for batch, (X, y) in enumerate(dataloader):  # 迭代数据加载器中的每一个批次
        X, y = X.to(device), y.to(device)  # 将输入数据和标签移动到指定的设备（通常是GPU或CPU）

        # Compute prediction error
        pred = model(X)  # 前向传播，计算模型的预测值
        loss = loss_fn(pred, y)  # 计算损失，衡量预测值和真实值之间的差异
        
        # Backpropagation
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 根据计算出的梯度更新模型参数
        optimizer.zero_grad()  # 清零优化器的梯度缓存，为下一次迭代做准备

        running_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 100 == 0:  # 每100个批次打印一次当前的损失和进度
            loss_value = loss.item()  # 获取当前损失值的标量表示
            current = (batch + 1) * len(X)  # 当前处理过的数据量
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")  # 打印损失值和进度
    
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / size
    return avg_loss, accuracy


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # 获取整个数据集的大小
    num_batches = len(dataloader)  # 获取批次数量
    model.eval()  # 将模型设置为评估模式，禁用训练中特有的层（如dropout, batchnorm等）
    test_loss, correct = 0, 0  # 初始化测试损失和正确预测的计数
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 禁用梯度计算，加速计算并减少内存使用
        for X, y in dataloader:  # 迭代数据加载器中的每一个批次
            X, y = X.to(device), y.to(device)  # 将输入数据和标签移动到指定的设备（通常是GPU或CPU）

            pred = model(X)  # 前向传播，计算模型的预测值
            test_loss += loss_fn(pred, y).item()  # 累加损失值
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # 计算正确预测的数量

            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_loss /= num_batches  # 计算平均损失
    correct /= size  # 计算准确率
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")  # 打印测试结果
    return test_loss, correct, all_preds, all_labels

epochs = 30
last_loss = float('inf')

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss, train_accuracy=train(trainloader, model, loss_fn, optimizer)
    test_loss, test_accuracy, all_preds, all_labels=test(testloader, model, loss_fn)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    if test_loss < last_loss:
        torch.save(model.state_dict(), "./best_transfer.pth")
        print("Saved PyTorch Model State to best.pth")
        last_loss = test_loss
    lr = adjust_learning_rate_poly(optimizer, 0.01, t, epochs)
    print("Learning Rate = ",optimizer.param_groups[0]["lr"])
print("Done!")

torch.save(model.state_dict(), f"./last_transfer.pth")
print("Saved PyTorch Model State to last.pth")

# 绘制损失和准确率曲线图
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(range(epochs), train_losses, label='Train Loss')
plt.plot(range(epochs), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs Epochs')

plt.subplot(1, 2, 2)
plt.plot(range(epochs), train_accuracies, label='Train Accuracy')
plt.plot(range(epochs), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy vs Epochs')

plt.savefig('./loss_acc_transfer.png')

plt.show()


# 绘制混淆矩阵
cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3,4,5,6])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['bellflower', 'daisy', 'dandelion', 'lotus', 'rose', 'sunflower', 'tulip']
)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('./confusion_matrix_transfer.png')
plt.show()