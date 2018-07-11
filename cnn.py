import torch
import torch.nn as nn
import torchvision.datasets as normal_dataset
import torchvision.transforms as transforms
from torch.autograd import Variable
num_epoch=1
batch_size=100
learning_rate=0.001
train_dataset=normal_dataset.MNIST(root='./mnist/', #数据保存路径
                                   train=True,       #是否作为训练集
                                   transform=transforms.ToTensor(),  # 数据如何处理, 自己自定义
                                   download=True)      # 路径下没有的话, 可以下载
test_dataset=normal_dataset.MNIST(root='./mnist/',
                                  train=False,
                                  transform=transforms.ToTensor())
#使用 DataLoader 进行batch训练
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=batch_size,
                                        shuffle=False)
#建立计算图模型
#两层卷积
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        # 使用序列工具快速构建
        self.conv1=nn.Sequential(nn.Conv2d(1,16,kernel_size=5,padding=2),
                                 nn.BatchNorm2d(16),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.conv2=nn.Sequential(nn.Conv2d(16,32,kernel_size=5,padding=2),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.fc=nn.Linear(7*7*32,10)
    def forward(self,x):
        out=self.conv1(x)
        out=self.conv2(out)
        out=out.view(out.size(0),-1) #reshape
        out=self.fc(out)
        return out
cnn=CNN()
'''
如果有GPU
if torch.cuda.is_available():
    cnn = cnn.cuda()
'''
#定义优化器optimizer和损失
loss_fuuc=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(cnn.parameters(),lr=learning_rate)
for epoch in range(num_epoch):
    for i,(image,label) in enumerate(train_loader):
        images=Variable(image)
        labels=Variable(label)
        optimizer.zero_grad()
        outputs=cnn(images)
        loss=loss_fuuc(outputs,labels)
        loss.backward()
        optimizer.step()
        if (i+1)%100==0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'%(epoch+1,num_epoch,i+1,len(test_dataset)//batch_size,loss.data[0]))
cnn.eval() # 改成测试形态, 应用场景如: dropout
correct=0
total=0
for image,label in test_loader:
    images=Variable(image)
    labels=Variable(label)
    outputs=cnn(images)
    _,predicted=torch.max(outputs.data,1)
    total += labels.size(0)
    correct +=(predicted==labels.data).sum()
print('测试 准确率: %d %%'%(100*correct/total))
