import time
from tqdm.auto import tqdm
import numpy as np
from matplotlib import pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
import torch
from torch import nn
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler

from quick_cifar import CifarLoader


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class FirstConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
    def forward(self, x):
        return F.relu(self.bn1(self.conv1(x)))

class PoolFlatten(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, 4).view(len(x), -1)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.layer0 = FirstConv()
        self.pre_linear = PoolFlatten()
        self.in_planes = 64
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pre_linear(out)
        out = self.linear(out)
        return out
    
## sequential variant of https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
def ResNet18():
    net = ResNet(BasicBlock, [2, 2, 2, 2])
    net_seq = nn.Sequential(
        net.layer0,
        *net.layer1,
        *net.layer2,
        *net.layer3,
        *net.layer4,
        net.pre_linear,
        net.linear,
    )
    return net_seq.to(memory_format=torch.channels_last)


def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(), autocast():
        for inputs, labels in loader:
            correct += (model(inputs).argmax(1) == labels).sum().item()
            total += len(inputs)
    return correct / total

# when training for few steps, batchnorm stats don't get updated fast enough.
# so we manually set bn stats using a batch before evaluating
def reset_bn(inputs):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = None
            m.reset_running_stats()
    with torch.no_grad():
        out = model(inputs)

# no augmentations
train_loader = CifarLoader('/tmp', train=True, batch_size=1000)
iters_per_epoch = len(train_loader)
test_loader = CifarLoader('/tmp', train=False, batch_size=2000)


# train using SGD ----
print('training with SGD...')
epochs = 10
lr = 0.15

model = ResNet18().cuda()

opt = SGD(model.parameters(), lr=lr, momentum=0.9)
lr_schedule = np.interp(np.arange(epochs * iters_per_epoch + 1),
                        [0, 5 * iters_per_epoch, epochs * iters_per_epoch],
                        [0, 1, 0])
scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)

losses = []
save_every = 5
i = 0
sds = []
times = []
start_time = time.time()
for ep in tqdm(range(epochs)):
    for inputs, labels in train_loader:
        if i % save_every == 0:
            sds.append({k: v.clone() for k, v in model.state_dict().items()})
            times.append(time.time() - start_time)
        i += 1
        outs = model(inputs)
        loss = F.cross_entropy(outs, labels)
        losses.append(loss.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        scheduler.step()
times.append(time.time() - start_time)
sds.append({k: v.clone() for k, v in model.state_dict().items()})

reset_bn(inputs)
print('train loss:', sum(losses[-iters_per_epoch:])/iters_per_epoch)
print('train accuracy:', evaluate(train_loader))
print('test accuracy:', evaluate(test_loader))
train_accs = []
test_accs = []
for sd in tqdm(sds):
    model.load_state_dict(sd)
    reset_bn(inputs)
    train_accs.append(evaluate(train_loader))
    test_accs.append(evaluate(test_loader))
obj = {'time': times, 'train_acc': train_accs, 'test_acc': test_accs, 'losses': losses}
slow_obj = obj


# train using TopSGD -----
print('training with TopSGD...')
epochs = 25
lr = 0.03

model = ResNet18().cuda()

# cache the backbone features
feats = []
labels = []
with torch.no_grad():
    for inputs, labels_b in train_loader:
        feats_b = model[:-3](inputs)
        feats.append(feats_b)
        labels.append(labels_b)
new_loader = list(zip(feats, labels))

# optimize just the top two layers
opt = SGD(model[-3:].parameters(), lr=lr, momentum=0.9)
lr_schedule = np.interp(np.arange(epochs * iters_per_epoch + 1),
                        [0, 5 * iters_per_epoch, epochs * iters_per_epoch],
                        [0, 1, 0])
scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)

losses = []
save_every = 20
i = 0
sds = []
times = []
start_time = time.time()
for ep in tqdm(range(epochs)):
    for feats, labels in new_loader:
        if i % save_every == 0:
            sds.append({k: v.clone() for k, v in model.state_dict().items()})
            times.append(time.time() - start_time)
        i += 1
        outs = model[-3:](feats)
        loss = F.cross_entropy(outs, labels)
        losses.append(loss.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        scheduler.step()
times.append(time.time() - start_time)
sds.append({k: v.clone() for k, v in model.state_dict().items()})

reset_bn(inputs)
print('train loss:', sum(losses[-iters_per_epoch:])/iters_per_epoch)
print('train accuracy:', evaluate(train_loader))
print('test accuracy:', evaluate(test_loader))
train_accs = []
test_accs = []
for sd in tqdm(sds):
    model.load_state_dict(sd)
    reset_bn(inputs)
    train_accs.append(evaluate(train_loader))
    test_accs.append(evaluate(test_loader))
obj = {'time': times, 'train_acc': train_accs, 'test_acc': test_accs, 'losses': losses}
fast_obj = obj

print('speedup (topsgd vs sgd):', slow_obj['time'][-1] / fast_obj['time'][-1])

# save figure
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(7, 4))
plt.title('ResNet18 on CIFAR-10')
plt.plot(slow_obj['time'], slow_obj['train_acc'], label='SGD (train)',
         linewidth=2)
plt.plot(slow_obj['time'], slow_obj['test_acc'], label='SGD (test)',
         color=colors[0], linestyle='--', linewidth=2)
plt.plot(fast_obj['time'], fast_obj['train_acc'], label='TopSGD (train)',
         linewidth=2)
plt.plot(fast_obj['time'], fast_obj['test_acc'], label='TopSGD (test)',
         color=colors[1], linestyle='--', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Accuracy')
plt.ylim(0.1, 1.04)
plt.legend()
plt.savefig('./topsgd.png', bbox_inches='tight', dpi=150)

