import os
import torch.nn.functional as F
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from examples_YTC import dataset
from examples_YTC.dataset import CustomDataset
from GrNet_GitHub import *


class GrNet(nn.Module):
    def __init__(self, in_datadim=400, out_datadim1=300, out_datadim2=100, out_datadim3=150, embeddim=10):
        super().__init__()
        self.p = embeddim
        self.QR = QRComposition()
        self.ProjMap = Projmap()
        self.FR1 = FRMap(in_datadim, out_datadim1)
        self.FR2 = FRMap(out_datadim3, out_datadim2)
        self.Orth = Orthmap(self.p)
        self.Pool = MixedPoolLayer()
        self.fc = nn.Linear(10000, 47)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.Orth(x)
        x = self.FR1(x) # 400-300
        x = self.QR(x)
        x = self.ProjMap(x)
        # print(x.shape)
        x = self.Pool(x)# 300-150
        x = self.Orth(x)
        x = self.FR2(x) # 150-100
        x = self.QR(x) # 100 * 100
        x = self.ProjMap(x)
        m = x
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x, m

writer = SummaryWriter(log_dir='YTC_supervised_MixPool_Adam_0.1R')
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

data_dir = '/data/Disk_B/yusheng/SPDNet/spdnet_master_Python/examples_YTC'
transformed_dataset_train = CustomDataset(data_dir, split='train')
dataloader_train = DataLoader(transformed_dataset_train, batch_size=32,
                    shuffle=True, num_workers=8)
transformed_dataset_val = CustomDataset(data_dir, split='test')
dataloader_val = DataLoader(transformed_dataset_val, batch_size=32,
                    shuffle=False, num_workers=8)


use_cuda = True
model = GrNet()
# if use_cuda:
#     model = model.cuda()
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adadelta(model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def projection_metric_distance(x1, x2):
    difference = x1 - x2
    distance = torch.norm(difference, 'fro') / torch.sqrt(torch.tensor(2.0))
    return distance


def triplet_loss_with_projection_metric(inputs, targets, marginA, marginB):
    loss = 0.0
    batch_size = inputs.shape[0]

    # Select a random sample as the anchor
    anchor_idx = torch.randint(0, batch_size, (1,)).item()
    anchor = inputs[anchor_idx]
    anchor_label = targets[anchor_idx]

    # Iterate over the rest of the samples in the batch
    for idx in range(batch_size):
        if idx != anchor_idx:
            current_label = targets[idx]
            if current_label == anchor_label:
                positive = inputs[idx]
                positive_distance = projection_metric_distance(anchor, positive)
                # print("P:", positive_distance)
                loss += F.relu(positive_distance - marginA).pow(2)
            else:
                negative = inputs[idx]
                negative_distance = projection_metric_distance(anchor, negative)
                # print("N:", negative_distance)
                loss += F.relu(marginB - negative_distance).pow(2)

    loss /= (batch_size - 1)
    return loss

# Training
def train(epoch):
    global batch_idx, train_acc
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0.0
    total = 0.0
    bar = tqdm(enumerate(dataloader_train))
    for batch_idx, sample_batched in bar:
        inputs = sample_batched['data']
        targets = sample_batched['label'].squeeze()
        #
        # if use_cuda:
        #     inputs = inputs.cuda()
        #     targets = targets.cuda()

        optimizer.zero_grad()
        outputs1, outputs2 = model(inputs)
        loss1 = criterion(outputs1, targets)
        loss2 = triplet_loss_with_projection_metric(outputs2, targets, marginA=2, marginB=22)
        loss = loss1 + 0.05 * loss2

        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs1.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().data.item()

        train_acc = 100. * correct / total
        bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1.0), 100.*correct/total, correct, total))

    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_acc, epoch)
    return (train_loss/(batch_idx+1), train_acc)

best_acc = 0
def test(epoch):
    global batch_idx, test_acc
    model.eval()
    test_loss = 0
    correct = 0.0
    total = 0.0
    bar = tqdm(enumerate(dataloader_val))
    for batch_idx, sample_batched in bar:
        inputs = sample_batched['data']
        targets = sample_batched['label'].squeeze()
        #
        # if use_cuda:
        #     inputs = inputs.cuda()
        #     targets = targets.cuda()

        outputs1, outputs2 = model(inputs)
        loss = criterion(outputs1, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs1.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().data.item()

        test_acc = 100. * correct / total
        bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    writer.add_scalar('Loss/Test', test_loss, epoch)
    writer.add_scalar('Accuracy/Test', test_acc, epoch)
    return (test_loss/(batch_idx+1), test_acc)


start_epoch = 1
for epoch in range(start_epoch, start_epoch+300):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)