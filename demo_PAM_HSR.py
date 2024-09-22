import os
import torch.nn.functional as F
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_AFEW import AfewDataset
from GrasNet import *


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
        self.fc = nn.Linear(10000, 7)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.Orth(x)
        x = self.FR1(x)  # 400-300
        x = self.QR(x)
        x = self.ProjMap(x)
        x = self.Pool(x)  # 300-150
        x = self.Orth(x)
        x = self.FR2(x)  # 150-100
        m = x
        x = self.QR(x)  # 100 * 100
        x = self.ProjMap(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x, m


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
transformed_dataset_train = AfewDataset(train=True)
dataloader_train = DataLoader(transformed_dataset_train, batch_size=32,
                    shuffle=True, num_workers=16)
transformed_dataset_val = AfewDataset(train=False)
dataloader_val = DataLoader(transformed_dataset_val, batch_size=32,
                    shuffle=False, num_workers=16)

use_cuda = True
model = GrNet()
if use_cuda:
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.006)


def principal_angle_distance(x1, x2):
    # Perform SVD on the product of transpose of x1 and x2
    u, s, v = torch.svd(torch.matmul(x1.transpose(-2, -1), x2))
    # The singular values are cosines of the principal angles
    principal_angles = torch.acos(s.clamp(min=-1.0, max=1.0))
    # clamp to avoid numerical errors outside the domain of acos
    # Compute distance as sum of squared sines of principal angles
    distance = torch.sum(torch.sin(principal_angles) ** 2)
    return distance


def triplet_loss_with_principal_angle_metric_modified(inputs, targets, marginA, marginB):
    loss = 0.0
    batch_size = inputs.shape[0]

    # Select a random sample as the anchor
    anchor_idx = torch.randint(0, batch_size, (1,)).item()
    anchor = inputs[anchor_idx]
    anchor_label = targets[anchor_idx]

    positive_distances = []
    negative_distances = []

    # Iterate over the rest of the samples in the batch to calculate distances
    for idx in range(batch_size):
        if idx != anchor_idx:
            current_label = targets[idx]
            current_input = inputs[idx]
            current_distance = principal_angle_distance(anchor, current_input)
            if current_label == anchor_label:
                positive_distances.append(current_distance)
            else:
                negative_distances.append((current_distance, idx))

    # Compute loss for positive and negative pairs
    for pos_dist in positive_distances:
        # Print positive sample distance
        # print(f"Positive Distance: {pos_dist.item()}")
        for neg_dist, neg_idx in negative_distances:
            # Print negative sample distance
            # print(f"Negative Distance (Idx {neg_idx}): {neg_dist.item()}")
            # Adjust weight if the negative sample is closer than the positive sample
            if neg_dist < pos_dist:
                weight = 2  # Assign more weight if negative sample is closer
            else:
                weight = 1  # Normal weight for other cases
            # Compute the loss with the weights
            loss += F.relu(pos_dist - marginA).pow(2)
            loss += weight * F.relu(marginB - neg_dist).pow(2)

    # Normalize loss by the total number of pairs considered
    num_pairs = len(positive_distances) * len(negative_distances)
    if num_pairs > 0:
        loss /= num_pairs
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

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()
        outputs1, outputs2 = model(inputs)
        loss1 = criterion(outputs1, targets)
        loss2 = triplet_loss_with_principal_angle_metric_modified(outputs2, targets, marginA=2, marginB=22)
        loss = loss1 + 0.01 * loss2

        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs1.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().data.item()

        train_acc = 100. * correct / total
        bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss / (batch_idx + 1.0), 100. * correct / total, correct, total))

    return (train_loss / (batch_idx + 1), train_acc)


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

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        outputs1, outputs2 = model(inputs)
        loss = criterion(outputs1, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs1.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().data.item()

        test_acc = 100. * correct / total
        bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return (test_loss / (batch_idx + 1), test_acc)


start_epoch = 1
for epoch in range(start_epoch, start_epoch + 300):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
