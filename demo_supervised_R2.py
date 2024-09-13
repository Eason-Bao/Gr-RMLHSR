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
from sklearn.metrics import precision_score, recall_score, f1_score


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
        x = self.Pool(x)# 300-150
        x = self.Orth(x)
        x = self.FR2(x) # 150-100
        m = x
        x = self.QR(x) # 100 * 100
        x = self.ProjMap(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x, m

writer = SummaryWriter(log_dir='YTC_PAM_M_indicators')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_dir = '/root/autodl-tmp/spdnet_master_Python/examples_YTC'
transformed_dataset_train = CustomDataset(data_dir, split='train')
dataloader_train = DataLoader(transformed_dataset_train, batch_size=32,
                    shuffle=True, num_workers=8)
transformed_dataset_val = CustomDataset(data_dir, split='test')
dataloader_val = DataLoader(transformed_dataset_val, batch_size=32,
                    shuffle=False, num_workers=8)


use_cuda = True
model = GrNet()
if use_cuda:
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0006)

def principal_angle_distance(x1, x2):
    # Perform SVD on the product of transpose of x1 and x2
    u, s, v = torch.svd(torch.matmul(x1.transpose(-2, -1), x2))
    # The singular values are cosines of the principal angles
    principal_angles = torch.acos(s.clamp(min=-1.0, max=1.0))  # clamp to avoid numerical errors outside the domain of acos
    # Compute distance as sum of squared sines of principal angles
    # distance = torch.sum(torch.sin(principal_angles) ** 2)
    distance = torch.sum(principal_angles ** 2)
    return distance

def triplet_loss_with_principal_angle_metric(inputs, targets, marginA, marginB):
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
                positive_distance = principal_angle_distance(anchor, positive)
                # print("P:", positive_distance)
                loss += F.relu(positive_distance - marginA).pow(2)
            else:
                negative = inputs[idx]
                negative_distance = principal_angle_distance(anchor, negative)
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
    all_targets = []
    all_predicted = []
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
        loss2 = triplet_loss_with_principal_angle_metric(outputs2, targets, marginA=2, marginB=12)
        loss = loss1 + 0.01 * loss2

        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs1.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().data.item()

        all_targets.extend(targets.cpu().numpy())
        all_predicted.extend(predicted.cpu().numpy())

        train_acc = 100. * correct / total
        bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss / (batch_idx + 1.0), 100. * correct / total, correct, total))

    precision = precision_score(all_targets, all_predicted, average='macro')
    recall = recall_score(all_targets, all_predicted, average='macro')
    f1 = f1_score(all_targets, all_predicted, average='macro')

    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_acc, epoch)
    writer.add_scalar('Precision/Train', precision, epoch)
    writer.add_scalar('Recall/Train', recall, epoch)
    writer.add_scalar('F1/Train', f1, epoch)

    return (train_loss / (batch_idx + 1), train_acc, precision, recall, f1)


best_acc = 0


def test(epoch):
    global batch_idx, test_acc, best_acc
    model.eval()
    test_loss = 0
    correct = 0.0
    total = 0.0
    all_targets = []
    all_predicted = []
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

        all_targets.extend(targets.cpu().numpy())
        all_predicted.extend(predicted.cpu().numpy())

        test_acc = 100. * correct / total
        bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    precision = precision_score(all_targets, all_predicted, average='macro')
    recall = recall_score(all_targets, all_predicted, average='macro')
    f1 = f1_score(all_targets, all_predicted, average='macro')

    writer.add_scalar('Loss/Test', test_loss, epoch)
    writer.add_scalar('Accuracy/Test', test_acc, epoch)
    writer.add_scalar('Precision/Test', precision, epoch)
    writer.add_scalar('Recall/Test', recall, epoch)
    writer.add_scalar('F1/Test', f1, epoch)

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), f"best_model_epoch_{epoch}.pth")

    return (test_loss / (batch_idx + 1), test_acc, precision, recall, f1)


start_epoch = 1
for epoch in range(start_epoch, start_epoch + 500):
    train_loss, train_acc, train_precision, train_recall, train_f1 = train(epoch)
    test_loss, test_acc, test_precision, test_recall, test_f1 = test(epoch)
    print(
        f"Epoch {epoch}: Train Loss {train_loss}, Train Acc {train_acc}, Train Precision {train_precision}, Train Recall {train_recall}, Train F1 {train_f1}")
    print(
        f"Test Loss {test_loss}, Test Acc {test_acc}, Test Precision {test_precision}, Test Recall {test_recall}, Test F1 {test_f1}")
