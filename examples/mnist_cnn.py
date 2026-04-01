"""
MNIST CNN - Train & Save
========================

A ~5MB float32 CNN for MNIST digit classification.

Architecture:
  Conv2d(1, 64, 3, pad=1)        -> BN -> ReLU   [28x28]
  Conv2d(64, 128, 3, pad=1, s=2) -> BN -> ReLU   [14x14]
  Conv2d(128, 256, 3, pad=1, s=2)-> BN -> ReLU   [7x7]
  Conv2d(256, 384, 3, pad=1, s=2)-> BN -> ReLU   [4x4]
  mean(dim=[2,3])  (global avg pool)              [384]
  Linear(384, 10)                                 [10]

Uses strided convolutions for downsampling (no MaxPool2d).

Parameters: ~1.26M  =>  ~5.0 MB (float32)

Usage:
    python examples/mnist_cnn.py
    # Saves weights to examples/mnist_cnn_weights.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MNISTConvNet(nn.Module):
    """
    Simple CNN for MNIST using only ops supported by the Tiny-NN-in-C compiler:
    Conv2d, BatchNorm2d, ReLU, mean (global avg pool), Linear.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(256, 384, 3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(384)
        self.relu4 = nn.ReLU()

        self.fc = nn.Linear(384, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = x.mean(dim=[2, 3])
        x = self.fc(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        running_loss += loss.item()

        if (batch_idx + 1) % 200 == 0:
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}]  "
                  f"loss={running_loss/200:.4f}  acc={100.*correct/total:.1f}%")
            running_loss = 0.0

    print(f"  Epoch {epoch} train accuracy: {100.*correct/total:.2f}%")


def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    test_loss /= total
    accuracy = 100. * correct / total
    print(f"  Test:  loss={test_loss:.4f}  accuracy={accuracy:.2f}% ({correct}/{total})")
    return accuracy


def main():
    import os

    print("=" * 60)
    print("MNIST CNN - Training")
    print("=" * 60)

    SAVE_PATH = os.path.join(os.path.dirname(__file__), "mnist_cnn_weights.pth")
    EPOCHS = 5
    BATCH_SIZE = 128
    LR = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    print("Downloading/loading MNIST dataset...")
    train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = MNISTConvNet().to(device)
    total_params = count_parameters(model)
    size_mb = total_params * 4 / (1024 * 1024)
    print(f"Parameters: {total_params:,}  ({size_mb:.2f} MB float32)")

    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\nTraining for {EPOCHS} epochs...\n")
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch)
        acc = evaluate(model, device, test_loader)
        if acc > best_acc:
            best_acc = acc

    print(f"\nBest test accuracy: {best_acc:.2f}%")

    model.eval()
    model.cpu()
    torch.save(model.state_dict(), SAVE_PATH)
    file_size = os.path.getsize(SAVE_PATH) / (1024 * 1024)
    print(f"Saved weights to {SAVE_PATH} ({file_size:.2f} MB)")
    print("Done.")


if __name__ == "__main__":
    main()
