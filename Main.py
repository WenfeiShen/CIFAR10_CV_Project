import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import argparse
import sys
from Model import *
from tqdm import tqdm
# Define the models (EfficientNet, ViT, CoAtNet) as per the previous setup

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Training Models on Small CIFAR-10 Subset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--model_name', type=str, choices=['efficientnet', 'vit', 'coatnet'], default='efficientnet', help='Model name to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')

    if "ipykernel_launcher" in sys.argv[0]:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    
    return args

# Function to select model based on name
def get_model(model_name, num_classes=10, image_size=32):
    if model_name == 'efficientnet':
        return EfficientNet(num_classes=num_classes)
    elif model_name == 'vit':
        return ViT(num_classes=num_classes, img_size=image_size)
    elif model_name == 'coatnet':
        return CoAtNet(num_classes=num_classes, img_size=image_size)
    else:
        raise ValueError(f"Model {model_name} not recognized.")

# Main training and testing script
if __name__ == "__main__":
    args = parse_args()

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create small subsets
    np.random.seed(42)
    small_train_indices = np.random.choice(len(trainset), 600, replace=False)
    small_test_indices = np.random.choice(len(testset), 60, replace=False)

    small_trainset = Subset(trainset, small_train_indices)
    small_testset = Subset(testset, small_test_indices)

    # Create data loaders
    trainloader = DataLoader(small_trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(small_testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Initialize model, loss function, and optimizer
    model = get_model(args.model_name, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    print('Start')
    # Training loop
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader, desc=f"Training Epoch {epoch + 1}"), 0):
            inputs, labels = data
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print('Finished Training')

    # Evaluate the model
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on the small CIFAR-10 test set: {100 * correct / total:.2f}%")
