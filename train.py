import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime
import os

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1) 
        self.bn1 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 8, 3, stride=2, padding=1)  
        self.bn3 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2) 

        self.conv4 = nn.Conv2d(8, 16, 3, padding=1) 
        self.bn4 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 16, 3, stride=2, padding=1) 
        self.bn6 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2) 

        self.conv7 = nn.Conv2d(16, 32, 3, padding=1)  
        self.bn7 = nn.BatchNorm2d(32)
        self.conv9 = nn.Conv2d(32, 32, 3, stride=2, padding=1) 
        self.bn9 = nn.BatchNorm2d(32)
        
        self.dropout = nn.Dropout(0.15)
        self.fc = nn.Sequential(
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.bn3(self.conv3(nn.functional.relu(self.bn1(self.conv1(x)))))))
        x = self.dropout(x)
        x = self.pool2(nn.functional.relu(self.bn6(self.conv6(nn.functional.relu(self.bn4(self.conv4(x)))))))
        x = self.dropout(x)
        x = nn.functional.relu(self.bn9(self.conv9(nn.functional.relu(self.bn7(self.conv7(x))))))
        x = self.dropout(x)
        
        # Flatten the tensor before passing to fully connected layer
        x = x.view(x.size(0), -1)  # Flatten the tensor (batch_size, 32 * 1 * 1)
        
        # Fully connected layers
        #x = F.relu(self.fc1(x))
        x = self.fc(x)
        
        return nn.functional.log_softmax(x, dim=1)

def train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
    
    # Initialize model
    model = MNISTModel().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Train for 1 epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'model_mnist_{timestamp}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")

if __name__ == "__main__":
    train() 