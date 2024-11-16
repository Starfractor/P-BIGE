import numpy as np
import torch
from tqdm import tqdm
from dataset import dataset_VQ, dataset_TM_eval

action_to_desc = {
        "bend and pull full" : 0,
        "countermovement jump" : 1,
        "left countermovement jump" : 2,
        "left lunge and twist" : 3,
        "left lunge and twist full" : 4,
        "right countermovement jump" : 5,
        "right lunge and twist" : 6,
        "right lunge and twist full" : 7,
        "right single leg squat" : 8,
        "squat" : 9,
        "bend and pull" : 10,
        "left single leg squat" : 11,
        "push up" : 12
    }

def get_class(text):
    out = []
    for t in text:
        out.append(action_to_desc[t])
    return out

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class MotionClassifierCNN(nn.Module):
    def __init__(self, num_classes=13):
        super(MotionClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(3072, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: [batch_size, 1, 64, 263]
        x = self.pool(self.relu(self.conv1(x)))  # Output shape: [batch_size, 32, 44, 33]
        x = self.pool(self.relu(self.conv2(x)))  # Output shape: [batch_size, 64, 14, 11]
        # print(x.shape)
        x = x.view(-1, 3072)  # Flatten the tensor
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
def main():

    train_loader = dataset_VQ.DATALoader("mcs",
                                        64,
                                        window_size=64,
                                        unit_length=4)

    train_loader_iter = dataset_VQ.cycle(train_loader)

    model = MotionClassifierCNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 500

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for input,label in train_loader:
        
            # input,label = next(train_loader_iter)
            input = torch.tensor(np.asarray(input)).to(device)
            label = torch.tensor(get_class(label)).to(device)
            # print(input.shape, label.shape)
            
            optimizer.zero_grad()
            outputs = model(input)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}, Accuracy: {100 * correct / total}%')
        

    test_loader = dataset_VQ.DATALoader("mcs",
                                        64,
                                        window_size=64,
                                        unit_length=4,
                                        mode='eval')
    total = 0
    correct = 0
    with torch.no_grad():
        for input, label in test_loader:
            input = torch.tensor(np.asarray(input)).to(device)
            label = torch.tensor(get_class(label)).to(device)
            output = model(input)
            _, predicted = output.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

    print(f'Validation Accuracy: {100 * correct / total}%')

    torch.save(model.state_dict(), 'motion_classifier_cnn.pth')
