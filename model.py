import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import datasets,transforms
from efficientnet_pytorch import EfficientNet

num_classes = 11
num_epochs = 2
batch_size = 100
learning_rate = 0.001


transform = transforms.Compose([
    transforms.Resize((288,288)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean =[0.5185, 0.5265, 0.5070], std = [0.2507, 0.2384, 0.2739])
])

dataset = datasets.ImageFolder('dataset',transform = transform)

train_set , test_set = train_test_split(dataset, test_size=0.2, random_state= 42)
train_loader = DataLoader(dataset=train_set, 
                                batch_size=batch_size, 
                                shuffle=True)

test_loader = DataLoader(dataset=test_set, 
                                batch_size=batch_size, 
                                shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.model._fc = nn.Linear(1280, 11)
    def forward(self, x):
        x = self.model(x)
        return x
    
model = Model()
model.to(device)
optimizer = torch.optim.ASGD(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the test images: {acc} %')


torch.save(model.state_dict(), 'model.pth')

