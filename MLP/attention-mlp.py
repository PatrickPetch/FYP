import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self, input_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(input_channels, input_channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(input_channels, input_channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(query, key)
        attention = torch.softmax(energy, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, width * height)
        
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        out = self.gamma * out + x
        return out

class CustomMLP(nn.Module):
    def __init__(self):
        super(CustomMLP, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1) # Assuming grayscale image input
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.attention = SelfAttention(32)
        
        self.fc1 = nn.Linear(32 * 32 * 32, 512) # Assuming a 32x32 input image
        self.fc2 = nn.Linear(512 + 3, 128)  # Added 3 to accommodate the coordinates (x, y, z)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, image, coordinates):
        x = self.conv1(image)
        x = nn.ReLU()(x)
        
        x = self.conv2(x)
        x = nn.ReLU()(x)
        
        x = self.attention(x)
        
        x = x.view(x.size(0), -1)
        
        x = torch.cat((x, coordinates), dim=1)
        
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        
        return x

# Creating an instance of the model and printing it
model = CustomMLP()
print(model)
