import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

# Function to create Gabor filters
def gabor_filter(kernel_size, sigma, theta, Lambda, psi, gamma):
    y, x = np.meshgrid(np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))
    rotx = x * np.cos(theta) + y * np.sin(theta)
    roty = -x * np.sin(theta) + y * np.cos(theta)

    gabor = np.exp(-(rotx ** 2 + gamma ** 2 * roty ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * rotx / Lambda + psi)
    return torch.tensor(gabor, dtype=torch.float32)

# Function to create LoG filter
def laplacian_of_gaussian(kernel_size):
    log_filter = np.array([[0, 0, -1, 0, 0],
                           [0, -1, -2, -1, 0],
                           [-1, -2, 16, -2, -1],
                           [0, -1, -2, -1, 0],
                           [0, 0, -1, 0, 0]], dtype=np.float32)
    return torch.tensor(log_filter, dtype=torch.float32)

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()

        # First Convolutional Block (Gabor, LoG, and Random Kernels)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2, bias=False)
        self.init_conv1_weights()

        # Second Convolutional Block (Intermediate Features)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2, bias=False)
        self.init_conv2_weights()
        self.pool2 = nn.MaxPool2d(2, 2)  # Pooling after second block

        # Third Convolutional Block (Random)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(2, 2)  # Pooling after third block

        # Fourth Convolutional Block (Dilated)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2)

        # Upsampling Layer for residual connection and output
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # 1x1 convolution to match channel dimensions in the residual connection
        self.residual_conv = nn.Conv2d(32, 128, kernel_size=1)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def init_conv1_weights(self):
        kernel_size = 5
        sigma = 1.0
        Lambda = 2.0
        psi = 0
        gamma = 0.5
        gabor_filters = []

        # Add Gabor angles
        #thetas = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        thetas = [0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6, np.pi]

        for theta in thetas:
            gabor = gabor_filter(kernel_size, sigma, theta, Lambda, psi, gamma)
            gabor_filters.append(gabor)

        log_filter = laplacian_of_gaussian(5)
        gabor_filters.append(log_filter)

        with torch.no_grad():
            for i in range(len(gabor_filters)):
                self.conv1.weight[i, 0] = gabor_filters[i]
            nn.init.xavier_uniform_(self.conv1.weight[len(gabor_filters):])

    def init_conv2_weights(self):
        kernel_size = 5
        sigma = 1.0
        Lambda = 2.0
        psi = 0
        gamma = 0.5
        gabor_filters = []

        # Add Gabor angles
        #thetas = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        thetas = [0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6, np.pi]

        for theta in thetas:
            gabor = gabor_filter(kernel_size, sigma, theta, Lambda, psi, gamma)
            gabor_filters.append(gabor)

        with torch.no_grad():
            for i in range(len(gabor_filters)):
                self.conv2.weight[i, 0] = gabor_filters[i]
            nn.init.xavier_uniform_(self.conv2.weight[len(gabor_filters):])

    def forward(self, x):
        # Block 1: First convolution, no pooling
        x = F.relu(self.conv1(x))

        # Block 2: Second convolution, followed by pooling
        x = F.relu(self.conv2(x))
        x_shortcut = x  # Save shortcut connection before pooling
        x = self.pool2(x)  # Downsample to 32x32

        # Block 3: Third convolution, followed by pooling
        x = F.relu(self.conv3(x))
        x = self.pool3(x)  # Downsample to 16x16

        # Block 4: Fourth convolution (dilated)
        x_residual = self.pool2(x_shortcut)  # Downsample x_shortcut from 32x32 to 16x16
        x_residual = self.residual_conv(x_residual)  # 1x1 convolution to match channel dimensions (32 -> 128)
        x = F.relu(self.conv4(x))  # Output 16x16x128

        x = self.upsample(x)  # Upsample x to match 32x32 in case dilated conv doesn't downsample

        # Adding the residual connection
        x = x + x_residual  # Add residual connection

        # Print the shape before flattening
        #print(f"Shape before flattening: {x.shape}")

        # Dynamically flatten the feature map
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, num_features)

        # Print the shape after flattening
        #print(f"Shape after flattening: {x.shape}")

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)
    

class CustomCNNwBatchNorm(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()

        # First Convolutional Block (Gabor, LoG, and Random Kernels)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2, bias=False)
        self.init_conv1_weights()

        # Second Convolutional Block (Intermediate Features)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2, bias=False)
        self.init_conv2_weights()
        self.pool2 = nn.MaxPool2d(2, 2)  # Pooling after second block

        # Third Convolutional Block (Random)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(2, 2)  # Pooling after third block

        # Fourth Convolutional Block (Dilated)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2)

        # Upsampling Layer for residual connection and output
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # 1x1 convolution to match channel dimensions in the residual connection
        self.residual_conv = nn.Conv2d(32, 128, kernel_size=1)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def init_conv1_weights(self):
        kernel_size = 5
        sigma = 1.0
        Lambda = 2.0
        psi = 0
        gamma = 0.5
        gabor_filters = []

        # Add Gabor angles
        #thetas = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        thetas = [0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6, np.pi]

        for theta in thetas:
            gabor = gabor_filter(kernel_size, sigma, theta, Lambda, psi, gamma)
            gabor_filters.append(gabor)

        log_filter = laplacian_of_gaussian(5)
        gabor_filters.append(log_filter)

        with torch.no_grad():
            for i in range(len(gabor_filters)):
                self.conv1.weight[i, 0] = gabor_filters[i]
            nn.init.xavier_uniform_(self.conv1.weight[len(gabor_filters):])

    def init_conv2_weights(self):
        kernel_size = 5
        sigma = 1.0
        Lambda = 2.0
        psi = 0
        gamma = 0.5
        gabor_filters = []

        # Add Gabor angles
        #thetas = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        thetas = [0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6, np.pi]

        for theta in thetas:
            gabor = gabor_filter(kernel_size, sigma, theta, Lambda, psi, gamma)
            gabor_filters.append(gabor)

        with torch.no_grad():
            for i in range(len(gabor_filters)):
                self.conv2.weight[i, 0] = gabor_filters[i]
            nn.init.xavier_uniform_(self.conv2.weight[len(gabor_filters):])

    def forward(self, x):
        # Block 1: First convolution, no pooling
        x = F.relu(F.batch_norm(self.conv1(x)))

        # Block 2: Second convolution, followed by pooling
        x = F.relu(F.batch_norm(self.conv2(x)))
        x_shortcut = x  # Save shortcut connection before pooling
        x = self.pool2(x)  # Downsample to 32x32

        # Block 3: Third convolution, followed by pooling
        x = F.relu(F.batch_norm(self.conv3(x)))
        x = self.pool3(x)  # Downsample to 16x16

        # Block 4: Fourth convolution (dilated)
        x_residual = self.pool2(x_shortcut)  # Downsample x_shortcut from 32x32 to 16x16
        x_residual = self.residual_conv(x_residual)  # 1x1 convolution to match channel dimensions (32 -> 128)
        x = F.relu(self.conv4(x))  # Output 16x16x128

        x = self.upsample(x)  # Upsample x to match 32x32 in case dilated conv doesn't downsample

        # Adding the residual connection
        x = x + x_residual  # Add residual connection

        # Print the shape before flattening
        #print(f"Shape before flattening: {x.shape}")

        # Dynamically flatten the feature map
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, num_features)

        # Print the shape after flattening
        #print(f"Shape after flattening: {x.shape}")

        # Fully connected layers
        x = F.relu(F.batch_norm(self.fc1(x)))
        x = F.dropout(x, 0.5)
        x = F.relu(F.batch_norm(self.fc2(x)))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

model_dict = {
    "modeldw" : lambda classes : CustomCNN(num_classes=classes),
    "modeldw_batchnorm" : lambda classes : CustomCNNwBatchNorm(num_classes=classes)
}



    # class Flatten(torch.nn.Module):
#     def forward(self, x):
#         return x.view(-1, 64)
#     
# conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) 
# 
# conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=3, padding=1) 
# 
# conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) 
# conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
# 
# conv1.weight.data
# model_list = [conv1, 
#                             nn.BatchNorm2d(16),
#                             nn.ReLU(),
#                             conv2, 
#                             nn.BatchNorm2d(32),
#                             nn.ReLU(),
#                             nn.MaxPool2d(2,2),
#                             conv3,
#                             nn.BatchNorm2d(32),
#                             nn.ReLU(),
#                             conv4,
#                             nn.BatchNorm2d(16),
#                             nn.ReLU(),
#                             nn.MaxPool2d(3,4),
#                             Flatten(),
#                             nn.LogSoftmax(dim=1),
#                             nn.Linear(64,32),
#                             nn.Linear(32,2)]
# 
# model = nn.Sequential(*model_list)
# print(model)