import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os


class ModelCompiler(nn.Module):
    def __init__(self):
        super(ModelCompiler, self).__init__()
        self._initialize_layers()
        self._set_activation_function()
        self.device = self._set_device()

    def forward(self, x):
        x = self._apply_fc_layers(x)
        x = self._reshape(x)
        x = self._apply_deconv_layers(x)
        return x

    def compile(self):
        self._set_loss_function()
        self._set_optimizer()
        return self

    def _initialize_layers(self):
        self.fc1 = nn.Linear(1, 128) # Input is 1 because we are using a single value to generate an image
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, 7 * 7 * 128)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)

    def _apply_fc_layers(self, x):
        x = x.float() # Ensure x is of type Float
        x = x.view(-1, 1) # Reshape x to a single column
        x = self.activation_function(self.fc1(x))
        x = self.activation_function(self.fc2(x))
        x = self.activation_function(self.fc3(x))
        x = self.activation_function(self.fc4(x))
        x = self.activation_function(self.fc5(x))
        return x

    def _reshape(self, x):
        return x.view(-1, 128, 7, 7)  # Reshape to 7x7x128 feature maps

    def _apply_deconv_layers(self, x):
        x = self.activation_function(self.deconv1(x))
        x = torch.tanh(self.deconv2(x))
        return x

    def _set_activation_function(self):
        self.activation_function = F.relu

    def _set_loss_function(self):
        self.criterion = nn.MSELoss()  # Mean Squared Error loss

    def _set_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def _set_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        return device

    def generate_and_save_images(self, num_images=10, output_dir='outputs'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for i in range(num_images):
            # Generate an input tensor
            input = torch.tensor([[i]], dtype=torch.float).to(self.device)
            
            # Pass the input through the model
            with torch.no_grad():
                generated_image = self.forward(input)
            
            # Convert the output to a numpy array and reshape it
            generated_image = generated_image.squeeze().cpu().numpy()
            
            # Normalize the image to the range [0, 1]
            generated_image = (generated_image - generated_image.min()) / (generated_image.max() - generated_image.min())
            
            # Save the image
            plt.imsave(f'{output_dir}/image_{i}.png', generated_image, cmap='gray')
