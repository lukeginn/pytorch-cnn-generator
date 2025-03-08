import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from config.paths import Paths
from src.model.classes.perceptual_loss import PerceptualLoss

class ModelCompiler(nn.Module):
    def __init__(self, config):
        super(ModelCompiler, self).__init__()
        self._load_config(config)
        self._initialize_layers()
        self._set_activation_function()
        self.device = self._set_device()

    def forward(self, x):
        x = self._prepare_input(x)
        x = self._apply_fc_layers(x)
        x = self._reshape(x)
        x = self._apply_deconv_layers(x)
        return x

    def compile(self):
        self._set_loss_function()
        self._set_optimizer()
        return self
    
    def generate_images(self):
        for i in range(10):
            input = self._generate_input_tensor(i)
            generated_image = self._generate_image(input)
            self._save_image(generated_image, Paths.GENERATED_IMAGES_PATH.value, i)

    def _load_config(self, config):
        self.learning_rate = config['model']['learning_rate']
        self.optimizer_name = config['model']['optimizer']
        self.activation_function_name = config['model']['activation_function']
        self.loss_function_name = config['model']['loss_function']

        self.fc_layers_config = config['model']['fc_layers']
        self.deconv_layers_config = config['model']['deconv_layers']

        self.view_shape_channels = config['model']['view_shape']['channels']
        self.view_shape_height = config['model']['view_shape']['height']
        self.view_shape_width = config['model']['view_shape']['width']

    def _initialize_layers(self):
        # Initialize fully connected layers
        self.fc_layers = nn.ModuleList()
        for layer_cfg in self.fc_layers_config:
            self.fc_layers.append(nn.Linear(layer_cfg['in_features'], layer_cfg['out_features']))

        # Initialize deconvolutional layers
        self.deconv_layers = nn.ModuleList()
        for layer_cfg in self.deconv_layers_config:
            self.deconv_layers.append(nn.ConvTranspose2d(
                layer_cfg['in_channels'],
                layer_cfg['out_channels'],
                kernel_size=layer_cfg['kernel_size'],
                stride=layer_cfg['stride'],
                padding=layer_cfg['padding']
            ))

    def _set_activation_function(self):
        if self.activation_function_name == "relu":
            self.activation_function = F.relu
        elif self.activation_function_name == "leaky_relu":
            self.activation_function = F.leaky_relu
        elif self.activation_function_name == "sigmoid":
            self.activation_function = F.sigmoid
        elif self.activation_function_name == "tanh":
            self.activation_function = F.tanh
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function_name}")

    def _set_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        return device

    def _prepare_input(self, x):
        x = x.float()  # Ensure x is of type Float
        x = x.view(-1, self.fc_layers_config[0]['in_features'])  # Reshape x to match the input features of the first FC layer
        return x

    def _apply_fc_layers(self, x):
        for layer in self.fc_layers:
            x = self.activation_function(layer(x))
        return x

    def _reshape(self, x):
        return x.view(-1, self.view_shape_channels, self.view_shape_height, self.view_shape_width)

    def _apply_deconv_layers(self, x):
        for layer in self.deconv_layers:
            x = self.activation_function(layer(x))
        x = torch.tanh(x)
        return x

    def _set_loss_function(self):
        if self.loss_function_name == 'MSELoss':
            self.criterion = nn.MSELoss()  # Mean Squared Error loss
        elif self.loss_function_name == 'L1Loss':
            self.criterion = nn.L1Loss()  # Mean Absolute Error loss
        elif self.loss_function_name == 'BCEWithLogitsLoss':
            self.criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy loss
        elif self.loss_function_name == 'PerceptualLoss':
            self.criterion = PerceptualLoss()  # Custom Perceptual Loss
        else:
            raise ValueError(f"Invalid loss function name: {self.loss_function_name}")

    def _set_optimizer(self):
        if self.optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        elif self.optimizer_name == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        elif self.optimizer_name == 'RMSprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Invalid optimizer name: {self.optimizer_name}")

    def _generate_input_tensor(self, i):
        return torch.tensor([[i]], dtype=torch.float).to(self.device)

    def _generate_image(self, input):
        with torch.no_grad():
            generated_image = self.forward(input)
        generated_image = generated_image.squeeze().cpu().numpy()
        return (generated_image - generated_image.min()) / (generated_image.max() - generated_image.min())

    def _save_image(self, image, output_dir, index):
        plt.imsave(f'{output_dir}/image_{index}.png', image, cmap='gray')