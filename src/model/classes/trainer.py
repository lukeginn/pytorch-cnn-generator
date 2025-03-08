import logging as logger
import wandb
import numpy as np
from src.model.classes.trainer_utils import ModelTrainerUtils

# A bug was found in the numpy library that causes the int and bool types to be overwritten.
# This code snippet is a workaround to fix the issue.
np.int = int
np.bool = bool


class ModelTrainer:
    """
    A class to handle the training and evaluation of a generative model for image generation.

    This class provides methods to train the model on the training dataset and evaluate its performance on the test dataset.

    Attributes:
        batch_size (int): The number of samples per batch to load.
        epochs (int): The number of times to iterate over the training dataset.
        model (nn.Module): The generative model to be trained and evaluated.
        device (torch.device): The device on which to perform computations (CPU or GPU).

    Methods:
        train(train_images, train_labels, test_images, test_labels):
            Trains the model on the provided training dataset.
            Args:
                train_images (numpy.ndarray): The training images.
                train_labels (numpy.ndarray): The training labels.
                test_images (numpy.ndarray): The test images.
                test_labels (numpy.ndarray): The test labels.
            Purpose: Performs the training loop, including forward pass, loss computation, backward pass, and optimizer step.

        evaluate(images, labels, epoch):
            Evaluates the model on the provided test dataset.
            Args:
                images (numpy.ndarray): The images to evaluate.
                labels (numpy.ndarray): The labels to evaluate.
                epoch (int): The current epoch number.
            Purpose: Computes the quality of the generated images.
    """

    def __init__(self, model, config, log_to_wandb):
        """
        Initializes the ModelTrainer with the model and configuration.

        Args:
            model (nn.Module): The generative model to be trained and evaluated.
            config (object): The configuration object containing training parameters.
        """
        self._initialize_config(config)
        self.model, self.device = ModelTrainerUtils.initialize_model(model)
        wandb.config = ModelTrainerUtils.initialize_logging(config)

    def train(self, train_images, train_labels, test_images, test_labels):
        """
        Trains the model on the provided training dataset.

        Args:
            train_images (numpy.ndarray): The training images.
            train_labels (numpy.ndarray): The training labels.
            test_images (numpy.ndarray): The test images.
            test_labels (numpy.ndarray): The test labels.

        Purpose:
            - Sets the model to training mode.
            - Creates a DataLoader for the training dataset.
            - Iterates over the dataset for the specified number of epochs.
            - For each batch, performs the following steps:
                - Moves the images to the appropriate device (CPU or GPU).
                - Computes the loss between the generated images and the true images.
                - Performs a backward pass to compute the gradients.
                - Updates the model's parameters using the optimizer.
                - Accumulates the running loss for monitoring.
            - Logs the average loss for each epoch.
        """
        self.model.train()
        data_loader = ModelTrainerUtils.create_data_loader(
            train_images, train_labels, self.batch_size, self.training_shuffle
        )

        for epoch in range(self.epochs):
            avg_loss = ModelTrainerUtils.train_one_epoch(
                self.model, data_loader, self.device
            )
            logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
            if self.log_to_wandb:
                wandb.log({"epoch": epoch + 1, "loss": avg_loss}, step=epoch + 1)

            if (epoch + 1) % self.evaluation_frequency == 0:
                self.evaluate(test_images, test_labels, epoch)

        if self.log_to_wandb:
            wandb.finish()
            logger.info("WandB run has been stopped.")

    def evaluate(self, images, labels, epoch):
        """
        Evaluates the model on the provided dataset.

        Args:
            images (numpy.ndarray): The images to evaluate.
            labels (numpy.ndarray): The labels to evaluate.
            epoch (int): The current epoch number.

        Purpose:
            - Sets the model to evaluation mode.
            - Creates a DataLoader for the dataset.
            - Iterates over the dataset without computing gradients.
            - For each batch, performs the following steps:
                - Moves the images to the appropriate device (CPU or GPU).
                - Generates images using the model.
                - Computes the quality of the generated images.
            - Logs the evaluation metrics.
        """
        self.model.eval()
        data_loader = ModelTrainerUtils.create_data_loader(
            images, labels, self.batch_size, self.evaluation_shuffle
        )

        generated_images = ModelTrainerUtils.generate_images(
            self.model, data_loader, self.device
        )

        # Log generated images to WandB
        if self.log_to_wandb:
            wandb.log(
                {"generated_images": [wandb.Image(img) for img in generated_images]},
                step=epoch + 1,
            )

        return generated_images

    def _initialize_config(self, config):
        self.log_to_wandb = config["logging"]["log_to_wandb"]
        self.batch_size = config["model"]["batch_size"]
        self.epochs = config["model"]["epochs"]
        self.training_shuffle = config["model"]["shuffle"]
        self.evaluation_shuffle = config["evaluation"]["shuffle"]
        self.evaluation_frequency = config["evaluation"]["epoch_frequency"]
