import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import logging as logger
import wandb
import time
import numpy as np

class ModelTrainerUtils:
    @staticmethod
    def create_data_loader(images, labels, batch_size, shuffle):
        dataset = TensorDataset(
            torch.tensor(images, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def generate_images(model, data_loader, device):
        true_images = []
        generated_images = []
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(labels) 
                true_images.extend(images.cpu().numpy())
                generated_images.extend(outputs.cpu().numpy())
        return true_images, generated_images

    @staticmethod
    def compute_metrics(true_images, generated_images):
        true_images = np.array(true_images)
        generated_images = np.array(generated_images)
        mae = mean_absolute_error(true_images.flatten(), generated_images.flatten())
        mse = mean_squared_error(true_images.flatten(), generated_images.flatten())
        ssim_score = ssim(true_images, generated_images, win_size=7, channel_axis=1, data_range=1.0)
        psnr_score = psnr(true_images, generated_images, data_range=1.0)
        return mae, mse, ssim_score, psnr_score

    @staticmethod
    def compute_average_metrics(fold_metrics):
        avg_mae = sum([metrics[0] for metrics in fold_metrics]) / len(fold_metrics)
        avg_mse = sum([metrics[1] for metrics in fold_metrics]) / len(fold_metrics)
        avg_ssim = sum([metrics[2] for metrics in fold_metrics]) / len(fold_metrics)
        avg_psnr = sum([metrics[3] for metrics in fold_metrics]) / len(fold_metrics)

        scores = {
            "average_mae": avg_mae,
            "average_mse": avg_mse,
            "average_ssim": avg_ssim,
            "average_psnr": avg_psnr,
        }

        return scores

    @staticmethod
    def initialize_model(model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model, device

    @staticmethod
    def initialize_logging(config):
        if config["logging"]["log_to_wandb"]:
            experiment_name = "experiment_run_on_" + time.strftime("%Y-%m-%d_%H-%M-%S")
            wandb.init(
                project=config["logging"]["project_name"],
                config=config,
                name=experiment_name,
            )
            return wandb.config

    @staticmethod
    def train_one_epoch(model, data_loader, device):
        running_loss = 0.0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            model.optimizer.zero_grad()
            outputs = model(labels) 
            loss = model.criterion(outputs, images)
            loss.backward()
            model.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(data_loader)

    @staticmethod
    def evaluate_and_log(
        trainer, train_images, train_labels, test_images, test_labels, epoch
    ):
        train_metrics = trainer.evaluate(train_images, train_labels, "train")
        ModelTrainerUtils.log_metrics(trainer, train_metrics, "train", epoch)

        test_metrics = trainer.evaluate(test_images, test_labels, "test")
        ModelTrainerUtils.log_metrics(trainer, test_metrics, "test", epoch)

    @staticmethod
    def log_metrics(trainer, metrics, dataset_type, epoch):
        mae, mse, ssim, psnr = metrics
        if trainer.log_to_wandb:
            wandb.log(
                {
                    f"{dataset_type}_mae": mae,
                    f"{dataset_type}_mse": mse,
                    f"{dataset_type}_ssim": ssim,
                    f"{dataset_type}_psnr": psnr,
                },
                step=epoch + 1,
            )

    @staticmethod
    def log_evaluation_metrics(metrics, dataset_type):
        logger.debug(f"Metrics of the model on the {dataset_type} images: {metrics}")

    @staticmethod
    def log_generated_images(generated_images, epoch):
        wandb.log(
            {"generated_images": [wandb.Image(img) for img in generated_images]},
            step=epoch + 1,
        )

    @staticmethod
    def train_and_evaluate_fold(trainer, images, labels, train_index, val_index, fold):
        train_images, val_images = images[train_index], images[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]

        trainer.train(train_images, train_labels, val_images, val_labels)
        metrics = trainer.evaluate(
            val_images, val_labels, dataset_type=f"fold_{fold+1}"
        )
        return metrics

    @staticmethod
    def log_cross_validation_metrics(metrics, k_folds, log_to_wandb):
        mae = metrics["average_mae"]
        mse = metrics["average_mse"]
        ssim = metrics["average_ssim"]
        psnr = metrics["average_psnr"]

        logger.info(f"Cross-Validation with {k_folds} folds:")
        logger.info(f"Average MAE: {mae:.4f}")
        logger.info(f"Average MSE: {mse:.4f}")
        logger.info(f"Average SSIM: {ssim:.4f}")
        logger.info(f"Average PNSR: {psnr:.4f}")

        if log_to_wandb:
            wandb.log(
                {
                    "cv_average_mae": mae,
                    "cv_average_mse": mse,
                    "cv_average_ssim": ssim,
                    "cv_average_psnr": psnr,
                }
            )
