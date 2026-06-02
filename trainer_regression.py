"""
Trainer for regression tasks, adapted from the original classification trainer.
- Supports MSE/SmoothL1/Huber losses.
- Optionally adds a differentiable Pearson correlation loss.
- Early stopping can use MAE/RMSE/Pearson/R2 or a composite validation score.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import seed_everything, get_device, count_parameters, get_optimizer
from models.bscan_regression import BscanRegression
from models.deepcirccode_regression import DeepCircCodeRegression
from models.bscan_v2_regression import BSCANv2Regression
from models.bscan_seq_regression import BSCANSeqRegression
from models.bscan_seq_lite_regression import BSCANSeqLiteRegression
from models.bscan_plus_regression import BSCANPlusRegression
from models.bscan_seq_rcattn_regression import BSCANSeqRCAttnRegression
from models.bscan_mamba_xattn_regression import BSCANMambaXAttnRegression
from models.bscan_region_interact_regression import BSCANRegionInteractRegression
from models.circcnn_regression import CircCNNRegression


class RegressionTrainer:
    def __init__(
        self,
        seed=42,
        device='cpu',
        dir_save='./saved_models/',
        loss_name='smoothl1',
        corr_loss_weight=0.0,
        huber_delta=1.0,
        early_metric='r2',
    ):
        self.seed = seed
        self.device = device
        seed_everything(self.seed)
        self.loss_name = loss_name
        self.corr_loss_weight = corr_loss_weight
        self.huber_delta = huber_delta
        self.early_metric = early_metric
        self.loss_fn = self._build_loss(loss_name, huber_delta)
        
        self.dir_save = dir_save
        os.makedirs(self.dir_save, exist_ok=True)

    def _build_loss(self, loss_name, huber_delta):
        loss_name = loss_name.lower()
        if loss_name == 'mse':
            return nn.MSELoss()
        if loss_name in ('smoothl1', 'huber'):
            return nn.SmoothL1Loss(beta=huber_delta)
        if loss_name == 'mae':
            return nn.L1Loss()
        raise ValueError(f"Unsupported regression loss: {loss_name}")

    def _pearson_loss(self, preds, labels, eps=1e-8):
        preds = preds - preds.mean()
        labels = labels - labels.mean()
        denom = torch.sqrt(torch.sum(preds ** 2) + eps) * torch.sqrt(torch.sum(labels ** 2) + eps)
        corr = torch.sum(preds * labels) / denom
        return 1.0 - corr

    def compute_loss(self, preds, labels):
        if getattr(self, "use_target_norm", False):
            preds = (preds - self.target_mean) / self.target_std
            labels = (labels - self.target_mean) / self.target_std
        loss = self.loss_fn(preds, labels)
        if self.corr_loss_weight > 0:
            loss = loss + self.corr_loss_weight * self._pearson_loss(preds, labels)
        return loss

    def _validation_score(self, scores):
        mae, rmse, pearson_corr, _, r2 = scores
        metric = self.early_metric.lower()
        if metric == 'mae':
            return -mae
        if metric == 'rmse':
            return -rmse
        if metric == 'pearson':
            return pearson_corr
        if metric == 'r2':
            return r2
        if metric == 'composite':
            # Prefer explained variance and rank agreement while mildly penalizing absolute error.
            return r2 + pearson_corr - 0.1 * mae
        raise ValueError(f"Unsupported early stopping metric: {self.early_metric}")

    def define_model(self, model_name, **kwargs):
        """Define the model architecture."""
        self.model_name = model_name
        if model_name == 'bscan_regression':
            self.model = BscanRegression(**kwargs).to(self.device)
            self.num_inputs = 2
        elif model_name == 'bscan_v2_regression':
            self.model = BSCANv2Regression(**kwargs).to(self.device)
            self.num_inputs = 3
        elif model_name == 'bscan_seq_regression':
            self.model = BSCANSeqRegression(**kwargs).to(self.device)
            self.num_inputs = 2
        elif model_name == 'bscan_seq_lite_regression':
            self.model = BSCANSeqLiteRegression(**kwargs).to(self.device)
            self.num_inputs = 2
        elif model_name == 'bscan_seq_lite_xattn_regression':
            self.model = BSCANSeqLiteRegression(use_cross_attention=True, **kwargs).to(self.device)
            self.num_inputs = 2
        elif model_name == 'bscan_plus_regression':
            self.model = BSCANPlusRegression(**kwargs).to(self.device)
            self.num_inputs = 2
        elif model_name == 'bscan_seq_rcattn_regression':
            self.model = BSCANSeqRCAttnRegression(**kwargs).to(self.device)
            self.num_inputs = 2
        elif model_name == 'bscan_mamba_xattn_regression':
            self.model = BSCANMambaXAttnRegression(**kwargs).to(self.device)
            self.num_inputs = 2
        elif model_name == 'bscan_region_interact_regression':
            self.model = BSCANRegionInteractRegression(**kwargs).to(self.device)
            self.num_inputs = 2
        elif model_name == 'circcnn_regression':
            self.model = CircCNNRegression(**kwargs).to(self.device)
            self.num_inputs = 2
        elif model_name == 'deepcirccode_regression':
            self.model = DeepCircCodeRegression(**kwargs).to(self.device)
            self.num_inputs = 1
        else:
            raise ValueError(f"Invalid model name for regression: {model_name}")
        
        print(f"Model: {model_name}, Parameters: {count_parameters(self.model):,}")
        if getattr(self, "use_target_norm", False):
            self._init_output_bias()

    def _init_output_bias(self):
        """Initialize the last scalar output bias near the train label mean."""
        bias_value = float(self.target_mean.item())
        candidate = None
        if hasattr(self.model, "classifier"):
            candidate = getattr(self.model.classifier, "net", None)
            if candidate is None:
                candidate = getattr(self.model.classifier, "fc", None)
        if candidate is not None and isinstance(candidate, nn.Sequential):
            for module in reversed(candidate):
                if isinstance(module, nn.Linear) and module.out_features == 1:
                    nn.init.constant_(module.bias, bias_value)
                    break

    def set_dataloaders(self, train_dataset, valid_dataset, test_dataset, batch_size=32):
        """Set up DataLoaders for train, validation, and test sets."""
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        train_labels = train_dataset.y.float()
        self.target_mean = train_labels.mean().to(self.device)
        self.target_std = train_labels.std(unbiased=False).to(self.device).clamp_min(1e-6)
        self.use_target_norm = True

    def save_model(self):
        """Save the model's state dictionary."""
        dir_save = os.path.join(self.dir_save, self.model_name, str(self.seed))
        os.makedirs(dir_save, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(dir_save, 'model.pth'))

    def get_scores(self, y_true, y_pred):
        """Calculate and return various regression evaluation metrics."""
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Correlation can only be computed if there's variance
        if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
            pearson_corr, _ = pearsonr(y_true, y_pred)
            spearman_corr, _ = spearmanr(y_true, y_pred)
        else:
            pearson_corr, spearman_corr = 0.0, 0.0

        return mae, rmse, pearson_corr, spearman_corr, r2

    def train(self, optimizer_name="adamw", lr=1e-4, epochs=100, early_stop_patience=10, verbose=True):
        """Train the model with configurable early stopping."""
        self.verbose = verbose
        optimizer = get_optimizer(self.model.parameters(), optimizer_name, lr)
        
        best_score = -float('inf')
        patience_counter = 0
        best_epoch = 0

        if self.verbose:
            print(f'Training {self.model_name} for {epochs} epochs...')
            print(f'Loss: {self.loss_name} | Corr loss weight: {self.corr_loss_weight} | Early metric: {self.early_metric}')
            if getattr(self, "use_target_norm", False):
                print(f'Target norm: mean={self.target_mean.item():.4f} std={self.target_std.item():.4f}')
            print('-' * 80)
            print('| Split | Epoch | Loss   | MAE    | RMSE   | Pearson r | R2     |')

        for epoch in range(epochs):
            # Training step
            self.model.train()
            train_loss, train_preds, train_labels = self.run_epoch(self.train_loader, is_train=True, optimizer=optimizer)
            train_scores = self.get_scores(train_labels, train_preds)
            if self.verbose:
                print(f'| Train | {epoch+1:^5d} | {train_loss:.4f} | {train_scores[0]:.4f} | {train_scores[1]:.4f} | {train_scores[2]:.4f}    | {train_scores[4]:.4f} |')

            # Validation step
            self.model.eval()
            with torch.no_grad():
                val_loss, val_preds, val_labels = self.run_epoch(self.valid_loader, is_train=False)
                val_scores = self.get_scores(val_labels, val_preds)
                val_mae = val_scores[0]
                val_score = self._validation_score(val_scores)
                if self.verbose:
                    print(f'| Valid | {epoch+1:^5d} | {val_loss:.4f} | {val_mae:.4f} | {val_scores[1]:.4f} | {val_scores[2]:.4f}    | {val_scores[4]:.4f} | score={val_score:.4f}', end='')

                if val_score > best_score:
                    best_score = val_score
                    best_epoch = epoch
                    patience_counter = 0
                    self.save_model()
                    if self.verbose:
                        print(' -> Best model saved!')
                else:
                    patience_counter += 1
                    if self.verbose:
                        print()

            if patience_counter >= early_stop_patience:
                print(f'Early stopping at epoch {epoch+1}. Best validation {self.early_metric}: {best_score:.4f} at epoch {best_epoch+1}.')
                break
        
        # Load best model and evaluate on test set
        best_model_path = os.path.join(self.dir_save, self.model_name, str(self.seed), 'model.pth')
        self.model.load_state_dict(torch.load(best_model_path, weights_only=True))
        
        self.model.eval()
        with torch.no_grad():
            test_loss, test_preds, test_labels = self.run_epoch(self.test_loader, is_train=False)
        
        test_scores = self.get_scores(test_labels, test_preds)
        print('-' * 80)
        print("Test Set Performance of Best Model:")
        print(f'| Test  | Loss: {test_loss:.4f} | MAE: {test_scores[0]:.4f} | RMSE: {test_scores[1]:.4f} | Pearson r: {test_scores[2]:.4f} | R2: {test_scores[4]:.4f} |')
        print('-' * 80)
        
        return {
            "test_loss": test_loss,
            "test_mae": test_scores[0],
            "test_rmse": test_scores[1],
            "test_pearson": test_scores[2],
            "test_spearman": test_scores[3],
            "test_r2": test_scores[4],
        }

    def run_epoch(self, data_loader, is_train, optimizer=None):
        """Runs a single epoch of training or validation."""
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for data_batch in data_loader:
            labels = data_batch[-1].to(self.device)
            
            if self.num_inputs == 1:
                sequences = data_batch[0].to(self.device)
                if is_train:
                    optimizer.zero_grad()
                preds = self.model(sequences)
            elif self.num_inputs == 2:
                upper_seq, lower_seq = data_batch[0].to(self.device), data_batch[1].to(self.device)
                if is_train:
                    optimizer.zero_grad()
                preds = self.model(upper_seq, lower_seq)
            elif self.num_inputs == 3:
                upper_seq = data_batch[0].to(self.device)
                lower_seq = data_batch[1].to(self.device)
                lower_rc_seq = data_batch[2].to(self.device)
                if is_train:
                    optimizer.zero_grad()
                preds = self.model(upper_seq, lower_seq, lower_rc_seq)
            else:
                raise ValueError(f"Unsupported number of inputs: {self.num_inputs}")

            loss = self.compute_loss(preds, labels)
            
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            all_preds.append(preds.detach())
            all_labels.append(labels.detach())

        avg_loss = total_loss / len(data_loader)
        return avg_loss, torch.cat(all_preds), torch.cat(all_labels)
