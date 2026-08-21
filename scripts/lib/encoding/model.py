"""
Encoding model for predicting fMRI responses from audio features.

This module implements the EncodingModel class that uses CochResNet50 features
with ridge regression to predict voxel-wise fMRI responses.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import einops
import himalaya
from sklearn.model_selection import cross_val_predict
from tqdm import tqdm
import warnings

from cochresnet50 import instantiate_cochresnet50
from lib.utils.math import mycorr
from lib.encoding.preprocessing import Lag, StandardScaler
from lib.encoding.ridge_utils import check_ridge_alphas
import chcochleagram
from chcochleagram import compression
from compression import ClippedGradPowerCompression

# Layer name to module path mapping for CochResNet50
# Defined here to avoid circular imports
layers = {
    'relu1': '1.layer1.2.relu',
    'relu2': '1.layer2.3.relu',
    'relu3': '1.layer3.5.relu',
    'relu4': '1.layer4.2.relu',
    'avgpool': '1.avgpool',
}


class EncodingModel:
    """
    Encoding model that predicts fMRI responses from audio using CochResNet50 features.
    
    This model extracts features from multiple layers of a CochResNet50 network,
    applies PCA dimensionality reduction, and fits ridge regression models to predict
    voxel-wise fMRI responses. The best layer for each voxel is selected based on
    cross-validated performance.
    """
    
    def __init__(self, 
                 model_name='cochresnet50',
                 robust=True,
                 n_pcs=80,
                 n_lags=0,
                 ridge_kwargs=None,
                 layers=layers,
                 standardize_features=True,
                 device='cuda'):
        """
        Initialize encoding model.
        
        Args:
            model_name: Name of base model (default: 'cochresnet50')
            robust: Whether to use robust or standard pretrained weights
            n_pcs: Number of principal components for dimensionality reduction
            n_lags: Number of temporal lags to include (default: 0)
            ridge_kwargs: Keyword arguments for RidgeCV
            layers: Dictionary mapping layer names to module paths
            standardize_features: Whether to standardize features
            device: Device to use for computation ('cuda' or 'cpu')
        """
        if ridge_kwargs is None:
            ridge_kwargs = {
                'alphas': np.logspace(-3, 8, 200), 
                'fit_intercept': True, 
                'warn': False
            }
        
        self.model_name = model_name
        self.robust = robust
        self.n_pcs = n_pcs
        self.n_lags = n_lags
        self.lag = Lag(self.n_lags)
        self.ridge_kwargs = ridge_kwargs
        self.layers = layers
        self.standardize_features = standardize_features
        self.device = device

        # Initialize CochResNet50 model
        model = instantiate_cochresnet50(robust=robust)
        _ = model.eval()
        
        # Register hooks to capture activations
        for name, module in model.named_modules():
            module.register_forward_hook(self.get_activation(name))
        
        model.to(device)
        self.model = model
        self.activations = {}
        self.V = {}
        self.scalers = {}

    def cpu_(self):
        """Move model and parameters to CPU."""
        self.model.to('cpu')
        self.scalers = {k: v.to('cpu') for k, v in self.scalers.items()}
        self.activations = {}
        self.device = 'cpu'

    def cuda_(self, device='cuda'):
        """Move model and parameters to CUDA device."""
        self.model.to(device)
        self.coefs = {k: torch.from_numpy(v).to(device) for k, v in self.coefs.items()}
        self.intercepts = {k: torch.from_numpy(v).to(device) for k, v in self.intercepts.items()}
        self.V = {k: v.to(device) for k, v in self.V.items()}
        self.scalers = {k: v.to(device) for k, v in self.scalers.items()}
        self.activations = {}
        self.device = device
        
    def get_activation(self, name):
        """Create hook function to capture layer activations."""
        def hook(model, input, output):
            self.activations[name] = output
        return hook
    
    def reshape_activations(self, A):
        """
        Reshape activations to 2D (samples x features).
        
        Args:
            A: Activations tensor of shape (n, t, f) or (n, t, h, w)
        
        Returns:
            Reshaped activations of shape (n, features)
        """
        if A.ndim == 3:
            A = einops.rearrange(A, 'n t f -> n (t f)')
        elif A.ndim == 4:
            A = einops.rearrange(A, 'n t h w -> n (t h w)')
        else:
            raise ValueError(f"Activation shape {A.shape} not recognized")
        return A
    
    def fit(self, D, X, D_test=None, X_test=None):
        """
        Fit encoding model to training data.
        
        Args:
            D: Training fMRI responses (n_stimuli x n_voxels)
            X: Training audio stimuli (n_stimuli x n_channels x n_samples)
            D_test: Test fMRI responses (optional)
            X_test: Test audio stimuli (optional)
        """
        # Extract features from all layers
        _ = self.model(X.to(self.device))
        A = {layer: self.activations[self.layers[layer]].detach().cpu() 
             for layer in self.layers}
        A = {k: self.reshape_activations(v) for k, v in A.items()}
        
        # Initialize storage for predictions and metrics
        self.layerwise_train_predictions = {}
        self.layerwise_train_correlations = {}
        self.layerwise_train_MSEs = {}
        self.layerwise_test_predictions = {}
        self.layerwise_test_correlations = {}
        self.layerwise_test_MSEs = {}
        
        # Fit ridge regression for each layer
        for layer in tqdm(A):
            print(f"Working on layer {layer}")
            
            # Standardize features
            if self.standardize_features:
                self.scalers[layer] = StandardScaler(mean_dim=0, use_norm=True)
                A[layer] = self.scalers[layer].fit_transform(A[layer])
            
            # Apply PCA
            if self.n_pcs is not None:
                u, s, v = torch.linalg.svd(A[layer], full_matrices=False)
                if u.shape[-1] < self.n_pcs:
                    self.n_pcs = u.shape[-1]
                    print(f"Reducing n_pcs to {self.n_pcs}")
                A[layer] = u[:, :self.n_pcs] * s[:self.n_pcs]
                self.V[layer] = v[:self.n_pcs]
            
            # Apply temporal lagging
            A[layer] = self.lag(A[layer])
            assert A[layer].shape[0] == D.shape[0], \
                f"Number of samples in activations {A[layer].shape[0]} does not match data {D.shape[0]}"

            # Cross-validated predictions
            ridge = himalaya.ridge.RidgeCV(**self.ridge_kwargs)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.layerwise_train_predictions[layer] = cross_val_predict(ridge, A[layer], D, cv=5)
                self.layerwise_train_correlations[layer] = mycorr(D, self.layerwise_train_predictions[layer])
                self.layerwise_train_MSEs[layer] = ((D - self.layerwise_train_predictions[layer])**2).mean(0)
        
        # Select best layer for each voxel
        stacked_MSEs = np.stack([self.layerwise_train_MSEs[layer] for layer in self.layers])
        layer_names = list(self.layers.keys())
        best_layer_indices = np.argmin(stacked_MSEs, axis=0)
        self.best_layers = {i: layer_names[idx] for i, idx in enumerate(best_layer_indices)}
        print(f"Counts of best layers: {np.unique(list(self.best_layers.values()), return_counts=True)}")

        # Fit final ridge models on full training data
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            ridge_models = {layer: himalaya.ridge.RidgeCV(**self.ridge_kwargs).fit(A[layer], D) 
                          for layer in self.layers}
        
        self.alphas = {layer: ridge.best_alphas_ for layer, ridge in ridge_models.items()}
        for layer in self.layers:
            print(f"Layer: {layer}")
            check_ridge_alphas(ridge_models[layer])

        self.coefs = {layer: ridge_models[layer].coef_ for layer in self.layers}
        self.intercepts = {layer: ridge_models[layer].intercept_ for layer in self.layers}

        # Evaluate on test set if provided
        if X_test is not None:
            _ = self.model(X_test.to(self.device))
            A_test = {layer: self.activations[self.layers[layer]].cpu().detach() 
                     for layer in self.layers}
            A_test = {k: self.reshape_activations(v) for k, v in A_test.items()}
            
            if self.standardize_features:
                A_test = {layer: self.scalers[layer].transform(A_test[layer]) 
                         for layer in A_test}
            
            if self.n_pcs is not None:
                A_test = {layer: A_test[layer] @ self.V[layer].T for layer in A_test}
            
            A_test = {layer: self.lag(A_test[layer]) for layer in A_test}
            
            self.all_predictions = {layer: ridge_models[layer].predict(A_test[layer]) 
                                   for layer in self.layers}
            self.layerwise_test_predictions = self.all_predictions
            self.layerwise_test_correlations = {
                layer: mycorr(D_test, self.layerwise_test_predictions[layer]) 
                for layer in self.layers
            }
            self.layerwise_test_MSEs = {
                layer: ((D_test - self.layerwise_test_predictions[layer])**2).mean(0) 
                for layer in self.layers
            }
            
            # Combine best predictions
            best_predictions = []
            for voxel in self.best_layers:
                best_predictions.append(
                    self.all_predictions[self.best_layers[voxel]][:, voxel].reshape(-1, 1)
                )
            self.best_predictions = np.concatenate(best_predictions, axis=-1)
            
            if D_test is not None:
                self.corrs_test = mycorr(D_test, self.best_predictions)
                print(f"Correlation between predicted and actual D using best layers: "
                      f"{self.corrs_test.mean()}")
                self.MSE_test = ((D_test - self.best_predictions)**2).mean(0)
                print(f"Mean squared error between predicted and actual D using best layers: "
                      f"{self.MSE_test.mean()}")

    def predict(self, X, return_features=False):
        """
        Predict fMRI responses for new audio stimuli.
        
        Args:
            X: Audio stimuli tensor
            return_features: Whether to also return extracted features
        
        Returns:
            Predicted fMRI responses (and features if return_features=True)
        """
        if X.ndim == 1:
            X = X.reshape(1, 1, -1)
        elif X.ndim == 2:
            X = X.unsqueeze(1)

        _ = self.model(X.to(self.device))
        A = {layer: self.activations[self.layers[layer]] for layer in self.layers}
        A = {k: self.reshape_activations(v) for k, v in A.items()}
        
        if self.standardize_features:
            A = {layer: self.scalers[layer].transform(A[layer]) for layer in A}
        
        if self.n_pcs is not None:
            A = {layer: A[layer] @ self.V[layer].T for layer in A}
        
        if self.n_lags > 0:
            A = {layer: self.lag(A[layer]) for layer in A}

        predictions_all_layers = {
            layer: A[layer] @ self.coefs[layer] + self.intercepts[layer] 
            for layer in self.coefs
        }
        
        best_predictions = []
        for voxel in self.best_layers:
            best_predictions.append(
                predictions_all_layers[self.best_layers[voxel]][:, voxel].reshape(-1, 1)
            )
        best_predictions = torch.concatenate(best_predictions, axis=-1)
        
        if return_features:
            return best_predictions, A
        else:
            return best_predictions

