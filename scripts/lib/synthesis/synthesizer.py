"""
Sound synthesis using encoding model predictions.

This module implements synthesizers that generate audio stimuli to match
target encoding model predictions while decorrelating predictions across models.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from typing import Union, Optional, List, Dict
from pathlib import Path
import itertools
import numpy as np
import torch
import torchaudio
import einops
from himalaya.ridge import RidgeCV
from tqdm.auto import tqdm
import warnings
from torch.utils.tensorboard import SummaryWriter

# Import from new locations
from cochresnet50 import instantiate_cochresnet50
from lib.encoding import EncodingModel
from lib.utils.math import mycov

from lib.utils.audio import rms_normalize, CustomFade
from lib.synthesis.utils import get_keys, generate_unique_filename, layers


class IdentityProcessor:
    """Processor that returns input unchanged."""
    def __call__(self, X):
        return X


class CochresnetProcessor:
    """Processor that adds channel dimension for CochResNet."""
    def __call__(self, X):
        return X.unsqueeze(1)


class BestLayerSynthesizer:
    def __init__(
            self,
            encoding_models: dict,
            batch_size = 60,
            rms = .1,
            window = None,
            max_iters: int = 10000, 
            monitor: int = 5, 
            fixup_every: int = 1,
            match_variances: bool = True,
            checkpoint_dir: Optional[str] = None,
            checkpoint_interval: Optional[int] = None,
            checkpoint_name: Optional[str] = None,
            log_dir: Optional[str] = None,
            device: Union[str,torch.device,dict]='cuda'
            ):
        self.batch_size = batch_size
        self.rms = rms
        if window is not None:
            self.window = CustomFade(window, n_iters = max_iters//fixup_every)
        else:
            self.window = None
        self.max_iters = max_iters
        if checkpoint_interval is not None or checkpoint_dir is not None or checkpoint_name is not None:
                self.setup_checkpointing(checkpoint_dir,checkpoint_interval,checkpoint_name)
                self.has_checkpointing = True
        else:
            self.has_checkpointing = False
        self.monitor = monitor
        self.fixup_every = fixup_every
        self.match_variances = match_variances
        if log_dir is not None:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None

        self.models = {}
        self.variance_targets = {}
        if isinstance(device,str):
            self.device = torch.device(device)
            self.device_dict = None
        elif isinstance(device,torch.device):
            self.device = device
            self.device_dict = None
        else:
            k = list(device.keys())[0]
            if isinstance(k,str):
                self.device = torch.device(device[k])
            else:
                self.device = device[k]
            self.device_dict = device
        for model in encoding_models:
            if self.device_dict is not None:
                encoding_models[model].cuda_(self.device_dict[model])
            else:
                encoding_models[model].cuda_(self.device)
        self.encoding_models = encoding_models
        
        self.reference_variances = None
        self.prediction_multiplers = {}
        for encoding_model_name in encoding_models:
            variance_target = encoding_models[encoding_model_name].best_predictions.var(0)
            if isinstance(variance_target,np.ndarray):
                variance_target = torch.from_numpy(variance_target).to(self.device)
            self.variance_targets[encoding_model_name] = variance_target
            # if self.match_variances:
            #     if self.reference_variances is None:
            #         self.reference_variances = variance_target.clone()
            #         self.prediction_multiplers[encoding_model_name] = 1
            #     else:
            #         variance_target = variance_target.clone()
            #         self.prediction_multiplers[encoding_model_name] = torch.sqrt(torch.mean(self.reference_variances) / torch.mean(variance_target))
            #         self.variance_targets[encoding_model_name] *= self.prediction_multiplers[encoding_model_name]
                    



    def setup_checkpointing(self,checkpoint_dir = None, checkpoint_interval = None, checkpoint_name = None):
        if checkpoint_interval is None:
            self.checkpoint_interval = self.max_iters // 5
        else:
            self.checkpoint_interval = checkpoint_interval
        if checkpoint_dir is None:
            self.checkpoint_dir = os.getcwd()
        else:
            self.checkpoint_dir = checkpoint_dir
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        if checkpoint_name is None:
            self.checkpoint_name = generate_unique_filename(self.checkpoint_dir, 'checkpoint.npy')
        else:
            self.checkpoint_name = checkpoint_name
        print("Checkpoints will be saved to", os.path.join(self.checkpoint_dir,self.checkpoint_name), "every", self.checkpoint_interval, "iterations")

    def checkpoint(self):
         with torch.no_grad():
            X_ = self.X.detach()
            X_ = rms_normalize(X_, self.rms, dim = 1).numpy(force=True)
            X_long = einops.rearrange(X_, 'b s -> (b s)')
            np.save(
                os.path.join(self.checkpoint_dir,str(self.iter) + "_" + self.checkpoint_name),
                X_long)

    def forward(self, X: torch.Tensor):
        # X is matrix of waveforms: batch x sample
        # inputs = {m: self.processors[m](self.sampling_rate_transforms[m](X)) for m in self.models}
        # X_normalized = rms_normalize(X, self.rms, dim = 1) # over sample dimension
        if self.device_dict is None:
            preds = {k: self.encoding_models[k].predict(X) for k in self.encoding_models}
        else:
            preds = {}
            for k in self.encoding_models:
                current_device = self.device_dict[k]
                preds[k] = self.encoding_models[k].predict(X.to(current_device))
		#torch.cuda.synchronize(current_device)
            preds = {k:v.to(self.device) for k,v in preds.items()}
        if self.match_variances:
            preds = {k: v * self.prediction_multiplers[k] for k,v in preds.items()}
        return preds

    def vcov(self,preds):
        variances = {m: preds[m].var(0).to(self.device) for m in preds}
        all_model_pairs = list(itertools.combinations(list(preds.keys()),2))
        covariances = {}
        for pair in all_model_pairs:
            m1,m2 = pair
            covariances[(m1,m2)] = mycov(preds[m1],preds[m2])
        return variances,covariances
    
    def loss(self,preds,c, variance_scale, train_indices, maximize_variance,log = False):
        variances,covariances = self.vcov(preds)
        if train_indices is not None:
            variances = {m: variances[m][train_indices] for m in variances}
            covariances = {pair: covariances[pair][train_indices] for pair in covariances}

        variance_loss = 0
        covariance_loss = 0
        for m in variances:
            if maximize_variance:
                # variance_loss -= torch.linalg.norm(variances[m])**2
                variance_loss -= torch.square(variances[m]).sum()#.sqrt()
            else:
                # variance_loss += torch.linalg.norm(variances[m] - self.variance_targets[m][train_indices])**2 
                variance_loss += torch.square(variances[m] - self.variance_targets[m][train_indices] * variance_scale).sum()#.sqrt()
        variance_loss /= len(variances)
        for pair, cov in covariances.items():
            # covariance_loss += torch.linalg.norm(cov)**2
            covariance_loss += torch.square(cov).sum()#.sqrt()

        L = variance_loss + (c * covariance_loss)
        if log:
            with torch.no_grad():
                for m in variances:
                    relative_variances = variances[m]/self.variance_targets[m][train_indices]
                    self.writer.add_histogram(m + '_variances', variances[m], self.iter)
                    self.writer.add_histogram(m + '_relative_variances', relative_variances, self.iter)
                for pair in covariances:
                    self.writer.add_histogram(str(pair) + '_covariances', covariances[pair], self.iter)
            self.writer.add_histogram('relative_variances', relative_variances, self.iter)
            self.writer.add_scalar('variance_loss', variance_loss, self.iter)
            self.writer.add_scalar('covariance_loss', covariance_loss, self.iter)
            self.writer.add_scalar('loss', L, self.iter)
            self.writer.add_scalar('learning_rate', self.opt.param_groups[0]['lr'], self.iter)

        return L

    def log_audio(self):
        with torch.no_grad():
            tag = 'synthesized_audio'
            self.writer.add_audio(tag, self.X.flatten(), global_step=self.iter, sample_rate=20000, walltime=None)

    
    def initialize_X(self,X = None, **kwargs):
        if X is None:
            self.X = torch.normal(0,.01, size=(self.batch_size, 40000), device=self.device,requires_grad = True)
            # self.X.requires_grad = True
        else:
            self.X = X.clone().to(self.device)
            self.X.requires_grad = True
        self.opt = torch.optim.Adam([self.X], **kwargs)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(\
        #     self.opt, 
        #     factor=0.5, 
        #     patience=10, 
        #     threshold=0.0001, 
        #     threshold_mode='rel', 
        #     cooldown=0, 
        #     min_lr=0, 
        #     eps=1e-08)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.opt, 1000, eta_min=1E-5)

    def fit(self,X = None, c = 1, variance_scale = 1, train_indices = None, test_indices = None, maximize_variance = False,**kwargs):
        self.initialize_X(X = X, **kwargs)
        pbar = tqdm(total = self.max_iters + 1)
        self.iter = 0
        # if variance_scale is None:
        #     assert maximize_variance
        if maximize_variance and variance_scale is not None:
            warnings.warn("maximize_variance is set to True, but variance_scale is not None. variance_scale will be set to 1.")
            variance_scale = 1
        
        
        
            variance_target = encoding_models[encoding_model_name].best_predictions.var(0)
            if isinstance(variance_target,np.ndarray):
                variance_target = torch.from_numpy(variance_target).to(self.device)
            self.variance_targets[encoding_model_name] = variance_target

        # NOTE: only need test_indices to peek at variance of test predictions to natsounds
        if self.match_variances:
            print("Matching variances")
            for encoding_model_name in self.variance_targets:
                if self.reference_variances is None:
                    self.reference_variances = torch.nan * torch.zeros_like(self.variance_targets[encoding_model_name])
                    self.reference_variances[test_indices] = self.variance_targets[encoding_model_name][test_indices]
                    self.prediction_multiplers[encoding_model_name] = 1
                else:
                    variance_target = torch.nan * torch.zeros_like(self.variance_targets[encoding_model_name])
                    variance_target[test_indices] = self.variance_targets[encoding_model_name][test_indices]
                    self.prediction_multiplers[encoding_model_name] = torch.sqrt(torch.nanmean(self.reference_variances) / torch.nanmean(variance_target))
                    # variance_target = variance_target.clone()
                    # self.prediction_multiplers[encoding_model_name] = torch.sqrt(torch.mean(self.reference_variances) / torch.mean(variance_target))
                    # self.variance_targets[encoding_model_name] *= self.prediction_multiplers[encoding_model_name]
            print(f"Using the following prediction multipliers {self.prediction_multiplers}")
            print("Inflating variance targets to match reference variances")
            for encoding_model_name in self.variance_targets:
                self.variance_targets[encoding_model_name] *= self.prediction_multiplers[encoding_model_name]
        else:
            print("Not matching variances")
        while self.iter < self.max_iters:
            self.opt.zero_grad()
            # with torch.autocast(device_type=self.device):
            # amp not implemented for complexFloat
            preds = self.forward(self.X)
            loss = self.loss(
                preds,
                c,
                variance_scale, 
                train_indices, 
                maximize_variance,
                log = (self.iter % self.monitor == 0))
            loss.backward()
            if self.iter % self.monitor == 0:
                with torch.no_grad():
                    # track gradients
                    self.writer.add_histogram('gradient', self.X.grad.flatten(), self.iter)
                    self.log_audio()

            self.opt.step()

            # if self.window is not None:
            #     with torch.no_grad():
            #         self.window(self.X)
            if self.window is not None and self.iter % self.fixup_every == 0:
                self.fixups()

            self.iter += 1
            pbar.update(1)
            # self.scheduler.step(loss)
            self.scheduler.step(self.iter)
            if self.has_checkpointing and self.iter % self.checkpoint_interval == 0:
                self.checkpoint()

    def fixups(self):
        waveform = self.X
        assert waveform.ndim == 2
        filtered = torchaudio.functional.highpass_biquad(
            waveform = waveform,
            sample_rate = 20000,
            cutoff_freq = 50,
            Q = 0.707
        )
        faded = self.window(filtered)
        normalized = rms_normalize(faded, self.rms)
        self.X.data = normalized

class BestLayerSpectrogramSynthesizer(BestLayerSynthesizer):
    def __init__(
        self,
        encoding_models: dict,
        batch_size = 60,
        rms = .1,
        window = None,
        max_iters: int = 10000, 
        fixup_every: int = 1,
        match_variances: bool = True,
        monitor: int = 5, 
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: Optional[int] = None,
        checkpoint_name: Optional[str] = None,
        log_dir: Optional[str] = None,
        device: Union[str,torch.device]='cuda',
        n_fft = 400,
        win_length = 400,
        hop_length = 200,
        spectrogram_kwargs = {},
        ):

        super(BestLayerSpectrogramSynthesizer,self).__init__(
            encoding_models = encoding_models,
            batch_size = batch_size,
            rms = rms,
            window = window,
            max_iters = max_iters,
            match_variances = match_variances,
            fixup_every = fixup_every,
            monitor = monitor,
            checkpoint_dir = checkpoint_dir,
            checkpoint_interval = checkpoint_interval,
            checkpoint_name = checkpoint_name,
            log_dir = log_dir,
            device = device
        )
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.spectrogram_kwargs = spectrogram_kwargs
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length,power=None,normalized=False, **spectrogram_kwargs).to(self.device)
        self.inverse_spectrogram_transform = torchaudio.transforms.InverseSpectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length,normalized=False).to(self.device)

    def forward(self, X: torch.Tensor):
        # X is an array of spectrograms: batch x freq x time bin
        waveform = self.inverse_spectrogram_transform(X)
        return super(BestLayerSpectrogramSynthesizer,self).forward(waveform)

    def initialize_X(self,X = None, **kwargs):
        if X is None:
            self.X = torch.normal(0,.01, size=(self.batch_size, self.n_fft//2+1, 40000//self.hop_length+1), device=self.device,requires_grad = True, dtype=torch.complex64)
            # self.X.requires_grad = True
        else:
            self.X = X.clone().to(self.device)
            self.X.requires_grad = True
        self.opt = torch.optim.Adam([self.X], **kwargs)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(\
        #     self.opt, 
        #     factor=0.5, 
        #     patience=10, 
        #     threshold=0.0001, 
        #     threshold_mode='rel', 
        #     cooldown=0, 
        #     min_lr=0, 
        #     eps=1e-08)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.opt, 1000, eta_min=1E-5)
    
    def log_audio(self):
        with torch.no_grad():
            tag = 'synthesized_audio'
            waveform = self.inverse_spectrogram_transform(self.X.detach())
            self.writer.add_audio(tag, waveform.flatten(), global_step=self.iter, sample_rate=20000, walltime=None)

    def fixups(self):
        assert self.X.ndim == 3
        waveform = self.inverse_spectrogram_transform(self.X)
        filtered = torchaudio.functional.highpass_biquad(
            waveform = waveform,
            sample_rate = 20000,
            cutoff_freq = 40,
            Q = 0.707
        )
        faded = self.window(filtered)
        normalized = rms_normalize(faded, self.rms)
        self.X.data = self.spectrogram_transform(normalized)

    def checkpoint(self):
         with torch.no_grad():
            X_ = self.inverse_spectrogram_transform(self.X.detach())
            X_ = rms_normalize(X_, self.rms, dim = 1).numpy(force=True)
            X_long = einops.rearrange(X_, 'b s -> (b s)')
            np.save(
                os.path.join(self.checkpoint_dir,str(self.iter) + "_" + self.checkpoint_name),
                X_long)

    
