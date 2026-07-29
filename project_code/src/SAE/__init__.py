"""Modern sparse autoencoder package for the ViT position study.

Public surface:

    from SAE import SAE, ActivationStore, train_sae
    from SAE.metrics import reconstruction_metrics, mean_max_cosine

The legacy ``train_SAE.py`` and ``resample.py`` remain for the original notebook
path. New work should use this package.
"""

from .sae import SAE, auxiliary_loss, ARCHITECTURES
from .activation_store import ActivationStore
from .train import train_sae

__all__ = ["SAE", "ActivationStore", "train_sae", "auxiliary_loss", "ARCHITECTURES"]
