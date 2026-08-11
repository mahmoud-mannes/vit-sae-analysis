"""Modern sparse autoencoder package for the ViT position study.

Public surface:

    from SAE import SAE, ActivationStore, train_sae
    from SAE.metrics import reconstruction_metrics, mean_max_cosine
"""

from .sae import SAE, auxiliary_loss, ARCHITECTURES
from .activation_store import ActivationStore
from .activation_store_scaled import ActivationStoreMemmap
from .train import train_sae

__all__ = ["SAE", "ActivationStore", "ActivationStoreMemmap", "train_sae", "auxiliary_loss", "ARCHITECTURES"]
