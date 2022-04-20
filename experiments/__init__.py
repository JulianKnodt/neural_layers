from .mnist import main as mnist
from .vit import main as vit
from .pointnet import main as pointnet

__all__ = [k for k in globals().keys() if not k.startswith("_")]
