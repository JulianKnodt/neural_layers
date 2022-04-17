from .mnist import main as mnist
from .coco import main as coco

__all__ = [k for k in globals().keys() if not k.startswith("_")]
