"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from .arch import *

#
from .backbone import *
from .backbone import (
    FrozenBatchNorm2d,
    freeze_batch_norm2d,
    get_activation,
)
# criterion excluded — pulls in src.misc -> src.data (faster_coco_eval)
from .postprocessor import *
