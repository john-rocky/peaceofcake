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
# criterion is registered lazily — it depends on src.misc which
# requires training-only packages.  It gets imported when the
# solver / trainer actually needs it.
from .postprocessor import *
