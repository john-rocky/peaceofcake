"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from .dfine import DFINE
from .dfine_decoder import DFINETransformer
from .hybrid_encoder import HybridEncoder
from .postprocessor import DFINEPostProcessor

# DFINECriterion and HungarianMatcher depend on src.misc (training-only).
# They are registered lazily when training starts.
def _register_training_modules():
    from .dfine_criterion import DFINECriterion  # noqa: F401
    from .matcher import HungarianMatcher  # noqa: F401
