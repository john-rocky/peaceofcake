"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

# Register nn and zoo eagerly (needed for inference).
# data, optim, solver, misc are imported lazily at training time
# to avoid requiring training-only dependencies (faster_coco_eval, scipy).
from . import nn, zoo
