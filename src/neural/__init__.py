import sys

if sys.platform == "darwin":
    import mlx.core as mx
    from . import other

elif sys.platform == "win32":
    from .cuda_patch import backend_cupy as mx
    from .cuda_patch import other

    # Patch sys.modules AVANT tout autre import du package neural,
    # pour que "from neural.other import *" dans neuralLayers.py
    # trouve la version cuda_patch et non neural/other.py (macOS)
    sys.modules['neural.other'] = other

else:
    import numpy as mx
    from . import other

# neuralLayers importe "from neural.other import *" — doit venir APRÈS le patch
from . import neuralLayers
from . import neuralNetwork2

__all__ = ["mx", "neuralLayers", "neuralNetwork2", "other"]