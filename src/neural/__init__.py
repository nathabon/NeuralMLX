import sys

if sys.platform == "darwin":
    import mlx.core as mx
    from . import other
    from . import neuralLayers
elif sys.platform == "win32":
    from .cuda_patch import backend_cupy as mx
    from .cuda_patch import other
    from .cuda_patch import neuralLayers
else:
    import numpy as mx
    from . import other
    from . import neuralLayers

from . import neuralNetwork2

__all__ = ["mx", "neuralLayers", "neuralNetwork2", "other"]