"""Model architectures for thermal anomaly detection."""

from .swin_unet import SwinUNet, SwinUNetConfig

# Import other modules only if they exist
try:
    from .convlstm_fusion import ConvLSTMFusion, TemporalFusionConfig
    _has_convlstm = True
except ImportError:
    _has_convlstm = False

try:
    from .patchcore_head import PatchCoreHead, PatchCoreConfig
    _has_patchcore = True
except ImportError:
    _has_patchcore = False

try:
    from .hybrid_model import ThermalAnomalyModel
    _has_hybrid = True
except ImportError:
    _has_hybrid = False

try:
    from .model_utils import load_model, save_model, get_model_hash
    _has_utils = True
except ImportError:
    _has_utils = False

# Dynamic __all__ based on available modules
__all__ = ["SwinUNet", "SwinUNetConfig"]

if _has_convlstm:
    __all__.extend(["ConvLSTMFusion", "TemporalFusionConfig"])
if _has_patchcore:
    __all__.extend(["PatchCoreHead", "PatchCoreConfig"])
if _has_hybrid:
    __all__.append("ThermalAnomalyModel")
if _has_utils:
    __all__.extend(["load_model", "save_model", "get_model_hash"])

