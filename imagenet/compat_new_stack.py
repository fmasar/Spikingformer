"""compat_new_stack.py -- run Spikingformer on Python 3.12, torch 2.x/CUDA 13,
timm 1.0.x, Pillow 13, numpy 2.5 and current spikingjelly.

Import this before torchvision / PIL / timm / spikingjelly are used.
Every step is idempotent.
"""
import inspect
import sys

# ------------------------------------------------------------------ torch.load
import torch

if not getattr(torch.load, "_sf_patched", False):
    _orig_torch_load = torch.load

    def _torch_load(*args, **kwargs):
        # torch>=2.6 defaults weights_only=True, which refuses this repo's
        # checkpoints (argparse.Namespace, optimizer state) -> breaks --resume
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)

    _torch_load._sf_patched = True
    torch.load = _torch_load

# ------------------------------------------------------------- numpy>=2 aliases
# Only aliases the old code could plausibly use. np.object / np.str are
# deliberately NOT probed: numpy 2.5 emits a FutureWarning just for touching
# them, and nothing here uses them.
import numpy as _np

for _alias, _builtin in (("bool", bool), ("int", int), ("float", float),
                         ("complex", complex)):
    if _alias not in vars(_np):
        setattr(_np, _alias, _builtin)

# ------------------------------------------- torchvision functional_tensor move
try:
    import torchvision.transforms._functional_tensor as _sf_ft
    sys.modules.setdefault("torchvision.transforms.functional_tensor", _sf_ft)
except ImportError:
    pass

# ------------------------------------------------- THE CRASH: PIL posterize bits
# timm's RandAugment posterize computes bits_to_keep = 8 - magnitude, which hits
# 0 or below at high magnitudes (rand-m9). Old Pillow tolerated it; Pillow>=9
# raises "bits must be between or equal to 1 and 8". timm resolves
# ImageOps.posterize at call time and DataLoader workers are forked, so patching
# here covers them too.
from PIL import ImageOps as _ImageOps

if not getattr(_ImageOps.posterize, "_sf_patched", False):
    _orig_posterize = _ImageOps.posterize

    def _posterize(image, bits):
        return _orig_posterize(image, int(max(1, min(8, bits))))

    _posterize._sf_patched = True
    _ImageOps.posterize = _posterize

# ------------------------------------------- spikingjelly multi-step -> step_mode
try:
    import cupy  # noqa: F401
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False

_SJ_ALIASES = {
    # removed class name  ->  single-step class it became
    "MultiStepLIFNode": "LIFNode",
    "MultiStepIFNode": "IFNode",
    "MultiStepEIFNode": "EIFNode",
    "MultiStepParametricLIFNode": "ParametricLIFNode",
}


def _install_sj_compat():
    try:
        import spikingjelly.activation_based.neuron as neuron
    except ImportError:
        return

    for ms_name, ss_name in _SJ_ALIASES.items():
        if getattr(neuron, ms_name, None) is not None:
            continue                      # old spikingjelly: nothing to do
        base = getattr(neuron, ss_name, None)
        if base is None:
            continue                      # unexpected: let the import fail loudly
        accepted = set(inspect.signature(base.__init__).parameters)

        class _MultiStep(base):
            def __init__(self, *args, **kwargs):
                kwargs["step_mode"] = "m"
                if not _HAS_CUPY and kwargs.get("backend", "torch") != "torch":
                    kwargs["backend"] = "torch"   # no cupy wheel for CUDA 13
                kwargs = {k: v for k, v in kwargs.items() if k in accepted}
                super().__init__(*args, **kwargs)

            def extra_repr(self):
                return super().extra_repr() + ", step_mode=m"

        _MultiStep.__name__ = ms_name
        _MultiStep.__qualname__ = ms_name
        _MultiStep.__module__ = neuron.__name__
        setattr(neuron, ms_name, _MultiStep)
        # keep the package namespace consistent for `from ... import` styles
        try:
            import spikingjelly.activation_based as _sj_pkg
            setattr(_sj_pkg, ms_name, _MultiStep)
        except ImportError:
            pass


_install_sj_compat()

# ------------------------------------------------ timm 1.x convenience fallbacks
try:  # create_optimizer_v2 -> create_optimizer
    import timm.optim as _topt
    create_optimizer = getattr(_topt, "create_optimizer", None) or _topt.create_optimizer_v2
    optimizer_kwargs = _topt.optimizer_kwargs
except ImportError:
    pass
