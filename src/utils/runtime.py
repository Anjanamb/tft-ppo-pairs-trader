"""
Runtime tidiness for the CLI entry points and dashboard.

Centralizes the noise-reduction the scripts opt into: enable Tensor Cores (which
also clears the matmul-precision warning) and silence the benign
Lightning/pytorch-forecasting banners and warnings. Library modules stay quiet
about this — only the scripts/dashboard call it, so importing the library never
mutates global logging/warning state behind your back.
"""

import logging
import warnings

# Lightning sets these loggers to INFO on import (GPU/TPU banners, LOCAL_RANK,
# the litlogger/litmodels promo tips), so quieting the parent alone is not enough.
_LIGHTNING_LOGGERS = (
    "lightning",
    "lightning.pytorch",
    "lightning.pytorch.utilities.rank_zero",
    "lightning.pytorch.accelerators.cuda",
    "lightning.fabric",
    "lightning.fabric.utilities.rank_zero",
    "pytorch_lightning",
)


def configure_quiet_runtime() -> None:
    """Reduce third-party log/warning noise for interactive runs."""
    # Benign UserWarnings we cannot fix from outside pytorch-forecasting.
    warnings.filterwarnings("ignore", message=r".*does not have many workers.*")
    warnings.filterwarnings("ignore", message=r".*is an instance of `nn.Module`.*")

    # Use Tensor Cores on capable GPUs (and clear the matmul-precision warning).
    try:
        import torch

        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
    except Exception:  # torch optional / CPU-only
        pass

    # Import Lightning so its loggers exist (and have been set to INFO), then
    # turn them down to WARNING so the banners/tips stop printing.
    try:
        import lightning.pytorch  # noqa: F401
    except Exception:
        pass
    for name in (*_LIGHTNING_LOGGERS, *list(logging.root.manager.loggerDict)):
        if name.startswith(("lightning", "pytorch_lightning")):
            logging.getLogger(name).setLevel(logging.WARNING)
