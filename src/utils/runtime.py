"""
Runtime tidiness for the CLI entry points.

Centralizes the noise-reduction the scripts opt into: enable Tensor Cores (which
also clears the matmul-precision warning) and silence a few well-known, benign
Lightning/pytorch-forecasting log lines and warnings. Library modules stay quiet
about this — only the scripts call it, so importing the library never mutates
global logging/warning state behind your back.
"""

import logging
import warnings


def configure_quiet_runtime() -> None:
    """Reduce third-party log/warning noise for interactive script runs."""
    # Lightning emits GPU/TPU banners and promotional tips at INFO; drop to WARNING.
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    # Benign UserWarnings we cannot fix from outside pytorch-forecasting.
    warnings.filterwarnings("ignore", message=r".*does not have many workers.*")
    warnings.filterwarnings("ignore", message=r".*is an instance of `nn.Module`.*")

    # Use Tensor Cores on capable GPUs (and clear the matmul-precision warning).
    try:
        import torch

        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
    except Exception:  # torch optional / CPU-only — nothing to configure
        pass
