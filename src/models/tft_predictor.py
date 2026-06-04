"""
Temporal Fusion Transformer spread predictor.

A single TFT is trained across *all* pairs at once (``pair_id`` as a static
categorical). This is how Lim et al. (2021) intend the model to be used: the
shared backbone transfers structure between pairs, and per-pair target
normalization (handled in ``dataset.py`` via ``GroupNormalizer``) stops the
high-variance crypto spreads from dominating the loss.

The model emits a quantile forecast rather than a point estimate. The median
(q_50) is the prediction the backtester uses; the band ``q_90 - q_10`` is the
uncertainty signal the PPO agent keys off — wide band => trade smaller.

Known limitation (addressed later in walk-forward backtesting): the hedge ratio
behind the spread is fit over the full sample, so the spread definition carries
mild look-ahead. That is acceptable for fitting the forecasting pipeline; the
Phase 7 backtester re-estimates everything on a rolling basis.

References:
    - Lim et al. (2021) "Temporal Fusion Transformers" arXiv:1912.09363
"""

import logging
from pathlib import Path

import pandas as pd

from src.models.base import BasePredictor
from src.utils.config import load_config

logger = logging.getLogger(__name__)


class TFTPredictor(BasePredictor):
    """TFT spread forecaster with quantile (uncertainty) outputs."""

    def __init__(self, config: dict | None = None):
        self.cfg = config or load_config()
        self.tft_cfg = self.cfg["tft"]
        self.quantiles = list(self.tft_cfg["quantiles"])

        self.model = None          # TemporalFusionTransformer (lazy)
        self.trainer = None        # lightning.Trainer
        self._dataset_params = None  # encoders/scalers, so predict() can transform new data
        self._best_checkpoint: str | None = None

    @property
    def name(self) -> str:
        return "tft"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, train_data, val_data) -> dict:
        """
        Fit the TFT.

        Args:
            train_data: Training ``TimeSeriesDataSet`` (from ``TFTDatasetBuilder``).
            val_data: Validation ``TimeSeriesDataSet`` built via ``from_dataset``.

        Returns:
            Metrics dict: best validation loss and the checkpoint path.
        """
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from pytorch_forecasting import TemporalFusionTransformer
        from pytorch_forecasting.metrics import QuantileLoss

        pl.seed_everything(self.tft_cfg["seed"], workers=True)

        self._dataset_params = train_data.get_parameters()
        batch_size = self.tft_cfg["batch_size"]
        train_loader = train_data.to_dataloader(
            train=True, batch_size=batch_size, num_workers=0
        )
        val_loader = val_data.to_dataloader(
            train=False, batch_size=batch_size * 2, num_workers=0
        )

        self.model = TemporalFusionTransformer.from_dataset(
            train_data,
            learning_rate=self.tft_cfg["learning_rate"],
            hidden_size=self.tft_cfg["hidden_size"],
            attention_head_size=self.tft_cfg["attention_head_size"],
            dropout=self.tft_cfg["dropout"],
            hidden_continuous_size=self.tft_cfg["hidden_continuous_size"],
            loss=QuantileLoss(quantiles=self.quantiles),
            output_size=len(self.quantiles),
            reduce_on_plateau_patience=4,
            log_interval=-1,
        )
        logger.info("TFT initialized: %.1fk parameters", self.model.size() / 1e3)

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.tft_cfg["early_stopping_patience"],
            mode="min",
        )
        checkpoint = ModelCheckpoint(
            monitor="val_loss", mode="min", save_top_k=1, filename="tft-best"
        )

        self.trainer = pl.Trainer(
            max_epochs=self.tft_cfg["max_epochs"],
            accelerator=self.tft_cfg["accelerator"],
            devices=1,
            gradient_clip_val=self.tft_cfg["gradient_clip_val"],
            callbacks=[early_stop, checkpoint],
            default_root_dir=self.tft_cfg.get("work_dir", "models"),
            enable_progress_bar=False,
            logger=False,
        )
        self.trainer.fit(self.model, train_loader, val_loader)

        self._best_checkpoint = checkpoint.best_model_path or None
        best_val = float(checkpoint.best_model_score) if checkpoint.best_model_score is not None else None

        # Reload the best epoch so predict() reflects early-stopping, not the
        # last (possibly overfit) epoch.
        if self._best_checkpoint:
            self.model = TemporalFusionTransformer.load_from_checkpoint(
                self._best_checkpoint
            )

        logger.info("Training done — best val_loss=%s", best_val)
        return {
            "best_val_loss": best_val,
            "epochs_run": self.trainer.current_epoch,
            "checkpoint": self._best_checkpoint,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, data) -> pd.DataFrame:
        """
        Generate quantile forecasts.

        Args:
            data: Either a panel ``DataFrame`` (transformed using the training
                dataset's encoders) or a prepared ``TimeSeriesDataSet``.

        Returns:
            Tidy long DataFrame with one row per (pair_id, forecast timestep):
            columns ``[pair_id, time_idx, horizon, prediction, q_02, ..., q_98]``
            where ``prediction`` is the median (q_50).
        """
        if self.model is None:
            raise RuntimeError("Model is not trained or loaded. Call train()/load() first.")

        dataset = self._as_dataset(data)
        loader = dataset.to_dataloader(
            train=False, batch_size=self.tft_cfg["batch_size"] * 2, num_workers=0
        )
        out = self.model.predict(
            loader,
            mode="quantiles",
            return_index=True,
            trainer_kwargs={"enable_progress_bar": False, "logger": False},
        )
        return self._format_predictions(out.output, out.index)

    def _as_dataset(self, data):
        """Coerce a DataFrame into a TimeSeriesDataSet using training encoders."""
        if isinstance(data, pd.DataFrame):
            if self._dataset_params is None:
                raise RuntimeError(
                    "No dataset spec available to transform a DataFrame; "
                    "pass a TimeSeriesDataSet or train/load first."
                )
            from pytorch_forecasting import TimeSeriesDataSet

            return TimeSeriesDataSet.from_parameters(
                self._dataset_params, data, predict=True, stop_randomization=True
            )
        return data

    def _format_predictions(self, output, index: pd.DataFrame) -> pd.DataFrame:
        """Expand the (N, pred_len, n_quantiles) tensor into a tidy frame."""
        preds = output.detach().cpu().numpy()  # (N, pred_len, n_quantiles)
        n_samples, pred_len, _ = preds.shape
        q_cols = [f"q_{int(q * 100):02d}" for q in self.quantiles]
        median_col = f"q_{int(0.5 * 100):02d}"

        rows = []
        for i in range(n_samples):
            pair_id = index.iloc[i]["pair_id"]
            start_idx = int(index.iloc[i]["time_idx"])
            for h in range(pred_len):
                row = {"pair_id": pair_id, "time_idx": start_idx + h, "horizon": h + 1}
                for qi, qc in enumerate(q_cols):
                    row[qc] = float(preds[i, h, qi])
                row["prediction"] = row[median_col]
                rows.append(row)

        cols = ["pair_id", "time_idx", "horizon", "prediction", *q_cols]
        return pd.DataFrame(rows)[cols]

    def interpret(self, data) -> dict:
        """
        TFT interpretability: variable importances and temporal attention.

        Returns the raw ``interpret_output`` dict (attention, encoder/decoder
        variable importances) — the material for the README's attention plots.
        """
        if self.model is None:
            raise RuntimeError("Model is not trained or loaded.")

        dataset = self._as_dataset(data)
        loader = dataset.to_dataloader(
            train=False, batch_size=self.tft_cfg["batch_size"] * 2, num_workers=0
        )
        raw = self.model.predict(
            loader,
            mode="raw",
            return_x=True,
            trainer_kwargs={"enable_progress_bar": False, "logger": False},
        )
        return self.model.interpret_output(raw.output, reduction="sum")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path):
        """Save the Lightning checkpoint and the training dataset spec."""
        if self.model is None:
            raise RuntimeError("Nothing to save — model is not trained.")
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.trainer.save_checkpoint(str(path))
        if self._dataset_params is not None:
            torch.save(self._dataset_params, path.with_suffix(".dataset.pt"))
        logger.info("Saved TFT to %s", path)

    def load(self, path: Path):
        """Load a TFT checkpoint (and dataset spec if present)."""
        import torch
        from pytorch_forecasting import TemporalFusionTransformer

        path = Path(path)
        self.model = TemporalFusionTransformer.load_from_checkpoint(str(path))

        ds_path = path.with_suffix(".dataset.pt")
        if ds_path.exists():
            self._dataset_params = torch.load(ds_path, weights_only=False)
        logger.info("Loaded TFT from %s", path)
