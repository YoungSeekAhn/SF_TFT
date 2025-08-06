# model_FTF.py
# This code defines a PyTorch Lightning module for a Temporal Fusion Transformer model.

import pytorch_lightning as pl
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder   
import pandas as pd
from pytorch_forecasting.metrics import SMAPE, MAE
class TFTModel(pl.LightningModule):
    def __init__(self, training_data, validation_data, max_encoder_length=24, max_prediction_length=6):
        super().__init__()
        self.save_hyperparameters()
        
        # Create TimeSeriesDataSet for training
        self.training_data = TimeSeriesDataSet(
            training_data,
            time_idx="time_idx",
            target="target",
            group_ids=["series_id"],
            min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length,
            min_prediction_length=max_prediction_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=["series_id"],
            static_reals=[],
            time_varying_known_categoricals=["time_idx"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=["target"],
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True
        )

        # Create TimeSeriesDataSet for validation
        self.validation_data = TimeSeriesDataSet.from_dataset(self.training_data, validation_data)

        # Initialize the Temporal Fusion Transformer model
        self.model = TemporalFusionTransformer.from_dataset(
            self.training_data,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.model.loss(output, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.model.loss(output, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.03)