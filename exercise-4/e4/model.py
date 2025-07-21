import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

LANDMARK_INDICES = {
    1: [0, 25],
    2: [25, 58],
    3: [58, 89],
    4: [89, 128],
    5: [128, 143],
    6: [143, 158],
    7: [158, 168],
    8: [168, 182],
    9: [182, 190],
    10: [190, 219],
    11: [219, 256],
    12: [256, 275],
    13: [275, 294]
}

class KPNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Define the model architecture
        self.models = nn.ModuleList()

        for i in range(1, 14):
            n = (LANDMARK_INDICES[i][1] - LANDMARK_INDICES[i][0])*2

            self.models.append(nn.Sequential(
                nn.Linear(n+50, n+50),
                nn.ReLU(),
                nn.Linear(n+50, n+50),
                nn.ReLU(),
                nn.Linear(n+50, n+50),
                nn.ReLU(),
                nn.Linear(n+50, n+50),
                nn.ReLU(),
                nn.Linear(n+50, n),
            ))


    def training_step(self, batch, batch_idx):
        # Unpack the batch
        keypoints_fks = batch["keypoints_fks"].flatten(start_dim=1)
        keypoints_hk = batch["keypoints_hk"].flatten(start_dim=1)
        keypoints_fkp = batch["keypoints_fkp"].flatten(start_dim=1)        
        cat = batch["category"]

        # Forward pass through the model
        output = self.models[cat-1](torch.cat([keypoints_fks, keypoints_hk], dim=1))

        # Compute loss
        loss = F.mse_loss(output, keypoints_fkp)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Unpack the batch
        keypoints_fks = batch["keypoints_fks"].flatten(start_dim=1)
        keypoints_hk = batch["keypoints_hk"].flatten(start_dim=1)
        keypoints_fkp = batch["keypoints_fkp"].flatten(start_dim=1)        
        cat = batch["category"][0].item()

        # Forward pass through the model
        output = self.models[cat-1](torch.cat([keypoints_fks, keypoints_hk], dim=1))

        # Compute loss
        loss = F.mse_loss(output, keypoints_fkp)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        # Unpack the batch
        keypoints_fks = batch["keypoints_fks"].flatten(start_dim=1)
        keypoints_hk = batch["keypoints_hk"].flatten(start_dim=1)
        keypoints_fkp = batch["keypoints_fkp"].flatten(start_dim=1)        
        cat = batch["category"]

        # Forward pass through the model
        output = self.models[cat-1](torch.cat([keypoints_fks, keypoints_hk], dim=1))

        # Compute loss
        loss = F.mse_loss(output, keypoints_fkp)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def forward(self, x):
        # Forward pass through the model
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer