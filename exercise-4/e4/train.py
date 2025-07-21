import pytorch_lightning as pl
from model import KPNet
from dataset import DataModule
from logger import ImageLogger
import os
import torch

dataset = DataModule(batch_size=1, num_workers=0, split=[0.7, 0.2, 0.1])
model = KPNet()
checkpoint = os.path.join("..", "checkpoints", "epoch=11-step=464256.ckpt")

sd = torch.load(checkpoint, map_location="cpu")
if "state_dict" in sd.keys():
    model.load_state_dict(sd["state_dict"], strict=False)
else:
    model.load_state_dict(sd, strict=False)

imageLogger = ImageLogger(batch_frequency=1000, max_images=4)
trainer = pl.Trainer(max_epochs=100, callbacks=[imageLogger])  # Adjust the number of GPUs as needed
trainer.fit(model, dataset)
trainer.save_checkpoint("kpnet_model.ckpt")
