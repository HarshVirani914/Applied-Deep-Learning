import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
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
    13: [275, 294],
}


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=8, num_workers=0, split=None):
        super().__init__()

        if split is None:
            split = [0.7, 0.2, 0.1]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split

    def setup(self, stage=None):
        dataset = FashionDataset()

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset=dataset,
            lengths=[0.9, 0.05, 0.05],
            generator=torch.Generator().manual_seed(42),
        )

        print(
            "Dataset Split",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset),
        )

        self.datasets = {
            "train": self.train_dataset,
            "validation": self.val_dataset,
            "test": self.test_dataset,
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


class FashionDataset(Dataset):
    def __init__(self):
        self.data_root = os.path.join("data", "e4_training_data")

        self.filenames_fks = os.listdir(
            os.path.join(self.data_root, "fashion_keypoints_stock")
        )
        self.filenames_fks = [
            filename for filename in self.filenames_fks if filename.endswith(".json")
        ]

        self.filenames_hk = os.listdir(os.path.join(self.data_root, "human_keypoints"))
        self.filenames_hk = [
            filename for filename in self.filenames_hk if filename.endswith(".json")
        ]

        self.filenames_fkp = os.listdir(
            os.path.join(self.data_root, "fashion_keypoints_posed")
        )
        self.filenames_fkp = [
            filename for filename in self.filenames_fkp if filename.endswith(".json")
        ]

        self.filenames = (
            set(self.filenames_fks) & set(self.filenames_hk) & set(self.filenames_fkp)
        )
        self.filenames = sorted(list(self.filenames))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):

        filename = self.filenames[i]

        with open(
            os.path.join(self.data_root, "fashion_keypoints_stock", filename), "r"
        ) as f:
            data = json.load(f)
            category_s = data["category"]
            cat_indices = LANDMARK_INDICES[category_s]
            keypoints = data["keypoints"]

            t_keypoints_fks = torch.zeros(len(keypoints), 2)

            scale = torch.rand(1) / 2 + 0.75

            """
            for i in range(len(keypoints)):
                if i in range(cat_indices[0], cat_indices[1]):
                    t_keypoints_fks[i][0] = (keypoints[str(i)]["x"]/768)*scale
                    t_keypoints_fks[i][1] = (keypoints[str(i)]["y"]/1024)*scale
                else:
                    t_keypoints_fks[i][0] = 0
                    t_keypoints_fks[i][1] = 0
            """
            n_keypoints = cat_indices[1] - cat_indices[0]
            t_keypoints_fks = torch.zeros(n_keypoints, 2)

            for i in range(n_keypoints):
                t_keypoints_fks[i][0] = (
                    keypoints[str(i + cat_indices[0])]["x"] / 768
                ) * scale
                t_keypoints_fks[i][1] = (
                    keypoints[str(i + cat_indices[0])]["y"] / 1024
                ) * scale

        scale = torch.rand(1) + 0.25
        d_x = (torch.rand(1) / 2) - 0.25
        d_y = (torch.rand(1) / 2) - 0.25

        with open(os.path.join(self.data_root, "human_keypoints", filename), "r") as f:
            keypoints = json.load(f)
            t_keypoints_hk = torch.zeros(len(keypoints), 2)

            for i in range(len(keypoints)):
                t_keypoints_hk[i][0] = (keypoints[str(i)]["x"] / 768) * scale + d_x
                t_keypoints_hk[i][1] = (keypoints[str(i)]["y"] / 1024) * scale + d_y

        with open(
            os.path.join(self.data_root, "fashion_keypoints_posed", filename), "r"
        ) as f:
            data = json.load(f)
            category_p = data["category"]
            keypoints = data["keypoints"]
            t_keypoints_fkp = torch.zeros(len(keypoints), 2)

            """
            for i in range(len(keypoints)):
                if i in range(cat_indices[0], cat_indices[1]):
                    t_keypoints_fkp[i][0] = (keypoints[str(i)]["x"]/768)*scale + d_x
                    t_keypoints_fkp[i][1] = (keypoints[str(i)]["y"]/1024)*scale + d_y
                else:
                    t_keypoints_fks[i][0] = 0
                    t_keypoints_fks[i][1] = 0
            """
            n_keypoints = cat_indices[1] - cat_indices[0]
            t_keypoints_fkp = torch.zeros(n_keypoints, 2)

            for i in range(n_keypoints):
                t_keypoints_fkp[i][0] = (
                    keypoints[str(i + cat_indices[0])]["x"] / 768
                ) * scale + d_x
                t_keypoints_fkp[i][1] = (
                    keypoints[str(i + cat_indices[0])]["y"] / 1024
                ) * scale + d_y

        if category_s != category_p:
            print(filename)

        return {
            "keypoints_fks": t_keypoints_fks,
            "keypoints_hk": t_keypoints_hk,
            "keypoints_fkp": t_keypoints_fkp,
            "category": category_s,
            "filename": filename,
        }


if __name__ == "__main__":
    dataset = FashionDataset()
    print(len(dataset))
    sample = dataset[0]
    print(sample["keypoints_fks"])
    print(sample["keypoints_hk"])
    print(sample["keypoints_fkp"])
    print(sample["category"])
    print(sample["filename"])
