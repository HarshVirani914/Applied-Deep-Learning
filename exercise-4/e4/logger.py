from pytorch_lightning.callbacks import Callback
import torch
import numpy as np
import cv2

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


JOINT_PAIRS_MAP_ALL = {(0, 15): {'joint_names': ('Nose', 'REye')},
                       (0, 16): {'joint_names': ('Nose', 'LEye')},
                       (1, 0): {'joint_names': ('Neck', 'Nose')},
                       (1, 2): {'joint_names': ('Neck', 'RShoulder')},
                       (1, 5): {'joint_names': ('Neck', 'LShoulder')},
                       (1, 8): {'joint_names': ('Neck', 'MidHip')},
                       (2, 3): {'joint_names': ('RShoulder', 'RElbow')},
                       (2, 17): {'joint_names': ('RShoulder', 'REar')},
                       (3, 4): {'joint_names': ('RElbow', 'RWrist')},
                       (5, 6): {'joint_names': ('LShoulder', 'LElbow')},
                       (5, 18): {'joint_names': ('LShoulder', 'LEar')},
                       (6, 7): {'joint_names': ('LElbow', 'LWrist')},
                       (8, 9): {'joint_names': ('MidHip', 'RHip')},
                       (8, 12): {'joint_names': ('MidHip', 'LHip')},
                       (9, 10): {'joint_names': ('RHip', 'RKnee')},
                       (10, 11): {'joint_names': ('RKnee', 'RAnkle')},
                       (11, 22): {'joint_names': ('RAnkle', 'RBigToe')},
                       (11, 24): {'joint_names': ('RAnkle', 'RHeel')},
                       (12, 13): {'joint_names': ('LHip', 'LKnee')},
                       (13, 14): {'joint_names': ('LKnee', 'LAnkle')},
                       (14, 19): {'joint_names': ('LAnkle', 'LBigToe')},
                       (14, 21): {'joint_names': ('LAnkle', 'LHeel')},
                       (15, 17): {'joint_names': ('REye', 'REar')},
                       (16, 18): {'joint_names': ('LEye', 'LEar')},
                       (19, 20): {'joint_names': ('LBigToe', 'LSmallToe')},
                       (22, 23): {'joint_names': ('RBigToe', 'RSmallToe')}}

SPLIT_CATEGORIES = {
    "upper": [1,2,3,4,5,6],
    "lower": [7,8,9],
    "dress": [10,11,12,13]
}

DRAW_LANDMARK_PAIRS = {
    1: [(5,4),(4,3),(3,2),(2,1),(1,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),(18,19),(19,20),(20,21),(21,22),(22,23),(23,24),(24,5)],
    2: [(5,4),(4,3),(3,2),(2,1),(1,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),(18,19),(19,20),(20,21),(21,22),(22,23),(23,24),(24,25),(25,26),(26,27),(27,28),(28,29),(29,30),(30,31),(31,32),(32,5)],
    3: [(1,2),(2,3),(3,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,30),(30,29),(29,1),(5,4),(4,25),(25,26),(26,27),(27,28),(28,16),(16,17),(17,18),(18,19),(19,20),(20,21),(21,22),(22,23),(23,24),(24,5)],
    4: [(1,2),(2,3),(1,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(16,17),(17,18),(18,19),(19,38),(38,37),(37,3),(5,4),(4,33),(33,34),(34,35),(35,36),(36,20),(21,22),(22,23),(23,24),(24,25),(25,26),(26,27),(27,28),(28,29),(29,30),(30,31),(31,32),(32,5)],
    5: [(1,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,5),(5,4),(4,3),(3,2),(2,1)],
    6: [(1,6),(1,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,5),(5,14),(5,4),(4,3),(3,2),(2,1)],
    7: [(2,1),(1,0),(0,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,2)],
    8: [(2,1),(1,0),(0,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,2)],
    9: [(2,1),(1,0),(0,3),(3,4),(4,5),(5,6),(6,7),(7,2)],
    10: [(1,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),(18,19),(19,20),(20,21),(21,22),(22,23),(23,24),(24,25),(25,26),(26,27),(27,28),(28,5),(5,4),(4,3),(3,2),(2,1)],
    11: [(1,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),(18,19),(19,20),(20,21),(21,22),(22,23),(23,24),(24,25),(25,26),(26,27),(27,28),(28,29),(29,30),(30,31),(31,32),(32,33),(33,34),(34,35),(35,36),(36,5),(5,4),(4,3),(3,2),(2,1)],
    12: [(1,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),(18,5),(5,4),(4,3),(3,2),(2,1)],
    13: [(1,6),(1,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,5),(5,18),(5,4),(4,3),(3,2),(2,1)] 
}

class ImageLogger(Callback):
    def __init__(self, batch_frequency=100, max_images=4):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        
        self.log_steps = [self.batch_freq]

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        N = min(self.max_images, len(batch["keypoints_fks"]))

        keypoints_fks = batch["keypoints_fks"][:N]
        keypoints_hk = batch["keypoints_hk"][:N]
        keypoints_fkp = batch["keypoints_fkp"][:N]        
        category = batch["category"][:N].item()
        # Forward pass through the model
        output = pl_module.models[category-1](torch.cat([keypoints_fks.flatten(start_dim=1), keypoints_hk.flatten(start_dim=1)], dim=1))
        
        n = (LANDMARK_INDICES[category][1] - LANDMARK_INDICES[category][0])

        # draw output
        for i in range(N):
            img = np.zeros((1024, 768, 3), np.uint8)

            for j1, j2 in JOINT_PAIRS_MAP_ALL.keys():
                cv2.circle(img, (int(keypoints_hk[i][j1][0]*768), int(keypoints_hk[i][j1][1]*1024)), 5, (255, 0, 0), -1)
                cv2.circle(img, (int(keypoints_hk[i][j2][0]*768), int(keypoints_hk[i][j2][1]*1024)), 5, (255, 0, 0), -1)
                cv2.line(img, (int(keypoints_hk[i][j1][0]*768), int(keypoints_hk[i][j1][1]*1024)), (int(keypoints_hk[i][j2][0]*768), int(keypoints_hk[i][j2][1]*1024)), color=(255, 0, 0), thickness=2)

            for j in range(n):
                cv2.circle(img, (int(output[i][2*j]*768), int(output[i][2*j+1]*1024)), 5, (0, 0, 255), -1)
            for j in range(n):
                cv2.circle(img, (int(keypoints_fkp[i][j][0]*768), int(keypoints_fkp[i][j][1]*1024)), 5, (0, 255, 0), -1)

            """for c in range(1,14):
                for j1, j2 in DRAW_LANDMARK_PAIRS[c]:
                    cv2.circle(img, (int(keypoints_fkp[i][j1][0]*768), int(keypoints_fkp[i][j1][1]*1024)), 5, (0, 255, 0), -1)
                    cv2.circle(img, (int(keypoints_fkp[i][j2][0]*768), int(keypoints_fkp[i][j2][1]*1024)), 5, (0, 255, 0), -1)
                    cv2.line(img, (int(keypoints_fkp[i][j1][0]*768), int(keypoints_fkp[i][j1][1]*1024)), (int(keypoints_fkp[i][j2][0]*768), int(keypoints_fkp[i][j2][1]*1024)), color=(0, 255, 0), thickness=2)
            """
            path = f"logs/{split}/step_{pl_module.global_step}_image_{batch_idx}_{i}.jpg"
            cv2.imwrite(path, img)
    
    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                pass
            return True
        return False 
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        check_idx = pl_module.global_step
        if (self.check_frequency(check_idx)):
            self.log_img(pl_module, batch, batch_idx, split="train")