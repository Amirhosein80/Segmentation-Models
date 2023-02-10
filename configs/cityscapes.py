import torch

# Dataset info :)

DIR = "data/Cityscapes/"
PHASES = ["train", "val"]
TRAIN_MODES = ["supervised"]  # TODO Add self-supervised mode & self-distillation

CLASSES = {0: {"name": "road", "color": (128, 64, 128)}, 1: {"name": "sidewalk", "color": (244, 35, 232)},
           2: {"name": "building", "color": (70, 70, 70)}, 3: {"name": "wall", "color": (102, 102, 156)},
           4: {"name": "fence", "color": (190, 153, 153)}, 5: {"name": "pole", "color": (153, 153, 153)},
           6: {"name": "traffic light", "color": (250, 170, 30)},
           7: {"name": "traffic sign", "color": (220, 220, 0)},
           8: {"name": "vegetation", "color": (107, 142, 35)},
           9: {"name": "terrain", "color": (152, 251, 152)},
           10: {"name": "sky", "color": (70, 130, 180)}, 11: {"name": "person", "color": (220, 20, 60)},
           12: {"name": "rider", "color": (255, 0, 0)}, 13: {"name": "car", "color": (0, 0, 142)},
           14: {"name": "truck", "color": (0, 0, 70)}, 15: {"name": "bus", "color": (0, 60, 100)},
           16: {"name": "train", "color": (0, 80, 100)},
           17: {"name": "motorcycle", "color": (0, 0, 230)},
           18: {"name": "bicycle", "color": (119, 11, 32)}, }

NUM_CLASSES = 19
IGNORE_LABEL = 255
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CLASS_WEIGHTS = torch.tensor([0.8373, 0.918, 0.866, 1.0345,
                              1.0166, 0.9969, 0.9754, 1.0489,
                              0.8786, 1.0023, 0.9539, 0.9843,
                              1.1116, 0.9037, 1.0865, 1.0955,
                              1.0865, 1.1529, 1.0507])
LABEL_MAPPING = {-1: IGNORE_LABEL, 0: IGNORE_LABEL,
                 1: IGNORE_LABEL, 2: IGNORE_LABEL,
                 3: IGNORE_LABEL, 4: IGNORE_LABEL,
                 5: IGNORE_LABEL, 6: IGNORE_LABEL,
                 7: 0, 8: 1, 9: IGNORE_LABEL,
                 10: IGNORE_LABEL, 11: 2, 12: 3,
                 13: 4, 14: IGNORE_LABEL,
                 15: IGNORE_LABEL,
                 16: IGNORE_LABEL, 17: 5,
                 18: IGNORE_LABEL,
                 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                 25: 12, 26: 13, 27: 14, 28: 15,
                 29: IGNORE_LABEL, 30: IGNORE_LABEL,
                 31: 16, 32: 17, 33: 18}

# Augmentation settings :)

MIN_MAX_SIZE = [400, 1600]
TRAIN_SIZE = [768, 768]
VAL_SIZE = [1024, 1024]

NOISE_FACTOR = 15

# Train settings :)

OPTIMIZER = "SGD"  # Options: ["SGD", "ADAMW"]
BATCH_SIZE = 8
NUM_WORKER = 4
SGD_LR = 0.05
SGD_MOMENTUM = 0.9
ADAMW_LR = 5e-3
ADAMW_BETAS = (0.9, 0.999)
WEIGHT_DECAY = 1e-5
EPOCHS = 300
WARMUP_EPOCHS = 20
WARMUP_FACTOR = 0.1
SCHEDULER_METHOD = "COS"  # Options: ["POLY", "COS"]
POLY_POWER = 0.9
LABEL_SMOOTHING = 0.0
OHEM_THRESH = 0.3
MODEL = "bisenetv2"
LOSS = "OHEM"  # Options: ["OHEM", "CROSS"]
RESUME = False
OVERFIT_TEST = False
OVERFIT_EPOCHS = 100
