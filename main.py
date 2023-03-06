import argparse
import datetime
import gc

import tensorboardX as ts
import torch.cuda.amp as amp
import torch.optim as optim
from tabulate import tabulate
from torch.utils.data import DataLoader

from datasets import *
from models import *
from utils import *

WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
NOW = datetime.datetime.now()
NOW = f"{NOW.year}-{NOW.month}-{NOW.day}_{NOW.hour}-{NOW.minute}-{NOW.second}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
gc.collect()
torch.cuda.empty_cache()


def get_args():
    parser = argparse.ArgumentParser(description="RTS Segmentation Training", add_help=True)
    parser.add_argument("--name", default="bisenetv2", type=str,
                        help=f"experiment name ")
    parser.add_argument("--dataset", default="cityscapes", type=str,
                        help=f"dataset name ")

    return parser

def main():
    setup_env()
    args = get_args().parse_args()
    if args.dataset == "cityscapes":
        import configs.cityscapes as config
    elif args.dataset == "coco":
        import configs.coco as config
    else:
        raise NotImplemented
    num_classes = config.NUM_CLASSES

    train_ds = DATASETS[args.dataset](phase="train", root=config.DIR, transforms=Compose([
        RandomResize(min_max_size=config.MIN_MAX_SIZE),
        RandomCrop(crop_size=config.TRAIN_SIZE, ignore_label=config.IGNORE_LABEL),
        RandomHorizontalFlip(),
        RandAugment(),
        ToTensor(),
        Normalize(mean=config.MEAN, std=config.STD),
    ]))

    valid_ds = DATASETS[args.dataset](phase="val", root=config.DIR, transforms=Compose([
        Resize(size=config.VAL_SIZE),
        ToTensor(),
        Normalize(mean=config.MEAN, std=config.STD),
    ]))

    model = models_collections[config.MODEL](num_classes)

    nw = min([os.cpu_count() // WORLD_SIZE, config.BATCH_SIZE if config.BATCH_SIZE > 1 else 0, config.NUM_WORKER])

    all_configs = [["name", args.name],
                   ["model", config.MODEL],
                   ["num classes", num_classes],
                   ["num train data", len(train_ds)],
                   ["num valid data", len(valid_ds)],
                   ["batch size", config.BATCH_SIZE],
                   ["epochs", config.EPOCHS],
                   ["train size", config.TRAIN_SIZE],
                   ["val size", config.VAL_SIZE],
                   ["warmup epochs", config.WARMUP_EPOCHS],
                   ["warmup factor", config.WARMUP_FACTOR],
                   ["workers", nw],
                   ["lr", config.LR],
                   ["optimizer", config.OPTIMIZER],
                   ["weight decay", config.WEIGHT_DECAY],
                   ["poly power", config.POLY_POWER],
                   ["ignore label", config.IGNORE_LABEL],
                   ["label smoothing", config.LABEL_SMOOTHING],
                   ["resume", config.RESUME], ]

    print(tabulate(all_configs, headers=["name", "config"]))
    print()
    print(
        summary(model, (config.BATCH_SIZE, 3, config.TRAIN_SIZE[0], config.TRAIN_SIZE[1]), device=DEVICE, col_width=16,
                col_names=["output_size", "num_params", "mult_adds"], ))
    print()
    if not config.OVERFIT_TEST and args.dataset == "cityscapes":
        print(f"{config.MODEL} FPS: {fps_calculator(reparameterize_model(model), [3, 1024, 2048])}")
        print()

    train_sampler = torch.utils.data.RandomSampler(train_ds)
    test_sampler = torch.utils.data.SequentialSampler(valid_ds)
    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, sampler=train_sampler,
                          num_workers=nw, drop_last=True,
                          collate_fn=collate_fn,
                          worker_init_fn=worker_init_fn)
    valid_dl = DataLoader(valid_ds, batch_size=1, sampler=test_sampler,
                          num_workers=nw, drop_last=True,
                          collate_fn=collate_fn,
                          worker_init_fn=worker_init_fn)

    params = model.get_params(lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    if config.OPTIMIZER == "adamw":
        optimizer = optim.AdamW(params, lr=config.LR, weight_decay=0 if config.OVERFIT_TEST else config.WEIGHT_DECAY,
                                betas=config.ADAMW_BETAS)
    elif config.OPTIMIZER == "sgd":
        optimizer = optim.SGD(params, lr=config.LR, weight_decay=0 if config.OVERFIT_TEST else config.WEIGHT_DECAY,
                              momentum=config.MOMENTUM)
    else:
        raise NotImplemented
    scaler = amp.GradScaler()