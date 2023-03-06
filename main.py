import argparse
import datetime
import gc

import tensorboardX as ts
import torch.cuda.amp as amp
import torch.optim as optim
from tabulate import tabulate
from torch.utils.data import DataLoader
from torchinfo import summary

from datasets import *
from models import *
from utils import *

# Set Variables
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
gc.collect()
torch.cuda.empty_cache()


def get_args():
    """
    get arguments of program :)
    """
    parser = argparse.ArgumentParser(description="A Good Python Code for train your semantic segmentation models",
                                     add_help=True)
    parser.add_argument("--name", default="bisenetv2", type=str,
                        help=f"experiment name ")
    parser.add_argument("--dataset", default="cityscapes", type=str,
                        help=f"dataset name ")

    return parser


def main():
    """
    program main function :)
    """
    setup_env()
    args = get_args().parse_args()

    # loading configs for dataset
    if args.dataset == "cityscapes":
        import configs.cityscapes as config
    elif args.dataset == "coco":
        import configs.coco as config
    else:
        raise NotImplemented

    num_classes = config.NUM_CLASSES

    # create train & validation dataset
    train_transforms, val_transforms = get_augs(config)
    train_ds = DATASETS[args.dataset](phase="train", root=config.DIR, transforms=train_transforms)
    valid_ds = DATASETS[args.dataset](phase="val", root=config.DIR, transforms=val_transforms)

    # create dataloaders for train & validation
    nw = min([os.cpu_count() // WORLD_SIZE, config.BATCH_SIZE if config.BATCH_SIZE > 1 else 0, config.NUM_WORKER])
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

    # Load Model
    model = models_collections[config.MODEL](num_classes)

    # Show some configs to check
    all_configs = [["name", args.name],
                   ["model", config.MODEL],
                   ["num classes", num_classes],
                   ["num train data", len(train_ds)],
                   ["num valid data", len(valid_ds)],
                   ["train size", config.TRAIN_SIZE],
                   ["batch size", config.BATCH_SIZE],
                   ["val size", config.VAL_SIZE],
                   ["class weights", config.USE_CLASS_WEIGHTS],
                   ["train transforms", config.TRAIN_AUGS],
                   ["optimizer", config.OPTIMIZER],
                   ["lr", config.LR],
                   ["weight decay", config.WEIGHT_DECAY],
                   ["momentum", config.SGD_MOMENTUM],
                   ["adam betas", config.ADAMW_BETAS],
                   ["epochs", config.EPOCHS],
                   ["warmup epochs", config.WARMUP_EPOCHS],
                   ["workers", nw],
                   ["schedular", config.SCHEDULER_METHOD],
                   ["ignore label", config.IGNORE_LABEL],
                   ["label smoothing", config.LABEL_SMOOTHING],
                   ["ohem threshold", config.OHEM_THRESH],
                   ["focal loss alpha", config.FOCAL_ALPHA],
                   ["focal loss gamma", config.FOCAL_GAMMA],
                   ["use ema", config.USE_EMA],
                   ["overfit test", config.OVERFIT_TEST],
                   ["resume", config.RESUME], ]

    print(tabulate(all_configs, headers=["name", "config"]))
    print()
    # print model summary
    print(
        summary(model, (config.BATCH_SIZE, 3, config.TRAIN_SIZE[0], config.TRAIN_SIZE[1]), device=DEVICE, col_width=16,
                col_names=["output_size", "num_params", "mult_adds"], ))
    print()
    # calculate & print overfit test
    if not config.OVERFIT_TEST and args.dataset == "cityscapes":
        print(f"{config.MODEL} FPS: {fps_calculator(reparameterize_model(model), [3, 1024, 2048])}")
        print()

