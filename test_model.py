import argparse
import math
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from models.yolo import Model

from models.common import *
from models.experimental import MixConv2d, CrossConv, C3
from utils.general import check_anchor_order, make_divisible, check_file
from utils.torch_utils import (
    time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, select_device)

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
opt = parser.parse_args()
opt.cfg = check_file(opt.cfg)  # check file
device = select_device(opt.device)

# Create model
model = Model(opt.cfg).to(device)
model.train()

# Profile
img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
# y = model(img, profile=True)

# ONNX export
# model.model[-1].export = True
# torch.onnx.export(model, img, opt.cfg.replace('.yaml', '.onnx'), verbose=True, opset_version=11)

# Tensorboard
from torch.utils.tensorboard import SummaryWriter
tb_writer = SummaryWriter()
print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
tb_writer.add_graph(model.model, img)  # add model to tensorboard
tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard