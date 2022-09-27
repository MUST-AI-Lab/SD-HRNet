# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn), Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path
from collections import OrderedDict
import numpy as np

import torch


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
                        (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def count_parameters_in_MB(model):
    total_params = []
    for name, v in model.named_parameters():
        if "aux" not in name and "head." not in name and 'beta' not in name and 'alpha' not in name and \
                'input_block.0' not in name and 'input_block.1' not in name and 'input_block.3' not in name and \
                'input_block.4' not in name:
            total_params.append(np.prod(v.size()))

    return np.sum(total_params) / 1e6

def rewrite_keys_for_SDHRNet_300w(pretrained_dict):
    new_dict = OrderedDict()
    for i, (k, v) in enumerate(pretrained_dict.items()):
        key_name = ""
        lst_k = k.split(".")

        if lst_k[0] == "input_block":
            # stem and layer1 keys
            if lst_k[1] == "0":
                key_name = "conv1.{}".format(lst_k[-1])
            elif lst_k[1] == "1":
                key_name = "bn1.{}".format(lst_k[-1])
            elif lst_k[1] == "3":
                key_name = "conv2.{}".format(lst_k[-1])
            elif lst_k[1] == "4":
                key_name = "bn2.{}".format(lst_k[-1])
            elif lst_k[1] == "6":
                key_name = "layer1.{}.{}.{}.{}".format(lst_k[2], lst_k[3], lst_k[4], lst_k[-1])
        elif lst_k[0] == "Stage":
            stg_id = int(lst_k[1])
            if stg_id == 0:
                # 64 -> 18
                # 64 -> 36
                # 64 -> 144
                if lst_k[2] == "0" and lst_k[3] == "0":
                    key_name = "stage{}.0.fuse_layers.0.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1":
                    key_name = "stage{}.0.fuse_layers.1.0.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "3":
                    key_name = "stage{}.0.fuse_layers.2.0.{}.{}.{}".format(stg_id+2, int(int(lst_k[6])/2), lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "0":
                    key_name = "stage{}.0.branches.0.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "1":
                    key_name = "stage{}.0.branches.1.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "3":
                    key_name = "stage{}.0.branches.2.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
            elif stg_id == 1:
                # lst_k[5] -> lst_k[3]
                # 18 -> 18/36
                # 36 -> 36/144
                # 144 -> 18/36/144
                if lst_k[2] == "0" and lst_k[3] == "0" and lst_k[5] == "0":
                    # 144->18
                    key_name = "stage{}.0.fuse_layers.0.1.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "2":
                    # 18->36
                    key_name = "stage{}.0.fuse_layers.1.0.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "0":
                    # 144->36
                    key_name = "stage{}.0.fuse_layers.1.2.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "3" and lst_k[5] == "1":
                    #36->144
                    key_name = "stage{}.0.fuse_layers.2.0.{}.{}.{}".format(stg_id+2, int(int(lst_k[6])/2), lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "0":
                    key_name = "stage{}.0.branches.0.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "1":
                    key_name = "stage{}.0.branches.1.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "3":
                    key_name = "stage{}.0.branches.2.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
            elif stg_id == 2:
                # lst_k[5] -> lst_k[3]
                # 18 -> 18/36
                # 36 -> 18/36
                # 144 -> 36
                if lst_k[2] == "0" and lst_k[3] == "0" and lst_k[5] == "2":
                    # 36->18
                    key_name = "stage{}.0.fuse_layers.0.1.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "2":
                    # 18->36
                    key_name = "stage{}.0.fuse_layers.1.0.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "0":
                    # 144->36
                    key_name = "stage{}.0.fuse_layers.1.2.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1":
                    key_name = "stage{}.0.branches.{}.{}.{}.{}.{}".format(stg_id+2, lst_k[3], lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
            elif stg_id == 3:
                # lst_k[5] -> lst_k[3]
                # 18 -> 18
                # 36 -> 36
                if lst_k[2] == "1":
                    key_name = "stage{}.0.branches.{}.{}.{}.{}.{}".format(stg_id+2, lst_k[3], lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
            elif stg_id == 4:
                # lst_k[5] -> lst_k[3]
                # 18 -> 18
                # 36 -> 18/36
                if lst_k[2] == "0" and lst_k[3] == "0" and lst_k[5] == "2":
                    # 36->18
                    key_name = "stage{}.0.fuse_layers.0.1.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1":
                    key_name = "stage{}.0.branches.{}.{}.{}.{}.{}".format(stg_id+2, lst_k[3], lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
            elif stg_id == 5:
                # lst_k[5] -> lst_k[3]
                # 18 -> 18/36
                # 36 -> 18/36/72
                if lst_k[2] == "0" and lst_k[3] == "0" and lst_k[5] == "2":
                    # 36->18
                    key_name = "stage{}.0.fuse_layers.0.1.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "2":
                    # 18->36
                    key_name = "stage{}.0.fuse_layers.1.0.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "2" and lst_k[5] == "1":
                    # 36->72
                    key_name = "stage{}.0.fuse_layers.2.0.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1":
                    key_name = "stage{}.0.branches.{}.{}.{}.{}.{}".format(stg_id+2, lst_k[3], lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
            elif stg_id == 6:
                # lst_k[5] -> lst_k[3]
                # 18 -> 18/36
                # 36 -> 18/36
                # 72 -> 18/36/72
                if lst_k[2] == "0" and lst_k[3] == "0" and lst_k[5] == "2":
                    # 36->18
                    key_name = "stage{}.0.fuse_layers.0.1.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "0" and lst_k[5] == "1":
                    # 72->18
                    key_name = "stage{}.0.fuse_layers.0.2.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "2":
                    # 18->36
                    key_name = "stage{}.0.fuse_layers.1.0.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "1":
                    # 72->36
                    key_name = "stage{}.0.fuse_layers.1.2.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1":
                    key_name = "stage{}.0.branches.{}.{}.{}.{}.{}".format(stg_id+2, lst_k[3], lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
            elif stg_id == 7:
                # lst_k[5] -> lst_k[3]
                # 18 -> 18/36/72
                # 36 -> 18/36/72
                # 72 -> 18/36/72
                if lst_k[2] == "0" and lst_k[3] == "0" and lst_k[5] == "2":
                    # 36->18
                    key_name = "stage{}.0.fuse_layers.0.1.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "0" and lst_k[5] == "1":
                    # 72->18
                    key_name = "stage{}.0.fuse_layers.0.2.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "2":
                    # 18->36
                    key_name = "stage{}.0.fuse_layers.1.0.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "1":
                    # 72->36
                    key_name = "stage{}.0.fuse_layers.1.2.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "2" and lst_k[5] == "2":
                    # 18->72
                    key_name = "stage{}.0.fuse_layers.2.0.{}.{}.{}".format(stg_id+2, int(int(lst_k[6])/2), lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "2" and lst_k[5] == "1":
                    # 36->72
                    key_name = "stage{}.0.fuse_layers.2.1.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1":
                    key_name = "stage{}.0.branches.{}.{}.{}.{}.{}".format(stg_id+2, lst_k[3], lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
            elif stg_id == 8:
                # lst_k[5] -> lst_k[3]
                # 18 -> 18/36/72
                # 36 -> 18/36
                # 72 -> 18/36/72/144
                if lst_k[2] == "0" and lst_k[3] == "0" and lst_k[5] == "2":
                    # 36->18
                    key_name = "stage{}.0.fuse_layers.0.1.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "0" and lst_k[5] == "1":
                    # 72->18
                    key_name = "stage{}.0.fuse_layers.0.2.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "2":
                    # 18->36
                    key_name = "stage{}.0.fuse_layers.1.0.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "1":
                    # 72->36
                    key_name = "stage{}.0.fuse_layers.1.2.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "2" and lst_k[5] == "2":
                    # 18->72
                    key_name = "stage{}.0.fuse_layers.2.0.{}.{}.{}".format(stg_id+2, int(int(lst_k[6])/2), lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "3" and lst_k[5] == "0":
                    # 72->144
                    key_name = "stage{}.0.fuse_layers.3.0.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1":
                    key_name = "stage{}.0.branches.{}.{}.{}.{}.{}".format(stg_id+2, lst_k[3], lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
        elif lst_k[0] == "head":
            key_name = k
        elif lst_k[0] == "beta_weights":
            key_name = k

        new_dict[key_name] = v

    return new_dict

def rewrite_keys_for_SDHRNet_cofw(pretrained_dict):
    new_dict = OrderedDict()
    for i, (k, v) in enumerate(pretrained_dict.items()):
        key_name = ""
        lst_k = k.split(".")

        if lst_k[0] == "input_block":
            # stem and layer1 keys
            if lst_k[1] == "0":
                key_name = "conv1.{}".format(lst_k[-1])
            elif lst_k[1] == "1":
                key_name = "bn1.{}".format(lst_k[-1])
            elif lst_k[1] == "3":
                key_name = "conv2.{}".format(lst_k[-1])
            elif lst_k[1] == "4":
                key_name = "bn2.{}".format(lst_k[-1])
            elif lst_k[1] == "6":
                key_name = "layer1.{}.{}.{}.{}".format(lst_k[2], lst_k[3], lst_k[4], lst_k[-1])
        elif lst_k[0] == "Stage":
            stg_id = int(lst_k[1])
            if stg_id == 0:
                # 64 -> 18
                # 64 -> 36
                # 64 -> 72
                if lst_k[2] == "0" and lst_k[3] == "0":
                    key_name = "stage{}.0.fuse_layers.0.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1":
                    key_name = "stage{}.0.fuse_layers.1.0.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "2":
                    key_name = "stage{}.0.fuse_layers.2.0.{}.{}.{}".format(stg_id+2, int(int(lst_k[6])/2), lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "0":
                    key_name = "stage{}.0.branches.0.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "1":
                    key_name = "stage{}.0.branches.1.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "2":
                    key_name = "stage{}.0.branches.2.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
            elif stg_id == 1:
                # lst_k[5] -> lst_k[3]
                # 18 -> 18
                # 36 -> 72/144
                # 72 -> 144
                if lst_k[2] == "0" and lst_k[3] == "2" and lst_k[5] == "1":
                    # 36->72
                    key_name = "stage{}.0.fuse_layers.1.0.0.{}.{}".format(stg_id + 2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "3" and lst_k[5] == "1":
                    # 36->144
                    key_name = "stage{}.0.fuse_layers.2.0.{}.{}.{}".format(stg_id+2, int(int(lst_k[6])/2), lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "3" and lst_k[5] == "0":
                    # 72->144
                    key_name = "stage{}.0.fuse_layers.2.1.0.{}.{}".format(stg_id + 2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "0":
                    key_name = "stage{}.0.branches.0.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "2":
                    key_name = "stage{}.0.branches.1.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "3":
                    key_name = "stage{}.0.branches.2.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
            elif stg_id == 2:
                # lst_k[5] -> lst_k[3]
                # 18 -> 18/36
                # 72 -> 72
                # 144 -> 36
                if lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "2":
                    # 18->36
                    key_name = "stage{}.0.fuse_layers.1.0.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "0":
                    # 144->36
                    key_name = "stage{}.0.fuse_layers.1.1.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1":
                    key_name = "stage{}.0.branches.{}.{}.{}.{}.{}".format(stg_id+2, lst_k[3], lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
            elif stg_id == 3:
                # lst_k[5] -> lst_k[3]
                # 18 -> 18
                # 36 -> 18/144
                # 72 -> 36
                if lst_k[2] == "0" and lst_k[3] == "0" and lst_k[5] == "2":
                    # 36->18
                    key_name = "stage{}.0.fuse_layers.0.1.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "1":
                    # 72->36
                    key_name = "stage{}.0.fuse_layers.1.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "3" and lst_k[5] == "1":
                    #36->144
                    key_name = "stage{}.0.fuse_layers.2.0.{}.{}.{}".format(stg_id+2, int(int(lst_k[6])/2), lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "0":
                    key_name = "stage{}.0.branches.0.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "1":
                    key_name = "stage{}.0.branches.1.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "3":
                    key_name = "stage{}.0.branches.2.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
            elif stg_id == 4:
                # lst_k[5] -> lst_k[3]
                # 18 -> 18/36
                # 36 -> 72
                # 144 -> 36/72
                if lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "2":
                    # 18->36
                    key_name = "stage{}.0.fuse_layers.1.0.0.{}.{}".format(stg_id + 2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "0":
                    # 144->36
                    key_name = "stage{}.0.fuse_layers.1.1.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "2" and lst_k[5] == "1":
                    # 36->72
                    key_name = "stage{}.0.fuse_layers.2.0.0.{}.{}".format(stg_id + 2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "2" and lst_k[5] == "0":
                    # 144->72
                    key_name = "stage{}.0.fuse_layers.2.1.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1":
                    key_name = "stage{}.0.branches.{}.{}.{}.{}.{}".format(stg_id+2, lst_k[3], lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
            elif stg_id == 5:
                # lst_k[5] -> lst_k[3]
                # 18 -> 18
                # 36 -> 18/144
                # 72 -> 18/36
                if lst_k[2] == "0" and lst_k[3] == "0" and lst_k[5] == "2":
                    # 36->18
                    key_name = "stage{}.0.fuse_layers.0.1.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "0" and lst_k[5] == "1":
                    # 72->18
                    key_name = "stage{}.0.fuse_layers.0.2.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "1":
                    # 72->36
                    key_name = "stage{}.0.fuse_layers.1.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "3" and lst_k[5] == "1":
                    #36->144
                    key_name = "stage{}.0.fuse_layers.2.0.{}.{}.{}".format(stg_id+2, int(int(lst_k[6])/2), lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "0":
                    key_name = "stage{}.0.branches.0.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "1":
                    key_name = "stage{}.0.branches.1.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "3":
                    key_name = "stage{}.0.branches.2.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
            elif stg_id == 6:
                # lst_k[5] -> lst_k[3]
                # 18 -> 36
                # 36 -> 36
                # 144 -> 18/36/72
                if lst_k[2] == "0" and lst_k[3] == "0" and lst_k[5] == "0":
                    # 144->18
                    key_name = "stage{}.0.fuse_layers.0.0.{}.{}".format(stg_id + 2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "2":
                    # 18->36
                    key_name = "stage{}.0.fuse_layers.1.0.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "0":
                    # 144->36
                    key_name = "stage{}.0.fuse_layers.1.2.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "2" and lst_k[5] == "0":
                    # 144->72
                    key_name = "stage{}.0.fuse_layers.2.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1":
                    key_name = "stage{}.0.branches.{}.{}.{}.{}.{}".format(stg_id+2, lst_k[3], lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
            elif stg_id == 7:
                # lst_k[5] -> lst_k[3]
                # 18 -> 18/36
                # 36 -> 18/36
                # 72 -> 18/36/72
                if lst_k[2] == "0" and lst_k[3] == "0" and lst_k[5] == "2":
                    # 36->18
                    key_name = "stage{}.0.fuse_layers.0.1.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "0" and lst_k[5] == "1":
                    # 72->18
                    key_name = "stage{}.0.fuse_layers.0.2.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "2":
                    # 18->36
                    key_name = "stage{}.0.fuse_layers.1.0.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "1":
                    # 72->36
                    key_name = "stage{}.0.fuse_layers.1.2.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1":
                    key_name = "stage{}.0.branches.{}.{}.{}.{}.{}".format(stg_id+2, lst_k[3], lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
            elif stg_id == 8:
                # lst_k[5] -> lst_k[3]
                # 18 -> 18/36
                # 36 -> 36
                # 72 -> 36/72
                if lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "2":
                    # 18->36
                    key_name = "stage{}.0.fuse_layers.1.0.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "1":
                    # 72->36
                    key_name = "stage{}.0.fuse_layers.1.2.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1":
                    key_name = "stage{}.0.branches.{}.{}.{}.{}.{}".format(stg_id+2, lst_k[3], lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
        elif lst_k[0] == "head":
            key_name = k
        elif lst_k[0] == "beta_weights":
            key_name = k

        new_dict[key_name] = v

    return new_dict

def rewrite_keys_for_SDHRNet_wflw(pretrained_dict):
    new_dict = OrderedDict()
    for i, (k, v) in enumerate(pretrained_dict.items()):
        key_name = ""
        lst_k = k.split(".")

        if lst_k[0] == "input_block":
            # stem and layer1 keys
            if lst_k[1] == "0":
                key_name = "conv1.{}".format(lst_k[-1])
            elif lst_k[1] == "1":
                key_name = "bn1.{}".format(lst_k[-1])
            elif lst_k[1] == "3":
                key_name = "conv2.{}".format(lst_k[-1])
            elif lst_k[1] == "4":
                key_name = "bn2.{}".format(lst_k[-1])
            elif lst_k[1] == "6":
                key_name = "layer1.{}.{}.{}.{}".format(lst_k[2], lst_k[3], lst_k[4], lst_k[-1])
        elif lst_k[0] == "Stage":
            stg_id = int(lst_k[1])
            if stg_id == 0:
                # 64 -> 18
                # 64 -> 36
                # 64 -> 72
                if lst_k[2] == "0" and lst_k[3] == "0":
                    key_name = "stage{}.0.fuse_layers.0.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1":
                    key_name = "stage{}.0.fuse_layers.1.0.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "2":
                    key_name = "stage{}.0.fuse_layers.2.0.{}.{}.{}".format(stg_id+2, int(int(lst_k[6])/2), lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "0":
                    key_name = "stage{}.0.branches.0.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "1":
                    key_name = "stage{}.0.branches.1.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "2":
                    key_name = "stage{}.0.branches.2.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
            elif stg_id == 1:
                # lst_k[5] -> lst_k[3]
                # 18 -> 36/144
                # 36 -> 36/72/144
                # 72 -> 144
                if lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "2":
                    # 18->36
                    key_name = "stage{}.0.fuse_layers.0.0.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "2" and lst_k[5] == "1":
                    # 36->72
                    key_name = "stage{}.0.fuse_layers.1.0.0.{}.{}".format(stg_id + 2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "3" and lst_k[5] == "2":
                    # 18->144
                    key_name = "stage{}.0.fuse_layers.2.0.{}.{}.{}".format(stg_id+2, int(int(lst_k[6])/2), lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "3" and lst_k[5] == "1":
                    # 36->144
                    key_name = "stage{}.0.fuse_layers.2.1.{}.{}.{}".format(stg_id+2, int(int(lst_k[6])/2), lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "3" and lst_k[5] == "0":
                    # 72->144
                    key_name = "stage{}.0.fuse_layers.2.2.0.{}.{}".format(stg_id + 2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "1":
                    key_name = "stage{}.0.branches.0.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "2":
                    key_name = "stage{}.0.branches.1.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "3":
                    key_name = "stage{}.0.branches.2.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
            elif stg_id == 2:
                # lst_k[5] -> lst_k[3]
                # 36 -> 18
                # 72 -> 36
                # 144 -> 18/36/144
                if lst_k[2] == "0" and lst_k[3] == "0" and lst_k[5] == "2":
                    # 36->18
                    key_name = "stage{}.0.fuse_layers.0.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "0" and lst_k[5] == "0":
                    # 144->18
                    key_name = "stage{}.0.fuse_layers.0.1.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "1":
                    # 72->36
                    key_name = "stage{}.0.fuse_layers.1.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "0":
                    # 144->36
                    key_name = "stage{}.0.fuse_layers.1.1.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "0":
                    key_name = "stage{}.0.branches.0.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "1":
                    key_name = "stage{}.0.branches.1.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "3":
                    key_name = "stage{}.0.branches.2.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
            elif stg_id == 3:
                # lst_k[5] -> lst_k[3]
                # 18 -> 36/72
                # 36 -> 36
                # 144 -> 72
                if lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "2":
                    # 18->36
                    key_name = "stage{}.0.fuse_layers.0.0.0.{}.{}".format(stg_id + 2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "2" and lst_k[5] == "2":
                    # 18->72
                    key_name = "stage{}.0.fuse_layers.1.0.{}.{}.{}".format(stg_id+2, int(int(lst_k[6])/2), lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "2" and lst_k[5] == "0":
                    # 144->72
                    key_name = "stage{}.0.fuse_layers.1.1.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "1":
                    key_name = "stage{}.0.branches.0.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "2":
                    key_name = "stage{}.0.branches.1.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
            elif stg_id == 4:
                # lst_k[5] -> lst_k[3]
                # 36 -> 144
                # 72 -> 18/144
                if lst_k[2] == "0" and lst_k[3] == "0" and lst_k[5] == "1":
                    # 72->18
                    key_name = "stage{}.0.fuse_layers.0.0.{}.{}".format(stg_id + 2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "3" and lst_k[5] == "1":
                    # 36->144
                    key_name = "stage{}.0.fuse_layers.1.0.{}.{}.{}".format(stg_id+2, int(int(lst_k[6])/2), lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "3" and lst_k[5] == "0":
                    # 72->144
                    key_name = "stage{}.0.fuse_layers.1.1.0.{}.{}".format(stg_id + 2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "0":
                    key_name = "stage{}.0.branches.0.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "3":
                    key_name = "stage{}.0.branches.1.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
            elif stg_id == 5:
                # lst_k[5] -> lst_k[3]
                # 18 -> 72
                # 144 -> 72
                if lst_k[2] == "0" and lst_k[3] == "2" and lst_k[5] == "2":
                    # 18->72
                    key_name = "stage{}.0.fuse_layers.0.0.{}.{}.{}".format(stg_id + 2, int(int(lst_k[6]) / 2),
                                                                           lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "2" and lst_k[5] == "0":
                    # 144->72
                    key_name = "stage{}.0.fuse_layers.0.1.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "2":
                    key_name = "stage{}.0.branches.0.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
            elif stg_id == 6:
                # lst_k[5] -> lst_k[3]
                # 72 -> 18/72/144
                if lst_k[2] == "0" and lst_k[3] == "0" and lst_k[5] == "1":
                    # 72->18
                    key_name = "stage{}.0.fuse_layers.0.0.{}.{}".format(stg_id + 2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "3" and lst_k[5] == "0":
                    # 72->144
                    key_name = "stage{}.0.fuse_layers.2.0.0.{}.{}".format(stg_id + 2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "0":
                    key_name = "stage{}.0.branches.0.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "2":
                    key_name = "stage{}.0.branches.1.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1" and lst_k[3] == "3":
                    key_name = "stage{}.0.branches.2.{}.{}.{}.{}".format(stg_id+2, lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
            elif stg_id == 7:
                # lst_k[5] -> lst_k[3]
                # 18 -> 18/36
                # 72 -> 18
                # 144 -> 36/72
                if lst_k[2] == "0" and lst_k[3] == "0" and lst_k[5] == "1":
                    # 72->18
                    key_name = "stage{}.0.fuse_layers.0.1.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "2":
                    # 18->36
                    key_name = "stage{}.0.fuse_layers.1.0.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "0":
                    # 144->36
                    key_name = "stage{}.0.fuse_layers.1.1.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "2" and lst_k[5] == "0":
                    # 144->72
                    key_name = "stage{}.0.fuse_layers.2.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1":
                    key_name = "stage{}.0.branches.{}.{}.{}.{}.{}".format(stg_id+2, lst_k[3], lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
            elif stg_id == 8:
                # lst_k[5] -> lst_k[3]
                # 18 -> 18/36/72
                # 36 -> 18/36/72
                # 72 -> 18/36/72
                if lst_k[2] == "0" and lst_k[3] == "0" and lst_k[5] == "2":
                    # 36->18
                    key_name = "stage{}.0.fuse_layers.0.1.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "0" and lst_k[5] == "1":
                    # 72->18
                    key_name = "stage{}.0.fuse_layers.0.2.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "2":
                    # 18->36
                    key_name = "stage{}.0.fuse_layers.1.0.0.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "1" and lst_k[5] == "1":
                    # 72->36
                    key_name = "stage{}.0.fuse_layers.1.2.{}.{}".format(stg_id+2, lst_k[-2], lst_k[-1])
                if lst_k[2] == "0" and lst_k[3] == "2" and lst_k[5] == "2":
                    # 18->72
                    key_name = "stage{}.0.fuse_layers.2.0.{}.{}.{}".format(stg_id + 2, int(int(lst_k[6]) / 2),
                                                                           lst_k[-2], lst_k[-1])
                elif lst_k[2] == "0" and lst_k[3] == "2" and lst_k[5] == "1":
                    # 36->72
                    key_name = "stage{}.0.fuse_layers.2.1.0.{}.{}".format(stg_id + 2, lst_k[-2], lst_k[-1])
                elif lst_k[2] == "1":
                    key_name = "stage{}.0.branches.{}.{}.{}.{}.{}".format(stg_id+2, lst_k[3], lst_k[5], lst_k[-3], lst_k[-2], lst_k[-1])
        elif lst_k[0] == "head":
            key_name = k
        elif lst_k[0] == "beta_weights":
            key_name = k

        new_dict[key_name] = v

    return new_dict

def get_retained_blocks(config):
    list_retained_blocks = []
    if '300W' in config.DATASET.DATASET:
        # 300w
        list_retained_blocks.append(torch.tensor([1, 1, 0, 1], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([1, 1, 0, 1], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([1, 1, 0, 0], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([1, 1, 0, 0], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([1, 1, 0, 0], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([1, 1, 1, 0], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([1, 1, 1, 0], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([1, 1, 1, 0], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([1, 1, 1, 1], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([], dtype=torch.uint8).cuda())
    elif 'COFW' in config.DATASET.DATASET:
        # cofw
        list_retained_blocks.append(torch.tensor([1, 1, 1, 0], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([1, 0, 1, 1], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([1, 1, 1, 0], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([1, 1, 0, 1], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([1, 1, 1, 0], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([1, 1, 0, 1], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([1, 1, 1, 0], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([1, 1, 1, 0], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([1, 1, 1, 0], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([], dtype=torch.uint8).cuda())
    elif 'WFLW' in config.DATASET.DATASET:
        # wflw
        list_retained_blocks.append(torch.tensor([1, 1, 1, 0], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([0, 1, 1, 1], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([1, 1, 0, 1], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([0, 1, 1, 0], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([1, 0, 0, 1], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([0, 0, 1, 0], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([1, 0, 1, 1], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([1, 1, 1, 0], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([1, 1, 1, 0], dtype=torch.uint8).cuda())
        list_retained_blocks.append(torch.tensor([], dtype=torch.uint8).cuda())

    return list_retained_blocks

