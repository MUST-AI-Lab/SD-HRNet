import os
import pprint
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from lib.config import config, update_config
from lib.utils import utils
from lib.datasets import get_dataset
from lib.models.shrinking_hr_model import Shrinking_Network_HR
from lib.utils.multadds_count import comp_multadds
from lib.utils.utils import rewrite_keys_for_SDHRNet_300w, rewrite_keys_for_SDHRNet_cofw, rewrite_keys_for_SDHRNet_wflw
from tools.trainer import SearchTrainer


def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--model-file', help='model parameters', required=True, type=str)
    parser.add_argument('--mask-file', help='mask parameters', required=True, type=str)
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():

    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()

    gpus = list(config.GPUS)
    dataset_type = get_dataset(config)

    test_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train=False),
        batch_size=8 * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    model = Shrinking_Network_HR(config, args.model_file)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # load model
    state_dict = torch.load(args.model_file)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    model_dict = model.module.state_dict()

    if config.DATASET.DATASET == "300W":
        pretrained_dict = rewrite_keys_for_SDHRNet_300w(state_dict)
    elif config.DATASET.DATASET == "COFW":
        pretrained_dict = rewrite_keys_for_SDHRNet_cofw(state_dict)
    elif config.DATASET.DATASET == "WFLW":
        pretrained_dict = rewrite_keys_for_SDHRNet_wflw(state_dict)
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.module.load_state_dict(model_dict)

    search_trainer = SearchTrainer(train_data=None, val_data=test_loader, search_optim=None,
                                   criterion_train=None, criterion_val=None, scheduler=None, config=config)
    with torch.no_grad():
        nme, predictions = search_trainer.inference(model, epoch=-1, masks=None)
    torch.save(predictions, os.path.join(final_output_dir, 'predictions_sd-hrnet.pth'))


if __name__ == '__main__':
    main()

