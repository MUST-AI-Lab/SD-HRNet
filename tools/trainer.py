import logging
import time

import numpy as np
import torch.nn as nn

from tensorboardX import SummaryWriter

import torch
from lib.datasets.prefetch_data import data_prefetcher, mi_data_prefetcher
from lib.utils import utils
from lib.core.evaluation import decode_preds, compute_nme

logger = logging.getLogger(__name__)


class SearchTrainer(object):
    def __init__(self, train_data, val_data, search_optim, criterion_train, criterion_val, scheduler, config):
        self.train_data = train_data
        self.val_data = val_data
        self.search_optim = search_optim
        self.criterion_train = criterion_train
        self.criterion_val = criterion_val
        self.scheduler = scheduler
        self.config = config

    def inference(self, model, epoch, masks=None, weights=None):
        data_time = utils.AverageMeter()
        batch_time = utils.AverageMeter()
        losses = utils.AverageMeter()

        nme_count = 0
        nme_batch_sum = 0
        count_failure_008 = 0
        count_failure_010 = 0

        num_classes = self.config.MODEL.NUM_JOINTS
        predictions = torch.zeros((len(self.val_data.dataset), num_classes, 2))

        model.eval()
        start = time.time()
        prefetcher = data_prefetcher(self.val_data)
        input, target, meta = prefetcher.next()
        step = 0
        while input is not None:
            step += 1
            data_t = time.time() - start
            n = input.size(0)
            outputs = model(input)

            # NME
            score_map = outputs.data.cpu()
            preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])

            nme_temp = compute_nme(preds, meta)
            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            batch_t = time.time() - start
            data_time.update(data_t)
            batch_time.update(batch_t)

            start = time.time()
            input, target, meta = prefetcher.next()

        nme = nme_batch_sum / nme_count
        failure_008_rate = count_failure_008 / nme_count
        failure_010_rate = count_failure_010 / nme_count

        msg = 'Test Results time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
              '[010]:{:.4f}'.format(batch_time.avg, losses.avg, nme,
                                    failure_008_rate, failure_010_rate)
        logger.info(msg)

        return nme, predictions
