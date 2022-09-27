# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from .cofw import COFW, MaskedCOFW
from .face300w import Face300W, MaskedFace300W
from .wflw import WFLW, MaskedWFLW

__all__ = ['COFW', 'Face300W', 'WFLW', 'get_dataset', 'MaskedFace300W', 'MaskedCOFW', 'MaskedWFLW']


def get_dataset(config):

    if config.DATASET.DATASET == 'COFW':
        return COFW
    elif config.DATASET.DATASET == 'MaskedCOFW':
        return MaskedCOFW
    elif config.DATASET.DATASET == '300W':
        return Face300W
    elif config.DATASET.DATASET == 'Masked300W':
        return MaskedFace300W
    elif config.DATASET.DATASET == 'WFLW':
        return WFLW
    elif config.DATASET.DATASET == 'MaskedWFLW':
        return MaskedWFLW
    else:
        raise NotImplemented()

