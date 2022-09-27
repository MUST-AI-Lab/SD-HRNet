from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from .operations import MBConv


BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, num_retained_blocks, blocks, num_blocks, num_outchannels,
                 num_channels, fuse_method, beta_in_channels, stage_id, beta_in_id, beta_weights, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_retained_blocks, num_outchannels)

        self.num_outchannels = num_outchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.beta_in_channels = beta_in_channels
        self.beta_in_id = beta_in_id
        self.channels = [18, 36, 72, 144]
        self.size = [64, 32, 16, 8]
        self.beta_weights = beta_weights

        self.multi_scale_output = multi_scale_output

        # if stage_id > 2:
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)
        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)

    def _check_branches(self, num_retained_blocks, num_outchannels):
        if sum(num_retained_blocks) != len(num_outchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_OUTCHANNELS({})'.format(
                sum(num_retained_blocks), len(num_outchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        layers = []
        for i in range(num_blocks[branch_index]):
            layers.append(block(self.num_outchannels[branch_index],
                                num_channels[branch_index], 3, stride, 1, t=1))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        num_branches = self.num_branches
        out_chs = self.num_outchannels
        fuse_layers = []
        for i, in_chs in enumerate(self.beta_in_channels):
            fuse_layer = []
            for j, in_ch in enumerate(in_chs):
                if in_ch == 64:
                    if out_chs[i] == 18:
                        fuse_layer.append(nn.Sequential(
                            nn.Conv2d(in_ch,
                                      out_chs[i],
                                      3,
                                      1,
                                      1,
                                      bias=False),
                            BatchNorm2d(out_chs[i], momentum=BN_MOMENTUM)))
                    else:
                        conv3x3s = []
                        num_conv = int(math.log2(out_chs[i] // 18))
                        for k in range(num_conv):
                            if k == num_conv - 1:
                                conv3x3s.append(nn.Sequential(
                                    nn.Conv2d(in_ch,
                                              out_chs[i],
                                              3, 2, 1, bias=False),
                                    BatchNorm2d(out_chs[i], momentum=BN_MOMENTUM)))
                            else:
                                conv3x3s.append(nn.Sequential(
                                    nn.Conv2d(in_ch,
                                              in_ch,
                                              3, 2, 1, bias=False),
                                    BatchNorm2d(in_ch,
                                                momentum=BN_MOMENTUM),
                                    nn.ReLU(inplace=True)))
                        fuse_layer.append(nn.Sequential(*conv3x3s))
                else:
                    if in_ch > out_chs[i]:
                        fuse_layer.append(nn.Sequential(
                            nn.Conv2d(in_ch,
                                      out_chs[i],
                                      1,
                                      1,
                                      0,
                                      bias=False),
                            BatchNorm2d(out_chs[i], momentum=BN_MOMENTUM)))
                        # nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                    elif in_ch == out_chs[i]:
                        fuse_layer.append(None)
                    else:
                        conv3x3s = []
                        num_conv = int(math.log2(out_chs[i] // in_ch))
                        for k in range(num_conv):
                            if k == num_conv - 1:
                                conv3x3s.append(nn.Sequential(
                                    nn.Conv2d(in_ch,
                                              out_chs[i],
                                              3, 2, 1, bias=False),
                                    BatchNorm2d(out_chs[i], momentum=BN_MOMENTUM)))
                            else:
                                conv3x3s.append(nn.Sequential(
                                    nn.Conv2d(in_ch,
                                              in_ch,
                                              3, 2, 1, bias=False),
                                    BatchNorm2d(in_ch,
                                                momentum=BN_MOMENTUM),
                                    nn.ReLU(inplace=True)))
                        fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_outchannels(self):
        return self.num_outchannels

    def forward(self, x):
        in_chs = self.beta_in_channels
        in_ids = self.beta_in_id
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            in_ch = in_chs[i]
            in_id = in_ids[i]
            beta_w = self.beta_weights[i]
            y_list = []
            flag_skip = False
            for j, (ch, d, beta) in enumerate(zip(in_ch, in_id, beta_w)):
                if ch == 64:
                    y = self.fuse_layers[i][j](x[d])
                else:
                    if self.num_outchannels[i] == ch:
                        y = x[d]
                        flag_skip = True
                    elif ch > self.num_outchannels[i]:
                        size = self.size[self.channels.index(self.num_outchannels[i])]
                        y = F.interpolate(
                            self.fuse_layers[i][j](x[d]),
                            size=[size, size],
                            mode='bilinear')
                        flag_skip = False
                    else:
                        y = self.fuse_layers[i][j](x[d])
                        flag_skip = False

                y_list.append(beta * y)
            if flag_skip and len(y_list) == 1:
                x_fuse.append(sum(y_list))
            else:
                x_fuse.append(self.relu(sum(y_list)))

        for i in range(self.num_branches):
            if i == len(x):
                x.append(self.branches[i](x_fuse[i]))
            else:
                x[i] = self.branches[i](x_fuse[i])

        return x

blocks_dict = {
    'mbconv': MBConv
}


def load_betas(model_file):
    beta_weights = nn.ParameterList()

    state_dict = torch.load(model_file)
    for i, (k, v) in enumerate(state_dict.items()):
        if "beta_weights" in k:
            beta_weights.append(nn.Parameter(v))

    return beta_weights


class Shrinking_Network_HR(nn.Module):
    def __init__(self, config, mask_file, **kwargs):
        # self.inplanes = 64
        extra = config.MODEL.EXTRA
        super(Shrinking_Network_HR, self).__init__()

        self.size = config.MODEL.HEATMAP_SIZE
        beta_weights = load_betas(mask_file)
        self.beta_weights = beta_weights

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer_mbconv(MBConv, 64, 64, 4, t=1)

        self.stage2_cfg = extra['STAGE2']
        stage_id = self.stage2_cfg['STG_ID']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        retained = self.stage2_cfg['RETAINED_BLOCKS']
        num_channels = [
            num_channels[i] for i in range(len(num_channels)) if retained[i]]
        # self.transition1 = self._make_transition_layer(
        #     [64], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, stage_id)

        self.stage3_cfg = extra['STAGE3']
        stage_id = self.stage3_cfg['STG_ID']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        retained = self.stage3_cfg['RETAINED_BLOCKS']
        num_channels = [
            num_channels[i] for i in range(len(num_channels)) if retained[i]]
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels, stage_id)

        self.stage4_cfg = extra['STAGE4']
        stage_id = self.stage4_cfg['STG_ID']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        retained = self.stage4_cfg['RETAINED_BLOCKS']
        num_channels = [
            num_channels[i] for i in range(len(num_channels)) if retained[i]]
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, stage_id)

        self.stage5_cfg = extra['STAGE5']
        stage_id = self.stage5_cfg['STG_ID']
        num_channels = self.stage5_cfg['NUM_CHANNELS']
        retained = self.stage5_cfg['RETAINED_BLOCKS']
        num_channels = [
            num_channels[i] for i in range(len(num_channels)) if retained[i]]
        self.stage5, pre_stage_channels = self._make_stage(
            self.stage5_cfg, num_channels, stage_id)

        self.stage6_cfg = extra['STAGE6']
        stage_id = self.stage6_cfg['STG_ID']
        num_channels = self.stage6_cfg['NUM_CHANNELS']
        retained = self.stage6_cfg['RETAINED_BLOCKS']
        num_channels = [
            num_channels[i] for i in range(len(num_channels)) if retained[i]]
        self.stage6, pre_stage_channels = self._make_stage(
            self.stage6_cfg, num_channels, stage_id)

        self.stage7_cfg = extra['STAGE7']
        stage_id = self.stage7_cfg['STG_ID']
        num_channels = self.stage7_cfg['NUM_CHANNELS']
        retained = self.stage7_cfg['RETAINED_BLOCKS']
        num_channels = [
            num_channels[i] for i in range(len(num_channels)) if retained[i]]
        self.stage7, pre_stage_channels = self._make_stage(
            self.stage7_cfg, num_channels, stage_id)

        self.stage8_cfg = extra['STAGE8']
        stage_id = self.stage8_cfg['STG_ID']
        num_channels = self.stage8_cfg['NUM_CHANNELS']
        retained = self.stage8_cfg['RETAINED_BLOCKS']
        num_channels = [
            num_channels[i] for i in range(len(num_channels)) if retained[i]]
        self.stage8, pre_stage_channels = self._make_stage(
            self.stage8_cfg, num_channels, stage_id)

        self.stage9_cfg = extra['STAGE9']
        stage_id = self.stage9_cfg['STG_ID']
        num_channels = self.stage9_cfg['NUM_CHANNELS']
        retained = self.stage9_cfg['RETAINED_BLOCKS']
        num_channels = [
            num_channels[i] for i in range(len(num_channels)) if retained[i]]
        self.stage9, pre_stage_channels = self._make_stage(
            self.stage9_cfg, num_channels, stage_id)

        self.stage10_cfg = extra['STAGE10']
        stage_id = self.stage10_cfg['STG_ID']
        num_channels = self.stage10_cfg['NUM_CHANNELS']
        retained = self.stage10_cfg['RETAINED_BLOCKS']
        num_channels = [
            num_channels[i] for i in range(len(num_channels)) if retained[i]]
        self.stage10, pre_stage_channels = self._make_stage(
            self.stage10_cfg, num_channels, stage_id, multi_scale_output=True)

        final_inp_channels = sum(pre_stage_channels)

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=final_inp_channels,
                out_channels=final_inp_channels,
                kernel_size=1,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0),
            BatchNorm2d(final_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=final_inp_channels,
                out_channels=config.MODEL.NUM_JOINTS,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)
        )

    def _make_layer_mbconv(self, block, inplanes, planes, blocks, kernel_size=3, stride=1, t=3):
        layers = []
        layers.append(block(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=1, t=t))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=1, t=t))

        return nn.Sequential(*layers)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_outchannels, stage_id,
                    multi_scale_output=True):
        num_retained_blocks = layer_config['RETAINED_BLOCKS']
        num_modules = layer_config['NUM_MODULES']
        num_branches = sum(num_retained_blocks)
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = num_outchannels
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        beta_in_channels = layer_config['BETA_IN_CHANNELS']
        beta_in_id = layer_config['BETA_BRANCH_ID']
        blocks_id = layer_config['BLOCKS_ID']
        beta_weights_id = layer_config['BETA_WEIGHTS_ID']
        # beta_weights = layer_config['BETA_WEIGHTS']

        beta_weights = []
        for i in range(layer_config['NUM_BRANCHES']):
            if num_retained_blocks[i]:
                b_w = []
                for block_id, beta_id in zip(blocks_id[i], beta_weights_id[i]):
                    b_w.append(self.beta_weights[block_id][beta_id])
                beta_weights.append(b_w)

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     num_retained_blocks,
                                     block,
                                     num_blocks,
                                     num_outchannels,
                                     num_channels,
                                     fuse_method,
                                     beta_in_channels,
                                     stage_id,
                                     beta_in_id,
                                     beta_weights,
                                     reset_multi_scale_output)
            )
            num_outchannels = modules[-1].get_num_outchannels()

        return nn.Sequential(*modules), num_outchannels

    def forward(self, x):
        # h, w = x.size(2), x.size(3)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = [x]
        # for i in range(sum(self.stage2_cfg['RETAINED_BLOCKS'])):
        #     if self.transition1[i] is not None:
        #         x_list.append(self.transition1[i](x))
        #     else:
        #         x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(sum(self.stage2_cfg['RETAINED_BLOCKS'])):
            x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(sum(self.stage3_cfg['RETAINED_BLOCKS'])):
            x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        x_list = []
        for i in range(sum(self.stage4_cfg['RETAINED_BLOCKS'])):
            x_list.append(y_list[i])
        y_list = self.stage5(x_list)

        x_list = []
        for i in range(sum(self.stage5_cfg['RETAINED_BLOCKS'])):
            x_list.append(y_list[i])
        y_list = self.stage6(x_list)

        x_list = []
        for i in range(sum(self.stage6_cfg['RETAINED_BLOCKS'])):
            x_list.append(y_list[i])
        y_list = self.stage7(x_list)

        x_list = []
        for i in range(sum(self.stage7_cfg['RETAINED_BLOCKS'])):
            x_list.append(y_list[i])
        y_list = self.stage8(x_list)

        x_list = []
        for i in range(sum(self.stage8_cfg['RETAINED_BLOCKS'])):
            x_list.append(y_list[i])
        y_list = self.stage9(x_list)

        x_list = []
        for i in range(sum(self.stage9_cfg['RETAINED_BLOCKS'])):
            x_list.append(y_list[i])
        y_list = self.stage10(x_list)

        # Head Part
        height, width = self.size[0], self.size[1]

        fused_list = []
        for fused_block in y_list:
            x = F.interpolate(fused_block, size=(height, width), mode='bilinear', align_corners=False)
            fused_list.append(x)
        x = torch.cat(fused_list, dim=1)

        # x1 = F.interpolate(y_list[1], size=(height, width), mode='bilinear', align_corners=False)
        # x2 = F.interpolate(y_list[2], size=(height, width), mode='bilinear', align_corners=False)
        # x3 = F.interpolate(y_list[3], size=(height, width), mode='bilinear', align_corners=False)
        # x = torch.cat([y_list[0], x1, x2, x3], 1)

        x = self.head(x)

        return x
