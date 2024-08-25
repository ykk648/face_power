from torch.nn import Module
import torch.nn.functional as F
import math
import numpy as np
import torch

'''
Net work's common utility
'''


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class L2Norm(Module):
    def forward(self, input):
        return F.normalize(input)


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
        # for onnx model convert
        # batch_size = np.array(input.size(0))
        # batch_size.astype(dtype=np.int32)
        # return input.view(batch_size, 512)


def Get_Conv_Size(height, width, kernel, stride, padding, rpt_num):
    conv_h = height
    conv_w = width
    for _ in range(rpt_num):
        conv_h = int((conv_h - kernel[0] + 2 * padding[0]) / stride[0] + 1)
        conv_w = int((conv_w - kernel[1] + 2 * padding[1]) / stride[1] + 1)
    return conv_h * conv_w


def Get_Conv_kernel(height, width, kernel, stride, padding, rpt_num):
    conv_h = height
    conv_w = width
    for _ in range(rpt_num):
        conv_h = math.ceil((conv_h - kernel[0] + 2 * padding[0]) / stride[0] + 1)
        conv_w = math.ceil((conv_w - kernel[1] + 2 * padding[1]) / stride[1] + 1)
        print(conv_h, conv_w)
    return (int(conv_h), int(conv_w))


def Get_Conv_kernel_floor(height, width, kernel, stride, padding, rpt_num):
    conv_h = height
    conv_w = width
    for _ in range(rpt_num):
        conv_h = math.floor((conv_h - kernel[0] + 2 * padding[0]) / stride[0] + 1)
        conv_w = math.floor((conv_w - kernel[1] + 2 * padding[1]) / stride[1] + 1)
        print(conv_h, conv_w)
        # print(conv_h, conv_w)
    return (int(conv_h), int(conv_w))


def get_dense_ave_pooling_size(height, width, block_config):
    size1 = Get_Conv_kernel(height, width, (3, 3), (2, 2), (1, 1), 1)
    size2 = Get_Conv_kernel(size1[0], size1[1], (2, 2), (2, 2), (1, 1), 1)
    # print(size1)
    size3 = Get_Conv_kernel(size2[0], size2[1], (2, 2), (2, 2), (0, 0), len(block_config) - 1)
    return size3


def get_shuffle_ave_pooling_size(height, width, using_pool=False):
    first_batch_num = 2
    if using_pool:
        first_batch_num = 3

    size1 = Get_Conv_kernel(height, width, (3, 3), (2, 2), (0, 0), first_batch_num)
    # print(size1)
    size2 = Get_Conv_kernel(size1[0], size1[1], (2, 2), (2, 2), (0, 0), 2)
    return size2


def get_ghost_dw_size(height, width):
    size1 = Get_Conv_kernel_floor(height, width, (3, 3), (2, 2), (3 // 2, 3 // 2), 3)
    size1 = Get_Conv_kernel_floor(size1[0], size1[1], (5, 5), (2, 2), (5 // 2, 5 // 2), 2)
    return size1


if __name__ == "__main__":
    # get_dense_ave_pooling_size(112,112, [1,2,3,4])
    # print("="*10)
    # get_shuffle_ave_pooling_size(112,112,True)
    # print("=" * 10)
    # get_shuffle_ave_pooling_size(112, 112, False)
    # print("=" * 10)
    Get_Conv_kernel_floor(112, 112, (3, 3), (2, 2), (1, 1), 4)
    print("=" * 10)
