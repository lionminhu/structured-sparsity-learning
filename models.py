import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, \
                 padding=0, dilation=1, groups=1, bias=False, use_gpu=False):
        super(MaskedConv2d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride, padding,
                                           dilation, groups, bias)
        self.masked_channels = []
        self.mask_flag = False
        self.masks = None
        self.use_gpu = use_gpu

    def forward(self, x):
        if self.mask_flag == True:
            self._expand_masks(x.size())
            weight = self.weight * self.masks
            return F.conv2d(x, weight, self.bias, self.stride, self.padding,
                            self.dilation, self.groups)
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)


    def set_masked_channels(self, masked_channels):
        self.masked_channels = masked_channels
        if len(masked_channels) == 0:
            self.mask_flag = False
        else:
            self.mask_flag = True

    def get_masked_channels(self):
        return self.masked_channels

    def _expand_masks(self, input_size):
        if len(self.masked_channels) == 0:
            self.masks = None
        masks = []
        batch_size, _, height, width = [int(input_size[i].item()) for i in range(4)]
        for mask_idx in range(len(self.masked_channels)):
            channel = [b[i].item()] * width
            channel = [channel] * height
            masks.append(channel)
        masks = [masks] * batch_size
        masks = Tensor(masks)
        if self.use_gpu:
            masks = masks.cuda()
        self.masks = Variable(masks, requires_grad=False, volatile=False)


class CustomNet(nn.Module):
    def __init__(self, num_classes, use_gpu=False):
        super(CustomNet, self).__init__()
        self.conv1_1 = MaskedConv2d(3, 64, 3, padding=1, use_gpu=use_gpu)
        self.conv2_1 = MaskedConv2d(64, 128, 3, padding=1, use_gpu=use_gpu)
        self.conv3_1 = MaskedConv2d(128, 256, 3, padding=1, use_gpu=use_gpu)

        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = F.relu(self.conv1_1(x))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv2_1(out))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv3_1(out))
        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return self.softmax(out)
