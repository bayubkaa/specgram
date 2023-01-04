import torch
import torch.nn as nn
import numpy as np

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def get_new_conv(conv, filter_index):
    new_conv = \
        torch.nn.Conv2d(in_channels = conv.in_channels, \
            out_channels = conv.out_channels - 1,
            kernel_size = conv.kernel_size, \
            stride = conv.stride,
            padding = conv.padding,
            dilation = conv.dilation,
            groups = conv.groups,
            bias = (conv.bias is not None))

    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()
    
    new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
    new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]

    new_conv.weight.data = torch.from_numpy(new_weights)
    use_cuda = False
    if use_cuda:
            new_conv.weight.data = new_conv.weight.data.cuda()

    if new_conv.bias is not None:
        bias_numpy = conv.bias.data.cpu().numpy()

        bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)
        bias[:filter_index] = bias_numpy[:filter_index]
        bias[filter_index : ] = bias_numpy[filter_index + 1 :]
        new_conv.bias.data = torch.from_numpy(bias)
        if use_cuda:
            new_conv.bias.data = new_conv.bias.data.cuda()

    return new_conv

def get_new_dependant_conv(conv, filter_index, concat_weights=None):
    new_conv = \
        torch.nn.Conv2d(in_channels = conv.in_channels - 1, \
            out_channels = conv.out_channels,
            kernel_size = conv.kernel_size, \
            stride = conv.stride,
            padding = conv.padding,
            dilation = conv.dilation,
            groups = conv.groups,
            bias = (conv.bias is not None))

    old_weights = conv.weight.data.cpu().numpy() if concat_weights is None else concat_weights.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()

    new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
    new_weights[:, filter_index : , :, :] = old_weights[:, filter_index + 1 :, :, :]

    new_conv.weight.data = torch.from_numpy(new_weights)
    use_cuda = False
    if use_cuda:
            new_conv.weight.data = new_conv.weight.data.cuda()

    if new_conv.bias is not None:
        bias_numpy = conv.bias.data.cpu().numpy()

        bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)
        bias[:filter_index] = bias_numpy[:filter_index]
        bias[filter_index : ] = bias_numpy[filter_index + 1 :]
        new_conv.bias.data = torch.from_numpy(bias)
        if use_cuda:
            new_conv.bias.data = new_conv.bias.data.cuda()

    return new_conv

def get_new_bn(bn, filter_index):
    new_bn = torch.nn.BatchNorm2d(num_features=bn.num_features - 1,
                                    eps=bn.eps,
                                    momentum=bn.momentum,
                                    affine=bn.affine,
                                    track_running_stats=bn.track_running_stats)

    old_weights = bn.weight.data.cpu().numpy()
    new_weights = new_bn.weight.data.cpu().numpy()
    new_weights[:filter_index] = old_weights[:filter_index]
    new_weights[filter_index : ] = old_weights[filter_index + 1 :]
    new_bn.weight.data = torch.from_numpy(new_weights)
    return new_bn

def pick_filter_to_prune(conv, norm_ord, dim=(1,2)):
    weights = conv.weight.data
    weights = weights.reshape(weights.shape[0], weights.shape[1], weights.shape[2]*weights.shape[3])
    # print(weights.shape)
    norm = torch.norm(weights, p=norm_ord, dim=dim)
    
    filter_index = torch.argmin(norm)
    return filter_index


def pick_neuron_to_remove(linear, norm):
    old_weights = linear.weight.data
    norm = torch.norm(old_weights, p=1, dim=0)
    neuron_index = np.argmin(norm.cpu())
    return neuron_index

def get_new_dependant_nn(linear, neuron_index):
    new_nn = torch.nn.Linear(in_features=linear.in_features-1, out_features=linear.out_features)
    old_weights = linear.weight.data.cpu().numpy()

    # catatan: untuk layer terakhir saja, selebihnya harus riset lagi
    new_weights = np.delete(old_weights, [neuron_index], 1)
    new_nn.weight.data = torch.from_numpy(new_weights)
    return new_nn


# lin = torch.nn.Linear(in_features=10, out_features=2)
# neuron_index = pick_neuron_to_remove(lin, 1)
# print(neuron_index, get_new_dependant_nn(lin, neuron_index)) 



# new_weights[:filter_index] = old_weights[:filter_index]
# new_weights[filter_index : ] = old_weights[filter_index + 1 :]
# new_nn.weight.data = torch.from_numpy(new_weights)


#bn = torch.nn.BatchNorm2d(10)
#new_bn = get_new_bn(bn, 5)
#print(list(new_bn.parameters()))



