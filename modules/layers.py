import torch
from torch import nn
import torch.nn.functional as F

class Mask(nn.Module):
    """A mask layer whose output matrix == 1 if not padded else 0
    """
    def __init__(self, pad=0):
        super(Mask, self).__init__()
        self.pad = pad

    def forward(self, inputs):
        return (inputs!=self.pad).float()


class Flatten(nn.Module):
    """Flatten all dimension except for first(batch) dimension
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, inputs):
        batch_size = inputs.size(0)
        return inputs.view(batch_size, -1)


class PackingRNN(nn.Module):
    """use packing and unpacking functions to avoid applying rnn to mask part
    Only used when the 'c' memory is not used in next step
    Args:
        rnn_instance: a LSTM or GRU instance
    """

    def __init__(self, rnn_instance):
        super(PackingRNN, self).__init__()
        self.rnn_instance = rnn_instance

    def forward(self, inputs, mask=None):
        if mask is not None:
            sorted_inputs, sorted_length, backpointer = self.sort_by_length(
                inputs, mask
            )
            sorted_inputs = nn.utils.rnn.pack_padded_sequence(
                sorted_inputs,
                sorted_length,
                batch_first=True
            )  # pack seqs for rnn
            out, _ = self.rnn_instance(sorted_inputs, None)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True
            )  # pad back for next step
            original_order_out = out.index_select(index=backpointer,dim=0)
        else: # do nothing
            original_order_out, _ = self.rnn_instance(inputs)
        return original_order_out

    def sort_by_length(self, inputs, mask):
        original_length = mask.long().sum(dim=1)
        sorted_length, sorted_ind = original_length.sort(descending=True)
        sorted_inputs = inputs.index_select(index=sorted_ind, dim=0)
        _, backpointer = sorted_ind.sort(descending=False)
        return sorted_inputs, sorted_length, backpointer


class SamePaddingConv1d(nn.Module):
    """Apply Conv1d to sequence with seq_len unchanged
    """

    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1, bias=True):
        super(SamePaddingConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.bias = bias

        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=dilation,
            bias=bias
        )
        pad_plus = True if (kernel_size - 1) * dilation % 2 else False
        pad_one_side = (kernel_size - 1) * dilation // 2
        if pad_plus:
            self.pad = (pad_one_side, pad_one_side + 1)
        else:
            self.pad = (pad_one_side, pad_one_side)

    def forward(self, inputs):
        return self.conv1d(F.pad(inputs, self.pad))


class LocalInferenceLayer(nn.Module):
    """Local inference modeling part of ESIM
    """

    def __init__(self):
        super(LocalInferenceLayer, self).__init__()

    def forward(self, seq_1, seq_2):
        # seq_1.shape = batch_size, seq_1_len, feature_dim
        # seq_2.shape = batch_size, seq_2_len, feature_dim

        # batch_size, seq_1_len, seq_2_len
        e_ij = torch.matmul(seq_1, seq_2.permute(0, 2, 1))

        # weighted for inference
        weighted_seq2 = F.softmax(e_ij, dim=2)
        weighted_seq1 = F.softmax(e_ij, dim=1)

        # inference
        seq_1_hat = torch.matmul(weighted_seq2, seq_2)  # same shape as seq_1
        seq_2_hat = torch.matmul(weighted_seq1.permute(0, 2, 1), seq_1)

        return seq_1_hat, seq_2_hat


class MultiKernelConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[2,3,4,5], dilation=1, bias=True):
        super(MultiKernelConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.bias = bias

        if isinstance(kernel_size, list):
            self.channels_per_kernel = out_channels // len(kernel_size)
        else:
            self.channels_per_kernel = out_channels
            self.kernel_size = [kernel_size]
        self.conv_kernels = nn.ModuleList([
            SamePaddingConv1d(
                in_channels,
                self.channels_per_kernel,
                kernel_size=k,
                dilation=dilation,
                bias=bias
            ) for k in self.kernel_size
        ])

    def forward(self, inputs, mask=None):
        if mask is not None:
            inputs = inputs * mask
        conv_outputs = []
        for conv_kernel in self.conv_kernels:
            conv_part = conv_kernel(inputs.permute(0,2,1)).permute(0,2,1)
            conv_outputs.append(conv_part)
        conv_outputs = torch.cat(conv_outputs, dim=-1)
        if mask is not None:
            return conv_outputs * mask
        return conv_outputs
