import torch
from torch.nn.functional import conv1d, pad

LOSS_EPSILON = 1e-5


def esr_loss(y, y_hat):
    loss = torch.mean(torch.square(y - y_hat))
    energy = torch.mean(torch.square(y)) + LOSS_EPSILON
    return torch.div(loss, energy)


def dc_loss(y, y_hat):
    loss = torch.mean(torch.square(torch.mean(y, dim=1) - torch.mean(y_hat, dim=1)))
    energy = torch.mean(torch.square(y)) + LOSS_EPSILON
    return torch.div(loss, energy)


def pre_emphasize(y, coeffs):
    pad_len = coeffs.shape[-1] - 1
    y = pad(y, (0, 0, pad_len, 0))
    y = torch.transpose(y, 1, 2)
    y = conv1d(y, coeffs, padding=0)
    return torch.transpose(y, 1, 2)
