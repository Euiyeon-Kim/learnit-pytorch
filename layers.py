import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    def __init__(self, in_f, out_f, bias=True):
        super(Linear, self).__init__(in_f, out_f, bias=bias)

    def forward(self, x, params=None):
        if params is None:
            x = super(Linear, self).forward(x)
        else:
            weight, bias = params.get('weight'), params.get('bias')
            if weight is None:
                weight = self.weight
            if bias is None:
                bias = self.bias
            x = F.linear(x, weight, bias)
        return x
