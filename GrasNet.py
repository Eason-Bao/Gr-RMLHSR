import math
import torch
from torch import nn
from torch.autograd import Function

dtype = torch.double
device = torch.device('cpu', 1)

def calcuK(S: torch.Tensor) -> torch.Tensor:
    """
    Calculate the matrix K for the SVD backpropagation.

    Formula:
        K_{ij} = 1.0 / (S_i - S_j)
    Parameters:
        S(torch.Tensor): The singular values from SVD, with shape (b, c, h).
    Returns:
        torch.Tensor: The resulting K matrix.
    """
    # b, c, h = S.shape
    Sr = S.unsqueeze(-2)
    Sc = S.unsqueeze(-1)
    K = Sc - Sr
    K = 1.0 / K
    K[torch.isinf(K)] = 0
    K[torch.isnan(K)] = 0
    return K


class FRMap(nn.Module):
    def __init__(self, input_size, output_size):
        super(FRMap, self).__init__()
        self.weight = nn.Parameter(torch.rand(input_size, output_size, dtype=torch.float32) * 2 - 1.0)

    def forward(self, x):
        x = x.to(torch.float32)
        weight, _ = torch.linalg.qr(self.weight)
        weight = weight.transpose(-1, -2)
        weight = weight.to(torch.float32)
        output = torch.matmul(weight, x)
        return output


class QRComposition(nn.Module):
    def __init__(self):
        super(QRComposition, self).__init__()

    def forward(self, x):
        Q, R = torch.linalg.qr(x)
        # flipping
        output = torch.matmul(Q, torch.diag_embed(torch.sign(torch.sign(torch.diagonal(R, dim1=-2, dim2=-1)) + 0.5)))
        return output


class Projmap(nn.Module):
    def __init__(self):
        super(Projmap, self).__init__()

    def forward(self, x):
        return torch.matmul(x, x.transpose(-1, -2))


class Orthmap(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        return OrthmapFunction.apply(x, self.p)


class OrthmapFunction(Function):
    @staticmethod
    def forward(ctx, x, p):
        # U, S, V = torch.linalg.svd(x)
        S, U = torch.linalg.eigh(x)
        S, indices = torch.sort(S, descending=True)
        U = torch.gather(U, -1, indices.unsqueeze(-2).expand_as(U))
        ctx.save_for_backward(U, S)
        return U[..., :p]

    @staticmethod
    def backward(ctx, grad_output):
        U, S = ctx.saved_tensors
        *batch_dims, h, w = grad_output.shape
        p = h - w
        pad_zero = torch.zeros(*batch_dims, h, p)
        pad_zero = pad_zero.cuda()
        # 调整输出格式
        grad_output = grad_output.cuda()
        grad_output = torch.cat((grad_output, pad_zero), -1)
        Ut = U.transpose(-1, -2)
        K = calcuK(S)
        mid_1 = K.transpose(-1, -2) * torch.matmul(Ut, grad_output)
        mid_2 = torch.matmul(U, mid_1)
        return torch.matmul(mid_2, Ut), None


class ProjPoolLayer_A(torch.autograd.Function):
    # AProjPooling  c/n ==0
    @staticmethod
    def forward(ctx, x, n=4):
        b, c, h, w = x.shape
        ctx.save_for_backward(n)
        new_c = int(math.ceil(c / n))
        new_x = [x[:, i:i + n].mean(1) for i in range(0, c, n)]
        return torch.cat(new_x, 1).reshape(b, new_c, h, w)

    @staticmethod
    def backward(ctx, grad_output):
        n = ctx.saved_variables
        return torch.repeat_interleave(grad_output / n, n, 1)


class ProjPoolLayer(nn.Module):
    """ W-ProjPooling"""

    def __init__(self, n=4):
        super().__init__()
        self.n = n

    def forward(self, x):
        avgpool = torch.nn.AvgPool2d(int(math.sqrt(self.n)))
        return avgpool(x)

class MixedPoolLayer(nn.Module):

    def __init__(self, n=4, mix_ratio=0.5):
        """
        初始化混合池化层
        参数:
        n (int): 池化窗口的总尺寸
        mix_ratio (float): 最大池化和平均池化结果的混合比例，范围0到1，0为全平均池化，1为全最大池化
        """
        super().__init__()
        self.n = n
        self.mix_ratio = mix_ratio
        self.pool_size = int(math.sqrt(n))
        self.avgpool = nn.AvgPool2d(self.pool_size)
        self.maxpool = nn.MaxPool2d(self.pool_size)

    def forward(self, x):
        avg_pooled = self.avgpool(x)
        max_pooled = self.maxpool(x)
        # 混合最大池化和平均池化的结果
        return self.mix_ratio * max_pooled + (1 - self.mix_ratio) * avg_pooled
