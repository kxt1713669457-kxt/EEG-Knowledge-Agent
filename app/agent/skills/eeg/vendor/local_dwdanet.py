# 从用户原网络提取的推理定义；不含训练入口。
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
def numberClassChannel(database_type):
    return {'A':(4,22),'B':(2,3),'C':(2,32)}[database_type]
class PatchEmbeddingCNN(nn.Module):

    def __init__(self, f1=16, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.3, number_channel=22, emb_size=40):
        super().__init__()
        f2 = D * f1
        self.cnn_module = nn.Sequential(nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding='same', bias=False), nn.BatchNorm2d(f1), nn.Conv2d(f1, f2, (number_channel, 1), (1, 1), groups=f1, padding='valid', bias=False), nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1, pooling_size1)), nn.Dropout(dropout_rate), nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False), nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1, pooling_size2)), nn.Dropout(dropout_rate))
        self.projection = nn.Sequential(Rearrange('b e (h) (w) -> b (h w) e'))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.cnn_module(x)
        x = self.projection(x)
        return x

class MultiScalePatchEmbeddingCNN(nn.Module):
    """Multi-scale Patch Embedding for EEG.

    Input:  x [B, 1, C, T]
    Output: tokens [B, N_total, E]

    Notes:
    - When kernel_sizes has multiple values, each kernel produces one token sequence (one "view").
    - We concatenate tokens from all views along the sequence dimension (N_total = sum N_view).
    """

    def __init__(self, f1=16, kernel_sizes=(32, 64, 128), D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.3, number_channel=22, emb_size=40, add_scale_embedding=True):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = (kernel_sizes,)
        elif isinstance(kernel_sizes, (list, tuple)):
            kernel_sizes = tuple(kernel_sizes)
        else:
            raise TypeError(f'kernel_sizes must be int/list/tuple, got {type(kernel_sizes)}')
        self.kernel_sizes = kernel_sizes
        self.add_scale_embedding = add_scale_embedding and len(self.kernel_sizes) > 1
        f2 = D * f1
        self.branches = nn.ModuleList()
        for k in self.kernel_sizes:
            self.branches.append(nn.Sequential(nn.Conv2d(1, f1, (1, k), (1, 1), padding='same', bias=False), nn.BatchNorm2d(f1), nn.Conv2d(f1, f2, (number_channel, 1), (1, 1), groups=f1, padding='valid', bias=False), nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1, pooling_size1)), nn.Dropout(dropout_rate), nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False), nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1, pooling_size2)), nn.Dropout(dropout_rate), nn.Conv2d(f2, emb_size, (1, 1), stride=(1, 1), bias=False), nn.BatchNorm2d(emb_size), nn.ELU()))
        if self.add_scale_embedding:
            self.scale_embedding = nn.Parameter(torch.randn(1, len(self.kernel_sizes), emb_size))
        self.projection = Rearrange('b e (h) (w) -> b (h w) e')

    def forward(self, x: Tensor) -> Tensor:
        tokens = []
        for i, branch in enumerate(self.branches):
            feat = branch(x)
            tok = self.projection(feat)
            if self.add_scale_embedding:
                tok = tok + self.scale_embedding[:, i:i + 1, :].to(tok.device)
            tokens.append(tok)
        if len(tokens) == 1:
            return tokens[0]
        return torch.cat(tokens, dim=1)

class MultiHeadAttention(nn.Module):
    """
    标准多头自注意力（MHA）：
    输入 x: [B, N, E]
    输出:   [B, N, E]
    """

    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor=None) -> Tensor:
        queries = rearrange(self.queries(x), 'b n (h d) -> b h n d', h=self.num_heads)
        keys = rearrange(self.keys(x), 'b n (h d) -> b h n d', h=self.num_heads)
        values = rearrange(self.values(x), 'b n (h d) -> b h n d', h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        return out

class FeedForwardBlock(nn.Sequential):
    """
    Transformer 的 FFN：
    Linear(E -> expansion*E) -> GELU -> Dropout -> Linear(expansion*E -> E)
    """

    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(nn.Linear(emb_size, expansion * emb_size), nn.GELU(), nn.Dropout(drop_p), nn.Linear(expansion * emb_size, emb_size))

class ClassificationHead(nn.Module):

    def __init__(self, flatten_number, n_classes, drop_p=0.5):
        super().__init__()
        self.drop = nn.Dropout(drop_p)
        self.fc = nn.LazyLinear(n_classes)

    def forward(self, x):
        x = self.drop(x)
        return self.fc(x)

class ResidualAdd(nn.Module):

    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)
        out = self.layernorm(self.drop(res) + x_input)
        return out

class TransformerEncoderBlock(nn.Sequential):
    """
    一个标准 Transformer Encoder Block：
    1) MHA + Residual
    2) FFN + Residual
    """

    def __init__(self, emb_size, num_heads=4, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__(ResidualAdd(nn.Sequential(MultiHeadAttention(emb_size, num_heads, drop_p)), emb_size, drop_p), ResidualAdd(nn.Sequential(FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p)), emb_size, drop_p))

class TransformerEncoder(nn.Sequential):
    """
    叠加 depth 个 TransformerEncoderBlock
    """

    def __init__(self, heads, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size, heads) for _ in range(depth)])

class BranchEEGNetTransformer(nn.Module):

    def __init__(self, heads=4, depth=6, emb_size=40, number_channel=22, f1=20, kernel_size=(32, 64, 128), D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.3, add_scale_embedding=True, **kwargs):
        super().__init__()
        is_multiscale = isinstance(kernel_size, (list, tuple)) and len(kernel_size) > 1
        if is_multiscale:
            self.patch = MultiScalePatchEmbeddingCNN(f1=f1, kernel_sizes=kernel_size, D=D, pooling_size1=pooling_size1, pooling_size2=pooling_size2, dropout_rate=dropout_rate, number_channel=number_channel, emb_size=emb_size, add_scale_embedding=add_scale_embedding)
        else:
            k = kernel_size[0] if isinstance(kernel_size, (list, tuple)) else kernel_size
            self.patch = PatchEmbeddingCNN(f1=f1, kernel_size=int(k), D=D, pooling_size1=pooling_size1, pooling_size2=pooling_size2, dropout_rate=dropout_rate, number_channel=number_channel, emb_size=emb_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.patch(x)

class PositioinalEncoding(nn.Module):
    """
    可学习位置编码：
    encoding: [1, length, embedding]
    输入 x: [B, N, E]
    输出: x + encoding[:, :N, :]
    """

    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))

    def forward(self, x):
        x = x + self.encoding[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)

class EEGTransformer(nn.Module):

    def __init__(self, heads=4, emb_size=40, depth=6, database_type='A', eeg1_f1=20, eeg1_kernel_size=(32, 64, 128), eeg1_D=2, eeg1_pooling_size1=8, eeg1_pooling_size2=8, eeg1_dropout_rate=0.3, eeg1_number_channel=22, flatten_eeg1=600, **kwargs):
        super().__init__()
        self.number_class, self.number_channel = numberClassChannel(database_type)
        self.emb_size = emb_size
        self.flatten_eeg1 = flatten_eeg1
        self.flatten = nn.Flatten()
        self.cnn = BranchEEGNetTransformer(heads, depth, emb_size, number_channel=self.number_channel, f1=eeg1_f1, kernel_size=eeg1_kernel_size, D=eeg1_D, pooling_size1=eeg1_pooling_size1, pooling_size2=eeg1_pooling_size2, dropout_rate=eeg1_dropout_rate)
        self.position = PositioinalEncoding(emb_size, dropout=0.1)
        self.trans = TransformerEncoder(heads, depth, emb_size)
        self.flatten = nn.Flatten()
        self.classification = ClassificationHead(self.flatten_eeg1, self.number_class)

    def forward(self, x):
        cnn = self.cnn(x)
        cnn = cnn * math.sqrt(self.emb_size)
        cnn = self.position(cnn)
        trans = self.trans(cnn)
        features = cnn + trans
        out = self.classification(self.flatten(features))
        return (features, out)
