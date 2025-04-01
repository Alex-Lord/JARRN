from torch import Tensor
import torch.nn as nn
from torch.nn import Conv2d, Sequential, Module, Parameter, SyncBatchNorm, LeakyReLU
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.cuda.amp import autocast
from torchvision.ops import StochasticDepth

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    
    We use channel_first mode for SP-Norm.
    """

    def __init__(self, normalized_shape, eps=1e-6, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, normalized_shape, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, normalized_shape, 1, 1))

    def forward(self, x):
        s, u = torch.std_mean(x, dim=1, keepdim=True)
        x = (x - u) / (s + self.eps)
        if self.affine:
            x = self.weight * x + self.bias
        return x


class NormLayer(nn.Module):
    def __init__(self, normalized_shape, norm_type):
        super(NormLayer, self).__init__()
        self.norm_type = norm_type

        if self.norm_type == 'LN':
            self.norm = LayerNorm(normalized_shape, affine=True)
        elif self.norm_type == 'BN':
            self.norm = nn.BatchNorm2d(normalized_shape, affine=True)
        elif self.norm_type == 'IN':
            self.norm = nn.InstanceNorm2d(normalized_shape, affine=True)
        elif self.norm_type == 'RZ':
            self.norm = nn.Identity()
        elif self.norm_type in ['CNX', 'CN+X', 'GRN']:
            self.norm = LayerNorm(normalized_shape, affine=False)
            # Use 1*1 conv to implement SLP in the channel dimension. 
            self.conv = nn.Conv2d(normalized_shape, normalized_shape, kernel_size=1)
        elif self.norm_type == 'NX':
            self.norm = LayerNorm(normalized_shape, affine=True)
        elif self.norm_type == 'CX':
            self.conv = nn.Conv2d(normalized_shape, normalized_shape, kernel_size=1)
        else:
            raise ValueError('norm_type error')

    def forward(self, x):
        if self.norm_type in ['LN', 'BN', 'IN', 'RZ']:
            x = self.norm(x)
        elif self.norm_type in ['CNX', 'GRN']:
            x = self.conv(self.norm(x)) * x
        elif self.norm_type == 'CN+X':
            x = self.conv(self.norm(x)) + x
        elif self.norm_type == 'NX':
            x = self.norm(x) * x
        elif self.norm_type == 'CX':
            x = self.conv(x) * x
        else:
            raise ValueError('norm_type error')
        return x


class CNBlock(nn.Module):
    def __init__(self, dim: int, norm_type: str, dp_rate: float):
        super(CNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
            NormLayer(dim, norm_type),
            nn.Conv2d(dim, 4 * dim, kernel_size=1),
            nn.ReLU(inplace=True),
            GRN(4 * dim) if norm_type == 'GRN' else nn.Identity(),
            nn.Conv2d(4 * dim, dim, kernel_size=1),
        )
        self.drop_path = StochasticDepth(dp_rate, mode='batch')
        self.norm_type = norm_type
        if self.norm_type == 'RZ':
            self.alpha = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        res = self.block(x)
        if self.norm_type == 'RZ':
            res = self.alpha * res
        x = x + self.drop_path(res)
        return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + self.eps)
        x = (1 + self.gamma * Nx) * x + self.beta
        return x

class ResNeXtBottleneck(Module):

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        rezero: bool,
    ):
        super(ResNeXtBottleneck, self).__init__()
        self.rezero = rezero
        mid_channels = int(in_channel / 2)

        if self.rezero:
            self.left = Sequential(
                LeakyReLU(0.2, inplace=True),
                Conv2d(in_channel, mid_channels, 1, 1),
                LeakyReLU(0.2, inplace=True),
                Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=32),
                
                LeakyReLU(0.2, inplace=True),
                Conv2d(mid_channels, out_channel, 1, 1),
            )
            self.alpha = Parameter(torch.tensor(0.0))
        else:
            self.left = Sequential(
                SyncBatchNorm(in_channel),
                LeakyReLU(0.2, inplace=True),
                Conv2d(in_channel, mid_channels, 1, 1),
                SyncBatchNorm(mid_channels),
                LeakyReLU(0.2, inplace=True),
                Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=32),
                SyncBatchNorm(mid_channels),
                LeakyReLU(0.2, inplace=True),
                Conv2d(mid_channels, out_channel, 1, 1),
            )
        if in_channel == out_channel:
            self.right = None
        else:
            self.right = Conv2d(in_channel, out_channel, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        factor = 1.0
        if self.rezero:
            factor = self.alpha
        x1 = factor * self.left(x) + (x
                                      if self.right is None else self.right(x))
        return x1

class BottleNeck(Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rezero: bool,
    ) -> None:
        super(BottleNeck, self).__init__()
        self.rezero = rezero
        mid_channels = int(in_channels / 4.0)
        if self.rezero:
            self.left = Sequential(
                LeakyReLU(0.2, inplace=True),
                Conv2d(in_channels, mid_channels, 1, 1),  
                LeakyReLU(0.2, inplace=True),
                Conv2d(mid_channels, mid_channels, 3, 1, 1), 
                LeakyReLU(0.2, inplace=True),
                Conv2d(mid_channels, out_channels, 1, 1),
            )
            self.alpha = Parameter(torch.tensor(0.0))
        else:
            self.left = Sequential(
                SyncBatchNorm(in_channels),
                LeakyReLU(0.2, inplace=True),
                Conv2d(in_channels, mid_channels, 1, 1),
                SyncBatchNorm(mid_channels),
                LeakyReLU(0.2, inplace=True),
                Conv2d(mid_channels, mid_channels, 3, 1, 1),
                SyncBatchNorm(mid_channels),
                LeakyReLU(0.2, inplace=True),
                Conv2d(mid_channels, out_channels, 1, 1),
            )

        if in_channels == out_channels:
            self.right = None
        else:
            self.right = Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        factor = 1.0
        if self.rezero:
            factor = self.alpha
        
        x1 = factor * self.left(x) + (x
                                      if self.right is None else self.right(x))
        return x1

class RZConv2dRL(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rezero: bool,
    ) -> None:
        super(RZConv2dRL, self).__init__()
        self.rezero = rezero
        if self.rezero:
            self.left = Sequential(
                Conv2d(in_channels, out_channels, 3, 1, 1),  # 3*3卷积
                LeakyReLU(0.2, inplace=True),
            )
            self.alpha = Parameter(torch.tensor(0.0))
        else:
            self.left = Sequential(
                Conv2d(in_channels, out_channels, 3, 1, 1),  # 3*3卷积
                SyncBatchNorm(out_channels),
                LeakyReLU(0.2, inplace=True),
            )

        if in_channels == out_channels:
            self.right = None
        else:
            self.right = Conv2d(in_channels, out_channels, 1, 1, 0)
    @autocast()
    def forward(self, x: Tensor) -> Tensor:
        factor = 1.0
        if self.rezero:
            factor = self.alpha
        x1 = factor * self.left(x) + (x
                                      if self.right is None else self.right(x))
        return x1

class CRPBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv3x3(in_channel if (i == 0) else out_channel,
                            out_channel, stride=1,
                            bias=False))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x


stages_suffixes = {0 : '_conv',
                   1 : '_conv_relu_varout_dimred'}
    
class RCUBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel, n_blocks, n_stages):
        super(RCUBlock, self).__init__()
        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, '{}{}'.format(i + 1, stages_suffixes[j]),
                        conv3x3(in_channel if (i == 0) and (j == 0) else out_channel,
                                out_channel, stride=1,
                                bias=(j == 0)))
        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
    
    def forward(self, x):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = F.relu(x)
                x = getattr(self, '{}{}'.format(i + 1, stages_suffixes[j]))(x)
            x += residual
        return x

def batchnorm(in_channel):
    "batch norm 2d"
    return nn.BatchNorm2d(in_channel, affine=True, eps=1e-5, momentum=0.1)

def conv3x3(in_channel, out_channel, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

def conv1x1(in_channel, out_channel, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride,
                     padding=0, bias=bias)

def convbnrelu(in_channel, out_channel, kernel_size, stride=1, groups=1, act=True):
    "conv-batchnorm-relu"
    if act:
        return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups, bias=False),
                             batchnorm(out_channel),
                             nn.ReLU6(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups, bias=False),
                             batchnorm(out_channel))


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1) # 沿最后一个dim划成3个qkv
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # PreNorm : (dim ,fn) fn(LayerNorm(..))
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, head_block, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 4, dim_head = 64, dropout = 0., emb_dropout = 0.):
        # dim: Last dimension of output tensor after linear transformation 
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # 生成patch
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        ) # 输出维度为b*hw*dim

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity() # 什么也不干

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes) # 投影到num_class的维度上
        # )
        self.mlp_head = nn.Identity() #暂时没有head
        self.head_block = head_block
        
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        # x.shape = b*hw*dim, hw为patch数量，b张图片，每张被分为hw个patch
        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # 去掉了cls token的嵌入
        # x += self.pos_embedding[:, :(n + 1)]
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.to_latent(x)
        return self.head_block(x), x

class ViTBlock(nn.Module):
    def __init__(self, head_block, dim, depth, heads, mlp_dim, dim_head=64, dropout=0.):
        super(ViTBlock, self).__init__()
        self.head_block = head_block
        self.dropout = nn.Dropout(dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.rearrange = rearrange
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.transformer(x)
        return self.head_block(x), x

class head1(Module):
    # 20,20,1024 -> 80,80,256
    def __init__(self, in_channel, out_channel):
        super(head1, self).__init__()
        self.tran_conv = nn.ConvTranspose2d(in_channel, out_channel, 3, stride=5, padding=10, padding_mode='zeros', dilation=2)
        self.rearrange = rearrange
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.bottleneck = Bottleneck(out_channel, out_channel // 4)
        
    def forward(self, x):
        x = self.rearrange(x, 'b (h w) c -> b c h w', h=20)
        x = self.tran_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.bottleneck(x)
    
class head2(Module):
    # 20,20,1024 -> 40,40,256
    def __init__(self, in_channel, out_channel):
        super(head2, self).__init__()
        self.tran_conv = nn.ConvTranspose2d(in_channel, out_channel, 3, stride=3, padding=11, padding_mode='zeros', dilation=2)
        self.rearrange = rearrange
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.bottleneck = Bottleneck(out_channel, out_channel // 4)

    def forward(self, x):
        x = self.rearrange(x, 'b (h w) c -> b c h w', h=20)
        
        x = self.tran_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return self.bottleneck(x)

class head3(Module):
    # 20,20,1024 -> 20,20,256
    def __init__(self, in_channel, out_channel):
        super(head3, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1)
        self.rearrange = rearrange
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.bottleneck = Bottleneck(out_channel, out_channel // 4)

    def forward(self, x):
        x = self.rearrange(x, 'b (h w) c -> b c h w', h=20)
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return self.bottleneck(x)

    
class head4(Module):
    # 20,20,1024 -> 10,10,256
    def __init__(self, in_channel, out_channel):
        super(head4, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,3 ,stride=3, padding=5)
        self.rearrange = rearrange
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.bottleneck = Bottleneck(out_channel, out_channel // 4)

    def forward(self, x):
        x = self.rearrange(x, 'b (h w) c -> b c h w', h=20)
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return self.bottleneck(x)
    

class Bottleneck(nn.Module):
    
    # in_channel -> out_channel *4
    # H_out = (H_in-1)/s +1, if s=1, H_in=H_out
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.alpha = Parameter(torch.tensor(0.0))  # 改成Rezero
        
        self.left_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x) + self.alpha * self.left_conv(x)
        return x
    
    
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

# UNet with SoftPlus activation
class UNetSP(nn.Module):
    def __init__(self, n_channels, n_classes, m=8):
        super().__init__()
        self.inc = inconv(n_channels, m*4)
        self.down1 = down(m*4, m*4)
        self.down2 = down(m*4, m*8)
        self.down3 = down(m*8, m*8)
        #self.down4 = down(128, 128)
        #self.up1 = up(256, 64)
        self.up2 = up(m*8+m*8, m*8)
        self.up3 = up(m*8+m*4, m*4)
        self.up4 = up(m*4+m*4, m*4)
        self.outc = outconv(m*4, n_classes)

    def forward(self, x):
        x1 = self.inc(x) #32
        x2 = self.down1(x1) #64
        x3 = self.down2(x2) #64
        x4 = self.down3(x3) #128
        x = self.up2(x4, x3) #128
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return F.softplus(x)


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

