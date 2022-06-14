import torch
import torch.nn.functional as F
from ..model.utils import bilinear_sampler, coords_grid

from perceiverpytorch.perceiver_pytorch.perceiver_pytorch import fourier_encode
from perceiverpytorch.perceiver_pytorch.perceiver_io import PerceiverIO, PreNorm, Attention

from models.position_encoding import PositionEncoder


class CorrSubsampling(torch.nn.Module) :
    def __init__(self, norm_fn='none') :
        super().__init__()
        self.norm_fn = norm_fn
        if self.norm_fn == 'batch' :
            self.norm1 = torch.nn.BatchNorm2d(16)
            self.norm2= torch.nn.BatchNorm2d(32)
            self.norm3 = torch.nn.BatchNorm2d(64)
        else :
            self.norm1 = torch.nn.Sequential()
            self.norm2 = torch.nn.Sequential()
            self.norm3 = torch.nn.Sequential()
        # TODO: padding
        self.layer1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.layer2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.layer3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)


    def forward(self, x) :
        N, d, H, W = x.shape
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        x = F.pad(x, (pad_w // 2, (pad_w+1)//2, pad_h // 2, (pad_h+1)//2))
        _, _, H, W = x.shape
        assert H % 8 == 0
        assert W % 8 == 0

        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return x


class AttentionCorr(torch.nn.Module) :
    def __init__(self, perceiver_params=None) :
        super().__init__()
        self.corr_sub = CorrSubsampling()
        # perceiver


        self.pos_bands = 32

        self.perceiver_params = dict(
            dim=64 + self.pos_bands*4+2,  # dimension of sequence to be encoded
            queries_dim=9*9 + self.pos_bands*4+2,  # dimension of decoder queries
            logits_dim=256,  # dimension of final logits
            depth=1,  # depth of net
            num_latents=8,
            # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim=128,  # latent dimension
            cross_heads=1,  # number of heads for cross attention. paper said 1
            latent_heads=4,  # number of heads for latent self attention, 8
            cross_dim_head=128,  # number of dimensions per cross attention head
            latent_dim_head=128,  # number of dimensions per latent self attention head
            weight_tie_layers=False,  # whether to weight tie layers (optional, as indicated in the diagram)
            decoder_ff=False
        )
        if perceiver_params is None:
            perceiver_params = {}

        self.perceiver_params.update(perceiver_params)

        self.perceiver = PerceiverIO(**self.perceiver_params)

        self.pos_encoder = PositionEncoder(num_bands=self.pos_bands)

    def forward(self, fmap1, fmap2) :
        N, D, H, W = fmap1.shape
        corr = AttentionCorr.corr(fmap1, fmap2)
        corr_flat = corr.reshape(-1, 1, H, W)
        corr_en = self.corr_sub(corr_flat) # has shape N*H*W, 64, H/8, W/8
        NHW, _, Hdown, Wdown = corr_en.shape

        # add position encodings
        # make position encodign at H, W resolution
        # repeat N times
        coords = coords_grid(1, Hdown, Wdown).to(fmap1.device) * 8 + 4
        # TODO: what shape is this
        coords_array = coords.reshape(-1, 2, Hdown*Wdown).permute(0, 2, 1)
        pos_en = torch.cat(self.pos_encoder.encode_positions(coords_array, res=(W, H)), dim=-1)
        pos_en = pos_en.repeat(NHW, 1, 1)

        corr_array = corr_en.reshape(NHW, -1, Hdown*Wdown).transpose(-1, -2)
        array = torch.cat([pos_en, corr_array], dim=-1)
        latents = self.perceiver(array)

        return AttentionCorrBlock(corr, latents, self)

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


# TODO is this actually a module
class AttentionCorrBlock :
    def __init__(self, corr, corr_memory, model:AttentionCorr) :
        N, H, W, D, H2, W2 = corr.shape
        self.corr = corr.reshape(N*H*W, D, H2, W2)
        self.corr_memory = corr_memory
        self.radius = 4

        self.model = model

    def __call__(self, coords) :
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        N, H, W, _ = coords.shape

        dx = torch.linspace(-r, r, 2*r+1)
        dy = torch.linspace(-r, r, 2*r+1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

        centroid_lvl = coords.reshape(N*H*W, 1, 1, 2)
        delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
        coords_lvl = centroid_lvl + delta_lvl
        # TODO: what is coords shape

        corr = bilinear_sampler(self.corr, coords_lvl)
        corr = corr.view(N, H, W, -1)

        # TODO: reshape coords

        coords_pe = torch.cat(self.model.pos_encoder.encode_positions(coords.reshape(-1, 2), (W, H)),
                              dim = -1)
        query = torch.cat([coords_pe.reshape(N*H*W, 1, -1), corr.reshape(N*H*W, 1, -1)], dim=-1)

        x = self.model.perceiver.decoder_cross_attn(query, context=self.corr_memory)
        x = self.model.perceiver.to_logits(x)
        x = x.reshape(N, H, W, -1)
        x = x.permute(0, 3, 1, 2)
        return x




