"""
Implement model described by Michela Testolina et al. (2021).

Decoder, whole HyperPrior network, whole ManyPriors network
cf "Towards image denoising in the latent space of learning-based compression"
"""
import math
import sys
import torch

sys.path.append("..")
from compression.models import Balle2018PT_compressor
from compression.models import abstract_model
from compression.models import GDN
from compression.models import manynets_compressor


class Testolina_Synthesis_net(torch.nn.Module):
    def __init__(
        self,
        out_channel_N=192,
        out_channel_M=320,
        out_channel_fin=3,
    ):
        super().__init__()
        self.igdn1 = GDN.GDN(out_channel_N, inverse=True)
        self.igdn2 = GDN.GDN(out_channel_N, inverse=True)
        self.igdn3 = GDN.GDN(out_channel_N, inverse=True)
        self.relu = torch.nn.ReLU()

        # added layers
        self.convred1 = torch.nn.Conv2d(
            out_channel_M, out_channel_N, 5, stride=1, padding=2
        )
        torch.nn.init.xavier_normal_(
            self.convred1.weight.data,
            (
                math.sqrt(
                    2
                    * 1
                    * (out_channel_M + out_channel_N)
                    / (out_channel_M + out_channel_M)
                )
            ),
        )
        torch.nn.init.constant_(self.convred1.bias.data, 0.01)

        self.convred2 = torch.nn.Conv2d(
            out_channel_N, out_channel_N, 5, stride=1, padding=2
        )
        torch.nn.init.xavier_normal_(self.convred2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.convred2.bias.data, 0.01)
        self.convred3 = torch.nn.Conv2d(
            out_channel_N, out_channel_N, 5, stride=1, padding=2
        )
        torch.nn.init.xavier_normal_(self.convred3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.convred3.bias.data, 0.01)
        self.convred4 = torch.nn.Conv2d(
            out_channel_N, out_channel_N, 5, stride=1, padding=2
        )
        torch.nn.init.xavier_normal_(self.convred4.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.convred4.bias.data, 0.01)
        # back to normal
        self.deconv1 = torch.nn.ConvTranspose2d(
            out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1
        )
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        # self.igdn1 = GDN.GDN(out_channel_N, inverse=True)
        self.deconv2 = torch.nn.ConvTranspose2d(
            out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1
        )
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        # self.igdn2 = GDN.GDN(out_channel_N, inverse=True)
        self.deconv3 = torch.nn.ConvTranspose2d(
            out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1
        )
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        # self.igdn3 = GDN.GDN(out_channel_N, inverse=True)
        self.deconv4 = torch.nn.ConvTranspose2d(
            out_channel_N, out_channel_fin, 5, stride=2, padding=2, output_padding=1
        )
        torch.nn.init.xavier_normal_(
            self.deconv4.weight.data,
            (math.sqrt(2 * 1 * (out_channel_N + 3) / (out_channel_N + out_channel_N))),
        )
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)

    def forward(self, x):
        x = self.relu(self.convred1(x))
        x = self.relu(self.convred2(x))
        x = self.relu(self.convred3(x))
        x = self.relu(self.convred4(x))
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.igdn3(self.deconv3(x))
        x = self.deconv4(x)
        return x


class TestolinaHP_ImageCompressor(abstract_model.AbstractHyperpriorImageCompressor):
    """Model definition; Balle2018 with extended decoder."""

    def __init__(
        self,
        out_channel_N=192,
        out_channel_M=320,
        lossf="mse",
        device="cuda:0",
        entropy_coding=False,
        **kwargs,
    ):
        super().__init__(
            out_channel_N=out_channel_N,
            out_channel_M=out_channel_M,
            lossf=lossf,
            device=device,
            entropy_coding=entropy_coding,
            **kwargs,
        )
        self.Encoder = Balle2018PT_compressor.Analysis_net(
            out_channel_N=out_channel_N, out_channel_M=out_channel_M
        )
        self.Decoder = Testolina_Synthesis_net(
            out_channel_N=out_channel_N, out_channel_M=out_channel_M
        )
        self.priorDecoder = Balle2018PT_compressor.SynthesisTFCodeExp_prior_net(
            out_channel_N=out_channel_N, out_channel_M=out_channel_M
        )
        self.priorEncoder = Balle2018PT_compressor.Analysis_prior_net(
            out_channel_N=out_channel_N, out_channel_M=out_channel_M
        )


class TestolinaManyPriors_ImageCompressor(
    manynets_compressor.Balle2017ManyPriors_ImageCompressor
):
    """ManyPriors architecture using Testolina (2021)'s decoder."""

    def __init__(
        self,
        out_channel_N=192,
        out_channel_M=320,
        lossf="mse",
        device="cuda:0",
        num_distributions=64,
        dist_patch_size=1,
        nchans_per_prior=None,
        min_feat=-127,
        max_feat=128,
        q_intv=1,
        precision=16,
        entropy_coding=False,
        model_param="2018",
        activation_function="GDN",
        passthrough_ae=False,
        encoder_cls=None,
        decoder_cls=None,
        **kwargs,
    ):
        super().__init__(
            out_channel_N=out_channel_N,
            out_channel_M=out_channel_M,
            lossf=lossf,
            device=device,
            num_distributions=num_distributions,
            dist_patch_size=dist_patch_size,
            nchans_per_prior=nchans_per_prior,
            min_feat=min_feat,
            max_feat=max_feat,
            q_intv=q_intv,
            precision=precision,
            entropy_coding=entropy_coding,
            model_param=model_param,
            activation_function=activation_function,
            passthrough_ae=passthrough_ae,
            encoder_cls=encoder_cls,
            decoder_cls=decoder_cls,
            **kwargs,
        )
        self.Decoder = Testolina_Synthesis_net(
            out_channel_N=out_channel_N, out_channel_M=out_channel_M
        )
