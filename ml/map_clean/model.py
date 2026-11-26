"""U-Net generator (dirty map -> clean map, 3 classes) and a PatchGAN discriminator, pix2pix style."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(inplace=True))


class UNet(nn.Module):
    """Input: one-hot dirty map (3 x H x W). Output: logits for free / occupied / unknown (3 x H x W).
    H and W must be multiples of 16. Four down-sampling stages give a receptive field of a few metres,
    enough to tell a wall (long, continuous) from clutter (small, isolated)."""

    def __init__(self, width=48, n_out=3):
        super().__init__()
        w = [width, width * 2, width * 4, width * 8, width * 8]
        self.inc = block(3, w[0])
        self.down = nn.ModuleList([block(w[i], w[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([block(w[i + 1] + w[i], w[i]) for i in reversed(range(4))])
        self.out = nn.Conv2d(w[0], n_out, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.down:
            skips.append(d(F.max_pool2d(skips[-1], 2)))
        h = skips.pop()
        for u in self.up:
            s = skips.pop()
            h = u(torch.cat([F.interpolate(h, size=s.shape[-2:], mode='nearest'), s], 1))
        return self.out(h)


class PatchD(nn.Module):
    """Judges (dirty map, clean map) pairs patch by patch: real target or generator output? 70 x 70 receptive field."""

    def __init__(self, width=64):
        super().__init__()
        sn = nn.utils.spectral_norm
        layers, c = [sn(nn.Conv2d(6, width, 4, 2, 1)), nn.LeakyReLU(0.2, True)], width
        for i, stride in enumerate((2, 2, 1)):
            layers += [sn(nn.Conv2d(c, min(c * 2, 512), 4, stride, 1)), nn.LeakyReLU(0.2, True)]
            c = min(c * 2, 512)
        layers += [nn.Conv2d(c, 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, dirty_onehot, clean_probs):
        return self.net(torch.cat([dirty_onehot, clean_probs], 1))
