import torch
from subsampling import Conv2dSubsampling, VggSubsampling


def test_conv2d_subsampling():
    B, idim, odim = 1, 80, 512
    model = Conv2dSubsampling(idim, odim)
    for t in range(4, 50):
        x = torch.randn(B, t, idim)
        outputs = model(x)
        assert outputs.shape == (B, t // 4, odim)


def test_vgg_subsampling():
    B, idim, odim = 1, 80, 512
    model = VggSubsampling(idim, odim)
    for t in range(4, 50):
        x = torch.randn(B, t, idim)
        outputs = model(x)
        assert outputs.shape == (B, t // 4, odim)


if __name__ == "__main__":
    test_conv2d_subsampling()
    test_vgg_subsampling()
