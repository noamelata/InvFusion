import torch
from einops import rearrange, reduce, repeat

from motionblur import Kernel


class Degradation(object):
    def __init__(self, imshape, deg_dim=(2, 128), device=torch.device("cuda"), generator=None, gen_cpu=None, **kwargs):
        self.imshape = imshape
        b, c, h, w = imshape
        self.outsize_min, self.outsize_max = deg_dim
        try:
            self.deg_matrix = torch.randn((b, self.outsize_max, c * h * w), device=device, dtype=torch.double, generator=generator)
            self.deg_matrix = self.deg_matrix / (torch.linalg.vector_norm(self.deg_matrix, dim=-1, keepdim=True) + 1e-8)
            mask = (torch.arange(self.outsize_max, device=device).unsqueeze(0).expand(b, self.outsize_max) <
                    torch.randint(self.outsize_min, self.outsize_max+1, (b,1), device=device, generator=generator)).double()
            self.deg_matrix = self.deg_matrix * mask.unsqueeze(-1)
            self.inv_matrix = torch.pinverse(self.deg_matrix).float()
            self.deg_matrix = self.deg_matrix.float()
        except torch._C._LinAlgError as e:
            print(e)
            self.deg_matrix = torch.ones((b, self.outsize_max, c * h * w), device=device, dtype=torch.float)
            self.deg_matrix = self.deg_matrix / (torch.linalg.vector_norm(self.deg_matrix, dim=-1, keepdim=True))
            mask = (torch.arange(self.outsize_max, device=device).unsqueeze(0).expand(b, self.outsize_max) <
                    torch.randint(self.outsize_min, self.outsize_max + 1, (b, 1), device=device, generator=generator)).float()
            self.deg_matrix = self.deg_matrix * mask.unsqueeze(-1)
            self.inv_matrix = torch.pinverse(self.deg_matrix)
        self.ndim = 5

    def H(self, x):
        b, c, h, w = self.imshape
        if x.ndim == 4:
            _x = rearrange(x, "b (k c) h w -> k b (c h w)", c=c)
            self.ndim = 4
        else:
            _x = rearrange(x, "b k c h w -> k b (c h w)")
            self.ndim = 5
        return torch.einsum("k b w, b o w -> b k o", _x, self.deg_matrix.to(dtype=_x.dtype))

    def H_pinv(self, y, noise_level=0):
        b, c, h, w = self.imshape
        y = y / (1 + noise_level**2)
        _x = torch.einsum("b k o, b w o -> b k w", y, self.inv_matrix.to(dtype=y.dtype))
        if self.ndim == 4:
            return rearrange(_x, "b k (c h w) -> b (k c) h w", h=h, w=w, c=c)
        return rearrange(_x, "b k (c h w) -> b k c h w ", h=h, w=w, c=c)


class MissingBox(object):
    def __init__(self, imshape, device=torch.device("cuda"), generator=None, gen_cpu=None, **kwargs):
        self.imshape = imshape
        b, c, h, w = imshape
        box_x = torch.ones((b, w + 1), dtype=torch.float, device=device).multinomial(2, generator=generator).sort().values
        box_y = torch.ones((b, h + 1), dtype=torch.float, device=device).multinomial(2, generator=generator).sort().values
        x_range = torch.arange(w, device=device).unsqueeze(0)
        y_range = torch.arange(h, device=device).unsqueeze(0)
        mask_zero_x = (x_range >= box_x[:, 0:1]) & (x_range < box_x[:, 1:2])
        mask_zero_y = (y_range >= box_y[:, 0:1]) & (y_range < box_y[:, 1:2])
        self.deg_mask = torch.logical_not(mask_zero_y[:, :, None].float() * mask_zero_x[:, None, :])[:, None, None, :, :].float()
        self.ndim = 5

    def H(self, x):
        if x.ndim == 4:
            self.ndim = 4
            x = rearrange(x, "b (k c) h w -> b k c h w", c=self.imshape[1])
        else:
            self.ndim = 5
        return rearrange(self.deg_mask * x, "b k c h w -> b k (c h w)")

    def H_pinv(self, y, noise_level=0):
        _, c, h, w = self.imshape
        y = y / (1 + noise_level**2)
        _x = self.deg_mask * rearrange(y, "b k (c h w) -> b k c h w", c=c, h=h, w=w)
        if self.ndim == 4:
            return rearrange(_x, "b k c h w -> b (k c) h w")
        return _x


class MissingPatches(object):
    def __init__(self, imshape, patch_size=(0, None), p=(0, 0.1), device=torch.device("cuda"), generator=None, gen_cpu=None, **kwargs):
        self.imshape = imshape
        b, c, h, w = imshape
        patch_max = patch_size[1] if patch_size[1] is not None else torch.log2(torch.tensor(min(h, w))).long().item()
        self.patch_size = 2 ** torch.randint(low=patch_size[0], high=1 + patch_max, size=(1,), generator=gen_cpu).item()
        prob = torch.rand((1,), device=device, generator=generator) * (p[1] - p[0]) + p[0]
        while not torch.all(torch.any(mask := (torch.rand((b, 1, 1, h // self.patch_size, w // self.patch_size),
                                                          device=device, generator=generator) < prob
                    ).repeat_interleave(self.patch_size, dim=-2).repeat_interleave(self.patch_size, dim=-1), dim=(1, 2, 3, 4))):
            self.patch_size = 2 ** torch.randint(low=patch_size[0], high=1 + patch_max, size=(1,),
                                                 generator=gen_cpu).item()
            prob = torch.randn((1,), device=device, generator=generator) * (p[1] - p[0]) + p[0]
        self.deg_mask = mask.float()
        self.ndim = 5

    def H(self, x):
        if x.ndim == 4:
            self.ndim = 4
            x = rearrange(x, "b (k c) h w -> b k c h w", c=self.imshape[1])
        else:
            self.ndim = 5
        return rearrange(self.deg_mask * x, "b k c h w -> b k (c h w)")

    def H_pinv(self, y, noise_level=0):
        _, c, h, w = self.imshape
        y = y / (1 + noise_level**2)
        _x = self.deg_mask * rearrange(y, "b k (c h w) -> b k c h w", c=c, h=h, w=w)
        if self.ndim == 4:
            return rearrange(_x, "b k c h w -> b (k c) h w")
        return _x


class Box(object):
    def __init__(self, imshape, device=torch.device("cuda"), generator=None, gen_cpu=None, **kwargs):
        self.imshape = imshape
        b, c, h, w = imshape
        box_x = torch.ones((b, w+1), dtype=torch.float, device=device).multinomial(2, generator=generator).sort().values
        box_y = torch.ones((b, h+1), dtype=torch.float, device=device).multinomial(2, generator=generator).sort().values
        x_range = torch.arange(w, device=device).unsqueeze(0)
        y_range = torch.arange(h, device=device).unsqueeze(0)
        mask_zero_x = (x_range >= box_x[:, 0:1]) & (x_range < box_x[:, 1:2])
        mask_zero_y = (y_range >= box_y[:, 0:1]) & (y_range < box_y[:, 1:2])
        self.deg_mask = (mask_zero_y[:, :, None].float() * mask_zero_x[:, None, :])[:, None, None, :, :].float()
        self.ndim = 5

    def H(self, x):
        if x.ndim == 4:
            self.ndim = 4
            x = rearrange(x, "b (k c) h w -> b k c h w", c=self.imshape[1])
        else:
            self.ndim = 5
        return rearrange(self.deg_mask * x, "b k c h w -> b k (c h w)")

    def H_pinv(self, y, noise_level=0):
        _, c, h, w = self.imshape
        y = y / (1 + noise_level**2)
        _x = self.deg_mask * rearrange(y, "b k (c h w) -> b k c h w", c=c, h=h, w=w)
        if self.ndim == 4:
            return rearrange(_x, "b k c h w -> b (k c) h w")
        return _x


class NoDegradation(object):
    def __init__(self, imshape, device=torch.device("cuda"), **kwargs):
        self.imshape = imshape
        self.ndim = 5

    def H(self, x):
        b, c, h, w = self.imshape
        self.ndim = x.ndim
        k = x.shape[1] if x.ndim == 5 else x.shape[1] // c
        return torch.zeros((b, k, c), device=x.device, dtype=x.dtype)

    def H_pinv(self, y, noise_level=0):
        b, c, h, w = self.imshape
        k = y.shape[1]
        if self.ndim == 5:
            return torch.zeros((b, k, c, h, w), device=y.device, dtype=y.dtype) * y.reshape((b, k, c, 1, 1))
        return torch.zeros((b, k*c, h, w), device=y.device, dtype=y.dtype) * y.reshape((b, -1, 1, 1))


class SRConv(object):
    def __init__(self, imshape, stride=(3, None), mix_color=False, device=torch.device("cuda"), generator=None, gen_cpu=None, **kwargs):
        self.imshape = imshape
        b, c, h, w = self.imshape
        stride = (stride[0], stride[1] if stride[1] is not None else torch.log2(torch.tensor(min(h, w))).long().item())
        self.stride = 2 ** torch.randint(low=stride[0], high=stride[1], size=(1,), generator=gen_cpu).item()
        self.fft_size = (2*h, 2*w)
        self.outsize = 1
        self.mix_color = mix_color
        kernel_size = (min(h, 2 * self.stride), min(w, 2 * self.stride))
        kernel = torch.randn((b, 3 if mix_color else 1, kernel_size[0], kernel_size[1]),
                             device=device, dtype=torch.float,
                            generator=generator
                            )
        kernel = kernel / kernel.sum((1, 2, 3), keepdim=True)
        self.compute_fft_kernel(kernel)
        self.ndim = 5

    def compute_fft_kernel(self, kernel):
        b, c, h, w = self.imshape
        threshold = 1e-4
        kernel_size = kernel.shape[-2:]
        kernel = torch.nn.functional.pad(kernel, (0, kernel_size[0], 0, kernel_size[1]))
        kernel = kernel.roll(kernel_size[0] // 2, dims=-2).roll(kernel_size[1] // 2, dims=-1)
        self.fft_kernel = torch.fft.fft2(kernel.float(), self.fft_size)
        self.fft_kernel.imag *= -1
        threshold = threshold * (self.fft_kernel.abs().max() - self.fft_kernel.abs().min())
        self.fft_kernel = torch.where(self.fft_kernel.abs() > threshold, self.fft_kernel,
                                      torch.zeros_like(self.fft_kernel))
        self.fft_inner_kernel = self.fft_kernel.abs().pow(2)
        if self.mix_color:
            self.fft_inner_kernel = self.fft_inner_kernel.sum(1)
        self.fft_inner_kernel = self.fft_inner_kernel.reshape(
            (b, -1, self.stride, 2 * h // self.stride, self.stride, 2 * w // self.stride)).mean(-4).mean(-2)
        self.fft_inner_kernel = torch.where(self.fft_inner_kernel.abs() > threshold, 1 / self.fft_inner_kernel,
                                            torch.zeros_like(self.fft_inner_kernel))
        self.fft_transpose_kernel = self.fft_kernel.clone()
        self.fft_transpose_kernel.imag *= -1

    def H(self, x):
        _, c, h, w = self.imshape
        if x.ndim == 4:
            self.ndim = 4
            x = rearrange(x, "b (k c) h w -> b k c h w", c=self.imshape[1])
        else:
            self.ndim = 5
        fft_x = torch.fft.fft2(x.float(), self.fft_size)
        if self.mix_color:
            fft_x = (fft_x.unsqueeze(-4) * self.fft_kernel.unsqueeze(-4).unsqueeze(-5)).sum(-3) # b k 1 c h w, b 1 1 c h w -> b k 1 h w
        else:
            fft_x = fft_x * self.fft_kernel.unsqueeze(-4) # b k c h w, b 1 1 h w -> b k c h w
        fft_x = torch.view_as_real(fft_x).to(x)
        fft_x = rearrange(fft_x, "b k c (sh h) (sw w) i -> b k c sh sw h w i", sh=self.stride, sw=self.stride)
        fft_x = reduce(fft_x, "b k c sh sw h w i -> b k c h w i", "mean")
        return rearrange(fft_x, "b k c h w i -> b k (c h w i)")

    def H_pinv(self, y, noise_level=0):
        _, c, h, w = self.imshape
        _y = rearrange(y, "b k (c h w i) -> b k c h w i", c=1 if self.mix_color else c, h=2*h//self.stride, w=2*w//self.stride, i=2)

        fft_inner_kernel = self.fft_inner_kernel
        if noise_level > 0:
            fft_inner_kernel = torch.where(fft_inner_kernel.abs() > 0, 1 / (noise_level**2 + 1/fft_inner_kernel), torch.zeros_like(fft_inner_kernel))
        _y = fft_inner_kernel.unsqueeze(-4).unsqueeze(-1) * _y
        _y = repeat(_y, "b k c h w i -> b k c (rh h) (rw w) i", rh=self.stride, rw=self.stride)
        _y = torch.view_as_complex(_y.float())
        if self.mix_color:
            _y = (_y.unsqueeze(-4) * self.fft_transpose_kernel.unsqueeze(-4).unsqueeze(-5)).sum(-4) # b k 1 1 h w, b 1 1 c h w -> b k c h w
        else:
            _y = _y * self.fft_transpose_kernel.unsqueeze(-4) # b k c h w, b 1 1 h w -> b k c h w
        _x = torch.fft.ifft2(_y.contiguous(), self.fft_size).real[..., :h, :w].to(y)
        if self.ndim == 4:
            return rearrange(_x, "b k c h w -> b (k c) h w")
        return _x


class MotionBlur(SRConv):
    def __init__(self, imshape, stride=(1, 4), device=torch.device("cuda"), generator=None, gen_cpu=None, **kwargs):
        self.imshape = imshape
        b, c, h, w = self.imshape
        stride = (stride[0], stride[1] if stride[1] is not None else torch.log2(torch.tensor(min(h, w))).long().item())
        self.stride = 2 ** torch.randint(low=stride[0], high=stride[1], size=(1,), generator=gen_cpu).item()
        self.fft_size = (2*h, 2*w)
        self.mix_color = False
        kernel_size = (h//2, w//2)
        internsity = torch.rand(size=(1,), generator=gen_cpu).item()
        seed = torch.randint(0, 65000, (1,), generator = gen_cpu).item() if gen_cpu is not None else None
        motion_blur = torch.from_numpy(Kernel(size=kernel_size, intensity=internsity, seed=seed).kernelMatrix
                                  ).to(device=device, dtype=torch.float).reshape((1, 1) + kernel_size)
        Gaussian_blur = torch.stack(torch.meshgrid([torch.arange(-((kernel_size[0]-1)//2), 1+kernel_size[0]//2, device=device)]*2, indexing="ij"), -1)
        Gaussian_blur = torch.exp(-Gaussian_blur.float().pow(2).sum(-1) / (2 * self.stride**2))
        Gaussian_blur = Gaussian_blur / Gaussian_blur.sum()
        Gaussian_blur = Gaussian_blur.reshape((1, 1) + kernel_size)
        kernel = torch.nn.functional.conv2d(motion_blur, Gaussian_blur, padding="same").repeat([b, 1, 1, 1])
        kernel = kernel / kernel.sum((1, 2, 3), keepdim=True)
        self.compute_fft_kernel(kernel)
        self.ndim = 5


class RandomDegradation(object):
    def __init__(self, imshape, sync=False, device=torch.device("cuda"), generator=None, gen_cpu=None, **kwargs):
        possibilities = [Degradation, MotionBlur, MissingPatches, NoDegradation]
        random_numbers = torch.randint(0, len(possibilities), (1,), device=device, generator=generator)
        if sync:
            torch.distributed.broadcast(random_numbers, 0)
        self.degradation = (possibilities[random_numbers.item()] (imshape=imshape, device=device))

    def H(self, x):
        return self.degradation.H(x)

    def H_pinv(self, y, noise_level=0.0):
        return self.degradation.H_pinv(y, noise_level)

    @property
    def ndim(self):
        return self.degradation.ndim

    @ndim.setter
    def ndim(self, value):
        self.degradation.ndim = value

