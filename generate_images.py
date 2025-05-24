import os
import re
import warnings
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
from einops import rearrange, einsum

import dnnlib
from torch_utils import dist, misc

warnings.filterwarnings('ignore', '`resume_download` is deprecated')


def mse(net, noise, H=None, y=None, class_labels=None, gnet=None, sigma=1e2, guidance=1,
    dtype=torch.float32, randn_like=torch.randn_like, noise_level=0.0,
):
    Dx = net(noise.to(dtype) * sigma, torch.ones((1,), dtype=dtype, device=noise.device) * sigma, H=H, y=y, class_labels=class_labels).to(dtype)
    if guidance == 1:
        return Dx
    ref_Dx = gnet(noise.to(dtype) * sigma, torch.ones((1,), dtype=dtype, device=noise.device) * sigma, H=H, y=y).to(dtype)
    return ref_Dx.lerp(Dx, guidance)

# EDM sampler from the paper
# "Elucidating the Design Space of Diffusion-Based Generative Models"
def edm_sampler(
    net, noise, H=None, y=None, class_labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like, noise_level=0.0,
):
    # Guided denoiser.
    def denoise(x, t):
        Dx = net(x, t, H=H, y=y, class_labels=class_labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t, H=H, y=y).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = noise.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        d_cur = (x_hat - denoise(x_hat, t_hat)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            d_prime = (x_next - denoise(x_next, t_next)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

def nppc_sampler(net, noise, H=None, y=None, class_labels=None, gnet=None, sigma=1e2, guidance=1,
    dtype=torch.float32, randn_like=torch.randn_like, noise_level=0.0, return_images=False,
):
    b, c, h, w = noise.shape
    denoised = net(noise.to(dtype) * sigma, torch.ones((1,), dtype=dtype, device=noise.device) * sigma, H=H, y=y, class_labels=class_labels, return_npc=True).to(dtype)
    denoised, nppcs = denoised[:, :c], denoised[:, c:]
    from training.loss import gram_schmidt
    nppcs, norms = gram_schmidt(rearrange(nppcs, 'b (k c) h w -> b k c h w', c=c))
    nppcs = norms.reshape(b, -1, 1, 1, 1) * nppcs
    images = [denoised] + [denoised + j * nppcs[:, i] for j in [-1, 1] for i in range(nppcs.shape[1])]
    images = images + [nppcs[:, i].mul(5).clamp(-1, 1) for i in range(nppcs.shape[1])]
    images = torch.stack(images, 1)
    return images

# DDRM sampler from the paper
# "Denoising Diffusion Restoration Models"
def ddrm_sampler(
    net, noise, H=None, y=None, class_labels=None, gnet=None,
    num_steps=25, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like, eta=0.85, noise_level=0.0,
):
    # Guided denoiser.
    def denoise(x, t):
        Dx = net(x, t, H=None, y=None, class_labels=class_labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t, H=None, y=None).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = H.H_pinv(y) + noise.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        x_0 = denoise(x_hat, t_hat)
        d_cur = (x_hat - x_0) / t_hat


        if t_next >= noise_level:
            x_null = x_0 + t_next * (((1 - eta ** 2) ** 0.5) * d_cur + eta * randn_like(d_cur))
            x_next = x_null + H.H_pinv(y - H.H(x_null + (t_next - noise_level) * randn_like(d_cur)))
        else:
            x_null = x_0 + t_next * (((1 - eta ** 2) ** 0.5) * d_cur + eta * randn_like(d_cur))
            y_hat = ((1 - eta ** 2) ** 0.5) * (H.H_pinv(y) - x_0) / noise_level + eta *  H.H_pinv(H.H(randn_like(d_cur)))
            x_next = x_null - H.H_pinv(H.H(x_null)) + y_hat * t_next

    return x_next

# project sampler from the paper
# "Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model"
def project_sampler(
    net, noise, H=None, y=None, class_labels=None, gnet=None,
    num_steps=100, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like, eta=0.85, noise_level=0.0,
):
    # Guided denoiser.
    def denoise(x, t):
        Dx = net(x, t, H=None, y=None, class_labels=class_labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t, H=None, y=None).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = noise.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        x_0 = denoise(x_hat, t_hat)
        d_cur = (x_hat - x_0) / t_hat
        d_cur = d_cur * ((1 - eta) ** 2) ** 0.5 + randn_like(d_cur) * eta
        if t_next >= noise_level:
            lambda_t = 1.
            gamma_t = (t_next ** 2 - noise_level ** 2).sqrt()
        else:
            lambda_t = (t_next / noise_level)
            gamma_t = 0.
        x_0 = x_0 + lambda_t * H.H_pinv(y - H.H(x_0))
        x_next = x_0 + gamma_t * d_cur

    return x_next

# PiGDM sampler from the paper
# "Pseudoinverse-Guided Diffusion Models for Inverse Problems"
def pigdm_sampler(
    net, noise, H=None, y=None, class_labels=None, gnet=None,
    num_steps=100, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like, eta=0, noise_level=0.0,
):
    # Guided denoiser.
    def denoise(x, t):
        Dx = net(x, t, H=None, y=None, class_labels=class_labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t, H=None, y=None).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = noise.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        with torch.enable_grad():
            x_hat.requires_grad = True
            x_0 = denoise(x_hat, t_hat)
            diff = H.H_pinv(y - H.H(x_0.detach()), noise_level)
            g = (diff.detach() * x_0).sum()
            grad = torch.autograd.grad(outputs=g, inputs=x_hat)[0].detach()

        d_cur = (x_hat - x_0) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur + grad

    return x_next

# DPS sampler from the paper
# "Diffusion Posterior Sampling for General Noisy Inverse Problems"
def dps_sampler(
    net, noise, H=None, y=None, class_labels=None, gnet=None,
    num_steps=1000, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like, eta=1, noise_level=0.0,
):
    # Guided denoiser.
    def denoise(x, t):
        Dx = net(x, t, H=None, y=None, class_labels=class_labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t, H=None, y=None).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = noise.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        with torch.enable_grad():
            x_hat.requires_grad = True
            x_0 = denoise(x_hat, t_hat)
            diff = torch.linalg.norm((y - H.H(x_0)))
            grad = torch.autograd.grad(outputs=diff, inputs=x_hat)[0].detach()

        d_cur = (x_hat - x_0) / t_hat
        c = eta * ((t_next * (t_hat ** 2 - t_next ** 2) ** 0.5) / t_hat)
        x_next = x_0 + (t_next ** 2 - c ** 2) ** 0.5 * d_cur + c * randn_like(d_cur) - grad * 0.5

    return x_next

# DAPS sampler from the paper
# "Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing"
def daps_sampler(
    net, noise, H=None, y=None, class_labels=None, gnet=None,
    num_steps=200, sigma_min=0.1, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like, eta=0.85, noise_level=0.0,
):
    # Guided denoiser.
    def denoise(x, t):
        Dx = net(x, t, H=None, y=None, class_labels=class_labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t, H=None, y=None).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    def sampler(xt, sigma_min, sigma_max, num_steps=5):

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

        # Main sampling loop.
        x_next = xt.to(dtype)
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next
            d_cur = (x_cur - denoise(x_cur, t_cur)) / t_cur
            x_next = x_cur + (t_next - t_cur) * d_cur

        return x_next

    def langevin(x0_hat, H, y, sigma, ratio, num_steps=100, lr=1e-4):
        with torch.enable_grad():
            p = 1
            lr = (1  + ratio * (0.01 * (1/p) - 1)) ** p * lr
            x = x0_hat.clone().detach().requires_grad_(True)
            optimizer = torch.optim.SGD([x], lr)
            for _ in range(num_steps):
                optimizer.zero_grad()
                loss = ((H.H(x) - y) ** 2).flatten(1).sum() / (2 * max(0.01, noise_level) ** 2)
                loss += ((x - x0_hat.detach()) ** 2).sum() / (2 * sigma ** 2)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    epsilon = torch.randn_like(x)
                    x.data = x.data + np.sqrt(2 * lr) * epsilon
                # early stopping with NaN
                if torch.isnan(x).any():
                    return torch.zeros_like(x)
        return x.detach()

    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    x_next = noise.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x0_hat = sampler(x_next, 0.01, t_cur)
        if i < num_steps - 1:
            x0_hat = langevin(x0_hat, H, y, t_cur, i/num_steps)
        x_next = x0_hat + t_next * randn_like(x0_hat)

    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Generate images for the given seeds in a distributed fashion.
# Returns an iterable that yields
# dnnlib.EasyDict(images, labels, noise, batch_idx, num_batches, indices, seeds)

def generate_images(
    net,                                        # Main network. Path, URL, or torch.nn.Module.
    gnet                = None,                 # Reference network for guidance. None = same as main network.
    encoder             = None,                 # Instance of training.encoders.Encoder. None = load from network pickle.
    outdir              = None,                 # Where to save the output images. None = do not save.
    subdirs             = False,                # Create subdirectory for every 1000 seeds?
    seeds               = range(16, 24),        # List of random seeds.
    class_idx           = None,                 # Class label. None = select randomly.
    max_batch_size      = 32,                   # Maximum batch size for the diffusion model.
    encoder_batch_size  = 4,                    # Maximum batch size for the encoder. None = default.
    verbose             = True,                 # Enable status prints?
    device              = torch.device('cuda'), # Which compute device to use.
    sampler_fn          = edm_sampler,          # Which sampler function to use.
    hyperparams         = "FFHQ64",             # Which sampler function to use.
    uncond              = False,
    blind               = False,
    **sampler_kwargs,                           # Additional arguments for the sampler function.
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    if encoder is None:
        encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')

    # Load main network.
    if isinstance(net, str):
        network_kwargs = dnnlib.EasyDict(class_name="models.hdit.HDiT",
                                         data=hyperparams,
                                         joint=not blind and not uncond,
                                         in_mult=1 if uncond else 2)
        img_resolution = 64 if hyperparams == "FFHQ64" or hyperparams == "ImageNet64" else 256
        label_dim = 1000 if "ImageNet" in hyperparams else 0
        interface_kwargs = dict(img_resolution=img_resolution, img_channels=3, label_dim=label_dim)
        net = load_network(net, network_kwargs, interface_kwargs, True, device)
    assert net is not None

    # Load guidance network.
    if isinstance(gnet, str):
        if verbose:
            dist.print0(f'Loading guidance network from {gnet} ...')
        with dnnlib.util.open_url(gnet, verbose=(verbose and dist.get_rank() == 0)) as f:
            gnet = pickle.load(f)['ema'].to(device)
    if gnet is None:
        gnet = net

    # Initialize encoder.
    assert encoder is not None
    if verbose:
        dist.print0(f'Setting up {type(encoder).__name__}...')
    encoder.init(device)
    if encoder_batch_size is not None and hasattr(encoder, 'batch_size'):
        encoder.batch_size = encoder_batch_size

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide seeds into batches.
    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    if verbose:
        dist.print0(f'Generating {len(seeds)} images...')

    # Return an iterable over the batches.
    class ImageIterable:
        def __len__(self):
            return len(rank_batches)

        def __iter__(self):
            # Loop over batches.
            for batch_idx, indices in enumerate(rank_batches):
                r = dnnlib.EasyDict(images=None, labels=None, noise=None, batch_idx=batch_idx, num_batches=len(rank_batches), indices=indices)
                r.seeds = [seeds[idx] for idx in indices]
                if len(r.seeds) > 0:

                    # Pick noise and labels.
                    rnd = StackedRandomGenerator(device, r.seeds)
                    r.noise = rnd.randn([len(r.seeds), net.img_channels, net.img_resolution, net.img_resolution], device=device)
                    r.labels = None
                    if net.label_dim > 0:
                        r.labels = torch.eye(net.label_dim, device=device)[
                            rnd.randint(net.label_dim, size=[len(r.seeds)], device=device)]
                        if class_idx is not None:
                            r.labels[:, :] = 0
                            r.labels[:, class_idx] = 1

                    # Generate images.
                    with torch.no_grad():
                        latents = dnnlib.util.call_func_by_name(func_name=sampler_fn, net=net, noise=r.noise,
                                                                class_labels=r.labels, gnet=gnet, randn_like=rnd.randn_like,
                                                                **sampler_kwargs)
                        r.images = encoder.decode(latents)

                    # Save images.
                    if outdir is not None:
                        for seed, image in zip(r.seeds, r.images.permute(0, 2, 3, 1).cpu().numpy()):
                            image_dir = os.path.join(outdir, f'{seed//1000*1000:06d}') if subdirs else outdir
                            os.makedirs(image_dir, exist_ok=True)
                            PIL.Image.fromarray(image, 'RGB').save(os.path.join(image_dir, f'{seed:06d}.png'))

                # Yield results.
                torch.distributed.barrier() # keep the ranks in sync
                yield r

    return ImageIterable()


def generate_conditional_images(
    net,                                        # Main network. Path, URL, or torch.nn.Module.
    gnet                = None,                 # Reference network for guidance. None = same as main network.
    encoder             = None,                 # Instance of training.encoders.Encoder. None = load from network pickle.
    outdir              = None,                 # Where to save the output images. None = do not save.
    subdirs             = False,                # Create subdirectory for every 1000 seeds?
    seeds               = range(16, 24),        # List of random seeds.
    max_batch_size      = 32,                   # Maximum batch size for the diffusion model.
    encoder_batch_size  = 4,                    # Maximum batch size for the encoder. None = default.
    verbose             = True,                 # Enable status prints?
    device              = torch.device('cuda'), # Which compute device to use.
    sampler_fn          = edm_sampler,          # Which sampler function to use.
    dataset_kwargs      = None,
    degradation_kwargs  = None,
    noise_level         = 0.0,
    uncond              = False,
    blind               = False,
    **sampler_kwargs,                           # Additional arguments for the sampler function.
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    if encoder is None:
        encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')

    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    ref_image, ref_label = dataset_obj[0]
    ref_image = encoder.encode_latents(torch.as_tensor(ref_image).to(device).unsqueeze(0))
    # Load main network.
    if isinstance(net, str):
        if verbose:
            dist.print0(f'Loading network from {net} ...')
        network_kwargs = dnnlib.EasyDict(class_name="models.hdit.HDiT",
                                         data=dataset_kwargs.class_name.split(".")[-1].replace("10K", "").replace("1K", ""),
                                         joint=not blind and not uncond,
                                         in_mult=1 if uncond else 2)
        interface_kwargs = dict(img_resolution=ref_image.shape[-1], img_channels=ref_image.shape[1],
                                label_dim=ref_label.shape[-1] if ref_label is not None else 0)
        net = load_network(net, network_kwargs, interface_kwargs, True, device)
    assert net is not None

    if gnet is None:
        gnet = net

    # Initialize encoder.
    assert encoder is not None
    if verbose:
        dist.print0(f'Setting up {type(encoder).__name__}...')
    encoder.init(device)
    if encoder_batch_size is not None and hasattr(encoder, 'batch_size'):
        encoder.batch_size = encoder_batch_size

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide seeds into batches.
    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank():: dist.get_world_size()]
    data_loader_kwargs = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=2,
                              prefetch_factor=2)
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=dataset_obj, batch_sampler=rank_batches,
                                                                **data_loader_kwargs))

    if verbose:
        dist.print0(f'Generating {len(seeds)} images...')

    # Return an iterable over the batches.
    class ImageIterable:
        def __len__(self):
            return len(rank_batches)

        def __iter__(self):
            # Loop over batches.
            for batch_idx, indices in enumerate(rank_batches):
                r = dnnlib.EasyDict(images=None, labels=None, noise=None, batch_idx=batch_idx, num_batches=len(rank_batches), indices=indices)
                r.seeds = [seeds[idx] for idx in indices]
                if len(r.seeds) > 0:
                    r.gts, r.labels = next(dataset_iterator)[:len(r.seeds)]
                    generator = torch.Generator(device=device).manual_seed(r.seeds[0])
                    cpu_generator = torch.Generator(device="cpu").manual_seed(r.seeds[0])

                    # Pick noise and labels.
                    gt = encoder.encode_latents(r.gts.to(device))
                    class_labels = r.labels.to(device) if r.labels is not None else None
                    rnd = StackedRandomGenerator(device, r.seeds)
                    r.noise = rnd.randn_like(gt)
                    degradation = dnnlib.util.construct_class_by_name(**degradation_kwargs, imshape=gt.shape,
                                         device=device, generator=generator,
                                                    gen_cpu=cpu_generator,
                                        ) if degradation_kwargs is not None else None
                    y = degradation.H(gt) if degradation is not None else None
                    if noise_level > 0:
                        y = y + noise_level * torch.randn(y.shape, device=device, dtype=y.dtype, generator=generator)
                    y_pinv = encoder.decode(degradation.H_pinv(y)).cpu() if degradation is not None else None
                    r.y = y
                    r.y_pinv = y_pinv

                    # Generate images.
                    with torch.no_grad():
                        latents = dnnlib.util.call_func_by_name(func_name=sampler_fn, net=net, gnet=gnet,
                                                                noise=r.noise, H=degradation,
                                                                y=y, class_labels=class_labels,
                                                                randn_like=rnd.randn_like, noise_level=noise_level,
                                                                **sampler_kwargs)
                        r.deg_latents = degradation.H(latents) if degradation is not None else None
                        r.projected_images = encoder.decode(latents + degradation.H_pinv(y - r.deg_latents))
                        r.images = encoder.decode(latents)

                    # Save images.
                    if outdir is not None:
                        for seed, image, gt, ys in zip(r.seeds,
                                                           r.images.permute(0, 2, 3, 1).cpu().numpy(),
                                                           r.gts.permute(0, 2, 3, 1).cpu().numpy(),
                                                           y_pinv.permute(0, 2, 3, 1).cpu().numpy()):
                            image_dir = os.path.join(outdir, f'{seed//1000*1000:06d}') if subdirs else outdir
                            os.makedirs(image_dir, exist_ok=True)
                            PIL.Image.fromarray(image, 'RGB').save(os.path.join(image_dir, f'{seed:06d}_im.png'))
                            PIL.Image.fromarray(gt, 'RGB').save(os.path.join(image_dir, f'{seed:06d}_gt.png'))
                            PIL.Image.fromarray(ys, 'RGB').save(os.path.join(image_dir, f'{seed:06d}_y.png'))

                # Yield results.
                torch.distributed.barrier() # keep the ranks in sync
                yield r

    return ImageIterable()


def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges



def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-v', '--verbose', action="store_true")
    parser.add_argument('--model-path', type=str, default="")
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--data-class', type=str, default="datautils.ffhq.ffhq64")
    parser.add_argument('--degradation', type=str, default="degradation.RandomDegradation")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-images', type=int, default=10000)
    parser.add_argument('--noise-level', type=float, default=0.0)
    parser.add_argument('--guidance', type=float, default=1.0)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--subdirs', action="store_true")
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()
    return args

def load_network(path, network_kwargs, interface_kwargs, fp16, device, verbose=True):
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    from training.precond import Precond
    net = Precond(net, use_fp16=fp16, **interface_kwargs).to(device)
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    m, u = net.load_state_dict(torch.load(path, map_location=device, weights_only=False)["ema"])
    assert len(m) == 0 and len(u) == 0
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    return net

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from training.encoders import StandardRGBEncoder
    encoder = StandardRGBEncoder()
    dataset_kwargs = dnnlib.EasyDict(class_name=args.data_class.replace("10K", "").replace("1K", ""), split="test")
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    ref_image, ref_label = dataset_obj[0]
    del dataset_obj
    ref_image = encoder.encode_latents(torch.as_tensor(ref_image).to(device).unsqueeze(0))
    interface_kwargs = dict(img_resolution=ref_image.shape[-1], img_channels=ref_image.shape[1],
                            label_dim=ref_label.shape[-1] if ref_label is not None else 0)
    network_kwargs = dnnlib.EasyDict(class_name="models.hdit.HDiT",
                                     data=args.data_class.split(".")[-1].replace("10K", "").replace("1K", ""))

    model = load_network(args.model_path, {**network_kwargs, "joint": True, "in_mult": 2},
                                    interface_kwargs, True, device)
    opts = dnnlib.EasyDict(encoder=encoder,
                           device=device,
                           verbose=args.verbose,
                           subdirs=args.subdirs,
                           noise_level=args.noise_level,
                           guidance=args.guidance,
                           outdir=args.outdir,
                           dataset_kwargs=dataset_kwargs,
                           degradation_kwargs=dnnlib.EasyDict(class_name=args.degradation),
                           )
    image_iter = generate_conditional_images(net=model, max_batch_size=args.batch_size,
                                                             seeds=range(args.seed, args.seed + args.num_images),
                                                             **opts)
    for _ in image_iter:
        pass


def cmdline():
    args = parse_args()
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    run(args)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
