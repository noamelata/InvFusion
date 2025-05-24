
import PIL
import torch
import torchvision

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from generate_images import edm_sampler, ddrm_sampler
import dnnlib

def generate_conditional_sample_images(net, images, class_labels, sampler, encoder, state, name, degradation_kwargs,
                                       noise_level, device=torch.device("cuda")):
    generator = torch.Generator(device=device).manual_seed(state.cur_nimg // 1024)
    gen_cpu = torch.Generator(device="cpu").manual_seed(state.cur_nimg // 1024)
    degradation = dnnlib.util.construct_class_by_name(**degradation_kwargs,
                            imshape=images.shape, device=device, generator=generator, gen_cpu=gen_cpu)
    y = degradation.H(images)
    if noise_level > 0:
        y = y + noise_level * torch.randn(y.shape, device=device, dtype=y.dtype, generator=generator)
    y_pinv = encoder.decode(degradation.H_pinv(y)).cpu()
    sample_images = encoder.decode(
        sampler(net=net, noise=torch.randn(images.shape, device=device, dtype=images.dtype, generator=generator),
                H=degradation, y=y, class_labels=class_labels)).cpu()
    more_sample_images = encoder.decode(
        sampler(net=net, noise=torch.randn(images.shape, device=device, dtype=images.dtype, generator=generator),
                H=degradation, y=y, class_labels=class_labels)).cpu()
    sample_grid = torchvision.utils.make_grid(torch.cat([torch.cat([encoder.decode(images).cpu(), y_pinv], dim=-1),
                                                         torch.cat([sample_images, more_sample_images], dim=-1)], -2),
                                              4).to(torch.uint8).cpu()
    wandb.log({f"Media/{name}_samples": wandb.Image(PIL.Image.fromarray(sample_grid.permute(1, 2, 0).numpy()),
                                                 caption=f"{name}_samples")}, step=state.cur_nimg // 1024)


def generate_sample_images(net, test_iterator, encoder, degradation_kwargs, state, sampler_fn=edm_sampler,
                           noise_level=0.0, device=torch.device("cuda")):
    net.eval()
    _images, class_labels = next(iter(test_iterator))
    images = encoder.encode_latents(_images.to(device))
    class_labels = class_labels.to(device) if class_labels is not None else None
    sample_images = encoder.decode(sampler_fn(net=net, noise=torch.randn_like(images))).cpu()
    sample_grid = torchvision.utils.make_grid(sample_images, 4).to(torch.uint8).cpu()
    wandb.log({"Media/samples": wandb.Image(PIL.Image.fromarray(sample_grid.permute(1, 2, 0).numpy()),
                                                caption="samples")}, step=state.cur_nimg // 1024)
    if degradation_kwargs is not None:
        sampler = sampler_fn
        generate_conditional_sample_images(net, images, class_labels, sampler, encoder, state, name="Cond",
                                           degradation_kwargs=degradation_kwargs, noise_level=noise_level,
                                           device=device)
    else:
        sampler = ddrm_sampler
        generate_conditional_sample_images(net, images, class_labels, sampler, encoder, state, name="Cond",
                                           degradation_kwargs={"class_name": "degradation.RandomDegradation"},
                                           noise_level=noise_level, device=device)
    """
    Adding a persistent degradation to for logging by adding:
    generate_conditional_sample_images(net, images, class_labels, sampler, encoder, state, name="<Degradation Name>", 
        degradation_kwargs={"class_name": "degradation.MissingPatches"}, 
        noise_level=noise_level, device=device
    )
    """

def generate_conditional_nppc_images(net, images, class_labels, sampler, encoder, state, name, degradation_kwargs,
                                       noise_level, device=torch.device("cuda")):
    generator = torch.Generator(device=device).manual_seed(state.cur_nimg // 1024)
    gen_cpu = torch.Generator(device="cpu").manual_seed(state.cur_nimg // 1024)
    degradation = dnnlib.util.construct_class_by_name(**degradation_kwargs,
                                                      imshape=images.shape, device=device, generator=generator,
                                                      gen_cpu=gen_cpu)
    y = degradation.H(images)
    if noise_level > 0:
        y = y + noise_level * torch.randn(y.shape, device=device, dtype=y.dtype, generator=generator)
    y_pinv = encoder.decode(degradation.H_pinv(y)).cpu()
    from generate_images import nppc_sampler
    sample_images = encoder.decode(
        nppc_sampler(net=net, noise=torch.randn(images.shape, device=device, dtype=images.dtype, generator=generator),
                H=degradation, y=y, class_labels=class_labels, return_images=True)).cpu()
    num_vecs = (sample_images.shape[1] - 1)//3
    sample_grid = torchvision.utils.make_grid(
        torch.cat([torch.cat([encoder.decode(images).cpu(), y_pinv], dim=-1),
        torch.cat([sample_images[:, 1], sample_images[:, 1 + num_vecs]], dim=-1)], -2),
                                              4).to(torch.uint8).cpu()
    wandb.log({f"Media/{name}_nppc": wandb.Image(PIL.Image.fromarray(sample_grid.permute(1, 2, 0).numpy()),
                                                    caption=f"{name}_nppc")}, step=state.cur_nimg // 1024)

def generate_nppc_images(net, test_iterator, encoder, degradation_kwargs, state, sampler_fn=edm_sampler,
                           noise_level=0.0, device=torch.device("cuda")):
    net.eval()
    _images, class_labels = next(iter(test_iterator))
    images = encoder.encode_latents(_images.to(device))
    if degradation_kwargs is not None:
        sampler = sampler_fn
        generate_conditional_nppc_images(net, images, class_labels, sampler, encoder, state, name="Cond",
                                           degradation_kwargs=degradation_kwargs, noise_level=noise_level,
                                           device=device)
