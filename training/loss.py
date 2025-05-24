import torch
from einops import rearrange, einsum


def gram_schmidt(x):
    b, k, c, h, w = x.shape
    vecs = rearrange(x, 'b k c h w -> b k (c h w)')
    new_vecs = torch.zeros_like(vecs[:, :0, :])
    norms = torch.zeros_like(vecs[:, :0, 0])
    for i in range(k):
        cur_vec = vecs[:, i]
        prev_vecs = new_vecs[:, :i].detach()
        cur_vec = cur_vec + 1e-8 * torch.randn_like(cur_vec)
        cur_vec = cur_vec - (einsum(cur_vec, prev_vecs.detach(), 'b d, b j d -> b j').unsqueeze(-1) * prev_vecs.detach()).sum(1)
        norm = cur_vec.norm(dim=-1, keepdim=True)
        cur_vec = cur_vec / (norm + 1e-12)
        new_vecs = torch.cat([new_vecs, cur_vec.unsqueeze(1)], dim=1)
        norms = torch.cat([norms, norm], dim=1)
    return rearrange(new_vecs, 'b k (c h w) -> b k c h w', c=c, h=h, w=w), norms


class EDM2Loss:
    def __init__(self, P_mean=-0.8, P_std=1.6, sigma_data=0.5, noise_level=0.0, label_drop=0.0):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.noise_level = noise_level
        self.label_drop = label_drop

    def __call__(self, net, images, degradation, augment_pipe=None, class_labels=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(images) * sigma
        images, aug_cond = augment_pipe(images) if augment_pipe is not None else (images, None)
        y = degradation.H(images) if degradation is not None else None
        if degradation is not None and self.noise_level > 0:
            y = y + self.noise_level * torch.randn_like(y)
        if self.label_drop > 0:
            class_labels = class_labels * (torch.rand((class_labels.shape[0], 1, 1), device=images.device, dtype=images.dtype) > self.label_drop)
        denoised, logvar = net(images + noise, sigma, degradation, y, class_labels=class_labels, aug_cond=aug_cond, return_logvar=True)
        consistency_loss = 0
        if degradation is not None and y is not None:
            consistency_loss = degradation.H_pinv(degradation.H(denoised - images)) ** 2
        loss = (weight / logvar.exp()) * ((denoised - images) ** 2) + logvar
        return loss, consistency_loss, 0, 0


class MSELoss:
    def __init__(self, sigma_data=0.5, noise_level=0.0, label_drop=0.0):
        self.sigma_data = sigma_data
        self.noise_level = noise_level
        self.label_drop = label_drop

    def __call__(self, net, images, degradation, augment_pipe=None, class_labels=None):
        sigma = 1e2 * torch.ones([images.shape[0], 1, 1, 1], device=images.device)
        noise = torch.randn_like(images) * sigma
        images, aug_cond = augment_pipe(images) if augment_pipe is not None else (images, None)
        y = degradation.H(images) if degradation is not None else None
        if degradation is not None and self.noise_level > 0:
            y = y + self.noise_level * torch.randn_like(y)
        if self.label_drop > 0:
            class_labels = class_labels * (torch.rand((class_labels.shape[0], 1, 1), device=images.device,
                                                      dtype=images.dtype) > self.label_drop)
        denoised, logvar = net(noise, sigma, degradation, y, class_labels=class_labels, aug_cond=aug_cond, return_logvar=True)
        consistency_loss = 0
        if degradation is not None and y is not None:
            consistency_loss = degradation.H_pinv(degradation.H(denoised - images)) ** 2
        loss = ((denoised - images) ** 2) + 0 * logvar
        return loss, consistency_loss, 0, 0


class NPPCLossMSE:
    def __init__(self, sigma_data=0.5):
        self.sigma_data = sigma_data

    def __call__(self, net, images, degradation, augment_pipe=None, class_labels=None):
        b, c, h, w = images.shape
        sigma = 1e2 * torch.ones([images.shape[0], 1, 1, 1], device=images.device)
        noise = torch.randn_like(images) * sigma
        images, aug_cond = augment_pipe(images) if augment_pipe is not None else (images, None)
        y = degradation.H(images) if degradation is not None else None
        denoised, logvar = net(noise, sigma, degradation, y, aug_cond=aug_cond, return_logvar=True, return_nppc=True)
        denoised, nppcs = denoised[:, :c], denoised[:, c:]
        nppcs, norms = gram_schmidt(rearrange(nppcs, 'b (k c) h w -> b k c h w', c=c))
        consistency_loss = 0
        if degradation is not None and y is not None:
            consistency_loss = degradation.H_pinv(degradation.H(denoised - images)) ** 2
        loss = ((denoised - images) ** 2) + 0 * logvar

        err = denoised.detach() - images
        err_norm = (torch.linalg.vector_norm(err, dim=(1, 2, 3), keepdim=True) + 1e-16)
        err = err / err_norm
        proj_err_sqr = einsum(err, nppcs, 'b c h w, b k c h w -> b k').pow(2)
        norms = norms / err_norm.reshape(norms.shape[0], 1)
        nppc_loss = (1-proj_err_sqr).sum(-1)
        nppc_norm_loss = (norms.pow(2) - proj_err_sqr.detach()).pow(2).sum(-1)

        return loss, consistency_loss, nppc_loss, nppc_norm_loss


