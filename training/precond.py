import numpy as np
import torch



class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)



class Precond(torch.nn.Module):
    def __init__(self,
        model,
        img_resolution,         # Image resolution.
        img_channels,           # Image channels.
        label_dim       = 0,    # Class label dimensionality. 0 = unconditional.
        use_fp16        = True, # Run the model at FP16 precision?
        sigma_data      = 0.5,  # Expected standard deviation of the training data.
        logvar_channels = 128,  # Intermediate dimensionality for uncertainty estimation.
        **unet_kwargs,          # Keyword arguments for UNet.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.unet = model
        self.logvar_fourier = MPFourier(logvar_channels)
        self.logvar_linear = torch.nn.Linear(logvar_channels, 1)

    def forward(self, x, sigma, H, y, class_labels=None, force_fp32=False, return_logvar=False, return_nppc=False, **unet_kwargs):
        x = x.to(torch.float32)
        y = y.to(torch.float32) if y is not None else y
        NPPC = None
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if (
                class_labels is None) else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        # Preconditioning weights.
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = (sigma.flatten().log() / 4).to(dtype)

        # Run the model.
        x_in = (c_in * x).to(dtype)
        y_in = y.to(dtype) if y is not None else y
        with torch.autocast(device_type=x.device.type, dtype=dtype, enabled=self.use_fp16):
            F_x = self.unet(x_in, c_noise, H, y_in, class_cond=class_labels, **unet_kwargs)

        # For NPPC training, split the output into two parts.
        if F_x.shape[1] > self.img_channels:
            NPPC = F_x[:, self.img_channels:]
            F_x = F_x[:, :self.img_channels]
        # Use post-conditioning
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        if return_nppc and NPPC is not None:
            D_x = torch.cat([D_x, NPPC.to(torch.float32)], dim=1)

        # Estimate uncertainty if requested.
        if return_logvar:
            logvar = self.logvar_linear(self.logvar_fourier(c_noise.to(torch.float32))).reshape(-1, 1, 1, 1)
            return D_x, logvar
        return D_x

#----------------------------------------------------------------------------
