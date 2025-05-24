import os
import random
import time

import numpy as np
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from training.log_utils import generate_sample_images, generate_nppc_images
from calculate_metrics import get_metrics, get_conditional_metrics
from generate_images import edm_sampler, ddrm_sampler
from torch_utils import dist, misc, training_stats
from training.precond import Precond
import dnnlib



def worker_init_fn(worker_id):
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    return 0

def learning_rate_schedule(cur_nimg, batch_size, ref_lr=1e-4, ref_batches=2 ** 5, rampup_img=2 ** 22):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_img > 0:
        lr *= min(cur_nimg / rampup_img, 1)
    return lr

def training_loop(
    dataset_kwargs      = dict(class_name='datautils.ffhq.FFHQ64', path=None),
    encoder_kwargs      = dict(class_name='training.encoders.StandardRGBEncoder'),
    data_loader_kwargs  = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=8, prefetch_factor=4),
    network_kwargs      = dict(class_name='models.hdit.HDiT'),
    loss_kwargs         = dict(class_name='training.loss.EDM2Loss'),
    optimizer_kwargs    = dict(class_name='torch.optim.Adam', betas=(0.9, 0.99)),
    lr_kwargs           = dict(func_name='training.training_loop.learning_rate_schedule'),
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    ema_kwargs          = dict(class_name='training.ema.TraditionalEMA'),

    run_dir             = '.',      # Output directory.
    seed                = 0,        # Global random seed.
    batch_size          = 128,     # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU. None = no limit.
    duration            = 8<<30,    # Train for a total of N training images.
    status              = 128<<10,  # Report status every N training images. None = disable.
    checkpoint_freq     = 128 << 20,  # Save state checkpoint every N training images. None = disable.
    sample_freq         = 128 << 20,

    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    force_finite        = True,     # Get rid of NaN/Inf gradients before feeding them to the optimizer.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),

    debug               = False,
    wandb_log           = False,
    verbose             = False,
    degradation_kwargs  = None,
    sampler_fn          = edm_sampler,
    fp16                = True,
    ref_path            = "dataset-refs/ffhq-64-test.pkl",
    consistency_loss_weight = 0,
    nppc_loss_weight     = 0,
    noise_level         = 0.0,
):
    # Initialize.
    prev_status_time = time.time()
    misc.set_random_seed(seed, dist.get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Validate batch size.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    assert duration % batch_size == 0
    assert status is None or status % batch_size == 0
    assert checkpoint_freq is None or (checkpoint_freq % batch_size == 0 and checkpoint_freq % 1024 == 0)

    # Setup dataset, encoder, and network.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    ref_image, ref_label = dataset_obj[0]
    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    ref_image = encoder.encode_latents(torch.as_tensor(ref_image).to(device).unsqueeze(0))
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=ref_image.shape[-1], img_channels=ref_image.shape[1],
                            label_dim=ref_label.shape[-1] if ref_label is not None else 0)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net = Precond(net, use_fp16=fp16, **interface_kwargs)
    net.train().requires_grad_(True).to(device)
    if isinstance(sampler_fn, str): sampler_fn = dnnlib.util.get_obj_by_name(sampler_fn)

    # Print network summary.
    if dist.get_rank() == 0:
        sample_input = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        sample_t = torch.ones([batch_gpu], device=device)
        sample_deg = dnnlib.util.construct_class_by_name(**degradation_kwargs, imshape=[batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device) if degradation_kwargs is not None else None
        sample_y = sample_deg.H(sample_input) if sample_deg is not None else None
        sample_labels = torch.zeros([batch_gpu, net.label_dim], device=device) if net.label_dim else None
        misc.print_module_summary(net, [
            sample_input, sample_t, sample_deg, sample_y, sample_labels
        ], max_nesting=4)
        del sample_input, sample_t, sample_deg, sample_y, sample_labels

    # Setup training state.
    dist.print0('Setting up training state...')
    state = dnnlib.EasyDict(cur_nimg=0, total_elapsed_time=0)
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)
    ema = dnnlib.util.construct_class_by_name(net=net, **ema_kwargs) if ema_kwargs is not None else None

    # Load previous checkpoint and decide how long to train.
    checkpoint = dist.CheckpointIO(state=state, net=net, loss_fn=loss_fn, optimizer=optimizer, ema=ema)
    checkpoint.load_latest(run_dir)
    stop_at_nimg = duration
    assert stop_at_nimg > state.cur_nimg
    dist.print0(f'Training from {state.cur_nimg // 1024} kimg to {stop_at_nimg // 1024} kimg:')
    dist.print0()

    # Main training loop.
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, start_idx=state.cur_nimg)
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    if dist.get_rank() == 0 and not debug and wandb_log:
        eval_samples = 8
        test_dataset = dnnlib.util.construct_class_by_name(**{k: v for k, v in dataset_kwargs.items() if k != "split"}, split="test")
        test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=eval_samples, shuffle=True, num_workers=1,
                                                    worker_init_fn=worker_init_fn)
    prev_status = state.cur_nimg
    cumulative_training_time = 0
    start_nimg = state.cur_nimg
    stats_jsonl = None
    while True:
        done = (state.cur_nimg >= stop_at_nimg)

        # Report status.
        if status is not None and (done or state.cur_nimg % status == 0) and (state.cur_nimg != start_nimg or start_nimg == 0):
            cur_time = time.time()
            state.total_elapsed_time += cur_time - prev_status_time
            dist.print0(' '.join(['Status:',
                'kimg',         f"{training_stats.report0('Progress/kimg',                              state.cur_nimg / 1024):<9.1f}",
                'iters',         f"{training_stats.report0('Progress/iters',                            state.cur_nimg / batch_gpu):<9.1f}",
                'time',         f"{dnnlib.util.format_time(training_stats.report0('Timing/total_sec',   state.total_elapsed_time)):<12s}",
                'sec/tick',     f"{training_stats.report0('Timing/sec_per_tick',                        cur_time - prev_status_time):<8.2f}",
                'sec/kimg',     f"{training_stats.report0('Timing/sec_per_kimg',                        cumulative_training_time / max(state.cur_nimg - prev_status, 1) * 1024):<7.3f}",
                'maintenance',  f"{training_stats.report0('Timing/maintenance_sec',                     cur_time - prev_status_time - cumulative_training_time):<7.2f}",
                                  ]))
            cumulative_training_time = 0
            prev_status = state.cur_nimg
            prev_status_time = cur_time
            torch.cuda.reset_peak_memory_stats()

            # Flush training stats.
            training_stats.default_collector.update()
            if dist.get_rank() == 0:
                if stats_jsonl is None:
                    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
                fmt = {'Progress/tick': '%.0f', 'Progress/kimg': '%.3f', 'timestamp': '%.3f'}
                items = [(name, value.mean) for name, value in training_stats.default_collector.as_dict().items()] + [('timestamp', time.time())]
                if not debug and wandb_log:
                    wandb.log({k: v for k, v in items}, step=state.cur_nimg // 1024)
                items = [f'"{name}": ' + (fmt.get(name, '%g') % value if np.isfinite(value) else 'NaN') for name, value in items]
                stats_jsonl.write('{' + ', '.join(items) + '}\n')
                stats_jsonl.flush()

                if (not debug and sample_freq is not None and (done or state.cur_nimg % sample_freq == 0)
                        and (state.cur_nimg != start_nimg or start_nimg == 0) and wandb_log):
                    with torch.no_grad():
                        if nppc_loss_weight == 0:
                            generate_sample_images(ema.ema, test_iterator, encoder, degradation_kwargs, state,
                                               sampler_fn=sampler_fn, noise_level=noise_level,
                                               device=device)
                        else:
                            generate_nppc_images(ema.ema, test_iterator, encoder, degradation_kwargs, state,
                                                   sampler_fn=sampler_fn, noise_level=noise_level,
                                                   device=device)


            # Update progress and check for abort.
            dist.update_progress(state.cur_nimg // 1024, stop_at_nimg // 1024)
            if state.cur_nimg == stop_at_nimg and state.cur_nimg < duration:
                dist.request_suspend()
            if dist.should_stop() or dist.should_suspend():
                done = True

        # Save state checkpoint.
        if checkpoint_freq is not None and not debug and (done or state.cur_nimg % checkpoint_freq == 0) and state.cur_nimg != start_nimg:
            checkpoint.save(os.path.join(run_dir, f'training-state-{state.cur_nimg//1024:07d}.pt'))
            misc.check_ddp_consistency(net)
            if not debug:
                metrics = get_metrics(net=ema.ema, encoder=encoder, ref_path=ref_path, sampler_fn=sampler_fn, verbose=False)
                if dist.get_rank() == 0 and wandb_log:
                    wandb.log({f"Metrics/{k}": v for k, v in metrics.items()}, step=state.cur_nimg // 1024)
                if degradation_kwargs is None:
                    conditional_metrics = get_conditional_metrics(net=ema.ema, encoder=encoder,
                                                                  ref_path=ref_path,
                                                                  dataset_kwargs=dnnlib.EasyDict(**{k: v for k, v in dataset_kwargs.items() if k != "split"}, split="test"),
                                                                  degradation_kwargs={"class_name": "degradation.RandomDegradation"},
                                                                  sampler_fn=ddrm_sampler, noise_level=noise_level,
                                                                  verbose=verbose)
                    if dist.get_rank() == 0 and wandb_log:
                        wandb.log({f"Metrics/Cond_{k}": v for k, v in conditional_metrics.items()}, step=state.cur_nimg // 1024)
                else:
                    conditional_metrics = get_conditional_metrics(net=ema.ema, encoder=encoder,
                                                                  ref_path=ref_path,
                                                                  dataset_kwargs=dnnlib.EasyDict(**{k: v for k, v in dataset_kwargs.items() if k != "split"}, split="test"),
                                                                  degradation_kwargs=degradation_kwargs,
                                                                  sampler_fn=sampler_fn, noise_level=noise_level,
                                                                  verbose=verbose)
                    if dist.get_rank() == 0 and wandb_log:
                        wandb.log({f"Metrics/Cond_{k}": v for k, v in conditional_metrics.items()}, step=state.cur_nimg // 1024)

        # Done?
        if done:
            break

        # Evaluate loss and accumulate gradients.
        batch_start_time = time.time()
        misc.set_random_seed(seed, dist.get_rank(), state.cur_nimg)
        optimizer.zero_grad(set_to_none=True)
        net.train()
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, class_labels = next(dataset_iterator)
                images = encoder.encode_latents(images.to(device))
                class_labels = class_labels.to(device) if class_labels is not None else None
                degradation = dnnlib.util.construct_class_by_name(**degradation_kwargs, sync=True, imshape=images.shape, device=device) if degradation_kwargs is not None else None
                loss, consistency_loss, nppc_loss, nppc_norm_loss = loss_fn(net=ddp, class_labels=class_labels, images=images, degradation=degradation, augment_pipe=augment_pipe)
                training_stats.report('Loss/loss', loss)
                loss = loss.sum()
                if degradation is not None:
                    training_stats.report('Loss/consistency_loss', consistency_loss)
                    loss = loss + consistency_loss_weight * consistency_loss.sum() if consistency_loss_weight else loss
                    if nppc_loss_weight > 0:
                        training_stats.report('Loss/nppc_loss', nppc_loss / batch_gpu_total)
                        training_stats.report('Loss/nppc_norm_loss', nppc_norm_loss / batch_gpu_total)
                        loss = (loss +
                                nppc_loss_weight * (min(state.cur_nimg / (2 ** 5 * batch_size), 1) ** 2) * nppc_loss.sum() +
                                nppc_loss_weight * 0.1 * (min(state.cur_nimg / (2 ** 6 * batch_size), 1) ** 2) * nppc_norm_loss.sum())
                loss.mul(loss_scaling / batch_gpu_total).backward()

        # Run optimizer and update weights.
        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
        training_stats.report('Loss/learning_rate', lr)
        for g in optimizer.param_groups:
            g['lr'] = lr
        if force_finite:
            for param in net.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        optimizer.step()

        # Update EMA and training state.
        state.cur_nimg += batch_size
        if ema is not None:
            training_stats.report('Loss/ema', 0.5 ** (batch_size / max(min(ema.halflife_img, state.cur_nimg / ema.rampup_ratio), 1e-8)))
            ema.update(cur_nimg=state.cur_nimg, batch_size=batch_size)
        cumulative_training_time += time.time() - batch_start_time