# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Alias-Free Generative Adversarial Networks"."""

import os
import click
import re
import json
import tempfile
import torch
import legacy
import math

import dnnlib
from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops
from torch_utils import misc

from data_configs import DATASETS

#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)
    
    # additions
    num_nodes = math.ceil(int(os.environ['WORLD_SIZE']) / 4)
    local_rank = rank
    global_rank = int(local_rank + int(os.environ['SLURM_PROCID']) * (c.num_gpus//num_nodes))
    local_gpus = c.num_gpus//num_nodes
    global_gpus = int(os.environ['WORLD_SIZE'])
    
    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            # changes
            # init_method = f'file://{init_file}'
            init_method = 'env://'
            # torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=global_rank, world_size=global_gpus)

    # Init torch_utils.
    # changes
    sync_device = torch.device('cuda', local_rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=global_rank, sync_device=sync_device)
    if local_rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, local_rank=local_rank, global_rank=global_rank, local_gpus=local_gpus, global_gpus=global_gpus, **c)

#----------------------------------------------------------------------------
def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]

    matching_dirs = [re.fullmatch(desc, x) for x in prev_run_dirs if re.fullmatch(desc, x) is not None]
    if c.restart_every > 0 and len(matching_dirs) > 0:  # expect unique desc, continue in this directory
        assert len(matching_dirs) == 1, f'Multiple directories found for resuming: {matching_dirs}'
        c.run_dir = os.path.join(outdir, matching_dirs[0].group())
    else:                     # fallback to standard
        c.run_dir = os.path.join(outdir, desc)
        assert not os.path.exists(c.run_dir)
    
    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir, exist_ok=c.restart_every > 0)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt+') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    # changes
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    num_nodes = math.ceil(int(os.environ['WORLD_SIZE']) / 4)
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus//num_nodes)

#----------------------------------------------------------------------------

def init_dataset_kwargs(dataset_name, resolution):
    try:
        #if 'imagenet' in data:
        #    dataset_kwargs = dnnlib.EasyDict(class_name="training.dataset.ImageFolderDatasetWithPreprocessing", path=data, resolution=resolution, use_labels=True, max_size=None, xflip=False)
        #else:
        #    dataset_kwargs = dnnlib.EasyDict(class_name="training.dataset.ImageFolderDataset", path=data, use_labels=True, max_size=None, xflip=False)
        dataset_kwargs = dnnlib.EasyDict(**DATASETS[dataset_name], resolution=resolution, use_labels=False, max_size=None, xflip=False)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--cfg',          help='Base configuration',                                      type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2', 'fastgan']), required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)

# Optional features.
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                 type=click.IntRange(min=0), default=0, show_default=True)

# Misc hyperparameters.
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1))
@click.option('--cbase',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax',         help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0))
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1))

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=0), default=3, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)

# StyleGAN-XL additions
@click.option('--restart_every',help='Time interval in seconds to restart code', metavar='INT', type=int, default=999999999, show_default=True)
@click.option('--stem',         help='Train the stem.', is_flag=True)
@click.option('--syn_layers',   help='Number of layers in the stem', type=click.IntRange(min=1), default=14, show_default=True)
@click.option('--superres',     help='Train superresolution stage. You have to provide the path to a pretrained stem.', is_flag=True)
@click.option('--path_stem',    help='Path to pretrained stem',  type=str)
@click.option('--head_layers',  help='Layers of added superresolution head.', type=click.IntRange(min=1), default=7, show_default=True)
@click.option('--cls_weight',   help='class guidance weight', type=float, default=0.0, show_default=True)
@click.option('--up_factor',    help='Up sampling factor of superres head', type=click.IntRange(min=2), default=2, show_default=True)
@click.option('--resolution',    help='Image resolution', type=click.IntRange(min=1))
@click.option('--dataset_name',    help='Dataset name', type=str)

def main(**kwargs):
    # Initialize config.
    opts = dnnlib.EasyDict(kwargs)  # Command line arguments
    c = dnnlib.EasyDict()  # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0, 0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0, 0.99], eps=1e-8)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)
    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(dataset_name=opts.dataset_name, resolution=opts.resolution)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror

    if opts.dataset_name is None:
        opts.dataset_name = dataset_name

    # Hyperparameters & settings.
    c.num_gpus = int(os.environ['WORLD_SIZE']) # global number of gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // int(os.environ['WORLD_SIZE'])
    c.G_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = opts.cmax
    c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    if opts.cfg == 'stylegan2':
        c.G_kwargs.class_name = 'training.networks_stylegan2.Generator'
        c.G_reg_interval = 4  # Enable lazy regularization for G.
        c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.

    elif opts.cfg == 'fastgan':
        c.G_kwargs = dnnlib.EasyDict(class_name='training.networks_fastgan.Generator',
                                     cond=opts.cond, mapping_kwargs=dnnlib.EasyDict(),
                                     synthesis_kwargs=dnnlib.EasyDict())
        c.G_kwargs.synthesis_kwargs.lite = True
        c.G_opt_kwargs.lr = c.D_opt_kwargs.lr = 0.0002
        c.G_opt_kwargs.lr = 0.002

    else:
        c.G_kwargs.class_name = 'training.networks_stylegan3_resetting.Generator'
        c.G_kwargs.magnitude_ema_beta = 0.5 ** (c.batch_size / (20 * 1e3))
        c.G_kwargs.channel_base *= 2  # increase for StyleGAN-XL
        c.G_kwargs.channel_max *= 2   # increase for StyleGAN-XL
        c.G_kwargs.conv_kernel = 1 if opts.cfg == 'stylegan3-r' else 3
        c.G_kwargs.use_radial_filters = True if opts.cfg == 'stylegan3-r' else False

        if opts.cfg == 'stylegan3-r':
            c.G_kwargs.channel_base *= 2
            c.G_kwargs.channel_max *= 2

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ada_kimg = 100  # Make ADA react faster at the beginning.
        c.ema_rampup = None  # Disable EMA rampup.

    # Restart.
    c.restart_every = opts.restart_every

    # Performance-related toggles.
    if opts.fp32:
        c.G_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{opts.cfg:s}-{opts.dataset_name:s}{opts.resolution:d}-gpus{c.num_gpus:d}-batch{c.batch_size:d}'
    if opts.desc is not None:
        desc=opts.desc
        #desc += f'-{opts.desc}'

    ##################################
    ########## StyleGAN-XL ###########
    ##################################

    # Generator
    c.G_kwargs.w_dim = 512
    c.G_kwargs.z_dim = 64
    c.G_kwargs.mapping_kwargs.rand_embedding = False
    c.G_kwargs.num_layers = opts.syn_layers
    c.G_kwargs.mapping_kwargs.num_layers = 2

    # Discriminator
    c.D_kwargs = dnnlib.EasyDict(
        class_name='pg_modules.discriminator.ProjectedDiscriminator',
        backbones=['deit_base_distilled_patch16_224', 'tf_efficientnet_lite0'],
        diffaug=True,
        interp224=(c.training_set_kwargs.resolution < 224),
        backbone_kwargs=dnnlib.EasyDict(),
    )
    c.D_kwargs.backbone_kwargs.cout = 64
    c.D_kwargs.backbone_kwargs.expand = True
    c.D_kwargs.backbone_kwargs.proj_type = 2 if c.training_set_kwargs.resolution <= 16 else 2  # CCM only works better on very low resolutions
    c.D_kwargs.backbone_kwargs.num_discs = 4
    c.D_kwargs.backbone_kwargs.cond = opts.cond

    # Loss
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.ProjectedGANLoss')
    c.loss_kwargs.blur_init_sigma = 2  # Blur the images seen by the discriminator.
    c.loss_kwargs.blur_fade_kimg = 300
    c.loss_kwargs.pl_weight = 2.0
    c.loss_kwargs.pl_no_weight_grad = True
    c.loss_kwargs.style_mixing_prob = 0.0
    c.loss_kwargs.cls_weight = 0.0  # use classifier guidance only for superresolution training (i.e., with pretrained stem)
    c.loss_kwargs.cls_model = 'deit_small_distilled_patch16_224'
    c.loss_kwargs.train_head_only = False

    if opts.superres:
        assert opts.path_stem is not None, "When training superres head, provide path to stem"

        # Generator
        if opts.cfg == 'stylegan2':
            c.G_kwargs = dnnlib.EasyDict(
                class_name='training.networks_stylegan2_resetting.SuperresGenerator',
                path_stem=opts.path_stem,
                up_factor=opts.up_factor,
                channel_base=opts.cbase,
                channel_max=opts.cmax
            )
        else:
            c.G_kwargs = dnnlib.EasyDict(
                class_name='training.networks_stylegan3_resetting.SuperresGenerator',
                path_stem=opts.path_stem,
                head_layers=opts.head_layers,
                up_factor=opts.up_factor,
            )

        
        # Loss
        c.loss_kwargs.pl_weight = 0.0
        c.loss_kwargs.cls_weight = opts.cls_weight if opts.cond else 0
        c.loss_kwargs.train_head_only = True

    ##################################
    ##################################
    ##################################

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)

    # Check for restart
    last_snapshot = misc.get_ckpt_path(c.run_dir)
    if os.path.isfile(last_snapshot):
        # get current number of training images
        with dnnlib.util.open_url(last_snapshot) as f:
            cur_nimg = legacy.load_network_pkl(f)['progress']['cur_nimg'].item()
        if (cur_nimg//1000) < c.total_kimg:
            print('Restart: exit with code 3')
            exit(3)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
