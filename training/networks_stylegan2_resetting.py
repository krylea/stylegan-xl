

from training.networks_stylegan2 import *
import dnnlib
import legacy

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 4,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_fp16_res = num_fp16_res
        self.channel_base = channel_base
        self.channel_max = channel_max
        self.block_kwargs = block_kwargs
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])


@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        if 'rand_embedding' in mapping_kwargs:
            del mapping_kwargs['rand_embedding']
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        return img
    



@persistence.persistent_class
class SuperresGenerator(torch.nn.Module):
    def __init__(self,
        img_resolution,
        path_stem,
        up_factor,
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        **synthesis_kwargs,
    ):
        assert up_factor in [2, 4, 8, 16], "Supported up_factors: [2, 4, 8, 16]"

        super().__init__()
        self.path_stem = path_stem
        self.up_factor = up_factor

        # load pretrained stem
        with dnnlib.util.open_url(path_stem) as f:
            G_stem = legacy.load_network_pkl(f)['G_ema']
        self.mapping = G_stem.mapping
        self.synthesis = G_stem.synthesis

        self.z_dim = G_stem.z_dim
        self.c_dim = G_stem.c_dim
        self.w_dim = G_stem.w_dim
        self.img_channels = G_stem.img_channels

        channel_base = getattr(self.synthesis, 'channel_base', channel_base)    
        channel_max = getattr(self.synthesis, 'channel_max', channel_max)

        head_layers = int(np.log2(up_factor))

        self.img_resolution = G_stem.img_resolution * up_factor
        self.synthesis.img_resolution = self.img_resolution
        assert img_resolution == self.img_resolution, f"Resolution mismatch. Dataset: {img_resolution}, G output: {self.img_resolution}"
        self.synthesis.img_resolution_log2 = self.synthesis.img_resolution_log2 + head_layers
        assert self.synthesis.img_resolution_log2 == int(np.log2(self.synthesis.img_resolution)), f"Log resolution mismatch."

        fp16_resolution = max(2 ** (self.synthesis.img_resolution_log2 + 1 - self.synthesis.num_fp16_res), 8)

        for res in self.synthesis.block_resolutions[-4:]:
            block = getattr(self.synthesis, f'b{res}')
            block.use_fp16 = (res >= fp16_resolution)
            if block.is_last:
                self.synthesis.num_ws -= block.num_torgb
                block.is_last = False

        self.synthesis.block_resolutions = [2 ** i for i in range(2, self.synthesis.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.synthesis.block_resolutions}

        for res in self.synthesis.block_resolutions[-head_layers:]:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=self.w_dim, resolution=res,
                img_channels=self.synthesis.img_channels, is_last=is_last, use_fp16=use_fp16, **synthesis_kwargs)
            self.synthesis.num_ws += block.num_conv
            if is_last:
                self.synthesis.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def reinit_stem(self):
        print("Reinitialize stem")
        with dnnlib.util.open_url(self.path_stem) as f:
            G_stem = legacy.load_network_pkl(f)['G_ema']

        # synthesis reinit
        for res in G_stem.synthesis.block_resolutions:
            layer_src = getattr(G_stem.synthesis, f'b{res}')
            layer_dst = getattr(self.synthesis, f'b{res}')
            misc.copy_params_and_buffers(layer_src, layer_dst)

        # mapping reinit
        misc.copy_params_and_buffers(G_stem.mapping, self.mapping)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        return img
    



@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class SuperresGenerator(torch.nn.Module):
    def __init__(
        self,
        img_resolution,
        path_stem,
        head_layers,
        up_factor,
        **synthesis_kwargs,
    ):
        assert up_factor in [2, 4, 8, 16], "Supported up_factors: [2, 4, 8, 16]"

        super().__init__()
        self.path_stem = path_stem
        self.up_factor = up_factor

        # load pretrained stem
        with dnnlib.util.open_url(path_stem) as f:
            G_stem = legacy.load_network_pkl(f)['G_ema']
        self.mapping = G_stem.mapping
        self.synthesis = G_stem.synthesis

        # update G params
        self.z_dim = G_stem.z_dim
        self.c_dim = G_stem.c_dim
        self.w_dim = G_stem.w_dim
        self.img_channels = G_stem.img_channels
        self.channel_base = G_stem.synthesis.channel_base
        self.channel_max = G_stem.synthesis.channel_max
        self.margin_size = G_stem.synthesis.margin_size
        self.last_stopband_rel = G_stem.synthesis.last_stopband_rel
        self.num_critical = G_stem.synthesis.num_critical
        self.num_fp16_res = G_stem.synthesis.num_fp16_res
        self.conv_kernel = getattr(G_stem.synthesis, 'conv_kernel', 3)
        self.use_radial_filters = getattr(G_stem.synthesis, 'use_radial_filters', False)

        # cut off critically sampled layers
        for name in reversed(self.synthesis.layer_names):
            if getattr(self.synthesis, name).is_critically_sampled:
                delattr(self.synthesis, name)
                self.synthesis.layer_names.pop()
        stem_len = len(self.synthesis.layer_names) + 1

        # update G and G.synthesis params
        self.img_resolution = G_stem.img_resolution * up_factor
        self.synthesis.img_resolution = self.img_resolution
        assert img_resolution == self.img_resolution, f"Resolution mismatch. Dataset: {img_resolution}, G output: {self.img_resolution}"

        self.num_layers = stem_len + head_layers
        self.synthesis.num_layers = self.num_layers

        self.num_ws = stem_len + head_layers + 1
        self.mapping.num_ws = self.num_ws
        self.synthesis.num_ws = self.num_ws

        # initialize new_layers
        last_stem_layer = getattr(self.synthesis, self.synthesis.layer_names[-1])
        fparams = self.compute_superres_filterparams(up_factor, self.img_resolution, last_stem_layer, head_layers)

        self.head_layer_names = []
        for idx in range(head_layers + 1):
            prev = max(idx - 1, 0)
            is_torgb = (idx == head_layers)
            is_critically_sampled = (idx >= head_layers - self.num_critical)
            use_fp16 = (fparams.sampling_rates[idx] * (2 ** self.num_fp16_res) > self.img_resolution)
            layer = SynthesisLayer(
                w_dim=self.w_dim, is_torgb=is_torgb, is_critically_sampled=is_critically_sampled, use_fp16=use_fp16,
                in_channels=int(fparams.channels[prev]), out_channels= int(fparams.channels[idx]),
                in_size=int(fparams.sizes[prev]), out_size=int(fparams.sizes[idx]),
                in_sampling_rate=int(fparams.sampling_rates[prev]), out_sampling_rate=int(fparams.sampling_rates[idx]),
                in_cutoff=fparams.cutoffs[prev], out_cutoff=fparams.cutoffs[idx],
                in_half_width=fparams.half_widths[prev], out_half_width=fparams.half_widths[idx],
                conv_kernel=self.conv_kernel, use_radial_filters=self.use_radial_filters,
            )
            name = f'L{idx+stem_len}_{layer.out_size[0]}_{layer.out_channels}'
            setattr(self.synthesis, name, layer)
            self.synthesis.layer_names.append(name)
            self.head_layer_names.append(name)

    def reinit_stem(self):
        print("Reinitialize stem")
        with dnnlib.util.open_url(self.path_stem) as f:
            G_stem = legacy.load_network_pkl(f)['G_ema']

        # cut off critically sampled layers
        for name in reversed(G_stem.synthesis.layer_names):
            if getattr(G_stem.synthesis, name).is_critically_sampled:
                delattr(G_stem.synthesis, name)
                G_stem.synthesis.layer_names.pop()

        # synthesis reinit
        for name in G_stem.synthesis.layer_names:
            layer_src = getattr(G_stem.synthesis, name)
            layer_dst = getattr(self.synthesis, name)
            misc.copy_params_and_buffers(layer_src, layer_dst)

        # mapping reinit
        misc.copy_params_and_buffers(G_stem.mapping, self.mapping)

    def reinit_mapping(self):
        print("Reinitialize mapping")
        self.mapping = self.new_mapping

    def compute_superres_filterparams(self, up_factor, img_resolution, last_stem_layer, head_layers, num_critical=2):
        # begin with output of last stem layer
        first_cutoff = last_stem_layer.out_cutoff
        first_stopband = last_stem_layer.out_half_width + first_cutoff

        # Geometric progression of layer cutoffs and min. stopbands.
        last_cutoff = img_resolution / 2  # f_{c,N}
        last_stopband = last_cutoff * self.last_stopband_rel # f_{t,N}
        exponents = np.minimum(np.arange(head_layers + 1) / (head_layers - num_critical), 1)
        cutoffs = first_cutoff * (last_cutoff / first_cutoff) ** exponents  # f_c[i]
        stopbands = first_stopband * (last_stopband / first_stopband) ** exponents  # f_t[i]

        # set sampling rates
        sampling_rates = np.exp2(np.ceil(np.log2(np.minimum(stopbands*2, img_resolution))))
        sampling_rates[0] = last_stem_layer.out_sampling_rate

        # Compute remaining layer parameters.
        half_widths = np.maximum(stopbands, sampling_rates / 2) - cutoffs  # f_h[i]
        sizes = sampling_rates + self.margin_size * 2
        sizes[-2:] = img_resolution
        channels = np.rint(np.minimum((self.channel_base / 2) / cutoffs, self.channel_max))
        channels[0] = last_stem_layer.out_channels
        channels[-1] = self.img_channels

        # save in dict
        fparams = dnnlib.EasyDict()
        fparams.cutoffs = cutoffs
        fparams.stopbands = stopbands
        fparams.sampling_rates = sampling_rates
        fparams.half_widths = half_widths
        fparams.sizes = sizes
        fparams.channels = channels
        return fparams

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        return img
