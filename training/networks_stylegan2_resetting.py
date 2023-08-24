

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
                img_channels=self.synthesis.img_channels, is_last=is_last, use_fp16=use_fp16)
            self.synthesis.num_ws += block.num_conv
            if is_last:
                self.synthesis.num_ws += block.num_torgb
            setattr(self.synthesis, f'b{res}', block)
        
        self.num_ws = self.synthesis.num_ws
        self.mapping.num_ws = self.synthesis.num_ws

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
    
