# ==============================================================================
# This file contains code that has been adapted or directly copied from the
# ComfyUI_IPAdapter_plus package by Matteo Spinelli ("Matt3o/Cubiq").
# Original source: https://github.com/cubiq/ComfyUI_IPAdapter_plus
#
# This code is used under the terms of the original license, with modifications
# made to suit the needs of this project.
# ==============================================================================

import torch
import comfy.model_management as model_management
import torch.nn as nn

from comfy.clip_vision import clip_preprocess, Output 

import math
from einops import rearrange
from einops.layers.torch import Rearrange

import torch.nn.functional as F
from comfy.ldm.modules.attention import optimized_attention

'''
CrossAttentionPatch.py:
'''

class Attn2Replace:
    def __init__(self, callback=None, **kwargs):
        self.callback = [callback]
        self.kwargs = [kwargs]

    def add(self, callback, **kwargs):
        self.callback.append(callback)
        self.kwargs.append(kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, q, k, v, extra_options):
        dtype = q.dtype
        out = optimized_attention(q, k, v, extra_options["n_heads"])
        sigma = extra_options["sigmas"].detach().cpu()[0].item() if 'sigmas' in extra_options else 999999999.9

        for i, callback in enumerate(self.callback):
            if sigma <= self.kwargs[i]["sigma_start"] and sigma >= self.kwargs[i]["sigma_end"]:
                out = out + callback(out, q, k, v, extra_options, **self.kwargs[i])

        return out.to(dtype=dtype)


def ipadapter_attention(out, q, k, v, extra_options, module_key='', ipadapter=None, weight=1.0, cond=None, cond_alt=None, uncond=None, weight_type="linear", mask=None, sigma_start=0.0, sigma_end=1.0, unfold_batch=False, embeds_scaling='V only', **kwargs):
    dtype = q.dtype
    cond_or_uncond = extra_options["cond_or_uncond"]
    block_type = extra_options["block"][0]
    #block_id = extra_options["block"][1]
    t_idx = extra_options["transformer_index"]
    layers = 11 if '101_to_k_ip' in ipadapter.ip_layers.to_kvs else 16
    k_key = module_key + "_to_k_ip"
    v_key = module_key + "_to_v_ip"

    # extra options for AnimateDiff
    ad_params = extra_options['ad_params'] if "ad_params" in extra_options else None

    b = q.shape[0]
    seq_len = q.shape[1]
    batch_prompt = b // len(cond_or_uncond)
    _, _, oh, ow = extra_options["original_shape"]

    if weight_type == 'ease in':
        weight = weight * (0.05 + 0.95 * (1 - t_idx / layers))
    elif weight_type == 'ease out':
        weight = weight * (0.05 + 0.95 * (t_idx / layers))
    elif weight_type == 'ease in-out':
        weight = weight * (0.05 + 0.95 * (1 - abs(t_idx - (layers/2)) / (layers/2)))
    elif weight_type == 'reverse in-out':
        weight = weight * (0.05 + 0.95 * (abs(t_idx - (layers/2)) / (layers/2)))
    elif weight_type == 'weak input' and block_type == 'input':
        weight = weight * 0.2
    elif weight_type == 'weak middle' and block_type == 'middle':
        weight = weight * 0.2
    elif weight_type == 'weak output' and block_type == 'output':
        weight = weight * 0.2
    elif weight_type == 'strong middle' and (block_type == 'input' or block_type == 'output'):
        weight = weight * 0.2
    elif isinstance(weight, dict):
        if t_idx not in weight:
            return 0

        if weight_type == "style transfer precise":
            if layers == 11 and t_idx == 3:
                uncond = cond
                cond = cond * 0
            elif layers == 16 and (t_idx == 4 or t_idx == 5):
                uncond = cond
                cond = cond * 0
        elif weight_type == "composition precise":
            if layers == 11 and t_idx != 3:
                uncond = cond
                cond = cond * 0
            elif layers == 16 and (t_idx != 4 and t_idx != 5):
                uncond = cond
                cond = cond * 0

        weight = weight[t_idx]

        if cond_alt is not None and t_idx in cond_alt:
            cond = cond_alt[t_idx]
            del cond_alt

    if unfold_batch:
        # Check AnimateDiff context window
        if ad_params is not None and ad_params["sub_idxs"] is not None:
            if isinstance(weight, torch.Tensor):
                weight = tensor_to_size(weight, ad_params["full_length"])
                weight = torch.Tensor(weight[ad_params["sub_idxs"]])
                if torch.all(weight == 0):
                    return 0
                weight = weight.repeat(len(cond_or_uncond), 1, 1) # repeat for cond and uncond
            elif weight == 0:
                return 0

            # if image length matches or exceeds full_length get sub_idx images
            if cond.shape[0] >= ad_params["full_length"]:
                cond = torch.Tensor(cond[ad_params["sub_idxs"]])
                uncond = torch.Tensor(uncond[ad_params["sub_idxs"]])
            # otherwise get sub_idxs images
            else:
                cond = tensor_to_size(cond, ad_params["full_length"])
                uncond = tensor_to_size(uncond, ad_params["full_length"])
                cond = cond[ad_params["sub_idxs"]]
                uncond = uncond[ad_params["sub_idxs"]]
        else:
            if isinstance(weight, torch.Tensor):
                weight = tensor_to_size(weight, batch_prompt)
                if torch.all(weight == 0):
                    return 0
                weight = weight.repeat(len(cond_or_uncond), 1, 1) # repeat for cond and uncond
            elif weight == 0:
                return 0

            cond = tensor_to_size(cond, batch_prompt)
            uncond = tensor_to_size(uncond, batch_prompt)

        k_cond = ipadapter.ip_layers.to_kvs[k_key](cond)
        k_uncond = ipadapter.ip_layers.to_kvs[k_key](uncond)
        v_cond = ipadapter.ip_layers.to_kvs[v_key](cond)
        v_uncond = ipadapter.ip_layers.to_kvs[v_key](uncond)
    else:
        # TODO: should we always convert the weights to a tensor?
        if isinstance(weight, torch.Tensor):
            weight = tensor_to_size(weight, batch_prompt)
            if torch.all(weight == 0):
                return 0
            weight = weight.repeat(len(cond_or_uncond), 1, 1) # repeat for cond and uncond
        elif weight == 0:
            return 0
        
        k_cond = ipadapter.ip_layers.to_kvs[k_key](cond).repeat(batch_prompt, 1, 1)
        k_uncond = ipadapter.ip_layers.to_kvs[k_key](uncond).repeat(batch_prompt, 1, 1)
        v_cond = ipadapter.ip_layers.to_kvs[v_key](cond).repeat(batch_prompt, 1, 1)
        v_uncond = ipadapter.ip_layers.to_kvs[v_key](uncond).repeat(batch_prompt, 1, 1)

    if len(cond_or_uncond) == 3: # TODO: cosxl, I need to check this
        ip_k = torch.cat([(k_cond, k_uncond, k_cond)[i] for i in cond_or_uncond], dim=0)
        ip_v = torch.cat([(v_cond, v_uncond, v_cond)[i] for i in cond_or_uncond], dim=0)
    else:
        ip_k = torch.cat([(k_cond, k_uncond)[i] for i in cond_or_uncond], dim=0)
        ip_v = torch.cat([(v_cond, v_uncond)[i] for i in cond_or_uncond], dim=0)

    if embeds_scaling == 'K+mean(V) w/ C penalty':
        scaling = float(ip_k.shape[2]) / 1280.0
        weight = weight * scaling
        ip_k = ip_k * weight
        ip_v_mean = torch.mean(ip_v, dim=1, keepdim=True)
        ip_v = (ip_v - ip_v_mean) + ip_v_mean * weight
        out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
        del ip_v_mean
    elif embeds_scaling == 'K+V w/ C penalty':
        scaling = float(ip_k.shape[2]) / 1280.0
        weight = weight * scaling
        ip_k = ip_k * weight
        ip_v = ip_v * weight
        out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
    elif embeds_scaling == 'K+V':
        ip_k = ip_k * weight
        ip_v = ip_v * weight
        out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
    else:
        #ip_v = ip_v * weight
        out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
        out_ip = out_ip * weight # I'm doing this to get the same results as before

    if mask is not None:
        mask_h = oh / math.sqrt(oh * ow / seq_len)
        mask_h = int(mask_h) + int((seq_len % int(mask_h)) != 0)
        mask_w = seq_len // mask_h

        # check if using AnimateDiff and sliding context window
        if (mask.shape[0] > 1 and ad_params is not None and ad_params["sub_idxs"] is not None):
            # if mask length matches or exceeds full_length, get sub_idx masks
            if mask.shape[0] >= ad_params["full_length"]:
                mask = torch.Tensor(mask[ad_params["sub_idxs"]])
                mask = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear").squeeze(1)
            else:
                mask = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear").squeeze(1)
                mask = tensor_to_size(mask, ad_params["full_length"])
                mask = mask[ad_params["sub_idxs"]]
        else:
            mask = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear").squeeze(1)
            mask = tensor_to_size(mask, batch_prompt)

        mask = mask.repeat(len(cond_or_uncond), 1, 1)
        mask = mask.view(mask.shape[0], -1, 1).repeat(1, 1, out.shape[2])

        # covers cases where extreme aspect ratios can cause the mask to have a wrong size
        mask_len = mask_h * mask_w
        if mask_len < seq_len:
            pad_len = seq_len - mask_len
            pad1 = pad_len // 2
            pad2 = pad_len - pad1
            mask = F.pad(mask, (0, 0, pad1, pad2), value=0.0)
        elif mask_len > seq_len:
            crop_start = (mask_len - seq_len) // 2
            mask = mask[:, crop_start:crop_start+seq_len, :]

        out_ip = out_ip * mask

    #out = out + out_ip

    return out_ip.to(dtype=dtype)



'''
image_proj_models.py:
'''

# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )



def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x



class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)



class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        max_seq_len: int = 257,  # CLIP tokens + CLS token
        apply_pos_emb: bool = False,
        num_latents_mean_pooled: int = 0,  # number of latents derived from mean pooled representation of the sequence
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        if self.pos_emb is not None:
            n, device = x.shape[1], x.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            x = x + pos_emb

        latents = self.latents.repeat(x.size(0), 1, 1)

        x = self.proj_in(x)

        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)




def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)



class FacePerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim=768,
        depth=4,
        dim_head=64,
        heads=16,
        embedding_dim=1280,
        output_dim=768,
        ff_mult=4,
    ):
        super().__init__()

        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, latents, x):
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)
    

class MLPProjModel(nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            nn.GELU(),
            nn.Linear(clip_embeddings_dim, cross_attention_dim),
            nn.LayerNorm(cross_attention_dim)
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class MLPProjModelFaceId(nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x


class ProjModelFaceIdPlus(nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, clip_embeddings_dim=1280, num_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

        self.perceiver_resampler = FacePerceiverResampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=cross_attention_dim // 64,
            embedding_dim=clip_embeddings_dim,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )

    def forward(self, id_embeds, clip_embeds, scale=1.0, shortcut=False):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        out = self.perceiver_resampler(x, clip_embeds)
        if shortcut:
            out = x + scale * out
        return out
    


class ImageProjModel(nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        x = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x



'''
utils.py:
'''

def split_tiles(embeds, num_split):
    _, H, W, _ = embeds.shape
    out = []
    for x in embeds:
        x = x.unsqueeze(0)
        h, w = H // num_split, W // num_split
        x_split = torch.cat([x[:, i*h:(i+1)*h, j*w:(j+1)*w, :] for i in range(num_split) for j in range(num_split)], dim=0)    
        out.append(x_split)
    
    x_split = torch.stack(out, dim=0)
    
    return x_split




def merge_hiddenstates(x, tiles):
    chunk_size = tiles*tiles
    x = x.split(chunk_size)

    out = []
    for embeds in x:
        num_tiles = embeds.shape[0]
        tile_size = int((embeds.shape[1]-1) ** 0.5)
        grid_size = int(num_tiles ** 0.5)

        # Extract class tokens
        class_tokens = embeds[:, 0, :]  # Save class tokens: [num_tiles, embeds[-1]]
        avg_class_token = class_tokens.mean(dim=0, keepdim=True).unsqueeze(0)  # Average token, shape: [1, 1, embeds[-1]]

        patch_embeds = embeds[:, 1:, :]  # Shape: [num_tiles, tile_size^2, embeds[-1]]
        reshaped = patch_embeds.reshape(grid_size, grid_size, tile_size, tile_size, embeds.shape[-1])

        merged = torch.cat([torch.cat([reshaped[i, j] for j in range(grid_size)], dim=1) 
                            for i in range(grid_size)], dim=0)
        
        merged = merged.unsqueeze(0)  # Shape: [1, grid_size*tile_size, grid_size*tile_size, embeds[-1]]
        
        # Pool to original size
        pooled = torch.nn.functional.adaptive_avg_pool2d(merged.permute(0, 3, 1, 2), (tile_size, tile_size)).permute(0, 2, 3, 1)
        flattened = pooled.reshape(1, tile_size*tile_size, embeds.shape[-1])
        
        # Add back the class token
        with_class = torch.cat([avg_class_token, flattened], dim=1)  # Shape: original shape
        out.append(with_class)
    
    out = torch.cat(out, dim=0)

    return out





def merge_embeddings(x, tiles): # TODO: this needs so much testing that I don't even
    chunk_size = tiles*tiles
    x = x.split(chunk_size)

    out = []
    for embeds in x:
        num_tiles = embeds.shape[0]
        grid_size = int(num_tiles ** 0.5)
        tile_size = int(embeds.shape[1] ** 0.5)
        reshaped = embeds.reshape(grid_size, grid_size, tile_size, tile_size)
        
        # Merge the tiles
        merged = torch.cat([torch.cat([reshaped[i, j] for j in range(grid_size)], dim=1) 
                            for i in range(grid_size)], dim=0)
        
        merged = merged.unsqueeze(0)  # Shape: [1, grid_size*tile_size, grid_size*tile_size]
        
        # Pool to original size
        pooled = torch.nn.functional.adaptive_avg_pool2d(merged, (tile_size, tile_size))  # pool to [1, tile_size, tile_size]
        pooled = pooled.flatten(1)  # flatten to [1, tile_size^2]
        out.append(pooled)
    out = torch.cat(out, dim=0)
    
    return out





def encode_image_masked(clip_vision, image, mask=None, batch_size=0, tiles=1, ratio=1.0, clipvision_size=224):
    # full image embeds
    embeds = encode_image_masked_(clip_vision, image, mask, batch_size, clipvision_size=clipvision_size)
    tiles = min(tiles, 16)

    if tiles > 1:
        # split in tiles
        image_split = split_tiles(image, tiles)

        # get the embeds for each tile
        embeds_split = Output()
        for i in image_split:
            encoded = encode_image_masked_(clip_vision, i, mask, batch_size, clipvision_size=clipvision_size)
            if not hasattr(embeds_split, "image_embeds"):
                #embeds_split["last_hidden_state"] = encoded["last_hidden_state"]
                embeds_split["image_embeds"] = encoded["image_embeds"]
                embeds_split["penultimate_hidden_states"] = encoded["penultimate_hidden_states"]
            else:
                #embeds_split["last_hidden_state"] = torch.cat((embeds_split["last_hidden_state"], encoded["last_hidden_state"]), dim=0)
                embeds_split["image_embeds"] = torch.cat((embeds_split["image_embeds"], encoded["image_embeds"]), dim=0)
                embeds_split["penultimate_hidden_states"] = torch.cat((embeds_split["penultimate_hidden_states"], encoded["penultimate_hidden_states"]), dim=0)

        #embeds_split['last_hidden_state'] = merge_hiddenstates(embeds_split['last_hidden_state'])
        embeds_split["image_embeds"] = merge_embeddings(embeds_split["image_embeds"], tiles)
        embeds_split["penultimate_hidden_states"] = merge_hiddenstates(embeds_split["penultimate_hidden_states"], tiles)

        #embeds['last_hidden_state'] = torch.cat([embeds_split['last_hidden_state'], embeds['last_hidden_state']])
        if embeds['image_embeds'].shape[0] > 1: # if we have more than one image we need to average the embeddings for consistency
            embeds['image_embeds'] = embeds['image_embeds']*ratio + embeds_split['image_embeds']*(1-ratio)
            embeds['penultimate_hidden_states'] = embeds['penultimate_hidden_states']*ratio + embeds_split['penultimate_hidden_states']*(1-ratio)
            #embeds['image_embeds'] = (embeds['image_embeds']*ratio + embeds_split['image_embeds']) / 2
            #embeds['penultimate_hidden_states'] = (embeds['penultimate_hidden_states']*ratio + embeds_split['penultimate_hidden_states']) / 2
        else: # otherwise we can concatenate them, they can be averaged later
            embeds['image_embeds'] = torch.cat([embeds['image_embeds']*ratio, embeds_split['image_embeds']])
            embeds['penultimate_hidden_states'] = torch.cat([embeds['penultimate_hidden_states']*ratio, embeds_split['penultimate_hidden_states']])

    #del embeds_split

    return embeds





def encode_image_masked_(clip_vision, image, mask=None, batch_size=0, clipvision_size=224):
    model_management.load_model_gpu(clip_vision.patcher)
    outputs = Output()

    if batch_size == 0:
        batch_size = image.shape[0]
    elif batch_size > image.shape[0]:
        batch_size = image.shape[0]

    image_batch = torch.split(image, batch_size, dim=0)

    for img in image_batch:
        img = img.to(clip_vision.load_device)
        pixel_values = clip_preprocess(img, size=clipvision_size).float()

        # TODO: support for multiple masks
        if mask is not None:
            pixel_values = pixel_values * mask.to(clip_vision.load_device)

        out = clip_vision.model(pixel_values=pixel_values, intermediate_output=-2)

        if not hasattr(outputs, "last_hidden_state"):
            outputs["last_hidden_state"] = out[0].to(model_management.intermediate_device())
            outputs["image_embeds"] = out[2].to(model_management.intermediate_device())
            outputs["penultimate_hidden_states"] = out[1].to(model_management.intermediate_device())
        else:
            outputs["last_hidden_state"] = torch.cat((outputs["last_hidden_state"], out[0].to(model_management.intermediate_device())), dim=0)
            outputs["image_embeds"] = torch.cat((outputs["image_embeds"], out[2].to(model_management.intermediate_device())), dim=0)
            outputs["penultimate_hidden_states"] = torch.cat((outputs["penultimate_hidden_states"], out[1].to(model_management.intermediate_device())), dim=0)

    del img, pixel_values, out
    torch.cuda.empty_cache()

    return outputs



def tensor_to_size(source, dest_size):
    if isinstance(dest_size, torch.Tensor):
        dest_size = dest_size.shape[0]
    source_size = source.shape[0]

    if source_size < dest_size:
        shape = [dest_size - source_size] + [1]*(source.dim()-1)
        source = torch.cat((source, source[-1:].repeat(shape)), dim=0)
    elif source_size > dest_size:
        source = source[:dest_size]

    return source




'''
IPAdapterPlus.py:
'''

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Main IPAdapter Class (IPAdapterPlus.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
class IPAdapter(nn.Module):
    def __init__(self, ipadapter_model, cross_attention_dim=1024, output_cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4, is_sdxl=False, is_plus=False, is_full=False, is_faceid=False, is_portrait_unnorm=False, is_kwai_kolors=False, encoder_hid_proj=None, weight_kolors=1.0):
        super().__init__()

        self.clip_embeddings_dim = clip_embeddings_dim
        self.cross_attention_dim = cross_attention_dim
        self.output_cross_attention_dim = output_cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.is_sdxl = is_sdxl
        self.is_full = is_full
        self.is_plus = is_plus
        self.is_portrait_unnorm = is_portrait_unnorm
        self.is_kwai_kolors = is_kwai_kolors

        if is_faceid and not is_portrait_unnorm:
            self.image_proj_model = self.init_proj_faceid()
        elif is_full:
            self.image_proj_model = self.init_proj_full()
        elif is_plus or is_portrait_unnorm:
            self.image_proj_model = self.init_proj_plus()
        else:
            self.image_proj_model = self.init_proj()

        self.image_proj_model.load_state_dict(ipadapter_model["image_proj"])
        self.ip_layers = To_KV(ipadapter_model["ip_adapter"], encoder_hid_proj=encoder_hid_proj, weight_kolors=weight_kolors)

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.cross_attention_dim,
            clip_embeddings_dim=self.clip_embeddings_dim,
            clip_extra_context_tokens=self.clip_extra_context_tokens
        )
        return image_proj_model

    def init_proj_plus(self):
        image_proj_model = Resampler(
            dim=self.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=20 if self.is_sdxl and not self.is_kwai_kolors else 12,
            num_queries=self.clip_extra_context_tokens,
            embedding_dim=self.clip_embeddings_dim,
            output_dim=self.output_cross_attention_dim,
            ff_mult=4
        )
        return image_proj_model

    def init_proj_full(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.cross_attention_dim,
            clip_embeddings_dim=self.clip_embeddings_dim
        )
        return image_proj_model

    def init_proj_faceid(self):
        if self.is_plus:
            image_proj_model = ProjModelFaceIdPlus(
                cross_attention_dim=self.cross_attention_dim,
                id_embeddings_dim=512,
                clip_embeddings_dim=self.clip_embeddings_dim,
                num_tokens=self.clip_extra_context_tokens,
            )
        else:
            image_proj_model = MLPProjModelFaceId(
                cross_attention_dim=self.cross_attention_dim,
                id_embeddings_dim=512,
                num_tokens=self.clip_extra_context_tokens,
            )
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, clip_embed, clip_embed_zeroed, batch_size):
        torch_device = model_management.get_torch_device()
        intermediate_device = model_management.intermediate_device()

        if batch_size == 0:
            batch_size = clip_embed.shape[0]
            intermediate_device = torch_device
        elif batch_size > clip_embed.shape[0]:
            batch_size = clip_embed.shape[0]

        clip_embed = torch.split(clip_embed, batch_size, dim=0)
        clip_embed_zeroed = torch.split(clip_embed_zeroed, batch_size, dim=0)
        
        image_prompt_embeds = []
        uncond_image_prompt_embeds = []

        for ce, cez in zip(clip_embed, clip_embed_zeroed):
            image_prompt_embeds.append(self.image_proj_model(ce.to(torch_device)).to(intermediate_device))
            uncond_image_prompt_embeds.append(self.image_proj_model(cez.to(torch_device)).to(intermediate_device))

        del clip_embed, clip_embed_zeroed

        image_prompt_embeds = torch.cat(image_prompt_embeds, dim=0)
        uncond_image_prompt_embeds = torch.cat(uncond_image_prompt_embeds, dim=0)

        torch.cuda.empty_cache()

        #image_prompt_embeds = self.image_proj_model(clip_embed)
        #uncond_image_prompt_embeds = self.image_proj_model(clip_embed_zeroed)
        return image_prompt_embeds, uncond_image_prompt_embeds

    @torch.inference_mode()
    def get_image_embeds_faceid_plus(self, face_embed, clip_embed, s_scale, shortcut, batch_size):
        torch_device = model_management.get_torch_device()
        intermediate_device = model_management.intermediate_device()

        if batch_size == 0:
            batch_size = clip_embed.shape[0]
            intermediate_device = torch_device
        elif batch_size > clip_embed.shape[0]:
            batch_size = clip_embed.shape[0]

        face_embed_batch = torch.split(face_embed, batch_size, dim=0)
        clip_embed_batch = torch.split(clip_embed, batch_size, dim=0)

        embeds = []
        for face_embed, clip_embed in zip(face_embed_batch, clip_embed_batch):
            embeds.append(self.image_proj_model(face_embed.to(torch_device), clip_embed.to(torch_device), scale=s_scale, shortcut=shortcut).to(intermediate_device))

        embeds = torch.cat(embeds, dim=0)
        del face_embed_batch, clip_embed_batch
        torch.cuda.empty_cache()
        #embeds = self.image_proj_model(face_embed, clip_embed, scale=s_scale, shortcut=shortcut)
        return embeds



class To_KV(nn.Module):
    def __init__(self, state_dict, encoder_hid_proj=None, weight_kolors=1.0):
        super().__init__()

        if encoder_hid_proj is not None:
            hid_proj = nn.Linear(encoder_hid_proj["weight"].shape[1], encoder_hid_proj["weight"].shape[0], bias=True)
            hid_proj.weight.data = encoder_hid_proj["weight"] * weight_kolors
            hid_proj.bias.data = encoder_hid_proj["bias"] * weight_kolors

        self.to_kvs = nn.ModuleDict()
        for key, value in state_dict.items():
            if encoder_hid_proj is not None:
                linear_proj = nn.Linear(value.shape[1], value.shape[0], bias=False)
                linear_proj.weight.data = value
                self.to_kvs[key.replace(".weight", "").replace(".", "_")] = nn.Sequential(hid_proj, linear_proj)
            else:
                self.to_kvs[key.replace(".weight", "").replace(".", "_")] = nn.Linear(value.shape[1], value.shape[0], bias=False)
                self.to_kvs[key.replace(".weight", "").replace(".", "_")].weight.data = value




def set_model_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"].copy()
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    else:
        to["patches_replace"]["attn2"] = to["patches_replace"]["attn2"].copy()

    if key not in to["patches_replace"]["attn2"]:
        to["patches_replace"]["attn2"][key] = Attn2Replace(ipadapter_attention, **patch_kwargs)
        model.model_options["transformer_options"] = to
    else:
        to["patches_replace"]["attn2"][key].add(ipadapter_attention, **patch_kwargs)




def ipadapter_execute(model,
                      ipadapter,
                      clipvision,
                      insightface=None,
                      image=None,
                      image_composition=None,
                      image_negative=None,
                      weight=1.0,
                      weight_composition=1.0,
                      weight_faceidv2=None,
                      weight_kolors=1.0,
                      weight_type="linear",
                      combine_embeds="concat",
                      start_at=0.0,
                      end_at=1.0,
                      attn_mask=None,
                      pos_embed=None,
                      neg_embed=None,
                      unfold_batch=False,
                      embeds_scaling='V only',
                      layer_weights=None,
                      encode_batch_size=0,
                      style_boost=None,
                      composition_boost=None,
                      enhance_tiles=1,
                      enhance_ratio=1.0,):
    device = model_management.get_torch_device()
    dtype = model_management.unet_dtype()
    if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        dtype = torch.float16 if model_management.should_use_fp16() else torch.float32

    is_full = "proj.3.weight" in ipadapter["image_proj"]
    is_portrait_unnorm = "portraitunnorm" in ipadapter
    is_plus = (is_full or "latents" in ipadapter["image_proj"] or "perceiver_resampler.proj_in.weight" in ipadapter["image_proj"]) and not is_portrait_unnorm
    output_cross_attention_dim = ipadapter["ip_adapter"]["1.to_k_ip.weight"].shape[1]
    is_sdxl = output_cross_attention_dim == 2048
    is_kwai_kolors_faceid = "perceiver_resampler.layers.0.0.to_out.weight" in ipadapter["image_proj"] and ipadapter["image_proj"]["perceiver_resampler.layers.0.0.to_out.weight"].shape[0] == 4096
    is_faceidv2 = "faceidplusv2" in ipadapter or is_kwai_kolors_faceid
    is_kwai_kolors = (is_sdxl and "layers.0.0.to_out.weight" in ipadapter["image_proj"] and ipadapter["image_proj"]["layers.0.0.to_out.weight"].shape[0] == 2048) or is_kwai_kolors_faceid
    is_portrait = "proj.2.weight" in ipadapter["image_proj"] and not "proj.3.weight" in ipadapter["image_proj"] and not "0.to_q_lora.down.weight" in ipadapter["ip_adapter"] and not is_kwai_kolors_faceid
    is_faceid = is_portrait or "0.to_q_lora.down.weight" in ipadapter["ip_adapter"] or is_portrait_unnorm or is_kwai_kolors_faceid

    if is_faceid and not insightface:
        raise Exception("insightface model is required for FaceID models")

    if is_faceidv2:
        weight_faceidv2 = weight_faceidv2 if weight_faceidv2 is not None else weight*2

    if is_kwai_kolors_faceid:
        cross_attention_dim = 4096
    elif is_kwai_kolors:
        cross_attention_dim = 2048
    elif (is_plus and is_sdxl and not is_faceid) or is_portrait_unnorm:
        cross_attention_dim = 1280
    else:
        cross_attention_dim = output_cross_attention_dim
    
    if is_kwai_kolors_faceid:
        clip_extra_context_tokens = 6
    elif (is_plus and not is_faceid) or is_portrait or is_portrait_unnorm:
        clip_extra_context_tokens = 16
    else:
        clip_extra_context_tokens = 4

    if image is not None and image.shape[1] != image.shape[2]:
        print("\033[33mINFO: the IPAdapter reference image is not a square, CLIPImageProcessor will resize and crop it at the center. If the main focus of the picture is not in the middle the result might not be what you are expecting.\033[0m")

    if isinstance(weight, list):
        weight = torch.tensor(weight).unsqueeze(-1).unsqueeze(-1).to(device, dtype=dtype) if unfold_batch else weight[0]

    if style_boost is not None:
        weight_type = "style transfer precise"
    elif composition_boost is not None:
        weight_type = "composition precise"

    # special weight types
    if layer_weights is not None and layer_weights != '':
        weight = { int(k): float(v)*weight for k, v in [x.split(":") for x in layer_weights.split(",")] }
        weight_type = weight_type if weight_type == "style transfer precise" or weight_type == "composition precise" else "linear"
    elif weight_type == "style transfer":
        weight = { 6:weight } if is_sdxl else { 0:weight, 1:weight, 2:weight, 3:weight, 9:weight, 10:weight, 11:weight, 12:weight, 13:weight, 14:weight, 15:weight }
    elif weight_type == "composition":
        weight = { 3:weight } if is_sdxl else { 4:weight*0.25, 5:weight }
    elif weight_type == "strong style transfer":
        if is_sdxl:
            weight = { 0:weight, 1:weight, 2:weight, 4:weight, 5:weight, 6:weight, 7:weight, 8:weight, 9:weight, 10:weight }
        else:
            weight = { 0:weight, 1:weight, 2:weight, 3:weight, 6:weight, 7:weight, 8:weight, 9:weight, 10:weight, 11:weight, 12:weight, 13:weight, 14:weight, 15:weight }
    elif weight_type == "style and composition":
        if is_sdxl:
            weight = { 3:weight_composition, 6:weight }
        else:
            weight = { 0:weight, 1:weight, 2:weight, 3:weight, 4:weight_composition*0.25, 5:weight_composition, 9:weight, 10:weight, 11:weight, 12:weight, 13:weight, 14:weight, 15:weight }
    elif weight_type == "strong style and composition":
        if is_sdxl:
            weight = { 0:weight, 1:weight, 2:weight, 3:weight_composition, 4:weight, 5:weight, 6:weight, 7:weight, 8:weight, 9:weight, 10:weight }
        else:
            weight = { 0:weight, 1:weight, 2:weight, 3:weight, 4:weight_composition, 5:weight_composition, 6:weight, 7:weight, 8:weight, 9:weight, 10:weight, 11:weight, 12:weight, 13:weight, 14:weight, 15:weight }
    elif weight_type == "style transfer precise":
        weight_composition = style_boost if style_boost is not None else weight
        if is_sdxl:
            weight = { 3:weight_composition, 6:weight }
        else:
            weight = { 0:weight, 1:weight, 2:weight, 3:weight, 4:weight_composition*0.25, 5:weight_composition, 9:weight, 10:weight, 11:weight, 12:weight, 13:weight, 14:weight, 15:weight }
    elif weight_type == "composition precise":
        weight_composition = weight
        weight = composition_boost if composition_boost is not None else weight
        if is_sdxl:
            weight = { 0:weight*.1, 1:weight*.1, 2:weight*.1, 3:weight_composition, 4:weight*.1, 5:weight*.1, 6:weight, 7:weight*.1, 8:weight*.1, 9:weight*.1, 10:weight*.1 }
        else:
            weight = { 0:weight, 1:weight, 2:weight, 3:weight, 4:weight_composition*0.25, 5:weight_composition, 6:weight*.1, 7:weight*.1, 8:weight*.1, 9:weight, 10:weight, 11:weight, 12:weight, 13:weight, 14:weight, 15:weight }

    clipvision_size = 224 if not is_kwai_kolors else 336

    img_comp_cond_embeds = None
    face_cond_embeds = None
    # if is_faceid:
    #     if insightface is None:
    #         raise Exception("Insightface model is required for FaceID models")

    #     from insightface.utils import face_align

    #     insightface.det_model.input_size = (640,640) # reset the detection size
    #     image_iface = tensor_to_image(image)
    #     face_cond_embeds = []
    #     image = []

    #     for i in range(image_iface.shape[0]):
    #         for size in [(size, size) for size in range(640, 256, -64)]:
    #             insightface.det_model.input_size = size # TODO: hacky but seems to be working
    #             face = insightface.get(image_iface[i])
    #             if face:
    #                 if not is_portrait_unnorm:
    #                     face_cond_embeds.append(torch.from_numpy(face[0].normed_embedding).unsqueeze(0))
    #                 else:
    #                     face_cond_embeds.append(torch.from_numpy(face[0].embedding).unsqueeze(0))
    #                 image.append(image_to_tensor(face_align.norm_crop(image_iface[i], landmark=face[0].kps, image_size=336 if is_kwai_kolors_faceid else 256 if is_sdxl else 224)))

    #                 if 640 not in size:
    #                     print(f"\033[33mINFO: InsightFace detection resolution lowered to {size}.\033[0m")
    #                 break
    #         else:
    #             raise Exception('InsightFace: No face detected.')
    #     face_cond_embeds = torch.stack(face_cond_embeds).to(device, dtype=dtype)
    #     image = torch.stack(image)
    #     del image_iface, face

    if image is not None:
        img_cond_embeds = encode_image_masked(clipvision, image, batch_size=encode_batch_size, tiles=enhance_tiles, ratio=enhance_ratio, clipvision_size=clipvision_size)
        if image_composition is not None:
            img_comp_cond_embeds = encode_image_masked(clipvision, image_composition, batch_size=encode_batch_size, tiles=enhance_tiles, ratio=enhance_ratio, clipvision_size=clipvision_size)

        if is_plus:
            img_cond_embeds = img_cond_embeds.penultimate_hidden_states
            image_negative = image_negative if image_negative is not None else torch.zeros([1, clipvision_size, clipvision_size, 3])
            img_uncond_embeds = encode_image_masked(clipvision, image_negative, batch_size=encode_batch_size, clipvision_size=clipvision_size).penultimate_hidden_states
            if image_composition is not None:
                img_comp_cond_embeds = img_comp_cond_embeds.penultimate_hidden_states
        else:
            img_cond_embeds = img_cond_embeds.image_embeds if not is_faceid else face_cond_embeds
            if image_negative is not None and not is_faceid:
                img_uncond_embeds = encode_image_masked(clipvision, image_negative, batch_size=encode_batch_size, clipvision_size=clipvision_size).image_embeds
            else:
                img_uncond_embeds = torch.zeros_like(img_cond_embeds)
            if image_composition is not None:
                img_comp_cond_embeds = img_comp_cond_embeds.image_embeds
        del image_negative, image_composition

        image = None if not is_faceid else image # if it's face_id we need the cropped face for later
    elif pos_embed is not None:
        img_cond_embeds = pos_embed

        if neg_embed is not None:
            img_uncond_embeds = neg_embed
        else:
            if is_plus:
                img_uncond_embeds = encode_image_masked(clipvision, torch.zeros([1, clipvision_size, clipvision_size, 3]), clipvision_size=clipvision_size).penultimate_hidden_states
            else:
                img_uncond_embeds = torch.zeros_like(img_cond_embeds)
        del pos_embed, neg_embed
    else:
        raise Exception("Images or Embeds are required")

    # ensure that cond and uncond have the same batch size
    img_uncond_embeds = tensor_to_size(img_uncond_embeds, img_cond_embeds.shape[0])

    img_cond_embeds = img_cond_embeds.to(device, dtype=dtype)
    img_uncond_embeds = img_uncond_embeds.to(device, dtype=dtype)
    if img_comp_cond_embeds is not None:
        img_comp_cond_embeds = img_comp_cond_embeds.to(device, dtype=dtype)

    # combine the embeddings if needed
    if combine_embeds != "concat" and img_cond_embeds.shape[0] > 1 and not unfold_batch:
        if combine_embeds == "add":
            img_cond_embeds = torch.sum(img_cond_embeds, dim=0).unsqueeze(0)
            if face_cond_embeds is not None:
                face_cond_embeds = torch.sum(face_cond_embeds, dim=0).unsqueeze(0)
            if img_comp_cond_embeds is not None:
                img_comp_cond_embeds = torch.sum(img_comp_cond_embeds, dim=0).unsqueeze(0)
        elif combine_embeds == "subtract":
            img_cond_embeds = img_cond_embeds[0] - torch.mean(img_cond_embeds[1:], dim=0)
            img_cond_embeds = img_cond_embeds.unsqueeze(0)
            if face_cond_embeds is not None:
                face_cond_embeds = face_cond_embeds[0] - torch.mean(face_cond_embeds[1:], dim=0)
                face_cond_embeds = face_cond_embeds.unsqueeze(0)
            if img_comp_cond_embeds is not None:
                img_comp_cond_embeds = img_comp_cond_embeds[0] - torch.mean(img_comp_cond_embeds[1:], dim=0)
                img_comp_cond_embeds = img_comp_cond_embeds.unsqueeze(0)
        elif combine_embeds == "average":
            img_cond_embeds = torch.mean(img_cond_embeds, dim=0).unsqueeze(0)
            if face_cond_embeds is not None:
                face_cond_embeds = torch.mean(face_cond_embeds, dim=0).unsqueeze(0)
            if img_comp_cond_embeds is not None:
                img_comp_cond_embeds = torch.mean(img_comp_cond_embeds, dim=0).unsqueeze(0)
        elif combine_embeds == "norm average":
            img_cond_embeds = torch.mean(img_cond_embeds / torch.norm(img_cond_embeds, dim=0, keepdim=True), dim=0).unsqueeze(0)
            if face_cond_embeds is not None:
                face_cond_embeds = torch.mean(face_cond_embeds / torch.norm(face_cond_embeds, dim=0, keepdim=True), dim=0).unsqueeze(0)
            if img_comp_cond_embeds is not None:
                img_comp_cond_embeds = torch.mean(img_comp_cond_embeds / torch.norm(img_comp_cond_embeds, dim=0, keepdim=True), dim=0).unsqueeze(0)
        img_uncond_embeds = img_uncond_embeds[0].unsqueeze(0) # TODO: better strategy for uncond could be to average them

    if attn_mask is not None:
        attn_mask = attn_mask.to(device, dtype=dtype)

    encoder_hid_proj = None

    if is_kwai_kolors_faceid and hasattr(model.model, "diffusion_model") and hasattr(model.model.diffusion_model, "encoder_hid_proj"):
        encoder_hid_proj = model.model.diffusion_model.encoder_hid_proj.state_dict()

    ipa = IPAdapter(
        ipadapter,
        cross_attention_dim=cross_attention_dim,
        output_cross_attention_dim=output_cross_attention_dim,
        clip_embeddings_dim=img_cond_embeds.shape[-1],
        clip_extra_context_tokens=clip_extra_context_tokens,
        is_sdxl=is_sdxl,
        is_plus=is_plus,
        is_full=is_full,
        is_faceid=is_faceid,
        is_portrait_unnorm=is_portrait_unnorm,
        is_kwai_kolors=is_kwai_kolors,
        encoder_hid_proj=encoder_hid_proj,
        weight_kolors=weight_kolors
    ).to(device, dtype=dtype)

    if is_faceid and is_plus:
        cond = ipa.get_image_embeds_faceid_plus(face_cond_embeds, img_cond_embeds, weight_faceidv2, is_faceidv2, encode_batch_size)
        # TODO: check if noise helps with the uncond face embeds
        uncond = ipa.get_image_embeds_faceid_plus(torch.zeros_like(face_cond_embeds), img_uncond_embeds, weight_faceidv2, is_faceidv2, encode_batch_size)
    else:
        cond, uncond = ipa.get_image_embeds(img_cond_embeds, img_uncond_embeds, encode_batch_size)
        if img_comp_cond_embeds is not None:
            cond_comp = ipa.get_image_embeds(img_comp_cond_embeds, img_uncond_embeds, encode_batch_size)[0]

    cond = cond.to(device, dtype=dtype)
    uncond = uncond.to(device, dtype=dtype)

    cond_alt = None
    if img_comp_cond_embeds is not None:
        cond_alt = { 3: cond_comp.to(device, dtype=dtype) }

    del img_cond_embeds, img_uncond_embeds, img_comp_cond_embeds, face_cond_embeds

    sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
    sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

    patch_kwargs = {
        "ipadapter": ipa,
        "weight": weight,
        "cond": cond,
        "cond_alt": cond_alt,
        "uncond": uncond,
        "weight_type": weight_type,
        "mask": attn_mask,
        "sigma_start": sigma_start,
        "sigma_end": sigma_end,
        "unfold_batch": unfold_batch,
        "embeds_scaling": embeds_scaling,
    }

    number = 0
    if not is_sdxl:
        for id in [1,2,4,5,7,8]: # id of input_blocks that have cross attention
            patch_kwargs["module_key"] = str(number*2+1)
            set_model_patch_replace(model, patch_kwargs, ("input", id))
            number += 1
        for id in [3,4,5,6,7,8,9,10,11]: # id of output_blocks that have cross attention
            patch_kwargs["module_key"] = str(number*2+1)
            set_model_patch_replace(model, patch_kwargs, ("output", id))
            number += 1
        patch_kwargs["module_key"] = str(number*2+1)
        set_model_patch_replace(model, patch_kwargs, ("middle", 0))
    else:
        for id in [4,5,7,8]: # id of input_blocks that have cross attention
            block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                set_model_patch_replace(model, patch_kwargs, ("input", id, index))
                number += 1
        for id in range(6): # id of output_blocks that have cross attention
            block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                set_model_patch_replace(model, patch_kwargs, ("output", id, index))
                number += 1
        for index in range(10):
            patch_kwargs["module_key"] = str(number*2+1)
            set_model_patch_replace(model, patch_kwargs, ("middle", 0, index))
            number += 1

    return (model, image)



def apply_ipadapter(model, ipadapter, image, weight, start_at, end_at, weight_type, attn_mask=None):
        if weight_type.startswith("style"):
            weight_type = "style transfer"
        elif weight_type == "prompt is more important":
            weight_type = "ease out"
        else:
            weight_type = "linear"

        ipa_args = {
            "image": image,
            "weight": weight,
            "start_at": start_at,
            "end_at": end_at,
            "attn_mask": attn_mask,
            "weight_type": weight_type,
            "insightface": ipadapter['insightface']['model'] if 'insightface' in ipadapter else None,
        }
        
        if 'ipadapter' not in ipadapter:
            raise Exception("IPAdapter model not present in the pipeline. Please load the models with the IPAdapterUnifiedLoader node.")
        if 'clipvision' not in ipadapter:
            raise Exception("CLIPVision model not present in the pipeline. Please load the models with the IPAdapterUnifiedLoader node.")

        return ipadapter_execute(model.clone(), ipadapter['ipadapter']['model'], ipadapter['clipvision']['model'], **ipa_args)