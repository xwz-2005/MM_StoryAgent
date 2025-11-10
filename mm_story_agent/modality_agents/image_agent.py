from typing import List, Dict
import json
import os
import random
import time
import base64
import requests
from io import BytesIO
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline, DDIMScheduler

from mm_story_agent.prompts_zh import role_extract_system, role_review_system, \
    story_to_image_reviser_system, story_to_image_review_system
from mm_story_agent.base import register_tool, init_tool_instance

"""

    StoryDiffusionAgent (本地版)  ← 继承相同逻辑 → DashScopeImageAgent (API版)
            ↓                               ↓
    StoryDiffusionSynthesizer         generate_image_from_prompt()
            ↓
    SpatialAttnProcessor2_0 (保持一致性)
            ↓
      AttnProcessor (基础注意力)
      
    ||---总结对比---||
    
    类名	                        类型	            主要作用	            使用场景
   |-----------------------------------------------------------------------|
    AttnProcessor	            基础组件	        标准注意力计算	        模型底层
    SpatialAttnProcessor2_0	    核心算法	        保持多图一致性	        故事连续性
    StoryDiffusionSynthesizer	生成引擎	        调用模型生成图片	    图像合成
    StoryDiffusionAgent	        流程控制器	    本地完整流程	        有GPU的用户
    DashScopeImageAgent	        流程控制器	    API完整流程	        无GPU的用户


"""

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

"""
    标准注意力计算（基础）
    模型底层
    处理文本和图像特征之间的交叉注意力
    用于Stable Diffusion模型中的常规注意力计算
"""
class AttnProcessor(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def cal_attn_mask_xl(total_length,
                     id_length,
                     sa32,
                     sa64,
                     height,
                     width,
                     device="cuda",
                     dtype=torch.float16):
    nums_1024 = (height // 32) * (width // 32)
    nums_4096 = (height // 16) * (width // 16)
    bool_matrix1024 = torch.rand((1, total_length * nums_1024),device = device,dtype = dtype) < sa32
    bool_matrix4096 = torch.rand((1, total_length * nums_4096),device = device,dtype = dtype) < sa64
    bool_matrix1024 = bool_matrix1024.repeat(total_length,1)
    bool_matrix4096 = bool_matrix4096.repeat(total_length,1)
    for i in range(total_length):
        bool_matrix1024[i:i+1,id_length*nums_1024:] = False
        bool_matrix4096[i:i+1,id_length*nums_4096:] = False
        bool_matrix1024[i:i+1,i*nums_1024:(i+1)*nums_1024] = True
        bool_matrix4096[i:i+1,i*nums_4096:(i+1)*nums_4096] = True
    mask1024 = bool_matrix1024.unsqueeze(1).repeat(1,nums_1024,1).reshape(-1,total_length * nums_1024)
    mask4096 = bool_matrix4096.unsqueeze(1).repeat(1,nums_4096,1).reshape(-1,total_length * nums_4096)
    return mask1024, mask4096


"""
    空间注意力处理器:
        作用：实现故事一致性的特殊注意力机制

        核心创新：确保多张故事图片中角色和风格保持一致
        
        通过特殊的注意力掩码控制不同图片区域之间的信息流动
        
        包含两种调用模式：
        
        __call1__: 使用空间注意力掩码，保持角色一致性
        
        __call2__: 常规注意力，用于早期生成步骤
"""
class SpatialAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        text_context_len (`int`, defaults to 77):
            The context length of the text features.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(self,
                 global_attn_args,
                 hidden_size=None,
                 cross_attention_dim=None,
                 id_length=4,
                 device="cuda",
                 dtype=torch.float16,
                 height=1280,
                 width=720,
                 sa32=0.5,
                 sa64=0.5,
                 ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.total_length = id_length + 1
        self.id_length = id_length
        self.id_bank = {}
        self.height = height
        self.width = width
        self.sa32 = sa32
        self.sa64 = sa64
        self.write = True

        self.global_attn_args = global_attn_args


    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None
    ):
        total_count = self.global_attn_args["total_count"]
        attn_count = self.global_attn_args["attn_count"]
        cur_step = self.global_attn_args["cur_step"]
        mask1024 = self.global_attn_args["mask1024"]
        mask4096 = self.global_attn_args["mask4096"]

        if self.write:
            self.id_bank[cur_step] = [hidden_states[:self.id_length], hidden_states[self.id_length:]]
        else:
            encoder_hidden_states = torch.cat((self.id_bank[cur_step][0].to(self.device),
                                               hidden_states[:1],
                                               self.id_bank[cur_step][1].to(self.device), hidden_states[1:]))
        # skip in early step
        if cur_step < 5:
            hidden_states = self.__call2__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        else:   # 256 1024 4096
            random_number = random.random()
            if cur_step < 20:
                rand_num = 0.3
            else:
                rand_num = 0.1
            if random_number > rand_num:
                if not self.write:
                    if hidden_states.shape[1] == (self.height // 32) * (self.width // 32):
                        attention_mask = mask1024[mask1024.shape[0] // self.total_length * self.id_length:]
                    else:
                        attention_mask = mask4096[mask4096.shape[0] // self.total_length * self.id_length:]
                else:
                    if hidden_states.shape[1] == (self.height // 32) * (self.width // 32):
                        attention_mask = mask1024[:mask1024.shape[0] // self.total_length * self.id_length,
                                                  :mask1024.shape[0] // self.total_length * self.id_length]
                    else:
                        attention_mask = mask4096[:mask4096.shape[0] // self.total_length * self.id_length, 
                                                  :mask4096.shape[0] // self.total_length * self.id_length]
                hidden_states = self.__call1__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
            else:
                hidden_states = self.__call2__(attn, hidden_states, None, attention_mask, temb)
        attn_count += 1
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1
            mask1024, mask4096 = cal_attn_mask_xl(self.total_length,
                                                  self.id_length,
                                                  self.sa32,
                                                  self.sa64,
                                                  self.height,
                                                  self.width,
                                                  device=self.device, 
                                                  dtype=self.dtype)
            self.global_attn_args["mask1024"] = mask1024
            self.global_attn_args["mask4096"] = mask4096

        self.global_attn_args["attn_count"] = attn_count
        self.global_attn_args["cur_step"] = cur_step

        return hidden_states
    
    def __call1__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            total_batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(total_batch_size, channel, height * width).transpose(1, 2)
        total_batch_size, nums_token, channel = hidden_states.shape
        img_nums = total_batch_size // 2
        hidden_states = hidden_states.view(-1, img_nums, nums_token, channel).reshape(-1, img_nums * nums_token, channel)

        batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1, self.id_length + 1, nums_token, channel).reshape(
                -1, (self.id_length + 1) * nums_token, channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)


        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(total_batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)



        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)


        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(total_batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        # print(hidden_states.shape)
        return hidden_states
    
    def __call2__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, channel = (
            hidden_states.shape
        )
        # print(hidden_states.shape)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1, self.id_length + 1, sequence_length, channel).reshape(
                -1, (self.id_length + 1) * sequence_length, channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


"""
    故事图像合成器: StoryDiffusionSynthesizer
        作用：核心的图像生成引擎，负责调用AI模型生成故事图片
        
        加载和管理Stable Diffusion XL模型
        
        配置不同的艺术风格（动漫、迪士尼、照片风等）
        
        实现多图片一致性生成：
        
        先生成前4张"锚点图片"（id_images）
        
        基于锚点图片生成剩余图片，保持风格和角色一致
        
        使用空间注意力确保角色外貌、场景风格统一
        
"""
class StoryDiffusionSynthesizer:

    def __init__(self,
                 num_pages: int,
                 height: int,
                 width: int,
                 model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 id_length: int = 4,
                 num_steps: int = 50):
        self.attn_args = {
            "attn_count": 0,
            "cur_step": 0,
            "total_count": 0,
        }
        self.sa32 = 0.5
        self.sa64 = 0.5
        self.id_length = id_length
        self.total_length = num_pages
        self.height = height
        self.width = width
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16
        self.num_steps = num_steps
        self.styles = {
            '(No style)': (
                '{prompt}',
                ''),
            'Japanese Anime': (
                'anime artwork illustrating {prompt}. created by japanese anime studio. highly emotional. best quality, high resolution, (Anime Style, Manga Style:1.3), Low detail, sketch, concept art, line art, webtoon, manhua, hand drawn, defined lines, simple shades, minimalistic, High contrast, Linear compositions, Scalable artwork, Digital art, High Contrast Shadows',
                'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'),
            'Digital/Oil Painting': (
                '{prompt} . (Extremely Detailed Oil Painting:1.2), glow effects, godrays, Hand drawn, render, 8k, octane render, cinema 4d, blender, dark, atmospheric 4k ultra detailed, cinematic sensual, Sharp focus, humorous illustration, big depth of field',
                'anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'),
            'Pixar/Disney Character': (
                'Create a Disney Pixar 3D style illustration on {prompt} . The scene is vibrant, motivational, filled with vivid colors and a sense of wonder.',
                'lowres, bad anatomy, bad hands, text, bad eyes, bad arms, bad legs, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, blurry, grayscale, noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo'),
            'Photographic': (
                'cinematic photo {prompt} . Hyperrealistic, Hyperdetailed, detailed skin, matte skin, soft lighting, realistic, best quality, ultra realistic, 8k, golden ratio, Intricate, High Detail, film photography, soft focus',
                'drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'),
            'Comic book': (
                'comic {prompt} . graphic illustration, comic art, graphic novel art, vibrant, highly detailed',
                'photograph, deformed, glitch, noisy, realistic, stock photo, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'),
            'Line art': (
                'line art drawing {prompt} . professional, sleek, modern, minimalist, graphic, line art, vector graphics',
                'anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'),
            'Black and White Film Noir': (
                '{prompt} . (b&w, Monochromatic, Film Photography:1.3), film noir, analog style, soft lighting, subsurface scattering, realistic, heavy shadow, masterpiece, best quality, ultra realistic, 8k',
                'anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'),
            'Isometric Rooms': (
                'Tiny cute isometric {prompt} . in a cutaway box, soft smooth lighting, soft colors, 100mm lens, 3d blender render',
                'anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'),
            'Storybook': (
                "Cartoon style, cute illustration of {prompt}.",
                'realism, photo, realistic, lowres, bad hands, bad eyes, bad arms, bad legs, error, missing fingers, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, grayscale, noisy, sloppy, messy, grainy, ultra textured'
            )
        }

        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors=True
        )

        pipe = pipe.to(self.device)
        
        # pipe.id_encoder.to(self.device)

        pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(num_steps)
        unet = pipe.unet

        attn_procs = {}
        ### Insert PairedAttention
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None and (name.startswith("up_blocks") ) :
                attn_procs[name] = SpatialAttnProcessor2_0(
                    id_length=self.id_length,
                    device=self.device,
                    height=self.height,
                    width=self.width,
                    sa32=self.sa32,
                    sa64=self.sa64,
                    global_attn_args=self.attn_args
                )
                self.attn_args["total_count"] += 1
            else:
                attn_procs[name] = AttnProcessor()
        print("successsfully load consistent self-attention")
        print(f"number of the processor : {self.attn_args['total_count']}")
        # unet.set_attn_processor(copy.deepcopy(attn_procs))
        unet.set_attn_processor(attn_procs)
        mask1024, mask4096 = cal_attn_mask_xl(
            self.total_length,
            self.id_length,
            self.sa32,
            self.sa64,
            self.height,
            self.width,
            device=self.device,
            dtype=torch.float16,
        )

        self.attn_args.update({
            "mask1024": mask1024,
            "mask4096": mask4096
        })

        self.pipe = pipe
        self.negative_prompt = "naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation," \
                               "extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating" \
                               "limbs, disconnected limbs, blurry, watermarks, oversaturated, distorted hands, amputation"

    def set_attn_write(self,
                       value: bool):
        unet = self.pipe.unet
        for name, processor in unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if cross_attention_dim is None:
                if name.startswith("up_blocks") :
                    assert isinstance(processor, SpatialAttnProcessor2_0)
                    processor.write = value

    def apply_style(self, style_name: str, positives: list, negative: str = ""):
        p, n = self.styles.get(style_name, self.styles["(No style)"])
        return [p.replace("{prompt}", positive) for positive in positives], n + ' ' + negative
    
    def apply_style_positive(self, style_name: str, positive: str):
        p, n = self.styles.get(style_name, self.styles["(No style)"])
        return p.replace("{prompt}", positive) 
    
    def call(self,
             prompts: List[str],        
             input_id_images = None,
             start_merge_step = None,
             style_name: str = "Pixar/Disney Character",
             guidance_scale: float = 5.0,
             seed: int = 2047):
        assert len(prompts) == self.total_length, "The number of prompts should be equal to the number of pages."
        setup_seed(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        torch.cuda.empty_cache()

        id_prompts = prompts[:self.id_length]
        real_prompts = prompts[self.id_length:]
        self.set_attn_write(True)
        self.attn_args.update({
            "cur_step": 0,
            "attn_count": 0
        })
        id_prompts, negative_prompt = self.apply_style(style_name, id_prompts, self.negative_prompt)
        id_images = self.pipe(
            id_prompts,
            input_id_images=input_id_images,
            start_merge_step=start_merge_step,
            num_inference_steps=self.num_steps,
            guidance_scale=guidance_scale,
            height=self.height, 
            width=self.width,
            negative_prompt=negative_prompt,
            generator=generator).images
    
        self.set_attn_write(False)
        real_images = []
        for real_prompt in real_prompts:
            self.attn_args["cur_step"] = 0
            real_prompt = self.apply_style_positive(style_name, real_prompt)
            real_images.append(self.pipe(
                real_prompt,
                num_inference_steps=self.num_steps,
                guidance_scale=guidance_scale, 
                height=self.height, 
                width=self.width,
                negative_prompt=negative_prompt,
                generator=generator).images[0]
            )

        images = id_images + real_images             
        return images


"""
    故事扩散代理：
    
        作用：完整的本地故事图像生成流程控制器
        
        从故事文本到最终图像的端到端处理
        
        包含两个核心子任务：
        
        extract_role_from_story: 提取故事角色信息
        
        generate_image_prompt_from_story: 生成图像描述
        
        调用StoryDiffusionSynthesizer实际生成图片
        
        使用本地Stable Diffusion模型
"""
@register_tool("story_diffusion_t2i")
class StoryDiffusionAgent:

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        
    def call(self, params: Dict):
        pages: List = params["pages"]
        save_path: str = params["save_path"]
        role_dict = self.extract_role_from_story(pages)
        image_prompts = self.generate_image_prompt_from_story(pages)
        image_prompts_with_role_desc = []
        for image_prompt in image_prompts:
            for role, role_desc in role_dict.items():
                if role in image_prompt:
                    image_prompt = image_prompt.replace(role, role_desc)
            image_prompts_with_role_desc.append(image_prompt)
        generation_agent = StoryDiffusionSynthesizer(
            num_pages=len(pages),
            height=self.cfg.get("height", 512),
            width=self.cfg.get("width", 512),
            model_name=self.cfg.get("model_name", "stabilityai/stable-diffusion-xl-base-1.0"),
            id_length=self.cfg.get("id_length", 4),
            num_steps=self.cfg.get("num_steps", 50)
        )
        images = generation_agent.call(
            image_prompts_with_role_desc,
            style_name=params.get("style_name", "Storybook"),
            guidance_scale=params.get("guidance_scale", 5.0),
            seed=params.get("seed", 2047)
        )
        for idx, image in enumerate(images):
            image.save(save_path / f"p{idx + 1}.png")
        return {
            "prompts": image_prompts_with_role_desc,
            "generation_results": images,
        }
        
    def extract_role_from_story(
            self,
            pages: List,
        ):
        num_turns = self.cfg.get("num_turns", 3)
        role_extractor = init_tool_instance({
            "tool": self.cfg.get("llm", "qwen"),
            "cfg": {
                "system_prompt": role_extract_system,
                "track_history": False
            }
        })
        role_reviewer = init_tool_instance({
            "tool": self.cfg.get("llm", "qwen"),
            "cfg": {
                "system_prompt": role_review_system,
                "track_history": False
            }
        })
        roles = {}
        review = ""
        for turn in range(num_turns):
            roles, success = role_extractor.call(json.dumps({
                    "story_content": pages,
                    "previous_result": roles,
                    "improvement_suggestions": review,
                }, ensure_ascii=False
            ))
            roles = json.loads(roles.strip("```json").strip("```"))
            review, success = role_reviewer.call(json.dumps({
                "story_content": pages,
                "role_descriptions": roles
            }, ensure_ascii=False))
            if review == "Check passed.":
                break
        return roles

    def generate_image_prompt_from_story(
            self,
            pages: List,
            num_turns: int = 3
        ):
        image_prompt_reviewer = init_tool_instance({
            "tool": self.cfg.get("llm", "qwen"),
            "cfg": {
                "system_prompt": story_to_image_review_system,
                "track_history": False
            }
        })
        image_prompt_reviser = init_tool_instance({
            "tool": self.cfg.get("llm", "qwen"),
            "cfg": {
                "system_prompt": story_to_image_reviser_system,
                "track_history": False
            }
        })
        image_prompts = []

        for page in pages:
            review = ""
            image_prompt = ""
            for turn in range(num_turns):
                image_prompt, success = image_prompt_reviser.call(json.dumps({
                    "all_pages": pages,
                    "current_page": page,
                    "previous_result": image_prompt,
                    "improvement_suggestions": review,
                }, ensure_ascii=False))
                if image_prompt.startswith("Image description:"):
                    image_prompt = image_prompt[len("Image description:"):]
                review, success = image_prompt_reviewer.call(json.dumps({
                    "all_pages": pages,
                    "current_page": page,
                    "image_description": image_prompt
                }, ensure_ascii=False))
                if review == "Check passed.":
                    break
            image_prompts.append(image_prompt)
        return image_prompts


# ==================== 任务2：基于API的图像生成Agent ====================

@register_tool("dashscope_image_api")
class DashScopeImageAgent:
    """
    使用通义万相API生成图像的Agent
    优点：
    1. 无需下载大模型
    2. 速度快
    3. 质量稳定
    """
    
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.api_key = os.environ.get('DASHSCOPE_API_KEY')
        if not self.api_key:
            raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
    
    def generate_image_from_prompt(self, prompt: str, style: str = "auto") -> Image.Image:
        """
        一句话 ———— 直接传入提示词调用API即可
        （会返回一个可下载的链接，下载就完了，最后再转换格式、异常处理）

        使用DashScope API生成单张图像
        
        Args:
            prompt: 图像描述文本
            style: 图像风格
            
        Returns:
            PIL Image对象
        """
        import dashscope
        from dashscope import ImageSynthesis
        
        # 调用通义万相API
        try:
            response = ImageSynthesis.call(
                model='wanx-v1',  # 通义万相模型
                prompt=prompt,
                n=1,  # 生成1张图片
                size='1024*1024',  # 图片尺寸
                api_key=self.api_key
            )
            
            if response.status_code == 200:
                # 获取图片URL
                image_url = response.output['results'][0]['url']
                """
                response = {
                    'status_code': 200,
                    'output': {
                        'results': [
                            {
                                'url': 'https://example.com/generated_image.jpg'
                            }
                        ]
                    }
                }
                """
                
                # 下载图片
                img_response = requests.get(image_url, timeout=30)
                img = Image.open(BytesIO(img_response.content))
                # 转换格式
                
                print(f"✅ 图像生成成功: {prompt[:50]}...")
                return img
            else:
                print(f"❌ API调用失败: {response.message}")
                # 返回一个空白图像作为fallback
                return Image.new('RGB', (1024, 1024), color='gray')
                
        except Exception as e:
            print(f"❌ 图像生成出错: {e}")
            # 返回一个空白图像作为fallback
            return Image.new('RGB', (1024, 1024), color='gray')
    
    def call(self, params: Dict):
        """
        主调用方法
        
        Args:
            params: 包含pages和save_path的字典
            
        Returns:
            包含prompts和generation_results的字典
        """
        pages: List = params["pages"]
        save_path = params["save_path"]
        
        # 1. 提取角色信息（复用原有逻辑）
        print("🔍 正在提取故事中的角色...")
        role_dict = self.extract_role_from_story(pages)
        print(f"✅ 提取到 {len(role_dict)} 个角色: {list(role_dict.keys())}")
        
        # 2. 生成图像prompt（复用原有逻辑）
        print("📝 正在生成图像描述...")
        image_prompts = self.generate_image_prompt_from_story(pages)
        
        # 3. 替换角色描述
        image_prompts_with_role_desc = []
        for image_prompt in image_prompts:
            for role, role_desc in role_dict.items():
                if role in image_prompt:
                    image_prompt = image_prompt.replace(role, role_desc)
            image_prompts_with_role_desc.append(image_prompt)
        """
        role_dict = {
            "小明": "一个戴着眼镜的胖胖小男孩，穿着蓝色校服",
            "小红": "一个扎着马尾辫的瘦高女孩，穿着红色裙子"
        }
        
        image_prompts = [
            "小明在公园里踢足球",
            "小红在旁边为小明加油"
        ]
        
        替换：
        image_prompt = "小明在公园里踢足球"
        # 替换"小明" → "一个戴着眼镜的胖胖小男孩，穿着蓝色校服"
        """

        # 4. 使用API生成图像
        print(f"🎨 开始生成 {len(image_prompts_with_role_desc)} 张图像...")
        images = []
        for idx, prompt in enumerate(image_prompts_with_role_desc):
            print(f"生成第 {idx + 1}/{len(image_prompts_with_role_desc)} 张图像...")
            
            # 调用API生成图像
            img = self.generate_image_from_prompt(prompt)
            
            # 调整图像尺寸（如果配置中指定了）
            target_width = self.cfg.get("width", 1024)
            target_height = self.cfg.get("height", 512)
            if img.size != (target_width, target_height):
                img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            images.append(img)
            
            # 保存图像
            img.save(save_path / f"p{idx + 1}.png")
            print(f"💾 已保存: {save_path / f'p{idx + 1}.png'}")
            
            # 避免API限流，稍微延时
            if idx < len(image_prompts_with_role_desc) - 1:
                time.sleep(1)
        
        print("✅ 所有图像生成完成！")
        
        return {
            "prompts": image_prompts_with_role_desc,
            "generation_results": images,
        }
    
    # 复用原有的角色提取和prompt生成方法
    """
        角色提取函数
        创建了两个助手：角色提取助手、角色检查助手
        （利用base中的init_tool_instance，忘记了再复习）
        然后循环调用,
            角色检查给出建议（improvement_suggestions）
            把建议给角色提取助手生成新的角色信息
            检查助手给出  "Check passed." 跳出循环      
    """
    def extract_role_from_story(self, pages: List):
        """提取故事中的角色（复用StoryDiffusionAgent的逻辑）"""
        num_turns = self.cfg.get("num_turns", 3)
        role_extractor = init_tool_instance({
            "tool": self.cfg.get("llm", "qwen"),
            "cfg": {
                "system_prompt": role_extract_system,
                "track_history": False
            }
        })
        role_reviewer = init_tool_instance({
            "tool": self.cfg.get("llm", "qwen"), # 从配置中获取LLM（大语言模型）的设置，如果没有找到，就使用默认值"qwen"。
            "cfg": {
                "system_prompt": role_review_system,
                "track_history": False
            }
        })
        roles = {}
        review = ""
        for turn in range(num_turns):  # dump相当于转为AI能看懂的json格式
            roles, success = role_extractor.call(json.dumps({
                    "story_content": pages,
                    "previous_result": roles,
                    "improvement_suggestions": review,
                }, ensure_ascii=False
            ))
            # strip:移除开头与结尾的……
            roles = json.loads(roles.strip("```json").strip("```"))
            review, success = role_reviewer.call(json.dumps({
                "story_content": pages,
                "role_descriptions": roles
            }, ensure_ascii=False))
            if review == "Check passed.":
                break
        return roles


    """
        跟上一个函数 extract_role_from_story 一样一样的。。。
        为故事书的每一页生成高质量的图像描述（prompt），通过"修订-检查"的循环机制确保描述准确且详细。
    """
    def generate_image_prompt_from_story(self, pages: List, num_turns: int = 3):
        """从故事生成图像prompt（复用StoryDiffusionAgent的逻辑）"""
        image_prompt_reviewer = init_tool_instance({
            "tool": self.cfg.get("llm", "qwen"),
            "cfg": {
                "system_prompt": story_to_image_review_system,
                "track_history": False
            }
        })
        image_prompt_reviser = init_tool_instance({
            "tool": self.cfg.get("llm", "qwen"),
            "cfg": {
                "system_prompt": story_to_image_reviser_system,
                "track_history": False
            }
        })
        image_prompts = []

        for page in pages:
            review = ""
            image_prompt = ""
            for turn in range(num_turns):
                image_prompt, success = image_prompt_reviser.call(json.dumps({
                    "all_pages": pages,
                    "current_page": page,
                    "previous_result": image_prompt,
                    "improvement_suggestions": review,
                }, ensure_ascii=False))
                if image_prompt.startswith("Image description:"):
                    image_prompt = image_prompt[len("Image description:"):]
                review, success = image_prompt_reviewer.call(json.dumps({
                    "all_pages": pages,
                    "current_page": page,
                    "image_description": image_prompt
                }, ensure_ascii=False))
                if review == "Check passed.":
                    break
            image_prompts.append(image_prompt)
        return image_prompts

