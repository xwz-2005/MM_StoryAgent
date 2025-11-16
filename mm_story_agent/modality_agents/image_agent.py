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
    
    def _get_user_choice(self, prompt="请选择操作: ", valid_options=['1', '2', '3'], max_attempts=5):
        """获取并验证用户选择，支持自定义选项和最大尝试次数"""
        attempts = 0
        while attempts < max_attempts:
            try:
                # 显示提示
                print(prompt)
                
                # 获取用户输入
                choice = input().strip().lower()
                
                # 支持帮助命令
                if choice in ['help', '帮助', '?']:
                    self._show_help()
                    continue
                    
                if choice in valid_options:
                    return choice
                else:
                    print(f"❌ 请输入正确选项 ({'/'.join(valid_options)})")
                    attempts += 1
                    if attempts >= max_attempts:
                        print(f"⚠️  多次输入错误，将使用默认选项: {valid_options[0]}")
                        return valid_options[0]
            except EOFError:
                # 处理EOF错误（非交互式环境）
                print("\n⚠️ 在非交互式环境中运行，使用默认选项")
                return valid_options[0]
            except KeyboardInterrupt:
                print("\n👋 用户中断操作，再见!")
                exit()
            except Exception as e:
                print(f"❌ 输入错误: {str(e)}，请重试")
                attempts += 1
        return valid_options[0]  # 默认返回第一个选项
    
    def _show_help(self):
        """显示帮助信息"""
        print("\n📚 图像生成系统帮助:")
        print("   • 在任何输入提示时，输入 'help'、'帮助' 或 '?' 可显示此帮助")
        print("   • 提示词优化技巧:")
        print("     - 详细描述角色特征、场景环境、光线效果和风格要求")
        print("     - 明确指定图像比例和构图要求")
        print("     - 添加情感和氛围描述以增强表现力")
        print("   • 图像生成注意事项:")
        print("     - 保持角色描述一致性，确保故事连续性")
        print("     - 复杂场景可尝试分步骤优化提示词")
        print("     - 遇到生成问题时，可尝试简化或调整提示词")
        print("   • 操作提示:")
        print("     - 输入 '1' 通常表示确认/保留当前选项")
        print("     - 输入 '2' 通常表示重新生成/修改选项")
        print("     - 连续多次重新生成会触发参数调整建议")
        print("   • 提示词历史记录会保存在prompt_history.json文件中")
        print("   • 如遇严重问题，可按Ctrl+C中断程序")
        print()
    
    def _adjust_generation_params(self):
        """调整生成参数"""
        print("\n⚙️  图像生成参数调整")
        print("1. 增加提示词详细度")
        print("2. 调整图像风格")
        print("3. 修改图像比例")
        print("4. 返回")
        
        choice = self._get_user_choice(valid_options=['1', '2', '3', '4'])
        
        if choice == '1':
            print("🔍 已设置为增加提示词详细度")
            return {"detail_level": "high"}
        elif choice == '2':
            print("\n🎨 请选择图像风格:")
            print("1. 写实风格")
            print("2. 卡通风格")
            print("3. 油画风格")
            print("4. 水彩风格")
            print("5. 赛博朋克风格")
            print("6. 奇幻风格")
            
            style_choice = self._get_user_choice(valid_options=['1', '2', '3', '4', '5', '6'])
            styles = {
                '1': '写实风格',
                '2': '卡通风格',
                '3': '油画风格',
                '4': '水彩风格',
                '5': '赛博朋克风格',
                '6': '奇幻风格'
            }
            selected_style = styles.get(style_choice, '写实风格')
            print(f"✅ 已选择风格: {selected_style}")
            return {"style": selected_style}
        elif choice == '3':
            print("\n📐 请选择图像比例:")
            print("1. 16:9 (宽屏)")
            print("2. 4:3 (标准)")
            print("3. 1:1 (方形)")
            print("4. 9:16 (竖屏)")
            
            ratio_choice = self._get_user_choice(valid_options=['1', '2', '3', '4'])
            ratios = {
                '1': {'width': 1024, 'height': 576},
                '2': {'width': 1024, 'height': 768},
                '3': {'width': 1024, 'height': 1024},
                '4': {'width': 576, 'height': 1024}
            }
            selected_ratio = ratios.get(ratio_choice, {'width': 1024, 'height': 512})
            print(f"✅ 已选择比例: {selected_ratio['width']}x{selected_ratio['height']}")
            return selected_ratio
        
        return None
    
    def _get_modification_input(self, current_prompt):
        """获取用户修改输入并提供指导"""
        print("\n✏️  提示词修改器")
        print("💡 修改建议:")
        print("   • 增加角色外观细节: '主角穿着蓝色外套，有棕色头发'")
        print("   • 调整场景环境: '在阳光明媚的公园里，有樱花树'")
        print("   • 指定光线效果: '温暖的午后阳光，柔和的阴影'")
        print("   • 添加情感氛围: '欢快的氛围，充满希望的场景'")
        print("   • 输入 'cancel' 取消修改")
        print("   • 输入 'help' 获取更多帮助")
        
        while True:
            print("\n当前提示词:")
            print(f"   {current_prompt[:100]}..." if len(current_prompt) > 100 else f"   {current_prompt}")
            
            new_prompt = input("\n请输入修改后的提示词: ").strip()
            
            if new_prompt.lower() == 'help':
                self._show_help()
                continue
            elif new_prompt.lower() == 'cancel':
                return None
            elif not new_prompt:
                print("⚠️  提示词不能为空，请重新输入")
            elif len(new_prompt) < 10:
                print("⚠️  提示词过于简短，请提供更详细的描述")
            else:
                return new_prompt
    
    def call(self, params: Dict):
        """
        主调用方法 - 增强版交互式提示词编辑和图像重生成功能
        
        Args:
            params: 包含pages和save_path的字典
            
        Returns:
            包含prompts和generation_results的字典
        """
        print(f"\n{'✨'*40}")
        print(f"🖼️  启动交互式图像生成与优化系统")
        print(f"{'✨'*40}")
        
        # 显示系统欢迎信息
        print("\n👋 欢迎使用交互式图像生成系统!")
        print("📝 系统将帮助您为故事生成高质量图像")
        print("💡 提示: 输入 'help' 在任何时候获取帮助")
        print(f"{'='*40}")
        
        # 初始化提示词历史记录
        prompt_history = []
        generation_stats = {
            "total_pages": len(params["pages"]),
            "success_count": 0,
            "error_count": 0,
            "optimization_rounds": 0
        }
        
        # 1. 提取角色信息（复用原有逻辑）
        print("\n🔍 正在提取故事中的角色信息...")
        try:
            start_time = time.time()
            role_dict = self.extract_role_from_story(params["pages"])
            end_time = time.time()
            
            print(f"✅ 角色提取完成 (耗时: {end_time - start_time:.2f} 秒)")
            print(f"   共提取到 {len(role_dict)} 个角色")
            
            # 显示角色详情
            if role_dict:
                print("\n👥 角色详情:")
                for role, details in role_dict.items():
                    # 检查details的类型，确保安全访问
                    if isinstance(details, dict):
                        description = details.get('description', '无描述')
                    elif isinstance(details, str):
                        # 如果details是字符串，直接使用它作为描述
                        description = details
                    else:
                        description = '无描述'
                    print(f"   • {role}: {description[:100]}..." if len(description) > 100 else f"   • {role}: {description}")
        except Exception as e:
            print(f"❌ 角色提取失败: {str(e)}")
            print("⚠️  将继续使用默认角色处理")
            role_dict = {}
        
        # 2. 生成图像prompt（复用原有逻辑）
        print("\n📝 正在为故事生成图像描述...")
        try:
            start_time = time.time()
            image_prompts = self.generate_image_prompt_from_story(params["pages"])
            end_time = time.time()
            print(f"✅ 提示词生成完成 (耗时: {end_time - start_time:.2f} 秒)")
        except Exception as e:
            print(f"❌ 提示词生成失败: {str(e)}")
            print("❌ 系统无法继续，请检查配置和网络连接")
            return {"error": "Prompt generation failed"}
        
        # 存储处理后的提示词和图像
        image_prompts_with_role_desc = []
        generation_results = []
        generation_params = {}
        
        # 逐个页面进行交互式处理
        for idx, (page, initial_prompt) in enumerate(zip(params["pages"], image_prompts)):
            print(f"\n{'='*60}")
            print(f"🖼️  处理第 {idx + 1}/{len(params['pages'])} 张图像")
            print(f"{'='*60}")
            
            # 显示当前页面内容摘要
            page_summary = page[:100] + "..." if len(page) > 100 else page
            print(f"\n📄 页面内容: {page_summary}")
            
            prompt_finalized = False
            retry_count = 0
            max_retries = 3
            current_prompt = initial_prompt
            
            while not prompt_finalized and retry_count < max_retries:
                # 3. 将角色描述替换为更详细的描述
                enhanced_prompt = current_prompt
                for role, role_desc in role_dict.items():
                    if role in current_prompt:
                        enhanced_prompt = enhanced_prompt.replace(role, role_desc)
                
                # 根据生成参数调整提示词
                if generation_params:
                    if generation_params.get("style"):
                        enhanced_prompt = f"{enhanced_prompt}, {generation_params['style']}"
                    if generation_params.get("detail_level") == "high":
                        enhanced_prompt = f"{enhanced_prompt}, 高细节，清晰，精确描述"
                
                # 显示提示词
                print(f"\n💬 生成的提示词:")
                if len(enhanced_prompt) > 150:
                    print(f"   {enhanced_prompt[:150]}...")
                    print(f"   [提示词过长，总长度: {len(enhanced_prompt)} 字符]")
                else:
                    print(f"   {enhanced_prompt}")
                
                # 提供提示词操作选项
                print("\n🔧 请选择提示词操作:")
                print("1. 保留当前提示词")
                print("2. 修改提示词")
                print("3. 重新生成提示词")
                print("4. 调整生成参数")
                print("5. 查看帮助")
                
                choice = self._get_user_choice(valid_options=['1', '2', '3', '4', '5'])
                
                if choice == '1':
                    # 保留当前提示词
                    print("✅ 提示词已确认")
                    image_prompts_with_role_desc.append(enhanced_prompt)
                    prompt_history.append({
                        "page_index": idx,
                        "action": "保留提示词",
                        "prompt": enhanced_prompt,
                        "params": generation_params.copy(),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    prompt_finalized = True
                    generation_stats["success_count"] += 1
                    
                elif choice == '2':
                    # 修改提示词
                    new_prompt = self._get_modification_input(enhanced_prompt)
                    
                    if new_prompt:
                        # 对修改后的提示词也进行角色增强
                        enhanced_new_prompt = new_prompt
                        for role, role_desc in role_dict.items():
                            if role in new_prompt:
                                enhanced_new_prompt = enhanced_new_prompt.replace(role, role_desc)
                        
                        # 根据当前参数调整
                        if generation_params:
                            if generation_params.get("style"):
                                enhanced_new_prompt = f"{enhanced_new_prompt}, {generation_params['style']}"
                            if generation_params.get("detail_level") == "high":
                                enhanced_new_prompt = f"{enhanced_new_prompt}, 高细节，清晰，精确描述"
                        
                        print(f"\n✅ 修改后的提示词:")
                        print(f"   {enhanced_new_prompt[:150]}..." if len(enhanced_new_prompt) > 150 else f"   {enhanced_new_prompt}")
                        
                        # 确认修改
                        print("\n🔧 确认修改:")
                        print("1. 确认使用修改后的提示词")
                        print("2. 重新修改")
                        print("3. 放弃修改，使用原始提示词")
                        
                        confirm_choice = self._get_user_choice(valid_options=['1', '2', '3'])
                        
                        if confirm_choice == '1':
                            image_prompts_with_role_desc.append(enhanced_new_prompt)
                            prompt_history.append({
                                "page_index": idx,
                                "action": "修改提示词",
                                "original_prompt": enhanced_prompt,
                                "modified_prompt": enhanced_new_prompt,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            })
                            prompt_finalized = True
                            generation_stats["optimization_rounds"] += 1
                            generation_stats["success_count"] += 1
                        elif confirm_choice == '2':
                            # 重新修改提示词
                            retry_modification = True
                            while retry_modification:
                                # 重新获取用户修改的提示词
                                retry_new_prompt = self._get_modification_input(enhanced_prompt)
                                if retry_new_prompt:
                                    # 对重新修改后的提示词进行角色增强
                                    enhanced_retry_new_prompt = retry_new_prompt
                                    for role, role_desc in role_dict.items():
                                        if role in retry_new_prompt:
                                            enhanced_retry_new_prompt = enhanced_retry_new_prompt.replace(role, role_desc)
                                    
                                    # 根据当前参数调整
                                    if generation_params:
                                        if generation_params.get("style"):
                                            enhanced_retry_new_prompt = f"{enhanced_retry_new_prompt}, {generation_params['style']}"
                                        if generation_params.get("detail_level") == "high":
                                            enhanced_retry_new_prompt = f"{enhanced_retry_new_prompt}, 高细节，清晰，精确描述"
                                    
                                    print(f"\n✅ 修改后的提示词:")
                                    print(f"   {enhanced_retry_new_prompt[:150]}..." if len(enhanced_retry_new_prompt) > 150 else f"   {enhanced_retry_new_prompt}")
                                    
                                    # 再次确认修改
                                    print("\n🔧 确认修改:")
                                    print("1. 确认使用修改后的提示词")
                                    print("2. 重新修改")
                                    print("3. 放弃修改，使用原始提示词")
                                    
                                    retry_confirm_choice = self._get_user_choice(valid_options=['1', '2', '3'])
                                    
                                    if retry_confirm_choice == '1':
                                        image_prompts_with_role_desc.append(enhanced_retry_new_prompt)
                                        prompt_history.append({
                                            "page_index": idx,
                                            "action": "修改提示词",
                                            "original_prompt": enhanced_prompt,
                                            "modified_prompt": enhanced_retry_new_prompt,
                                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                                        })
                                        prompt_finalized = True
                                        generation_stats["optimization_rounds"] += 1
                                        generation_stats["success_count"] += 1
                                        retry_modification = False
                                    elif retry_confirm_choice == '3':
                                        image_prompts_with_role_desc.append(enhanced_prompt)
                                        prompt_history.append({
                                            "page_index": idx,
                                            "action": "放弃修改，保留提示词",
                                            "prompt": enhanced_prompt,
                                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                                        })
                                        prompt_finalized = True
                                        generation_stats["success_count"] += 1
                                        retry_modification = False
                                    # 选择2将继续循环
                                else:
                                    # 如果用户取消修改
                                    retry_modification = False
                        elif confirm_choice == '3':
                            image_prompts_with_role_desc.append(enhanced_prompt)
                            prompt_history.append({
                                "page_index": idx,
                                "action": "放弃修改，保留提示词",
                                "prompt": enhanced_prompt,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            })
                            prompt_finalized = True
                            generation_stats["success_count"] += 1
                    
                elif choice == '3':
                    # 重新生成提示词
                    retry_count += 1
                    print(f"🔄 正在重新生成提示词... (重试 {retry_count}/{max_retries})")
                    generation_stats["optimization_rounds"] += 1
                    
                    try:
                        # 为特定页面重新生成提示词
                        new_prompt_list = self.generate_image_prompt_from_story([page])
                        if new_prompt_list:
                            current_prompt = new_prompt_list[0]
                            print("✅ 新提示词已生成")
                        else:
                            print("⚠️  无法生成新提示词，使用原提示词")
                    except Exception as e:
                        print(f"❌ 提示词生成失败: {str(e)}")
                    
                    # 连续重新生成时，询问是否调整参数
                    if retry_count >= 2:
                        print("\n⚠️  您已连续多次选择重新生成提示词")
                        print("🔧 是否调整生成参数？")
                        print("1. 是")
                        print("2. 否")
                        
                        param_choice = self._get_user_choice(valid_options=['1', '2'])
                        if param_choice == '1':
                            new_params = self._adjust_generation_params()
                            if new_params:
                                generation_params.update(new_params)
                
                elif choice == '4':
                    # 调整生成参数
                    new_params = self._adjust_generation_params()
                    if new_params:
                        generation_params.update(new_params)
                
                elif choice == '5':
                    # 显示帮助
                    self._show_help()
            
            if retry_count >= max_retries:
                # 达到最大重试次数，使用最后生成的提示词
                print("⚠️  已达到最大重试次数，使用当前生成的提示词")
                image_prompts_with_role_desc.append(enhanced_prompt)
                prompt_history.append({
                    "page_index": idx,
                    "action": "达到最大重试次数，使用提示词",
                    "prompt": enhanced_prompt,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                generation_stats["error_count"] += 1
        
        # 4. 使用API生成图像
        print("\n🎨 开始生成图像...")
        save_path = params["save_path"]
        
        # 确保保存路径存在
        os.makedirs(save_path, exist_ok=True)
        
        # 逐个生成图像，允许用户对每个图像进行交互
        for idx, prompt in enumerate(image_prompts_with_role_desc):
            image_accepted = False
            image_retry_count = 0
            max_image_retries = 3
            error_history = []
            
            while not image_accepted and image_retry_count < max_image_retries:
                print(f"\n{'='*60}")
                print(f"🖼️  生成第 {idx + 1}/{len(image_prompts_with_role_desc)} 张图像")
                print(f"{'='*60}")
                print(f"💬 使用提示词: {prompt[:50]}...")
                print(f"   提示词长度: {len(prompt)} 字符")
                
                # 生成单张图像
                try:
                    start_time = time.time()
                    print("\n⏳ 正在调用AI图像生成服务...")
                    print("   这可能需要几秒钟时间，请耐心等待...")
                    
                    img = self.generate_image_from_prompt(prompt)
                    end_time = time.time()
                    
                    print(f"✅ 图像生成成功 (耗时: {end_time - start_time:.2f} 秒)")
                    
                    # 调整图像尺寸（如果配置中指定了）
                    target_width = generation_params.get("width", self.cfg.get("width", 1024))
                    target_height = generation_params.get("height", self.cfg.get("height", 512))
                    if img.size != (target_width, target_height):
                        print(f"📐 调整图像尺寸为 {target_width}x{target_height}")
                        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    
                    # 临时保存图像
                    temp_path = save_path / f"temp_p{idx + 1}.png"
                    img.save(temp_path)
                    print(f"   临时保存路径: {temp_path}")
                    
                    # 提供图像操作选项
                    print("\n🔧 请选择图像操作:")
                    print("1. 保留当前图像")
                    print("2. 重新生成图像")
                    print("3. 修改提示词")
                    print("4. 调整生成参数")
                    print("5. 查看帮助")
                    
                    image_choice = self._get_user_choice(valid_options=['1', '2', '3', '4', '5'])
                    
                    if image_choice == '1':
                        # 保存最终图像
                        final_path = save_path / f"p{idx + 1}.png"
                        img.save(final_path)
                        generation_results.append(img)
                        print(f"✅ 图像已接受并保存至: {final_path}")
                        # 删除临时文件
                        if temp_path.exists() and temp_path != final_path:
                            os.remove(temp_path)
                        image_accepted = True
                        generation_stats["success_count"] += 1
                        
                    elif image_choice == '2':
                        # 重新生成图像
                        image_retry_count += 1
                        print(f"🔄 正在重新生成图像... (重试 {image_retry_count}/{max_image_retries})")
                        generation_stats["optimization_rounds"] += 1
                        
                    elif image_choice == '3':
                        # 修改提示词
                        print("\n✏️  修改图像提示词:")
                        new_prompt = self._get_modification_input(prompt)
                        if new_prompt:
                            # 更新当前提示词
                            image_prompts_with_role_desc[idx] = new_prompt
                            prompt_history.append({
                                "page_index": idx,
                                "action": "图像生成阶段修改提示词",
                                "original_prompt": prompt,
                                "modified_prompt": new_prompt,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            })
                            print("✅ 提示词已更新")
                            # 重置重试次数
                            image_retry_count = 0
                            # 更新当前提示词
                            prompt = new_prompt
                    
                    elif image_choice == '4':
                        # 调整生成参数
                        new_params = self._adjust_generation_params()
                        if new_params:
                            generation_params.update(new_params)
                            print("✅ 参数已更新")
                    
                    elif image_choice == '5':
                        # 显示帮助
                        self._show_help()
                except Exception as e:
                    error_msg = str(e)
                    error_history.append(error_msg)
                    print(f"❌ 图像生成失败: {error_msg}")
                    image_retry_count += 1
                    generation_stats["error_count"] += 1
                    
                    # 分析错误并给出建议
                    if "API" in error_msg or "connection" in error_msg.lower():
                        print("💡 建议: 检查网络连接和API配置")
                    elif "timeout" in error_msg.lower():
                        print("💡 建议: 尝试使用更短的提示词")
                    elif "invalid" in error_msg.lower():
                        print("💡 建议: 检查提示词内容是否包含不支持的内容")
                    
                    # 连续失败时提供更多选项
                    if image_retry_count < max_image_retries:
                        print("\n🔧 请选择操作:")
                        print("1. 重试生成")
                        print("2. 修改提示词")
                        print("3. 调整生成参数")
                        
                        error_choice = self._get_user_choice(valid_options=['1', '2', '3'])
                        if error_choice == '2':
                            new_prompt = self._get_modification_input(prompt)
                            if new_prompt:
                                image_prompts_with_role_desc[idx] = new_prompt
                                prompt = new_prompt
                                image_retry_count = 0
                        elif error_choice == '3':
                            new_params = self._adjust_generation_params()
                            if new_params:
                                generation_params.update(new_params)
            
            if not image_accepted:
                # 达到最大重试次数，使用最后生成的图像
                print("⚠️  已达到最大重试次数，使用当前生成的图像")
                if 'img' in locals():
                    final_path = save_path / f"p{idx + 1}.png"
                    img.save(final_path)
                    generation_results.append(img)
                    print(f"   已保存至: {final_path}")
        
        # 保存提示词历史记录和生成统计
        try:
            # 保存提示词历史
            history_file = save_path / "prompt_history.json"
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(prompt_history, f, ensure_ascii=False, indent=2)
            print(f"\n💾 提示词历史已保存至: {history_file}")
            
            # 保存生成统计
            stats_file = save_path / "generation_stats.json"
            generation_stats["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            generation_stats["total_images"] = len(generation_results)
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(generation_stats, f, ensure_ascii=False, indent=2)
            print(f"💾 生成统计已保存至: {stats_file}")
            
        except Exception as e:
            print(f"⚠️  保存历史记录失败: {str(e)}")
        
        # 显示最终统计
        print(f"\n{'='*60}")
        print(f"🎉 图像生成和优化完成")
        print(f"{'='*60}")
        print(f"📊 生成统计:")
        print(f"   • 总页数: {generation_stats['total_pages']}")
        print(f"   • 成功生成图像数: {len(generation_results)}")
        print(f"   • 优化轮数: {generation_stats['optimization_rounds']}")
        print(f"   • 错误次数: {generation_stats['error_count']}")
        print(f"   • 图像保存路径: {save_path}")
        print(f"{'='*60}")
        
        return {
            "prompts": image_prompts_with_role_desc,
            "generation_results": generation_results,
            "stats": generation_stats,
            "history": prompt_history
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
            "tool": self.cfg.get("llm", "qwen"),
            "cfg": {
                "system_prompt": role_review_system,
                "track_history": False
            }
        })
        roles = {}
        review = ""
        for turn in range(num_turns):  # dump相当于转为AI能看懂的json格式
            try:
                # 调用角色提取器
                roles_result, success = role_extractor.call(json.dumps({
                        "story_content": pages,
                        "previous_result": roles,
                        "improvement_suggestions": review,
                    }, ensure_ascii=False
                ))
                
                # 确保结果是字符串类型
                if isinstance(roles_result, str):
                    # 清理和解析JSON
                    cleaned_result = roles_result.strip("```json").strip("```")
                    try:
                        roles = json.loads(cleaned_result)
                        # 确保解析后的结果是字典
                        if not isinstance(roles, dict):
                            roles = {}
                    except json.JSONDecodeError:
                        print(f"⚠️  角色提取器返回的JSON格式无效，使用空角色字典")
                        roles = {}
                else:
                    # 如果不是字符串，尝试转换或使用空字典
                    roles = {}
                
                # 调用角色审查器
                review, success = role_reviewer.call(json.dumps({
                    "story_content": pages,
                    "role_descriptions": roles
                }, ensure_ascii=False))
                
                if review == "Check passed.":
                    break
            except Exception as e:
                print(f"⚠️  角色提取过程中出错: {str(e)}")
                break
        
        # 确保返回的是字典类型
        if not isinstance(roles, dict):
            roles = {}
        
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

