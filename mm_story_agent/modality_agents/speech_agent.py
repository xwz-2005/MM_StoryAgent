import os
import json
from pathlib import Path
from typing import List, Dict
import asyncio

from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
import nls

from mm_story_agent.base import register_tool

"""
    这个文件实现了一个文本转语音(TTS)系统，将故事文本转换为语音旁白，使用阿里云的CosyVoice服务。
"""
# Due to the trouble regarding environment, we use dashscope to deploy and call the API for CosyVoice.
class CosyVoiceSynthesizer:

    def __init__(self) -> None:
        self.access_key_id = os.environ.get('ALIYUN_ACCESS_KEY_ID')
        self.access_key_secret = os.environ.get('ALIYUN_ACCESS_KEY_SECRET')
        self.app_key = os.environ.get('ALIYUN_APP_KEY')
        self.setup_token()

    def setup_token(self):
        client = AcsClient(self.access_key_id, self.access_key_secret,
                           'cn-shanghai')
        request = CommonRequest()
        request.set_method('POST')
        request.set_domain('nls-meta.cn-shanghai.aliyuncs.com')
        request.set_version('2019-02-28')
        request.set_action_name('CreateToken')

        try:
            response = client.do_action_with_exception(request)
            jss = json.loads(response)
            if 'Token' in jss and 'Id' in jss['Token']:
                token = jss['Token']['Id']
                self.token = token
        except Exception as e:
            import traceback
            raise RuntimeError(
                f'Request token failed with error: {e}, with detail {traceback.format_exc()}'
            )

    def call(self, save_file, transcript, voice="longyuan", sample_rate=16000):
        writer = open(save_file, "wb")
        return_data = b''

        def write_data(data, *args):
            nonlocal return_data
            return_data += data
            if writer is not None:
                writer.write(data)

        def raise_error(error, *args):
            raise RuntimeError(
                f'Synthesizing speech failed with error: {error}')

        def close_file(*args):
            if writer is not None:
                writer.close()

        # 修复：使用正确的API - NlsSpeechSynthesizer
        # 启用 long_tts=True 以支持 CosyVoice 音色（如 longyuan）
        sdk = nls.NlsSpeechSynthesizer(
            url='wss://nls-gateway.cn-shanghai.aliyuncs.com/ws/v1',
            token=self.token,
            appkey=self.app_key,
            long_tts=True,  # 启用长文本/CosyVoice支持
            on_data=write_data,
            on_error=raise_error,
            on_close=close_file,
        )

        # 修复：使用 start() 方法，传入完整文本
        sdk.start(
            text=transcript,
            voice=voice,
            aformat='wav',
            sample_rate=sample_rate,
            wait_complete=True
        )


@register_tool("cosyvoice_tts")
class CosyVoiceAgent:

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def call(self, params: Dict):
        pages: List = params["pages"]
        save_path: str = params["save_path"]
        generation_agent = CosyVoiceSynthesizer()

        for idx, page in enumerate(pages):
            generation_agent.call(
                save_file=save_path / f"p{idx + 1}.wav",
                transcript=page,
                voice=params.get("voice", "longyuan"),
                sample_rate=self.cfg.get("sample_rate", 16000)
            )

        return {
            "modality": "speech"
        }


# ==================== 免费 Edge-TTS 语音合成 ====================

class EdgeTTSSynthesizer:
    """
    使用微软 Edge-TTS 的免费语音合成
    优点：
    1. 完全免费
    2. 无需API密钥
    3. 质量高
    4. 支持多语言
    """
    
    def __init__(self) -> None:
        pass
    
    async def _synthesize(self, text: str, voice: str, output_file: str):
        """异步合成语音"""
        import edge_tts
        
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)
    
    def call(self, save_file, transcript, voice="en-US-AriaNeural", sample_rate=16000):
        """
        同步调用接口
        
        常用语音：
        - 英文女声: en-US-AriaNeural
        - 英文男声: en-US-GuyNeural
        - 中文女声: zh-CN-XiaoxiaoNeural
        - 中文男声: zh-CN-YunxiNeural
        """
        # 安全地运行异步函数，避免事件循环关闭异常
        try:
            # 尝试获取当前事件循环
            loop = asyncio.get_event_loop()
            # 检查事件循环是否已关闭
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # 如果没有事件循环，创建一个新的
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            # 在当前事件循环中运行异步任务
            loop.run_until_complete(self._synthesize(transcript, voice, str(save_file)))
        finally:
            # 不要在子进程中关闭事件循环，让进程自然结束
            # 这样可以避免与多进程环境中的事件循环管理冲突
            pass


@register_tool("edge_tts")
class EdgeTTSAgent:
    """使用 Edge-TTS 的语音合成 Agent"""
    
    def __init__(self, cfg) -> None:
        self.cfg = cfg
    
    def call(self, params: Dict):
        pages: List = params["pages"]
        save_path: str = params["save_path"]
        generation_agent = EdgeTTSSynthesizer()
        
        for idx, page in enumerate(pages):
            generation_agent.call(
                save_file=save_path / f"p{idx + 1}.wav",
                transcript=page,
                voice=params.get("voice", "en-US-AriaNeural"),
                sample_rate=self.cfg.get("sample_rate", 16000)
            )
        
        return {
            "modality": "speech"
        }