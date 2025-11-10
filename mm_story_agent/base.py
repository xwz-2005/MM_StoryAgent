from abc import ABC

register_map = {
    'qwen': 'QwenAgent',
    'qa_outline_story_writer': 'QAOutlineStoryWriter',
    'musicgen_t2m': 'MusicGenAgent',
    'story_diffusion_t2i': 'StoryDiffusionAgent',
    'dashscope_image_api': 'DashScopeImageAgent',  # 任务2：新增API图像生成
    'cosyvoice_tts': 'CosyVoiceAgent',
    'edge_tts': 'EdgeTTSAgent',  # 免费Edge-TTS语音合成
    'audioldm2_t2a': 'AudioLDM2Agent',
    'slideshow_video_compose': 'SlideshowVideoComposeAgent',
    'freesound_sfx_retrieval': 'FreesoundSfxAgent',
    'freesound_music_retrieval': 'FreesoundMusicAgent',
}    


def import_from_register(key):
    value = register_map[key]
    if value == 'QAOutlineStoryWriter':
        from .modality_agents.story_agent import QAOutlineStoryWriter
        TOOL_REGISTRY[key] = QAOutlineStoryWriter
    elif value == 'MusicGenAgent':
        from .modality_agents.music_agent import MusicGenAgent
        TOOL_REGISTRY[key] = MusicGenAgent
    elif value == 'StoryDiffusionAgent':
        from .modality_agents.image_agent import StoryDiffusionAgent
        TOOL_REGISTRY[key] = StoryDiffusionAgent
    elif value == 'DashScopeImageAgent':
        from .modality_agents.image_agent import DashScopeImageAgent
        TOOL_REGISTRY[key] = DashScopeImageAgent
    elif value == 'CosyVoiceAgent':
        from .modality_agents.speech_agent import CosyVoiceAgent
        TOOL_REGISTRY[key] = CosyVoiceAgent
    elif value == 'EdgeTTSAgent':
        from .modality_agents.speech_agent import EdgeTTSAgent
        TOOL_REGISTRY[key] = EdgeTTSAgent
    elif value == 'AudioLDM2Agent':
        from .modality_agents.sound_agent import AudioLDM2Agent
        TOOL_REGISTRY[key] = AudioLDM2Agent
    elif value == 'SlideshowVideoComposeAgent':
        from .video_compose_agent import SlideshowVideoComposeAgent
        TOOL_REGISTRY[key] = SlideshowVideoComposeAgent
    elif value == 'FreesoundSfxAgent':
        from .modality_agents.freesound_agent import FreesoundSfxAgent
        TOOL_REGISTRY[key] = FreesoundSfxAgent
    elif value == 'FreesoundMusicAgent':
        from .modality_agents.freesound_agent import FreesoundMusicAgent
        TOOL_REGISTRY[key] = FreesoundMusicAgent
    elif value == 'QwenAgent':
        from .modality_agents.llm import QwenAgent
        TOOL_REGISTRY[key] = QwenAgent


class ToolRegistry(dict):

    def _import_key(self, key):
        try:
            import_from_register(key)
        except Exception as e:
            print(f'import {key} failed, details: {e}')

    def __getitem__(self, key):
        if key not in self.keys():
            self._import_key(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        self._import_key(key)
        return super().__contains__(key)
    

TOOL_REGISTRY = ToolRegistry()


def register_tool(name):

    def decorator(cls):
        TOOL_REGISTRY[name] = cls
        return cls
    
    return decorator


def init_tool_instance(cfg):
    return TOOL_REGISTRY[cfg["tool"]](cfg["cfg"])