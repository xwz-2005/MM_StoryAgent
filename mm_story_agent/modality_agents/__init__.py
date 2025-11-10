import sys

from ..utils.import_utils import _LazyModule

_import_structure = {
    'story_agent': [
        'QAOutlineStoryWriter',
    ],
    'music_agent': [
        'MusicGenAgent'
    ],
    'sound_agent': [
        'AudioLDM2Agent'
    ],
    'speech_agent': [
        'CosyVoiceAgent',
        'EdgeTTSAgent'  # 免费Edge-TTS语音合成
    ],
    'image_agent': [
        'StoryDiffusionAgent',
        'DashScopeImageAgent'  # 任务2：新增API图像生成
    ],
    'llm': [
        'QwenAgent'
    ],
    "freesound_agent": [
        "FreesoundSfxAgent",
        "FreesoundMusicAgent"
    ]
}

sys.modules[__name__] = _LazyModule(
    __name__,
    globals()['__file__'],
    _import_structure,
    module_spec=__spec__,
)