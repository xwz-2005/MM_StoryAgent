#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语言切换工具
用法: python switch_language.py zh 或 python switch_language.py en
"""

import sys
import yaml
import os
from pathlib import Path

def update_imports(language):
    """更新所有代理文件的导入语句"""
    
    # 需要修改的文件和对应的导入行
    files_to_update = {
        "mm_story_agent/modality_agents/story_agent.py": {
            "old_patterns": ["from ..prompts_en import", "from ..prompts_zh import"],
            "new": f"from ..prompts_{language} import"
        },
        "mm_story_agent/modality_agents/image_agent.py": {
            "old_patterns": ["from mm_story_agent.prompts_en import", "from mm_story_agent.prompts_zh import"],
            "new": f"from mm_story_agent.prompts_{language} import"
        },
        "mm_story_agent/modality_agents/music_agent.py": {
            "old_patterns": ["from mm_story_agent.prompts_en import", "from mm_story_agent.prompts_zh import"],
            "new": f"from mm_story_agent.prompts_{language} import"
        },
        "mm_story_agent/modality_agents/sound_agent.py": {
            "old_patterns": ["from mm_story_agent.prompts_en import", "from mm_story_agent.prompts_zh import"],
            "new": f"from mm_story_agent.prompts_{language} import"
        },
        "mm_story_agent/modality_agents/freesound_agent.py": {
            "old_patterns": ["from ..prompts_en import", "from ..prompts_zh import"],
            "new": f"from ..prompts_{language} import"
        }
    }
    
    for file_path, changes in files_to_update.items():
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否已经是目标语言
            if changes["new"] in content:
                print(f"✓ {file_path} 已经是 {language} 模式")
                continue
            
            # 尝试替换所有可能的模式
            updated = False
            for old_pattern in changes["old_patterns"]:
                if old_pattern in content:
                    content = content.replace(old_pattern, changes["new"])
                    updated = True
                    break
            
            if updated:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✓ 已更新 {file_path}")
            else:
                print(f"⚠ {file_path} 中未找到需要替换的内容")
        else:
            print(f"✗ 文件不存在: {file_path}")

def update_config(language):
    """更新配置文件"""
    
    # 加载语言配置
    with open('language_config.yaml', 'r', encoding='utf-8') as f:
        lang_config = yaml.load(f, Loader=yaml.FullLoader)
    
    config_data = lang_config[language]
    
    # 更新主配置文件
    config_files = [
        "configs/mm_story_agent.yaml",
        "configs/mm_story_agent_no_music.yaml", 
        "configs/mm_story_agent_api.yaml"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            
            # 更新故事参数
            if 'story_writer' in config and 'params' in config['story_writer']:
                config['story_writer']['params']['story_topic'] = config_data['story_topic']
                config['story_writer']['params']['main_role'] = config_data['main_role']
                config['story_writer']['params']['scene'] = config_data['scene']
            
            # 更新语音参数
            if 'speech_generation' in config and 'params' in config['speech_generation']:
                config['speech_generation']['params']['voice'] = config_data['voice']
                # 统一使用 Edge-TTS（避免 CosyVoice 试用过期问题）
                config['speech_generation']['tool'] = 'edge_tts'
            
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            print(f"✓ 已更新 {config_file}")

def main():
    if len(sys.argv) != 2:
        print("用法: python switch_language.py <zh|en>")
        print("示例: python switch_language.py zh  # 切换到中文")
        print("示例: python switch_language.py en  # 切换到英文")
        sys.exit(1)
    
    language = sys.argv[1].lower()
    
    if language not in ['zh', 'en']:
        print("错误: 语言参数只能是 'zh' 或 'en'")
        sys.exit(1)
    
    print(f"正在切换到 {language.upper()} 模式...")
    print()
    
    # 更新导入语句
    print("1. 更新代理文件导入...")
    update_imports(language)
    print()
    
    # 更新配置文件
    print("2. 更新配置文件...")
    update_config(language)
    print()
    
    print(f"✅ 已成功切换到 {language.upper()} 模式！")
    print()
    print("现在可以运行:")
    print("  python run.py -c configs/mm_story_agent_api.yaml")

if __name__ == "__main__":
    main()
