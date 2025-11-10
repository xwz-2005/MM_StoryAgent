import time
import json
from pathlib import Path

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

from .base import init_tool_instance





class MMStoryAgent:

    def __init__(self) -> None:
        # 任务要求：去除音乐和音效模块，只保留图像和语音
        self.modalities = ["image", "speech"]

    # 调用指定模态的代理处理任务，并将结果收集到返回字典中。
    def call_modality_agent(self, modality, agent, params, return_dict):
        result = agent.call(params)
        return_dict[modality] = result

    def write_story(self, config):
        cfg = config["story_writer"]
        story_writer = init_tool_instance(cfg) #都是先init再call
        pages = story_writer.call(cfg["params"])
        return pages

    # 并行生成图像和语音
    def generate_modality_assets(self, config, pages):
        script_data = {"pages": [{"story": page} for page in pages]}
        # script_data = {
        #     "pages": [
        #         {
        #             "story": "第一页：从前有座山，山里有个庙。",
        #             "image_prompt": "一座古老的山和寺庙",  # 后来添加的
        #             "sound_prompt": "风声和钟声"  # 后来添加的
        #         },
        #         {
        #             "story": "第二页：庙里有个老和尚在讲故事。",
        #             "image_prompt": "慈祥的老和尚",  # 后来添加的
        #             "sound_prompt": "讲故事的声音"  # 后来添加的
        #         },
        #         {
        #             "story": "第三页：他讲的故事是：从前有座山……"
        #             # 其他素材也会陆续添加进来...
        #         }
        #     ],
        #     "music_prompt": "整个故事的背景音乐提示词"
        # }

        # 在配置文件yaml中开头有
        story_dir = Path(config["story_dir"])

        for sub_dir in self.modalities:
            (story_dir / sub_dir).mkdir(exist_ok=True, parents=True)
            # 文件夹存在不要报错，如果父文件夹不存在，自动创建父文件夹

        agents = {}
        params = {}
        for modality in self.modalities:
            agents[modality] = init_tool_instance(config[modality + "_generation"])
            params[modality] = config[modality + "_generation"]["params"].copy()
            params[modality].update({
                "pages": pages,
                "save_path": story_dir / modality
            })

        processes = []
        return_dict = mp.Manager().dict()

        # 创建并启动进程
        for modality in self.modalities:
            p = mp.Process(
                target=self.call_modality_agent,  # 其实每个过程都与compose_storytelling_video()一样！这个看明白了其它都懂了
                args=(
                    modality,
                    agents[modality],
                    params[modality],
                    return_dict)
                )
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()

        images = None  # 初始化images变量，避免未定义错误
        for modality, result in return_dict.items():# 这里看上面的pages注释就能看懂的
            try:
                if modality == "image":
                    images = result["generation_results"]
                    for idx in range(len(pages)):
                        script_data["pages"][idx]["image_prompt"] = result["prompts"][idx]
                # 任务要求：已移除 sound 和 music 模态的处理
            except Exception as e:
                print(f"Error occurred during generation: {e}")
        
        with open(story_dir / "script_data.json", "w") as writer:
            json.dump(script_data, writer, ensure_ascii=False, indent=4)
        # indent=4让JSON数据自动缩进4
        # ensure_ascii=False：支持中文等非ASCII字符
        
        return images
    
    def compose_storytelling_video(self, config, pages):
        video_compose_agent = init_tool_instance(config["video_compose"])
        params = config["video_compose"]["params"].copy()
        params["pages"] = pages
        video_compose_agent.call(params)

    def call(self, config):
        pages = self.write_story(config)
        images = self.generate_modality_assets(config, pages)
        self.compose_storytelling_video(config, pages)
