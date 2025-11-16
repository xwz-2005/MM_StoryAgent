import argparse
import yaml
from mm_story_agent import MMStoryAgent
from  mm_story_agent.utils.visualization import VisualizationTool
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)

    args = parser.parse_args()

    with open(args.config, "r", encoding='utf-8') as reader:
        config = yaml.load(reader, Loader=yaml.FullLoader)
    
    mm_story_agent = MMStoryAgent()
    mm_story_agent.call(config)
     # 初始化故事生成代理（已集成可视化工具）
    mm_story_agent = MMStoryAgent()
    # 启动主流程（包含新增的图表生成菜单）
    mm_story_agent.call(config)
