from typing import Dict, Callable
import os

from dashscope import Generation

from mm_story_agent.base import register_tool


"""
    其实就是调用了千问的大模型，传入的参数看上去比较多
    这个文件创建了一个智能对话助手，可以调用阿里云的通义千问大模型，并进行对话历史管理和错误重试。
    
    通过系统指令让AI扮演不同专业角色
    在整个故事生成系统中作为"智能大脑"被各个模块使用（其它agent里面可以直接使用）
    
    参数：创造一个文艺青年还是严谨学术大师
    （希望你过几天还能记得住每个参数的意义）
    # 角色提取 - 需要稳定性
    role_extractor.call(
        prompt=story_text,
        temperature=0.3,    # 低创造性，确保角色提取准确
        top_p=0.8,          # 适当多样性
        seed=42             # 可重复结果
    )
    
    # 创意描述 - 需要创造性  
    image_describer.call(
        prompt=story_page, 
        temperature=1.0,    # 高创造性，生成生动描述
        top_p=0.95,         # 丰富词汇选择
        seed=123            # 仍保持一定稳定性
    )
"""

@register_tool("qwen")
class QwenAgent(object):

    # 初始化提示词、是否记录对话历史、对话历史
    def __init__(self,
                 config: Dict):
        
        self.system_prompt = config.get("system_prompt") # 系统角色指令
        track_history = config.get("track_history", False)   # 是否记录对话历史
        if self.system_prompt is None:
            self.history = []
        else:
            self.history = [
                {"role": "system", "content": self.system_prompt}
            ]
        self.track_history = track_history
    
    def basic_success_check(self, response):
        if not response or not response.output or not response.output.text:
            print(response)
            return False
        else:
            return True
    
    def call(self,
             prompt: str,
             model_name: str = "qwen2-72b-instruct",
             top_p: float = 0.95,
             temperature: float = 1.0,
             seed: int = 1,
             max_length: int = 1024,
             max_try: int = 5,
             success_check_fn: Callable = None
             ):

        # 1. 把用户问题添加到对话历史
        self.history.append({"role": "user", "content": prompt})

        success = False
        try_times = 0
        # 2. 重试循环
        while try_times < max_try:
            # 调用通义千问API
            response = Generation.call(
                model=model_name,  # 使用72B参数的大模型
                messages=self.history,  # 完整的对话历史
                top_p=top_p,  # 生成多样性控制
                temperature=temperature,  # 创造性控制
                api_key=os.environ.get('DASHSCOPE_API_KEY'),  # API密钥
                seed=seed,  # 随机种子，确保可重复性
                max_length=max_length  # 最大生成长度
            )
            # 3. 检查响应是否成功
            if success_check_fn is None:
                success_check_fn = lambda x: True
            if self.basic_success_check(response) and success_check_fn(response.output.text):
                response = response.output.text
                # 把AI回复添加到历史
                self.history.append({
                    "role": "assistant",
                    "content": response
                })
                success = True
                break
            else:
                try_times += 1

        # 4. 清理历史记录（如果不跟踪历史）
        if not self.track_history:
            if self.system_prompt is not None:
                self.history = self.history[:1]  # 只保留系统指令
            else:
                self.history = []  # 清空所有历史
        
        return response, success
   