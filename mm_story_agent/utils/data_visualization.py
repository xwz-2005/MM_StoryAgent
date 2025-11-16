"""
数据可视化配置和工具
"""
from typing import Dict, List, Optional
import json
from pathlib import Path

class DataVisualizationConfig:
    """数据可视化配置类"""
    
    def __init__(self, config: Dict = None):
        self.enable_visualization = config.get("enable_visualization", True) if config else True
        self.chart_types = config.get("chart_types", ["bar", "line", "scatter"]) if config else ["bar", "line", "scatter"]
        self.auto_detect = config.get("auto_detect", True) if config else True
        self.chart_layout = config.get("chart_layout", "bottom") if config else "bottom"  # bottom, side_by_side, overlay
        self.generate_interactive = config.get("generate_interactive", False) if config else False
    
    def should_generate_chart(self, story_text: str) -> bool:
        """判断是否应该为故事生成图表"""
        if not self.enable_visualization:
            return False
        
        if not self.auto_detect:
            return False
        
        # 简单的数据检测逻辑
        import re
        has_numbers = bool(re.search(r'\d+', story_text))
        has_percentage = bool(re.search(r'\d+%', story_text))
        has_comparison = bool(re.search(r'(从|到|增长|下降|上升)', story_text))
        
        return has_numbers and (has_percentage or has_comparison)