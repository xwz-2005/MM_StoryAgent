"""
快速测试数据可视化功能
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mm_story_agent.utils.visualization import VisualizationTool

# 初始化可视化工具
viz_tool = VisualizationTool(output_dir="test_output/visualizations")

# 测试数据检测
test_stories = [
    "公司的销售额从100万元增长到了150万元，增长了50%。",
    "第一季度销售额为30万元，第二季度为40万元，第三季度为45万元。",
    "线上渠道占比达到了70%，线下渠道占比30%。",
    "市场份额从15%提升到了22%，增长了7个百分点。"
]

print("=" * 60)
print("测试数据可视化功能")
print("=" * 60)

for idx, story in enumerate(test_stories, 1):
    print(f"\n📄 测试故事 {idx}: {story}")
    
    # 检测数据
    detected_data = viz_tool.detect_data_in_story(story)
    
    if detected_data:
        print(f"✅ 检测到数据: {detected_data}")
        
        # 生成图表
        chart_path = viz_tool.generate_chart_from_story(
            story,
            chart_type="auto",
            page_index=idx-1
        )
        
        if chart_path:
            print(f"✅ 图表已生成: {chart_path}")
        else:
            print("❌ 图表生成失败")
    else:
        print("⚠️  未检测到数据")

print("\n" + "=" * 60)
print("测试完成！请查看 test_output/visualizations/ 目录")
print("=" * 60)