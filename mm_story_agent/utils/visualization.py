# 工具类：封装静态/动态图表生成逻辑
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import plotly.express as px
import pandas as pd
from pathlib import Path
import platform
import warnings

# 确保中文显示正常 - 改进版字体配置
def setup_chinese_font():
    """配置中文字体，确保中文能正确显示"""
    if platform.system() == 'Windows':
        # Windows 系统字体配置
        # 尝试多种方法找到可用的中文字体
        font_candidates = [
            'Microsoft YaHei',      # 微软雅黑
            'SimHei',                # 黑体
            'SimSun',                # 宋体
            'KaiTi',                 # 楷体
            'FangSong',              # 仿宋
            'STSong',                # 华文宋体
            'STHeiti',               # 华文黑体
        ]
        
        # 方法1：使用字体管理器查找可用字体
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        chinese_font = None
        
        for font_name in font_candidates:
            if font_name in available_fonts:
                chinese_font = font_name
                break
        
        if chinese_font:
            plt.rcParams['font.sans-serif'] = [chinese_font] + font_candidates
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 已设置中文字体: {chinese_font}")
        else:
            # 方法2：尝试直接使用系统字体路径
            try:
                # Windows 字体路径
                font_paths = [
                    r'C:\Windows\Fonts\msyh.ttc',      # 微软雅黑
                    r'C:\Windows\Fonts\simhei.ttf',   # 黑体
                    r'C:\Windows\Fonts\simsun.ttc',    # 宋体
                ]
                
                for font_path in font_paths:
                    if Path(font_path).exists():
                        prop = fm.FontProperties(fname=font_path)
                        plt.rcParams['font.family'] = prop.get_name()
                        plt.rcParams['axes.unicode_minus'] = False
                        print(f"✅ 已从文件加载中文字体: {font_path}")
                        break
                else:
                    # 如果都找不到，使用默认配置
                    plt.rcParams['font.sans-serif'] = font_candidates
                    plt.rcParams['axes.unicode_minus'] = False
                    warnings.warn("⚠️  未找到中文字体，中文可能显示为方块")
            except Exception as e:
                warnings.warn(f"⚠️  字体配置失败: {e}")
                plt.rcParams['font.sans-serif'] = font_candidates
                plt.rcParams['axes.unicode_minus'] = False
    else:
        # Linux/Mac 系统
        plt.rcParams['font.family'] = ['WenQuanYi Micro Hei', 'SimHei', 'Heiti TC']
        plt.rcParams['axes.unicode_minus'] = False

# 初始化字体配置
setup_chinese_font()

sns.set(font_scale=1.2)

class VisualizationTool:
    def __init__(self, output_dir="output/visualizations"):
        # 创建输出目录（存放生成的图表）
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_static_chart(self, data, chart_type="bar", title="数据图表"):
        """生成静态图表（柱状图/折线图/散点图）"""
        # data格式：{"x": ["类别1", "类别2"], "y": [10, 20], "x_label": "x轴", "y_label": "y轴"}
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # 强制使用中文字体文件
        chinese_font_prop = None
        if platform.system() == 'Windows':
            font_paths = [
                r'C:\Windows\Fonts\msyh.ttc',
                r'C:\Windows\Fonts\simhei.ttf',
                r'C:\Windows\Fonts\simsun.ttc',
            ]
            for font_path in font_paths:
                if Path(font_path).exists():
                    try:
                        chinese_font_prop = fm.FontProperties(fname=font_path)
                        break
                    except:
                        continue
        
        # 使用matplotlib直接绘制，而不是seaborn（避免聚合问题）
        if chart_type == "bar":
            x_positions = range(len(data["x"]))
            ax.bar(x_positions, data["y"], width=0.6)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(data["x"])
        elif chart_type == "line":
            x_positions = range(len(data["x"]))
            ax.plot(x_positions, data["y"], marker="o", linewidth=2, markersize=8)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(data["x"])
        elif chart_type == "scatter":
            x_positions = range(len(data["x"]))
            ax.scatter(x_positions, data["y"], s=100)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(data["x"])
        
        # 使用字体属性设置标题和标签
        if chinese_font_prop:
            ax.set_title(title, fontproperties=chinese_font_prop, fontsize=14)
            ax.set_xlabel(data.get("x_label", "类别"), fontproperties=chinese_font_prop)
            ax.set_ylabel(data.get("y_label", "数值"), fontproperties=chinese_font_prop)
            # 设置x轴刻度标签的字体
            for label in ax.get_xticklabels():
                label.set_fontproperties(chinese_font_prop)
        else:
            ax.set_title(title)
            ax.set_xlabel(data.get("x_label", "类别"))
            ax.set_ylabel(data.get("y_label", "数值"))
        
        # 添加网格线
        ax.grid(True, alpha=0.3, linestyle='--')
        
        save_path = self.output_dir / f"static_{chart_type}_{len(list(self.output_dir.glob('*'))) + 1}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        return str(save_path)

    def generate_dynamic_chart(self, data, title="动态数据图表"):
        """生成动态交互式图表（Plotly，保存为HTML）"""
        df = pd.DataFrame(data)
        fig = px.line(df, x="x", y="y", title=title, markers=True)  # 默认为折线图，可扩展其他类型
        fig.update_layout(
            xaxis_title=data.get("x_label", "类别"),
            yaxis_title=data.get("y_label", "数值")
        )
        
        save_path = self.output_dir / f"dynamic_{len(list(self.output_dir.glob('*'))) + 1}.html"
        fig.write_html(save_path)
        return str(save_path)

    def detect_data_in_story(self, story_text: str):
        """从故事文本中检测是否包含数据信息"""
        import re
        
        patterns = {
            'comparison': r'从\s*(\d+(?:\.\d+)?%?)\s*(?:到|提升到|提升到了|增长到|增长到了|上升至|上升到了)\s*(\d+(?:\.\d+)?%?)',  # 修复：支持"提升到了"
            'percentage': r'(\d+(?:\.\d+)?%)',
            'number': r'(\d+(?:\.\d+)?)(?:万|千|亿)?(?:元|人|个|次|倍)',
            'trend': r'(增长|下降|上升|减少)\s*(\d+(?:\.\d+)?)',
        }
        
        detected_data = {}
        for data_type, pattern in patterns.items():
            matches = re.findall(pattern, story_text)
            if matches:
                detected_data[data_type] = matches
        
        return detected_data if detected_data else None
    
    def generate_chart_from_story(self, story_text: str, chart_type: str = "auto", 
                                   title: str = None, page_index: int = 0):
        """
        根据故事文本自动生成图表
        """
        detected_data = self.detect_data_in_story(story_text)
        
        if not detected_data:
            return None
        
        # 根据检测到的数据类型选择合适的图表
        if chart_type == "auto":
            if 'trend' in detected_data or 'comparison' in detected_data:
                chart_type = "line"
            elif 'percentage' in detected_data:
                chart_type = "bar"
            else:
                chart_type = "bar"
        
        # 提取数据并格式化
        chart_data = self._extract_chart_data(detected_data, story_text)
        
        if not chart_data:
            return None
        
        # 生成图表
        if chart_type == "bar":
            return self.generate_static_chart(chart_data, "bar", 
                                             title or f"数据图表 - 第{page_index+1}页")
        elif chart_type == "line":
            return self.generate_static_chart(chart_data, "line",
                                             title or f"趋势图表 - 第{page_index+1}页")
        elif chart_type == "scatter":
            return self.generate_static_chart(chart_data, "scatter",
                                             title or f"散点图 - 第{page_index+1}页")
        else:
            return self.generate_static_chart(chart_data, "bar",
                                             title or f"数据图表 - 第{page_index+1}页")
    
    def _extract_chart_data(self, detected_data: dict, story_text: str):
        """从检测到的数据中提取并格式化图表数据"""
        chart_data = {"x": [], "y": [], "x_label": "类别", "y_label": "数值"}
        import re
        
        # 优先处理比较数据（从X到Y）
        if 'comparison' in detected_data:
            comparisons = detected_data['comparison']
            comp = comparisons[0]
            start_val = comp[0]
            end_val = comp[1]
            
            # 判断是否包含百分比
            has_percent = '%' in start_val or '%' in end_val
            
            # 提取数值
            start_num = float(start_val.replace('%', ''))
            end_num = float(end_val.replace('%', ''))
            
            # 设置标签：如果有百分比，显示为"15%"和"22%"，否则显示"起始值"和"结束值"
            if has_percent:
                chart_data["x"] = [f"{start_val}", f"{end_val}"]
                chart_data["y"] = [start_num, end_num]
                chart_data["y_label"] = "百分比 (%)"
            else:
                chart_data["x"] = ["起始值", "结束值"]
                chart_data["y"] = [start_num, end_num]
                chart_data["y_label"] = "数值"
            return chart_data
        
        # 如果有多个number数据，优先使用（如季度数据）
        if 'number' in detected_data and len(detected_data['number']) >= 2:
            numbers = detected_data['number']
            quarter_labels = re.findall(r'(第[一二三四1-4]季度)', story_text)
            if len(quarter_labels) == len(numbers):
                chart_data["x"] = quarter_labels
            else:
                chart_data["x"] = [f"项目{i+1}" for i in range(len(numbers))]
            chart_data["y"] = [float(n) for n in numbers]
            chart_data["y_label"] = "数值"
            return chart_data
        
        # 处理百分比数据 - 改进标签提取逻辑
        if 'percentage' in detected_data:
            percentages = detected_data['percentage']
            labels = []
            
            # 为每个百分比单独提取标签
            for i, pct in enumerate(percentages):
                label = None
                
                # 找到百分比在文本中的位置
                pct_pos = story_text.find(pct)
                if pct_pos > 0:
                    # 只在这个百分比位置附近查找（向前查找最多30个字符）
                    search_start = max(0, pct_pos - 30)
                    search_text = story_text[search_start:pct_pos]
                    
                    # 方法1：匹配完整短语（如"线上渠道"、"线下渠道"）
                    # 从后往前匹配，找到最近的一个
                    pattern1 = r'([\u4e00-\u9fa5]{2,6}(?:渠道|市场|份额|占比|比例|销售|收入))'
                    matches1 = list(re.finditer(pattern1, search_text))
                    if matches1:
                        # 取最后一个匹配（最接近百分比的）
                        label = matches1[-1].group(1)
                    else:
                        # 方法2：匹配百分比前的2-4个中文字符
                        pattern2 = r'([\u4e00-\u9fa5]{2,4})(?=\s*(?:占比|达到|为|是|提升|增长))'
                        matches2 = list(re.finditer(pattern2, search_text))
                        if matches2:
                            candidate = matches2[-1].group(1)
                            # 过滤无意义词
                            if candidate not in ['占比', '达到', '提升', '增长', '为', '是', '了', '的', '渠道', '市场']:
                                label = candidate
                
                # 如果还是没找到，尝试在整个文本中查找（但只匹配这个百分比前的）
                if not label:
                    # 方法3：在整个文本中，但只匹配这个百分比前的部分
                    before_pct = story_text[:pct_pos]
                    # 从后往前查找"XX渠道"、"XX市场"等
                    pattern3 = r'([\u4e00-\u9fa5]{2,6}(?:渠道|市场|份额))'
                    matches3 = list(re.finditer(pattern3, before_pct))
                    if matches3:
                        label = matches3[-1].group(1)
                
                labels.append(label)
            
            # 处理标签：如果提取失败，使用默认标签
            final_labels = []
            for i, label in enumerate(labels):
                if label:
                    final_labels.append(label)
                else:
                    final_labels.append(f"项目{i+1}")
            
            chart_data["x"] = final_labels
            chart_data["y"] = [float(p.replace('%', '')) for p in percentages]
            chart_data["y_label"] = "百分比 (%)"
            return chart_data
        
        # 处理趋势数据
        if 'trend' in detected_data:
            trends = detected_data['trend']
            chart_data["x"] = [f"时间点{i+1}" for i in range(len(trends))]
            chart_data["y"] = [float(t[1]) for t in trends]
            chart_data["y_label"] = "变化值"
            return chart_data
        
        # 处理单个number数据
        if 'number' in detected_data:
            numbers = detected_data['number']
            chart_data["x"] = [f"项目{i+1}" for i in range(len(numbers))]
            chart_data["y"] = [float(n) for n in numbers]
            chart_data["y_label"] = "数值"
            return chart_data
        
        return None
    
    def generate_animated_chart(self, data_series: list, title: str = "动态图表"):
        """
        生成动态图表（GIF格式）
        data_series: 包含多个时间点的数据列表
        """
        import matplotlib.animation as animation
        from matplotlib import pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        def animate(frame):
            ax.clear()
            current_data = data_series[:frame+1]
            if current_data:
                x = [d["x"] for d in current_data]
                y = [d["y"] for d in current_data]
                ax.plot(x, y, marker='o', linewidth=2, markersize=8)
                ax.set_title(f"{title} - 帧 {frame+1}/{len(data_series)}")
                ax.set_xlabel("时间")
                ax.set_ylabel("数值")
                ax.grid(True, alpha=0.3)
        
        anim = animation.FuncAnimation(fig, animate, frames=len(data_series), 
                                       interval=200, repeat=True)
        
        save_path = self.output_dir / f"animated_{len(list(self.output_dir.glob('*')))+1}.gif"
        anim.save(save_path, writer='pillow', fps=5)
        plt.close()
        
        return str(save_path)
    
    def combine_image_with_chart(self, image_path: str, chart_path: str, 
                                 output_path: str, layout: str = "side_by_side"):
        """
        将静态图像与图表结合
        layout: "side_by_side" (并排), "overlay" (叠加), "bottom" (图表在底部)
        """
        from PIL import Image
        
        img = Image.open(image_path)
        chart = Image.open(chart_path)
        
        if layout == "side_by_side":
            # 并排布局
            total_width = img.width + chart.width
            max_height = max(img.height, chart.height)
            combined = Image.new('RGB', (total_width, max_height), color='white')
            combined.paste(img, (0, 0))
            combined.paste(chart, (img.width, 0))
        elif layout == "overlay":
            # 叠加布局（图表在右下角）
            combined = img.copy()
            chart_resized = chart.resize((img.width // 3, img.height // 3))
            combined.paste(chart_resized, (img.width - chart_resized.width - 20,
                                          img.height - chart_resized.height - 20))
        else:  # bottom
            # 图表在底部
            total_height = img.height + chart.height
            combined = Image.new('RGB', (img.width, total_height), color='white')
            combined.paste(img, (0, 0))
            combined.paste(chart, (0, img.height))
        
        combined.save(output_path)
        return output_path
