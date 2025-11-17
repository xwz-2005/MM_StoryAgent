import json
from typing import Dict, List
import random
import time
from tqdm import trange, tqdm
from ..utils.llm_output_check import parse_list
from ..base import register_tool, init_tool_instance
from ..prompts_zh import question_asker_system, expert_system, \
    dlg_based_writer_system, dlg_based_writer_prompt, chapter_writer_system

def json_parse_outline(outline):
    outline = outline.strip("```json").strip("```")
    try:
        outline = json.loads(outline)
        if not isinstance(outline, dict):
            return False
        if outline.keys() != {"story_title", "story_outline"}:
            return False
        for chapter in outline["story_outline"]:
            if chapter.keys() != {"chapter_title", "chapter_summary"}:
                return False
    except json.decoder.JSONDecodeError:
        return False
    return True

# 增强的JSON格式校验函数
def is_valid_json(text: str) -> bool:
    """验证文本是否为有效JSON格式"""
    try:
        json.loads(text.strip("```json").strip("```"))
        return True
    except (json.JSONDecodeError, TypeError):
        return False

# 敏感词过滤列表
SENSITIVE_WORDS = [
    # 暴力相关
    "杀人", "暴力", "抢劫", "斗殴", "凶器", "自杀", "自残",
    # 违规相关
    "毒品", "赌博", "色情", "嫖娼", "卖淫", "走私", "诈骗",
    # 政治敏感
    "反动", "颠覆", "分裂", "极端", "恐怖", "邪教",
    # 不适当表述
    "血腥", "恐怖", "惊悚", "虐待", "霸凌", "歧视"
]

# 合规替换映射（将敏感表述替换为合规表达）
COMPLIANCE_REPLACE = {
    "杀人": "制止不法行为",
    "暴力": "冲突",
    "抢劫": "抢夺财物",
    "血腥": "激烈",
    "恐怖": "紧张"
}

class ComplianceFilter:
    """合规内容过滤器"""
    @staticmethod
    def filter_sensitive(text: str) -> str:
        """过滤敏感词，替换为合规表述"""
        if not text:
            return text
        # 替换敏感词
        filtered_text = text
        for sensitive, replacement in COMPLIANCE_REPLACE.items():
            filtered_text = filtered_text.replace(sensitive, replacement)
        # 二次过滤剩余敏感词
        for word in SENSITIVE_WORDS:
            if word not in COMPLIANCE_REPLACE:
                filtered_text = filtered_text.replace(word, "")
        return filtered_text.strip()
    
    @staticmethod
    def check_compliance(text: str) -> (bool, List[str]):
        """检查文本是否合规，返回合规状态和敏感词列表"""
        sensitive_found = []
        for word in SENSITIVE_WORDS:
            if word in text:
                sensitive_found.append(word)
        return len(sensitive_found) == 0, sensitive_found

LONG_TEXT_SUMMARIZER_SYSTEM = """
你是一名专业的文本摘要分析师，需要从长文本中提取故事创作所需的关键信息。
请严格按照以下要求输出：
1. 仅返回JSON对象，无任何额外文字、注释或说明
2. JSON必须包含以下字段，字段值不能为空且合规：
   - 核心主题（积极正面，符合公序良俗）
   - 主要角色（名称及核心特征，无违规表述）
   - 关键场景（合规场景，无敏感地点）
   - 核心情节（积极向上，无暴力、血腥等敏感内容）
   - 情感基调（健康正面）
3. 确保JSON格式标准，无语法错误，使用双引号包裹字段名和值
4. 内容必须符合中国法律法规和公序良俗，禁止任何违规、敏感元素
示例输出：
{"核心主题": "友谊与成长", "主要角色": "小明（勇敢的小学生）、小红（细心的班长）", "关键场景": "周末的公园、学校教室", "核心情节": "小明丢失书包，小红帮忙寻找，两人克服困难成为好友", "情感基调": "温暖、积极"}
"""

CHAPTER_WRITER_SYSTEM_COMPLIANCE = """
你是专业的故事章节作家，需按照以下要求生成内容：
1. 内容必须符合中国法律法规和公序良俗，禁止暴力、血腥、色情、赌博等违规敏感元素
2. 角色行为积极正面，情节健康向上，具有教育意义或正向引导
3. 语言文明规范，无脏话、不适当表述
4. 严格遵循故事大纲，保持内容连贯性和逻辑性
5. 输出格式为列表字符串（如["页面1内容", "页面2内容"]），无额外文字
"""

@register_tool("qa_outline_story_writer")
class QAOutlineStoryWriter:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.temperature = cfg.get("temperature", 1.0)
        self.max_conv_turns = cfg.get("max_conv_turns", 3)
        self.num_outline = cfg.get("num_outline", 4)
        self.llm_type = cfg.get("llm", "qwen")
        self.long_text_threshold = cfg.get("long_text_threshold", 500)
        self.max_retry = cfg.get("max_retry", 2)
        self.compliance_filter = ComplianceFilter()  # 初始化合规过滤器

    def _summarize_long_text(self, long_text: str) -> Dict:
        """总结长文本并提取关键信息"""
        print("📄 正在处理长文本...")
        # 先过滤长文本中的敏感内容
        filtered_long_text = self.compliance_filter.filter_sensitive(long_text)
        
        summarizer = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": LONG_TEXT_SUMMARIZER_SYSTEM,
                "track_history": False
            }
        })
        
        chunks = self._split_long_text(filtered_long_text)
        chunk_summaries = []
        
        for i, chunk in enumerate(chunks):
            print(f"🔍 处理文本片段 {i+1}/{len(chunks)}")
            retry_count = 0
            valid_summary = None
            
            while retry_count < self.max_retry and valid_summary is None:
                # 提示词添加合规强调
                prompt = f"长文本片段 {i+1}：{chunk}\n严格按照系统提示格式输出JSON，仅返回JSON，无其他内容。确保所有字段内容合规，无敏感元素。"
                summary, success = summarizer.call(prompt, temperature=self.temperature)
                
                if success and summary.strip():
                    clean_summary = summary.strip("```json").strip("```").strip()
                    if is_valid_json(clean_summary):
                        # 验证摘要内容合规性
                        summary_json = json.loads(clean_summary)
                        summary_text = json.dumps(summary_json, ensure_ascii=False)
                        is合规, sensitive_words = self.compliance_filter.check_compliance(summary_text)
                        if is合规:
                            valid_summary = clean_summary
                            break
                        else:
                            print(f"⚠️  片段 {i+1} 摘要包含敏感词：{','.join(sensitive_words)}，正在重试")
                    else:
                        print(f"⚠️  片段 {i+1} 摘要格式无效，正在重试（{retry_count+1}/{self.max_retry}）")
                retry_count += 1
            
            if valid_summary:
                chunk_summaries.append(json.loads(valid_summary))
            else:
                print(f"❌ 片段 {i+1} 多次生成失败，跳过该片段")
        
        if len(chunk_summaries) == 0:
            print("⚠️  无有效片段摘要，返回默认合规结构")
            return {
                "核心主题": "积极向上的生活故事",
                "主要角色": "普通市民（善良、乐观）",
                "关键场景": "城市社区、公园、工作场所",
                "核心情节": "主角通过努力解决生活中的小困难，获得成长",
                "情感基调": "温暖、励志"
            }
        
        if len(chunk_summaries) > 1:
            print("🔄 正在整合片段摘要...")
            retry_count = 0
            final_summary = None
            
            while retry_count < self.max_retry and final_summary is None:
                combine_prompt = f"""
                    请整合以下{len(chunk_summaries)}个片段摘要，生成完整的长文本摘要。
                    要求：
                    1. 仅返回JSON对象，无任何额外内容
                    2. 合并重复信息，补充缺失信息，所有内容必须合规无敏感元素
                    3. 字段保持与片段摘要一致：核心主题、主要角色、关键场景、核心情节、情感基调
                    4. 确保JSON格式标准，无语法错误，符合中国法律法规和公序良俗

                    片段摘要：
                    {json.dumps(chunk_summaries, ensure_ascii=False, indent=2)}
                    """
                summary, success = summarizer.call(combine_prompt, temperature=self.temperature)
                if success and summary.strip():
                    clean_summary = summary.strip("```json").strip("```").strip()
                    if is_valid_json(clean_summary):
                        summary_text = json.dumps(json.loads(clean_summary), ensure_ascii=False)
                        is合规, sensitive_words = self.compliance_filter.check_compliance(summary_text)
                        if is合规:
                            final_summary = json.loads(clean_summary)
                            break
                        else:
                            print(f"⚠️  整合摘要包含敏感词：{','.join(sensitive_words)}，正在重试")
                    else:
                        print(f"⚠️  整合摘要格式无效，正在重试（{retry_count+1}/{self.max_retry}）")
                retry_count += 1
            
            if final_summary:
                return final_summary
            else:
                print("⚠️  整合摘要失败，返回第一个片段摘要")
                return chunk_summaries[0]
        
        return chunk_summaries[0]

    def _split_long_text(self, text: str, chunk_size: int = 2000) -> List[str]:
        """将长文本分割为适合模型处理的片段"""
        chunks = []
        if len(text) <= chunk_size:
            return [text]
        
        # 按段落分割优先，避免敏感内容跨段
        paragraphs = text.split("\n\n")
        current_chunk = ""
        for para in paragraphs:
            # 过滤段落中的敏感内容
            filtered_para = self.compliance_filter.filter_sensitive(para)
            if len(current_chunk) + len(filtered_para) + 2 <= chunk_size:
                current_chunk += filtered_para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = filtered_para + "\n\n"
                else:
                    # 单个段落超长，按字符分割
                    for i in range(0, len(filtered_para), chunk_size):
                        chunks.append(filtered_para[i:i+chunk_size])
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _analyze_story_style(self, story_setting):
        """主题风格自动分析"""
        print("🔍 正在进行主题风格自动分析...")
        
        # 先过滤故事设置中的敏感内容
        setting_str = self.compliance_filter.filter_sensitive(json.dumps(story_setting, ensure_ascii=False))
        
        keywords = {
            "日常写实": [
                "学校", "宿舍", "公司", "家庭", "医院", "公园", "商场", "街道", "办公室",
                "教室", "图书馆", "餐厅", "咖啡厅", "车站", "机场", "银行", "超市",
                "学生", "老师", "医生", "护士", "警察", "消防员", "工程师", "程序员",
                "设计师", "销售人员", "服务员", "厨师", "司机", "科学家", "研究员",
                "上课", "学习", "工作", "吃饭", "睡觉", "散步", "购物", "运动", "聊天",
                "阅读", "编程", "开会", "考试", "通勤", "打扫", "做饭", "健身"
            ],
            "奇幻": [
                "魔法", "巫师", "女巫", "咒语", "魔棒", "城堡", "王国", "公主", "王子",
                "龙", "精灵", "矮人", "兽人", "半兽人", "魔法书", "药水", "宝石", "神器",
                "超自然", "神秘力量", "异世界", "穿越", "时空", "预言", "传说", "神话",
                "会说话的动物", "妖怪", "怪物", "幽灵", "鬼魂", "吸血鬼", "狼人"
            ],
            "科幻": [
                "未来", "科技", "人工智能", "机器人", "太空", "宇宙", "星球", "飞船",
                "太空站", "宇航员", "外星生物", "UFO", "飞碟", "激光", "纳米技术", "克隆",
                "量子", "虚拟现实", "增强现实", "赛博朋克", "未来城市", "机械臂", "芯片",
                "基因工程", "时间旅行", "平行宇宙", "黑洞", "超光速"
            ]
        }
        
        scores = {style: 0 for style in keywords.keys()}
        for style, style_keywords in keywords.items():
            for keyword in style_keywords:
                if keyword in setting_str:
                    scores[style] += 1
        
        dominant_style = max(scores, key=scores.get)
        if scores[dominant_style] == 0:
            dominant_style = "日常写实"
        
        # 风格规则添加合规约束
        style_rules = {
            "日常写实": "仅包含现实中存在的场景（如学校、宿舍）、角色（如学生、老师）和行为（如上课、编程），禁止任何虚构元素和违规敏感内容，符合公序良俗",
            "奇幻": "允许包含魔法、虚构生物等奇幻元素，场景和角色可虚构，但需符合奇幻逻辑，禁止暴力、血腥等违规内容，符合公序良俗",
            "科幻": "聚焦未来科技、太空探索等元素，禁止无逻辑的超自然力量和违规敏感内容，需符合科学幻想设定和公序良俗"
        }
        
        print(f"✅ 风格分析完成: {dominant_style}")
        print(f"📝 生成规则: {style_rules[dominant_style]}")
        
        return dominant_style, style_rules[dominant_style]
    
    def _handle_compliance_error(self, input_content: str, error_source: str) -> str:
        """处理内容审核失败，引导用户修改输入"""
        print(f"\n❌ 【内容审核失败】{error_source} 包含可能违规的内容，已被系统拦截")
        print("🔍 可能的原因：")
        print("1. 包含敏感关键词（如暴力、违规行为等）")
        print("2. 表述方式存在歧义，触发审核规则")
        print("3. 内容不符合公序良俗")
        
        # 检测输入中的敏感词
        is合规, sensitive_words = self.compliance_filter.check_compliance(input_content)
        if sensitive_words:
            print(f"\n⚠️  检测到可能的敏感词：{','.join(sensitive_words)}")
        
        print(f"\n✏️  请修改{error_source}内容：")
        print(f"当前内容：{input_content[:100]}..." if len(input_content) > 100 else f"当前内容：{input_content}")
        
        while True:
            modified_content = input("\n修改后的内容：").strip()
            if not modified_content:
                print("❌ 内容不能为空，请重新输入")
                continue
            
            # 验证修改后的内容合规性
            is合规, sensitive_words = self.compliance_filter.check_compliance(modified_content)
            if is合规:
                # 自动过滤剩余潜在敏感词
                filtered_content = self.compliance_filter.filter_sensitive(modified_content)
                print(f"✅ 内容合规，已自动优化为：{filtered_content[:100]}..." if len(filtered_content) > 100 else f"✅ 内容合规，已确认：{filtered_content}")
                return filtered_content
            else:
                print(f"❌ 修改后的内容仍包含敏感词：{','.join(sensitive_words)}，请再次修改")
    
    def generate_outline(self, params):
        """生成故事大纲"""
        # 先过滤params中的敏感内容
        filtered_params = {}
        for key, value in params.items():
            if isinstance(value, str):
                filtered_params[key] = self.compliance_filter.filter_sensitive(value)
            else:
                filtered_params[key] = value
        
        # 检查长文本输入
        if "long_text" in filtered_params and len(filtered_params["long_text"]) > self.long_text_threshold:
            summary = self._summarize_long_text(filtered_params["long_text"])
            story_setting = {
                "story_topic": summary.get("核心主题", "基于长文本的合规故事"),
                "main_role": summary.get("主要角色", "未明确角色"),
                "scene": summary.get("关键场景", "未明确场景"),
                "emotional_tone": summary.get("情感基调", "积极"),
                "core_plot": summary.get("核心情节", "")
            }
            print("📝 长文本处理完成，生成合规故事设置")
        else:
            story_setting = filtered_params
        
        # 主题风格分析
        dominant_style, style_rule = self._analyze_story_style(story_setting)
        
        asker = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": question_asker_system + "\n注意：所有提问必须合规，无敏感内容",
                "track_history": False
            }
        })
        expert = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": expert_system + "\n注意：所有回答必须合规，无敏感内容，符合公序良俗",
                "track_history": False
            }
        })
        dialogue = []
        for turn in trange(self.max_conv_turns):
            dialogue_history = "\n".join(dialogue)
            
            question, success = asker.call(
                f"Story setting: {filtered_params}\nDialogue history: \n{dialogue_history}\n提问必须合规，无敏感内容",
                temperature=self.temperature
            )
            question = self.compliance_filter.filter_sensitive(question.strip())
            if question == "Thank you for your help!":
                break
            dialogue.append(f"You: {question}")
            
            answer, success = expert.call(
                f"Story setting: {filtered_params}\nQuestion: \n{question}\nAnswer: 回答必须合规，无敏感内容，符合公序良俗",
                temperature=self.temperature
            )
            answer = self.compliance_filter.filter_sensitive(answer.strip())
            dialogue.append(f"Expert: {answer}")
        
        writer = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": dlg_based_writer_system + "\n所有大纲内容必须合规，无敏感元素，符合公序良俗",
                "track_history": False
            }
        })
        writer_prompt = dlg_based_writer_prompt.format(
            story_setting=filtered_params,
            dialogue_history="\n".join(dialogue),
            num_outline=self.num_outline,
            style_type=dominant_style,
            style_rule=style_rule + "，内容必须合规，无敏感元素"
        )
        
        # 大纲生成重试校验
        retry_count = 0
        outline = None
        while retry_count < self.max_retry and outline is None:
            outline_str, success = writer.call(writer_prompt, success_check_fn=json_parse_outline)
            if success:
                outline_clean = outline_str.strip("```json").strip("```")
                if is_valid_json(outline_clean):
                    outline_json = json.loads(outline_clean)
                    # 验证大纲合规性
                    outline_text = json.dumps(outline_json, ensure_ascii=False)
                    is合规, sensitive_words = self.compliance_filter.check_compliance(outline_text)
                    if is合规:
                        outline = outline_json
                        break
                    else:
                        print(f"⚠️  大纲包含敏感词：{','.join(sensitive_words)}，正在重试")
                else:
                    print(f"⚠️  大纲格式无效，正在重试（{retry_count+1}/{self.max_retry}）")
            else:
                print(f"⚠️  大纲生成失败，正在重试（{retry_count+1}/{self.max_retry}）")
            retry_count += 1
        
        if not outline:
            # 生成默认合规大纲
            print("⚠️  大纲生成失败，使用默认合规大纲")
            outline = {
                "story_title": "积极向上的生活故事",
                "story_outline": [
                    {"chapter_title": "初识困境", "chapter_summary": "主角遇到生活中的小困难，感到迷茫"},
                    {"chapter_title": "寻求帮助", "chapter_summary": "主角向身边人求助，获得支持和建议"},
                    {"chapter_title": "努力克服", "chapter_summary": "主角通过自身努力和他人帮助，逐步解决问题"},
                    {"chapter_title": "收获成长", "chapter_summary": "主角成功解决困难，获得成长和感悟"}
                ]
            }
        
        return outline

    def _get_user_choice(self, prompt="请选择操作: ", valid_options=['1', '2', '3'], max_attempts=5):
        """获取并验证用户选择"""
        attempts = 0
        while attempts < max_attempts:
            try:
                if prompt.strip().lower() in ['help', '帮助', '?']:
                    self._show_help()
                    continue
                
                print(prompt)
                choice = input().strip()
                
                if choice.lower() in ['help', '帮助', '?']:
                    self._show_help()
                    continue
                    
                if choice in valid_options:
                    return choice
                else:
                    print(f"❌ 请输入正确选项 ({'/'.join(valid_options)})")
                    attempts += 1
                    if attempts >= max_attempts:
                        print(f"⚠️  多次输入错误，将使用默认选项: {valid_options[0]}")
                        return valid_options[0]
            except Exception as e:
                print(f"❌ 输入错误: {str(e)}，请重试")
                attempts += 1
        return valid_options[0]
    
    def _show_help(self):
        """显示帮助信息"""
        print("\n📚 帮助信息:")
        print("   • 在任何输入提示时，输入 'help'、'帮助' 或 '?' 可显示此帮助")
        print("   • 所有操作都会被记录在edit_history.json中")
        print("   • 连续多次重新生成会触发参数调整建议")
        print("   • 修改内容时，请尽量提供具体、明确的修改建议，且内容必须合规")
        print("   • 避免使用暴力、血腥、违规等敏感表述，否则会触发内容审核失败")
        print("   • 如遇问题，可按Ctrl+C中断程序")
        print()
    
    def _get_modification_input(self, content_type="内容"):
        """获取用户修改意见"""
        print(f"\n✏️  请输入您对{content_type}的修改意见（需合规，无敏感内容）:")
        print("   示例: '增加主角的心理活动' 或 '让场景更温暖一些'")
        print("   提示: 更具体的修改意见会获得更好的结果")
        print("   输入 'cancel' 可取消修改")
        
        while True:
            try:
                modification = input("\n修改意见: ").strip()
                
                if modification.lower() in ['help', '帮助', '?']:
                    self._show_help()
                    continue
                    
                if not modification:
                    print("⚠️  修改意见不能为空，请重新输入")
                elif modification.lower() == 'cancel':
                    return None
                elif len(modification) < 5:
                    print("⚠️  修改意见过于简短，请提供更详细的描述")
                else:
                    # 合规校验
                    is合规, sensitive_words = self.compliance_filter.check_compliance(modification)
                    if not is合规:
                        print(f"❌ 修改意见包含敏感词：{','.join(sensitive_words)}，请修改")
                        continue
                    return modification
            except Exception:
                print("❌ 输入错误，请重试")
    
    def _modify_content(self, original_content, chapter, all_pages):
        """处理内容修改"""
        modification = self._get_modification_input("章节")
        if modification is None:
            return original_content
        
        # 合规处理
        filtered_modification = self.compliance_filter.filter_sensitive(modification)
        filtered_original = self.compliance_filter.filter_sensitive(original_content)
        
        modify_prompt = json.dumps({
            "original_content": filtered_original,
            "chapter_info": chapter,
            "modification_request": filtered_modification,
            "completed_story": all_pages
        }, ensure_ascii=False)
        
        modifier = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": "你是一个专业的故事编辑助手。请根据用户的修改意见，对提供的故事内容进行精准修改。保持故事的连贯性和风格一致性。内容必须合规，无敏感元素，符合公序良俗。直接输出修改后的完整内容，不要添加任何额外的解释或标记。",
                "track_history": False
            }
        })
        
        print("🔄 正在根据您的意见修改内容...")
        modified_content, success = modifier.call(
            modify_prompt,
            temperature=max(0.5, self.temperature * 0.9),
            success_check_fn=lambda x: x.strip() != ""
        )
        
        if success:
            return self.compliance_filter.filter_sensitive(modified_content.strip())
        else:
            print("⚠️ 修改失败，保留原始内容")
            return filtered_original
    
    def _adjust_generation_params(self):
        """调整生成参数"""
        print("\n⚙️  调整生成参数")
        print("请选择要调整的参数:")
        print("1. temperature (创造性控制，当前值: {:.2f})".format(self.temperature))
        print("2. max_retry (解析重试次数，当前值: {})".format(self.max_retry))
        print("3. 返回")
        
        choice = self._get_user_choice(valid_options=['1', '2', '3'])
        
        if choice == '1':
            print("\n请输入新的temperature值 (0.1-1.0):")
            while True:
                try:
                    temp_value = input().strip()
                    if temp_value.lower() in ['help', '帮助', '?']:
                        self._show_help()
                        continue
                    
                    temp_value = float(temp_value)
                    if 0.1 <= temp_value <= 1.0:
                        self.temperature = temp_value
                        print(f"✅ 已设置temperature为: {temp_value:.2f}")
                        break
                    else:
                        print("❌ 无效值，请输入0.1-1.0之间的数字")
                except ValueError:
                    print("❌ 输入错误，请输入有效的数字")
        
        elif choice == '2':
            print("\n请输入新的max_retry值 (1-5):")
            while True:
                try:
                    retry_value = input().strip()
                    if retry_value.lower() in ['help', '帮助', '?']:
                        self._show_help()
                        continue
                    
                    retry_value = int(retry_value)
                    if 1 <= retry_value <= 5:
                        self.max_retry = retry_value
                        print(f"✅ 已设置max_retry为: {retry_value}")
                        break
                    else:
                        print("❌ 无效值，请输入1-5之间的整数")
                except ValueError:
                    print("❌ 输入错误，请输入有效的整数")
    
    def generate_story_from_outline(self, outline):
        """根据大纲生成故事"""
        # 初始化章节生成代理
        chapter_writer = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": CHAPTER_WRITER_SYSTEM_COMPLIANCE,
                "track_history": False
            }
        })
        all_pages = []
        edit_history = []
        modify_count = 0
        
        # 显示大纲概览
        print(f"\n📋 故事大纲概览:")
        for i, chap in enumerate(outline["story_outline"], 1):
            filtered_title = self.compliance_filter.filter_sensitive(chap['chapter_title'])
            filtered_summary = self.compliance_filter.filter_sensitive(chap['chapter_summary'])
            print(f"   {i}. {filtered_title} - {filtered_summary[:80]}...")
        
        for idx, chapter in enumerate(tqdm(outline["story_outline"])):
            print(f"\n{'='*60}")
            # 合规处理章节信息
            filtered_chapter = {
                "chapter_title": self.compliance_filter.filter_sensitive(chapter['chapter_title']),
                "chapter_summary": self.compliance_filter.filter_sensitive(chapter['chapter_summary'])
            }
            print(f"📖 正在生成第 {idx + 1} 章节: {filtered_chapter['chapter_title']}")
            print(f"{'='*60}")
            
            chapter_completed = False
            retry_count = 0
            max_retries = 3
            modify_attempts = 0
            max_modify_attempts = 3
            
            while not chapter_completed and retry_count < max_retries:
                print(f"🔄 正在生成章节内容... (temperature={self.temperature:.2f})")
                
                # 准备合规的输入参数
                input_data = json.dumps(
                    {
                        "completed_story": all_pages,
                        "current_chapter": filtered_chapter
                    },
                    ensure_ascii=False
                )
                filtered_input = self.compliance_filter.filter_sensitive(input_data)
                
                # 调用LLM生成
                chapter_detail, success = chapter_writer.call(
                    filtered_input,
                    success_check_fn=parse_list,
                    temperature=self.temperature
                )
                
                # 处理审核失败
                if not success:
                    # 检查是否为审核失败
                    if isinstance(chapter_detail, dict) and chapter_detail.get("code") == "DataInspectionFailed":
                        print(f"❌ 章节 {idx+1} 触发内容审核失败")
                        # 引导用户修改章节摘要
                        new_summary = self._handle_compliance_error(
                            filtered_chapter["chapter_summary"],
                            f"第 {idx+1} 章节摘要"
                        )
                        filtered_chapter["chapter_summary"] = new_summary
                        # 重新准备输入
                        input_data = json.dumps(
                            {
                                "completed_story": all_pages,
                                "current_chapter": filtered_chapter
                            },
                            ensure_ascii=False
                        )
                        filtered_input = self.compliance_filter.filter_sensitive(input_data)
                        # 重新调用
                        chapter_detail, success = chapter_writer.call(
                            filtered_input,
                            success_check_fn=parse_list,
                            temperature=self.temperature
                        )
                
                # 内部重试
                retry_internal = 0
                while success is False and retry_internal < 3:
                    retry_internal += 1
                    print(f"⚠️  生成失败，正在重试 ({retry_internal}/3)...")
                    
                    # 每次重试前优化输入
                    filtered_input = self.compliance_filter.filter_sensitive(input_data)
                    chapter_detail, success = chapter_writer.call(
                        filtered_input,
                        seed=random.randint(0, 100000),
                        temperature=self.temperature,
                        success_check_fn=parse_list
                    )
                
                if not success:
                    print("❌ 生成持续失败，跳过此章节")
                    break
                
                try:
                    pages = [self.compliance_filter.filter_sensitive(page.strip()) for page in eval(chapter_detail)]
                except Exception as e:
                    print(f"❌ 解析内容失败: {e}")
                    pages = ["内容生成失败，请重新尝试"]
                
                # 显示生成的内容
                print(f"\n✏️  第 {idx + 1} 章节内容生成完成:")
                for i, page in enumerate(pages):
                    print(f"\n--- 页面 {i + 1} ---")
                    if len(page) > 300:
                        print(page[:300] + "...")
                        print(f"[内容过长，仅显示前300字符。总长度: {len(page)}字符]")
                    else:
                        print(page)
                
                # 交互选项
                print("\n🔧 请选择操作:")
                print("1. 保留当前内容")
                print("2. 修改内容")
                print("3. 重新生成内容")
                print("4. 查看帮助")
                
                choice = self._get_user_choice(valid_options=['1', '2', '3', '4'])
                
                if choice == '1':
                    print("✅ 内容已保留")
                    all_pages.extend(pages)
                    edit_history.append({
                        "chapter_index": idx,
                        "chapter_title": filtered_chapter["chapter_title"],
                        "action": "保留原始内容",
                        "temperature": self.temperature,
                        "pages": pages,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    chapter_completed = True
                    modify_count = 0
                    
                elif choice == '2':
                    modify_attempts += 1
                    modify_count += 1
                    
                    if modify_count >= 3:
                        print("\n💡 提示: 您已经连续修改多次")
                        print("   建议：尝试提供更具体的修改意见，或考虑调整生成参数")
                    
                    combined_pages = "\n".join(pages)
                    modified_pages_str = self._modify_content(combined_pages, filtered_chapter, all_pages)
                    
                    try:
                        modified_pages = [p.strip() for p in modified_pages_str.split("\n") if p.strip()]
                        if not modified_pages:
                            modified_pages = pages
                    except Exception as e:
                        print(f"❌ 解析修改内容失败: {e}")
                        modified_pages = pages
                    
                    print("\n✅ 修改后的内容:")
                    for i, page in enumerate(modified_pages):
                        print(f"\n--- 页面 {i + 1} ---")
                        if len(page) > 300:
                            print(page[:300] + "...")
                            print(f"[内容过长，仅显示前300字符]")
                        else:
                            print(page)
                    
                    print("\n🔧 确认修改后的内容:")
                    print("1. 确认并保留")
                    print("2. 重新修改")
                    print("3. 放弃修改，使用原始内容")
                    print("4. 调整生成参数")
                    
                    confirm_choice = self._get_user_choice(valid_options=['1', '2', '3', '4'])
                    
                    if confirm_choice == '1':
                        all_pages.extend(modified_pages)
                        edit_history.append({
                            "chapter_index": idx,
                            "chapter_title": filtered_chapter["chapter_title"],
                            "action": "修改内容",
                            "temperature": self.temperature,
                            "original_pages": pages,
                            "modified_pages": modified_pages,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        chapter_completed = True
                        modify_count = 0
                    elif confirm_choice == '3':
                        all_pages.extend(pages)
                        edit_history.append({
                            "chapter_index": idx,
                            "chapter_title": filtered_chapter["chapter_title"],
                            "action": "放弃修改，保留原始内容",
                            "pages": pages,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        chapter_completed = True
                        modify_count = 0
                    elif confirm_choice == '4':
                        self._adjust_generation_params()
                    
                    if modify_attempts >= max_modify_attempts:
                        print(f"\n⚠️  已修改{modify_attempts}次，建议调整策略")
                        self._adjust_generation_params()
                    
                elif choice == '3':
                    retry_count += 1
                    print(f"🔄 正在重新生成内容... (重试 {retry_count}/{max_retries})")
                    
                    if retry_count >= 2:
                        print(f"\n⚠️  您已连续重新生成{retry_count}次")
                        print("🔧 是否调整生成参数？")
                        print("1. 调整参数")
                        print("2. 继续使用当前参数")
                        print("3. 使用随机参数")
                        
                        param_choice = self._get_user_choice(valid_options=['1', '2', '3'])
                        if param_choice == '1':
                            self._adjust_generation_params()
                        elif param_choice == '3':
                            self.temperature = round(random.uniform(0.3, 0.9), 2)
                            print(f"🎲 已设置随机temperature参数: {self.temperature}")
                            
                elif choice == '4':
                    self._show_help()
            
            if retry_count >= max_retries:
                print("⚠️  已达到最大重试次数，使用当前生成的内容")
                all_pages.extend(pages)
                edit_history.append({
                    "chapter_index": idx,
                    "chapter_title": filtered_chapter["chapter_title"],
                    "action": "达到最大重试次数，使用生成内容",
                    "temperature": self.temperature,
                    "pages": pages,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
        
        # 保存编辑历史
        outline["edit_history"] = edit_history
        try:
            with open("edit_history.json", "w", encoding="utf-8") as f:
                json.dump(edit_history, f, ensure_ascii=False, indent=2)
            print(f"\n💾 编辑历史已保存至 edit_history.json")
        except Exception as e:
            print(f"⚠️  保存编辑历史失败: {e}")
        
        # 统计信息
        print("\n📊 故事生成统计:")
        print(f"   • 总章节数: {len(outline['story_outline'])}")
        print(f"   • 总页数: {len(all_pages)}")
        print(f"   • 修改次数: {sum(1 for h in edit_history if h['action'] == '修改内容')}")
        
        return all_pages

    def call(self, params):
        print(f"\n{'✨'*30}")
        print(f"🎭 启动交互式合规故事生成系统")
        print(f"{'✨'*30}")
        
        # 系统提示（新增合规说明）
        print("\n💡 系统提示:")
        print("   • 您可以在每个环节进行内容审核和修改")
        print("   • 连续多次重新生成会触发参数调整建议")
        print("   • 所有编辑操作将被记录在edit_history.json中")
        print("   • 输入 'help' 随时获取帮助信息")
        print("   • 内容必须符合中国法律法规和公序良俗，避免敏感表述")
        
        # 参数显示
        print(f"\n⚙️ 当前设置:")
        print(f"   • 创造性参数(temperature): {self.temperature:.2f}")
        print(f"   • 最大对话轮数: {self.max_conv_turns}")
        print(f"   • 大纲章节数: {self.num_outline}")
        print(f"   • 解析重试次数(max_retry): {self.max_retry}")
        
        # 获取用户主题输入（含合规校验）
        print("\n📝 请指定故事主题（需积极合规，无敏感内容）:")
        print("   示例: '友谊与成长'、'科学探索'、'日常暖心故事'等")
        print("   输入 'help' 获取帮助，直接回车使用默认主题")
        
        if params is None:
            params = {}
        
        while True:
            try:
                story_topic = input("\n故事主题: ").strip()
                
                if story_topic.lower() in ['help', '帮助', '?']:
                    self._show_help()
                    continue
                
                if story_topic:
                    # 合规校验
                    is合规, sensitive_words = self.compliance_filter.check_compliance(story_topic)
                    if not is合规:
                        print(f"❌ 主题包含敏感词：{','.join(sensitive_words)}，请修改")
                        continue
                    params["story_topic"] = self.compliance_filter.filter_sensitive(story_topic)
                    print(f"✅ 已设置故事主题: {params['story_topic']}")
                else:
                    params["story_topic"] = "友谊与成长"
                    print("⚠️  未指定主题，将使用默认合规主题: 友谊与成长")
                
                # 额外设置
                print("\n🔧 是否添加更多故事设置?")
                print("1. 添加主角信息（需合规）")
                print("2. 添加背景设定（需合规）")
                print("3. 输入参考长文本") 
                print("4. 跳过，直接生成") 
                
                setting_choice = self._get_user_choice(valid_options=['1', '2', '3', '4'])
                
                if setting_choice == '1':
                    while True:
                        main_role = input("\n主角描述 (例如: '勇敢的小学生小明'): ").strip()
                        if not main_role:
                            print("⚠️  主角信息不能为空，请重新输入")
                            continue
                        # 合规校验
                        is合规, sensitive_words = self.compliance_filter.check_compliance(main_role)
                        if not is合规:
                            print(f"❌ 主角描述包含敏感词：{','.join(sensitive_words)}，请修改")
                            continue
                        params["main_role"] = self.compliance_filter.filter_sensitive(main_role)
                        print(f"✅ 已添加主角信息: {params['main_role']}")
                        break
                
                elif setting_choice == '2':
                    while True:
                        background = input("\n故事背景 (例如: '和平的校园'): ").strip()
                        if not background:
                            print("⚠️  背景设定不能为空，请重新输入")
                            continue
                        # 合规校验
                        is合规, sensitive_words = self.compliance_filter.check_compliance(background)
                        if not is合规:
                            print(f"❌ 背景设定包含敏感词：{','.join(sensitive_words)}，请修改")
                            continue
                        params["background"] = self.compliance_filter.filter_sensitive(background)
                        print(f"✅ 已添加背景设定: {params['background']}")
                        break
                elif setting_choice == '3':
                    print("\n📄 请输入参考长文本（支持小说片段、情节描述等，将自动总结）:")
                    print("   提示：输入完成后按Ctrl+D（Linux/Mac）或Ctrl+Z（Windows）结束输入")
                    print("   输入 'cancel' 可取消此操作")
                    
                    try:
                        # 读取多行输入
                        long_text_lines = []
                        while True:
                            line = input()
                            if line.lower() == 'cancel':
                                print("⚠️  已取消长文本输入")
                                long_text_lines = []
                                break
                            long_text_lines.append(line)
                        
                        if long_text_lines:
                            long_text = '\n'.join(long_text_lines)
                            # 合规校验
                            is合规, sensitive_words = self.compliance_filter.check_compliance(long_text)
                            if not is合规:
                                print(f"❌ 长文本包含敏感词：{','.join(sensitive_words)}")
                                # 调用合规错误处理
                                long_text = self._handle_compliance_error(long_text, "长文本")
                            
                            params["long_text"] = long_text
                            print(f"✅ 已接收长文本（{len(long_text)}字符），将自动进行内容总结")
                            # 提前展示长文本处理状态
                            if len(long_text) > self.long_text_threshold:
                                print(f"ℹ️  检测到长文本超过{self.long_text_threshold}字符，将进行分段总结")
                    except EOFError:
                        # 用户完成输入
                        if long_text_lines:
                            long_text = '\n'.join(long_text_lines)
                            params["long_text"] = self.compliance_filter.filter_sensitive(long_text)
                            print(f"✅ 已接收长文本（{len(long_text)}字符），将自动进行内容总结")
                        else:
                            print("⚠️  未输入任何内容，已取消")
                    except Exception as e:
                        print(f"❌ 长文本处理错误: {str(e)}")
                break
            except Exception as e:
                print(f"❌ 输入错误: {str(e)}，请重试")
        
        # 生成大纲
        print("\n🔄 正在生成合规故事大纲...")
        outline = self.generate_outline(params)
        
        # 显示大纲
        print(f"\n📋 故事大纲已生成完成")
        print(f"\n📖 故事标题: {outline.get('story_title', '未设置')}")
        print(f"\n📑 大纲章节 ({len(outline['story_outline'])} 章):")
        for i, chapter in enumerate(outline["story_outline"], 1):
            filtered_summary = self.compliance_filter.filter_sensitive(chapter['chapter_summary'])
            print(f"   {i}. {chapter['chapter_title']} - {filtered_summary[:80]}...")
        
        print("\n🔄 是否基于此大纲生成故事内容?")
        print("1. 是，开始生成")
        print("2. 调整生成参数")
        print("3. 重新生成大纲")
        
        choice = self._get_user_choice(valid_options=['1', '2', '3'])
        if choice == '2':
            self._adjust_generation_params()
        elif choice == '3':
            print("\n🔄 重新生成大纲...")
            outline = self.generate_outline(params)
        
        # 生成故事内容
        print("\n🚀 开始生成合规故事内容...")
        pages = self.generate_story_from_outline(outline)
        
        # 最终确认
        print(f"\n🎉 合规故事生成全部完成！共 {len(pages)} 页内容")
        print("\n📊 最终统计信息:")
        print(f"   • 总页数: {len(pages)}")
        print(f"   • 总字符数: {sum(len(page) for page in pages)}")
        print(f"   • 使用参数: temperature={self.temperature:.2f}, max_retry={self.max_retry}")
        print(f"\n💡 提示: 所有内容已通过合规过滤，可安全使用")
        print(f"\n💾 编辑历史已保存至 edit_history.json")
        
        return pages