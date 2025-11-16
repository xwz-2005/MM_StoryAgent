import json
from typing import Dict
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

LONG_TEXT_SUMMARIZER_SYSTEM = """
你是一名专业的文本摘要分析师，需要从长文本中提取故事创作所需的关键信息。
请分析输入的长文本，提炼出以下要素：
1. 核心主题（故事的中心思想）
2. 主要角色（名称及核心特征）
3. 关键场景（时间和地点）
4. 核心情节（起承转合的关键节点）
5. 情感基调（整体氛围）

输出格式为JSON对象，确保信息完整且简洁，便于后续故事创作使用。
"""

@register_tool("qa_outline_story_writer")
class QAOutlineStoryWriter:

    def __init__(self,
                 cfg: Dict):
        self.cfg = cfg
        self.temperature = cfg.get("temperature", 1.0)
        self.max_conv_turns = cfg.get("max_conv_turns", 3)
        self.num_outline = cfg.get("num_outline", 4)
        self.llm_type = cfg.get("llm", "qwen")
        self.long_text_threshold = cfg.get("long_text_threshold", 500)

    def _summarize_long_text(self, long_text: str) -> Dict:
        """总结长文本并提取关键信息"""
        print("📄 正在处理长文本...")
        summarizer = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": LONG_TEXT_SUMMARIZER_SYSTEM,
                "track_history": False
            }
        })
        
        # 处理超长文本（分段处理）
        chunks = self._split_long_text(long_text)
        chunk_summaries = []
        
        for i, chunk in enumerate(chunks):
            print(f"🔍 处理文本片段 {i+1}/{len(chunks)}")
            summary, success = summarizer.call(
                f"长文本片段 {i+1}：{chunk}\n请按照要求格式输出摘要JSON",
                temperature=self.temperature
            )
            if success:
                chunk_summaries.append(json.loads(summary))
        
        # 整合片段摘要
        if len(chunk_summaries) > 1:
            final_summary, success = summarizer.call(
                f"请整合以下片段摘要，生成完整的长文本摘要：{json.dumps(chunk_summaries, ensure_ascii=False)}",
                temperature=self.temperature
            )
            return json.loads(final_summary)
        
        return chunk_summaries[0] if chunk_summaries else {}

    def _split_long_text(self, text: str, chunk_size: int = 2000) -> List[str]:
        """将长文本分割为适合模型处理的片段"""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
        return chunks
    
    def _analyze_story_style(self, story_setting):
        """主题风格自动分析"""
        print("🔍 正在进行主题风格自动分析...")
        
        # 将故事设置转换为字符串便于分析
        setting_str = json.dumps(story_setting, ensure_ascii=False)
        
        # 定义关键词库
        keywords = {
            "日常写实": [
                # 现实场景
                "学校", "宿舍", "公司", "家庭", "医院", "公园", "商场", "街道", "办公室",
                "教室", "图书馆", "餐厅", "咖啡厅", "车站", "机场", "银行", "超市",
                # 真实职业
                "学生", "老师", "医生", "护士", "警察", "消防员", "工程师", "程序员",
                "设计师", "销售人员", "服务员", "厨师", "司机", "科学家", "研究员",
                # 日常行为
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
        
        # 计算每种风格的匹配分数
        scores = {style: 0 for style in keywords.keys()}
        for style, style_keywords in keywords.items():
            for keyword in style_keywords:
                if keyword in setting_str:
                    scores[style] += 1
        
        # 确定主导风格
        dominant_style = max(scores, key=scores.get)
        
        # 如果所有分数都很低，默认使用日常写实
        if scores[dominant_style] == 0:
            dominant_style = "日常写实"
        
        # 根据风格生成规则
        style_rules = {
            "日常写实": "仅包含现实中存在的场景（如学校、宿舍）、角色（如学生、老师）和行为（如上课、编程），禁止任何虚构元素（魔法、会说话的动物、神秘力量等）",
            "奇幻": "允许包含魔法、虚构生物等奇幻元素，场景和角色可虚构，但需符合奇幻逻辑",
            "科幻": "聚焦未来科技、太空探索等元素，禁止无逻辑的超自然力量，需符合科学幻想设定"
        }
        
        print(f"✅ 风格分析完成: {dominant_style}")
        print(f"📝 生成规则: {style_rules[dominant_style]}")
        
        return dominant_style, style_rules[dominant_style]
    
    def generate_outline(self, params):
        # `params`: story setting like 
        # {
        #     "story_title": "xxx",
        #     "main_role": "xxx",
        #     ......
        # }
        
        # 检查是否为长文本输入
        if "long_text" in params and len(params["long_text"]) > self.long_text_threshold:
            # 长文本处理流程：long_text → summarize → story setting
            summary = self._summarize_long_text(params["long_text"])
            
            # 从摘要构建故事设置
            story_setting = {
                "story_topic": summary.get("核心主题", "基于长文本的故事"),
                "main_role": summary.get("主要角色", "未明确角色"),
                "scene": summary.get("关键场景", "未明确场景"),
                "emotional_tone": summary.get("情感基调", ""),
                "core_plot": summary.get("核心情节", "")
            }
            print("📝 长文本处理完成，生成故事设置")
        else:
            story_setting = params
            
        # 添加主题风格自动分析
        dominant_style, style_rule = self._analyze_story_style(params)
        
        asker = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": question_asker_system,
                "track_history": False
            }
        })
        expert = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": expert_system,
                "track_history": False
            }
        })

        dialogue = []
        for turn in trange(self.max_conv_turns):
            dialogue_history = "\n".join(dialogue)
            
            question, success = asker.call(
                f"Story setting: {params}\nDialogue history: \n{dialogue_history}\n",
                temperature=self.temperature
            )
            question = question.strip()
            if question == "Thank you for your help!":
                break
            dialogue.append(f"You: {question}")
            answer, success = expert.call(
                f"Story setting: {params}\nQuestion: \n{question}\nAnswer: ",
                temperature=self.temperature
            )
            answer = answer.strip()
            dialogue.append(f"Expert: {answer}")

        # print("\n".join(dialogue))
        writer = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": dlg_based_writer_system,
                "track_history": False
            }
        })
        writer_prompt = dlg_based_writer_prompt.format(
            story_setting=params,
            dialogue_history="\n".join(dialogue),
            num_outline=self.num_outline,
            style_type=dominant_style,
            style_rule=style_rule
        )

        outline, success = writer.call(writer_prompt, success_check_fn=json_parse_outline)
        outline = json.loads(outline)
        # print(outline)
        return outline

    def _get_user_choice(self, prompt="请选择操作: ", valid_options=['1', '2', '3'], max_attempts=5):
        """获取并验证用户选择，支持自定义选项和最大尝试次数"""
        attempts = 0
        while attempts < max_attempts:
            try:
                # 检查用户是否需要帮助
                if prompt.strip().lower() in ['help', '帮助', '?']:
                    self._show_help()
                    continue
                
                print(prompt)
                choice = input().strip()
                
                # 支持帮助命令
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
        return valid_options[0]  # 默认返回第一个选项
    
    def _show_help(self):
        """显示帮助信息"""
        print("\n📚 帮助信息:")
        print("   • 在任何输入提示时，输入 'help'、'帮助' 或 '?' 可显示此帮助")
        print("   • 所有操作都会被记录在edit_history.json中")
        print("   • 连续多次重新生成会触发参数调整建议")
        print("   • 修改内容时，请尽量提供具体、明确的修改建议")
        print("   • 如遇问题，可按Ctrl+C中断程序")
        print()
    
    def _get_modification_input(self, content_type="内容"):
        """获取用户修改意见，提供引导和示例"""
        print(f"\n✏️  请输入您对{content_type}的修改意见:")
        print("   示例: '增加主角的心理活动' 或 '让场景更黑暗一些'")
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
                    return modification
            except Exception:
                print("❌ 输入错误，请重试")
    
    def _modify_content(self, original_content, chapter, all_pages):
        """处理内容修改"""
        # 使用增强的修改输入方法
        modification = self._get_modification_input("章节")
        if modification is None:
            return original_content
        
        # 创建修改提示词
        modify_prompt = json.dumps({
            "original_content": original_content,
            "chapter_info": chapter,
            "modification_request": modification,
            "completed_story": all_pages
        }, ensure_ascii=False)
        
        # 调用LLM进行修改，修改时略微降低创造性以保持一致性
        modifier = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": "你是一个专业的故事编辑助手。请根据用户的修改意见，对提供的故事内容进行精准修改。保持故事的连贯性和风格一致性。直接输出修改后的完整内容，不要添加任何额外的解释或标记。",
                "track_history": False
            }
        })
        
        print("🔄 正在根据您的意见修改内容...")
        modified_content, success = modifier.call(
            modify_prompt,
            temperature=max(0.5, self.temperature * 0.9),  # 修改时降低创造性
            success_check_fn=lambda x: x.strip() != ""
        )
        
        if success:
            return modified_content.strip()
        else:
            print("⚠️ 修改失败，保留原始内容")
            return original_content
    
    def _adjust_generation_params(self):
        """调整生成参数"""
        print("\n⚙️  调整生成参数")
        print("请选择要调整的参数:")
        print("1. temperature (创造性控制，当前值: {:.2f})".format(self.temperature))
        print("2. 返回")
        
        choice = self._get_user_choice(valid_options=['1', '2'])
        
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
    
    def generate_story_from_outline(self, outline):
        chapter_writer = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": chapter_writer_system,
                "track_history": False
            }
        })
        all_pages = []
        
        # 初始化编辑历史
        edit_history = []
        modify_count = 0  # 跟踪修改次数
        
        # 显示大纲概览
        print(f"\n📋 故事大纲概览:")
        for i, chap in enumerate(outline["story_outline"], 1):
            print(f"   {i}. {chap['chapter_title']}")
        
        for idx, chapter in enumerate(tqdm(outline["story_outline"])):
            print(f"\n{'='*60}")
            print(f"📖 正在生成第 {idx + 1} 章节: {chapter['chapter_title']}")
            print(f"{'='*60}")
            
            chapter_completed = False
            retry_count = 0
            max_retries = 3
            modify_attempts = 0
            max_modify_attempts = 3
            
            while not chapter_completed and retry_count < max_retries:
                # 生成章节内容
                print(f"🔄 正在生成章节内容... (temperature={self.temperature:.2f})")
                chapter_detail, success = chapter_writer.call(
                    json.dumps(
                        {
                            "completed_story": all_pages,
                            "current_chapter": chapter
                        },
                        ensure_ascii=False
                    ),
                    success_check_fn=parse_list,
                    temperature=self.temperature
                )
                
                # 重试直到成功生成
                retry_internal = 0
                while success is False and retry_internal < 3:
                    retry_internal += 1
                    print(f"⚠️  生成失败，正在重试 ({retry_internal}/3)...")
                    chapter_detail, success = chapter_writer.call(
                        json.dumps(
                            {
                                "completed_story": all_pages,
                                "current_chapter": chapter
                            },
                            ensure_ascii=False
                        ),
                        seed=random.randint(0, 100000),
                        temperature=self.temperature,
                        success_check_fn=parse_list
                    )
                
                if not success:
                    print("❌ 生成持续失败，跳过此章节")
                    break
                
                try:
                    pages = [page.strip() for page in eval(chapter_detail)]
                except Exception as e:
                    print(f"❌ 解析内容失败: {e}")
                    pages = ["内容生成失败，请重新尝试"]
                
                # 显示生成的内容（优化长内容显示）
                print(f"\n✏️  第 {idx + 1} 章节内容生成完成:")
                for i, page in enumerate(pages):
                    print(f"\n--- 页面 {i + 1} ---")
                    if len(page) > 300:
                        print(page[:300] + "...")
                        print(f"[内容过长，仅显示前300字符。总长度: {len(page)}字符]")
                    else:
                        print(page)
                
                # 提供交互选项
                print("\n🔧 请选择操作:")
                print("1. 保留当前内容")
                print("2. 修改内容")
                print("3. 重新生成内容")
                print("4. 查看帮助")
                
                choice = self._get_user_choice(valid_options=['1', '2', '3', '4'])
                
                if choice == '1':
                    # 保留当前内容
                    print("✅ 内容已保留")
                    all_pages.extend(pages)
                    edit_history.append({
                        "chapter_index": idx,
                        "chapter_title": chapter["chapter_title"],
                        "action": "保留原始内容",
                        "temperature": self.temperature,
                        "pages": pages,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    chapter_completed = True
                    # 重置修改计数
                    modify_count = 0
                    
                elif choice == '2':
                    # 修改内容
                    modify_attempts += 1
                    modify_count += 1
                    
                    # 检测连续修改
                    if modify_count >= 3:
                        print("\n💡 提示: 您已经连续修改多次")
                        print("   建议：尝试提供更具体的修改意见，或考虑调整生成参数")
                    
                    combined_pages = "\n".join(pages)
                    modified_pages_str = self._modify_content(combined_pages, chapter, all_pages)
                    
                    # 尝试将修改后的内容分割回原始格式
                    try:
                        modified_pages = [p.strip() for p in modified_pages_str.split("\n") if p.strip()]
                        if not modified_pages:
                            modified_pages = pages  # 如果分割失败，保留原始内容
                    except Exception as e:
                        print(f"❌ 解析修改内容失败: {e}")
                        modified_pages = pages
                    
                    # 显示修改后的内容
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
                            "chapter_title": chapter["chapter_title"],
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
                            "chapter_title": chapter["chapter_title"],
                            "action": "放弃修改，保留原始内容",
                            "pages": pages,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        chapter_completed = True
                        modify_count = 0
                    elif confirm_choice == '4':
                        self._adjust_generation_params()
                    # 选择2则继续循环修改
                    
                    # 检测多次修改
                    if modify_attempts >= max_modify_attempts:
                        print(f"\n⚠️  已修改{modify_attempts}次，建议调整策略")
                        self._adjust_generation_params()
                    
                elif choice == '3':
                    # 重新生成内容
                    retry_count += 1
                    print(f"🔄 正在重新生成内容... (重试 {retry_count}/{max_retries})")
                    
                    # 连续重新生成时，询问是否调整参数
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
                    # 显示帮助
                    self._show_help()
            
            if retry_count >= max_retries:
                # 达到最大重试次数，使用最后生成的内容
                print("⚠️  已达到最大重试次数，使用当前生成的内容")
                all_pages.extend(pages)
                edit_history.append({
                    "chapter_index": idx,
                    "chapter_title": chapter["chapter_title"],
                    "action": "达到最大重试次数，使用生成内容",
                    "temperature": self.temperature,
                    "pages": pages,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
        
        # 将编辑历史保存到outline中，并同时保存到文件
        outline["edit_history"] = edit_history
        
        # 保存编辑历史到文件
        try:
            with open("edit_history.json", "w", encoding="utf-8") as f:
                json.dump(edit_history, f, ensure_ascii=False, indent=2)
            print(f"\n💾 编辑历史已保存至 edit_history.json")
        except Exception as e:
            print(f"⚠️  保存编辑历史失败: {e}")
        
        # 显示统计信息
        print("\n📊 故事生成统计:")
        print(f"   • 总章节数: {len(outline['story_outline'])}")
        print(f"   • 总页数: {len(all_pages)}")
        print(f"   • 修改次数: {sum(1 for h in edit_history if h['action'] == '修改内容')}")
        
        return all_pages
        # print(all_pages)
        return all_pages

    def call(self, params):
        print(f"\n{'✨'*30}")
        print(f"🎭 启动交互式故事生成系统")
        print(f"{'✨'*30}")
        
        # 显示系统提示
        print("\n💡 系统提示:")
        print("   • 您可以在每个环节进行内容审核和修改")
        print("   • 连续多次重新生成会触发参数调整建议")
        print("   • 所有编辑操作将被记录在edit_history.json中")
        print("   • 输入 'help' 随时获取帮助信息")
        
        # 显示当前参数设置
        print(f"\n⚙️ 当前设置:")
        print(f"   • 创造性参数(temperature): {self.temperature:.2f}")
        print(f"   • 最大对话轮数: {self.max_conv_turns}")
        print(f"   • 大纲章节数: {self.num_outline}")
        
        # 获取用户主题输入
        print("\n📝 请指定故事主题:")
        print("   示例: '科幻冒险'、'童话故事'、'历史悬疑'等")
        print("   输入 'help' 获取帮助，直接回车使用默认主题")
        
        # 初始化params字典
        if params is None:
            params = {}
        
        while True:
            try:
                story_topic = input("\n故事主题: ").strip()
                
                if story_topic.lower() in ['help', '帮助', '?']:
                    self._show_help()
                    continue
                
                if story_topic:
                    params["story_topic"] = story_topic
                    print(f"✅ 已设置故事主题: {story_topic}")
                else:
                    params["story_topic"] = "奇幻冒险"
                    print("⚠️  未指定主题，将使用默认主题: 奇幻冒险")
                
                # 询问是否添加更多设置
                print("\n🔧 是否添加更多故事设置?")
                print("1. 添加主角信息")
                print("2. 添加背景设定")
                print("3. 跳过，直接生成")
                
                setting_choice = self._get_user_choice(valid_options=['1', '2', '3'])
                
                if setting_choice == '1':
                    main_role = input("\n主角描述 (例如: '勇敢的骑士亚瑟'): ").strip()
                    if main_role:
                        params["main_role"] = main_role
                        print(f"✅ 已添加主角信息: {main_role}")
                elif setting_choice == '2':
                    background = input("\n故事背景 (例如: '魔法王国艾尔文'): ").strip()
                    if background:
                        params["background"] = background
                        print(f"✅ 已添加背景设定: {background}")
                
                break
            except Exception as e:
                print(f"❌ 输入错误: {str(e)}，请重试")
        
        # 生成故事大纲
        print("\n🔄 正在生成故事大纲...")
        outline = self.generate_outline(params)
        
        # 显示大纲并确认
        print(f"\n📋 故事大纲已生成完成")
        print(f"\n📖 故事标题: {outline.get('story_title', '未设置')}")
        print(f"\n📑 大纲章节 ({len(outline['story_outline'])} 章):")
        for i, chapter in enumerate(outline["story_outline"], 1):
            print(f"   {i}. {chapter['chapter_title']} - {chapter['chapter_summary'][:80]}...")
        
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
        
        # 根据大纲生成故事内容
        print("\n🚀 开始生成故事内容...")
        pages = self.generate_story_from_outline(outline)
        
        # 最终确认
        print(f"\n🎉 故事生成全部完成！共 {len(pages)} 页内容")
        print("\n📊 最终统计信息:")
        print(f"   • 总页数: {len(pages)}")
        print(f"   • 总字符数: {sum(len(page) for page in pages)}")
        print(f"   • 使用参数: temperature={self.temperature:.2f}")
        print(f"\n💾 编辑历史已保存至 edit_history.json")
        
        return pages
