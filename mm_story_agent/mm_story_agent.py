import time
import json
import os
import re
from pathlib import Path
import traceback

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

from .base import init_tool_instance





class MMStoryAgent:

    def __init__(self) -> None:
        # 任务要求：去除音乐和音效模块，只保留图像和语音
        self.modalities = ["image", "speech"]
        # 用于操作回退的历史记录
        self.history_stack = []

    # 调用指定模态的代理处理任务，并将结果收集到返回字典中。
    def call_modality_agent(self, modality, agent, params, return_dict):
        result = agent.call(params)
        return_dict[modality] = result

    def _generate_default_prompt(self, user_input, story_type="realistic"):
        """
        生成默认的故事提示词，避免奇幻色彩
        
        Args:
            user_input: 用户输入的主题
            story_type: 故事类型，默认为realistic（现实）
            
        Returns:
            str: 生成的提示词
        """
        # 根据故事类型选择合适的提示词模板
        if story_type == "realistic":
            template = f"""
请生成一个基于以下主题的现实主义故事：
{user_input}

要求：
1. 故事必须基于现实世界，不包含任何魔法、超能力、虚构生物或幻想元素
2. 情节要合理，符合现实逻辑和自然规律
3. 人物和场景描述要具体、真实
4. 故事应具有一定的教育意义或积极正面的价值观
5. 语言流畅，叙述自然
6. 每个页面保持在150-200字左右，内容简洁明了
"""
        elif story_type == "scientific":
            template = f"""
请生成一个基于以下主题的科学知识故事：
{user_input}

要求：
1. 故事必须基于真实的科学知识和原理
2. 可以包含科学探索、发现或应用的情节
3. 避免任何虚构或不科学的元素
4. 内容应准确反映现实中的科学事实
5. 适合教育和科普目的
6. 每个页面保持在150-200字左右
"""
        else:  # 通用模板，仍避免奇幻元素
            template = f"""
请生成一个基于以下主题的故事：
{user_input}

要求：
1. 故事内容应贴近现实生活
2. 避免包含魔法、超能力、虚构生物或奇幻元素
3. 情节要连贯合理
4. 人物性格鲜明，场景描述具体
5. 语言简洁流畅
6. 每个页面保持在150-200字左右
"""
        
        return template.strip()
    
    def write_story(self, config):
        print(f"\n{'='*60}")
        topic = config.get('topic', '未指定主题')
        print(f"🎯 主题: {topic}")
        print(f"{'='*60}")
        
        # 让用户选择故事类型以避免奇幻元素
        print("\n📖 请选择故事类型:")
        print("1. 现实主义故事（默认，避免奇幻元素）")
        print("2. 科学知识故事（基于真实科学原理）")
        print("3. 普通故事（仍避免奇幻元素）")
        
        story_type_choice = input("请选择 (1-3，默认1): ").strip()
        story_type_map = {
            "1": "realistic",
            "2": "scientific",
            "3": "general"
        }
        story_type = story_type_map.get(story_type_choice, "realistic")
        
        # 生成默认提示词，避免奇幻色彩
        default_prompt = self._generate_default_prompt(topic, story_type)
        
        # 更新配置中的参数，添加无奇幻元素要求
        story_writer_params = config["story_writer"]["params"].copy()
        # 如果配置中有prompt参数，则修改它，否则添加
        if "prompt" in story_writer_params:
            story_writer_params["prompt"] += "\n\n" + default_prompt
        else:
            story_writer_params["prompt"] = default_prompt
        
        # 初始化故事生成代理
        story_writer = init_tool_instance(config["story_writer"])
        
        # 调用代理的call方法
        print("\n📝 正在生成故事...")
        pages = story_writer.call(story_writer_params)
        
        print(f"\n✅ 故事生成完成，共 {len(pages)} 页内容")
        return pages

    # 并行生成图像和语音
    def _get_user_choice(self, prompt="请选择操作: "):
        """获取并验证用户选择"""
        while True:
            try:
                choice = input(prompt).strip()
                if choice in ['1', '2', '3']:
                    return choice
                else:
                    print("❌ 请输入正确选项（1/2/3）")
            except Exception:
                print("❌ 输入错误，请重试")
                
    def find_story_files(self, story_dir=None):
        """
        递归查找所有可用的故事JSON文件，包括子文件夹中的文件
        
        Args:
            story_dir: 故事目录路径
            
        Returns:
            list: 找到的JSON文件列表，包含文件路径、相对路径、名称等信息
        """
        if story_dir is None:
            story_dir = Path("./outputs")
        else:
            story_dir = Path(story_dir)
            
        story_files = []
        
        # 检查目录是否存在
        if not story_dir.exists() or not story_dir.is_dir():
            print(f"⚠️  故事目录不存在: {story_dir}")
            return story_files
        
        print(f"🔍 正在递归扫描故事目录: {story_dir}")
        
        # 递归查找所有JSON文件
        try:
            # 使用glob递归查找所有层级的JSON文件
            for file in story_dir.glob("**/*.json"):
                if file.is_file():
                    # 获取文件大小和修改时间
                    file_size = file.stat().st_size
                    mod_time = file.stat().st_mtime
                    
                    # 获取相对路径，用于显示文件在哪个子文件夹中
                    rel_path = str(file.relative_to(story_dir))
                    
                    # 尝试解析文件以获取更多信息
                    story_info = {
                        'path': str(file),
                        'rel_path': rel_path,  # 添加相对路径信息
                        'name': file.name,
                        'size': file_size,
                        'mod_time': mod_time,
                        'parent_dir': str(file.parent)  # 添加父目录信息，用于会话关联
                    }
                    
                    # 尝试读取文件内容获取故事信息
                    try:
                        # 尝试多种编码格式
                        encodings = ['utf-8', 'gbk', 'gb2312', 'cp936']
                        data = None
                        
                        for encoding in encodings:
                            try:
                                with open(file, 'r', encoding=encoding) as f:
                                    data = json.load(f)
                                break
                            except (UnicodeDecodeError, json.JSONDecodeError):
                                continue
                        
                        if data is None:
                            raise Exception("无法使用尝试的编码格式解析文件")
                            # 尝试提取页面数量
                            if isinstance(data, list):
                                story_info['pages'] = len(data)
                                # 尝试获取第一页作为预览
                                if data:
                                    if isinstance(data[0], str):
                                        story_info['preview'] = data[0][:50] + '...' if len(data[0]) > 50 else data[0]
                                    elif isinstance(data[0], dict):
                                        story_info['preview'] = str(list(data[0].values())[0])[:50] + '...' if data[0] else "(空页面)"
                            elif isinstance(data, dict):
                                if 'pages' in data and isinstance(data['pages'], list):
                                    story_info['pages'] = len(data['pages'])
                                    # 尝试获取第一页作为预览
                                    if data['pages']:
                                        if isinstance(data['pages'][0], str):
                                            story_info['preview'] = data['pages'][0][:50] + '...' if len(data['pages'][0]) > 50 else data['pages'][0]
                                        elif isinstance(data['pages'][0], dict):
                                            story_info['preview'] = str(list(data['pages'][0].values())[0])[:50] + '...' if data['pages'][0] else "(空页面)"
                                else:
                                    story_info['pages'] = 1
                                    story_info['preview'] = str(list(data.values())[0])[:50] + '...' if data else "(空故事)"
                    except Exception as e:
                        # 如果无法解析文件，仍然添加到列表
                        story_info['pages'] = '未知'
                        story_info['preview'] = '(无法解析内容)'
                    
                    story_files.append(story_info)
                    
        except Exception as e:
            print(f"❌ 扫描故事目录时出错: {str(e)}")
        
        # 按修改时间排序，最新的在前
        story_files.sort(key=lambda x: x['mod_time'], reverse=True)
        
        print(f"✅ 共找到 {len(story_files)} 个故事文件")
        
        return story_files
        
    def select_story_file(self, config):
        """
        自动列出并让用户选择要加载的故事文件，支持子文件夹浏览和选择
        
        Args:
            config: 配置信息
            
        Returns:
            str: 选中的文件路径，如果未选择则返回None
        """
        story_dir = Path(config.get("story_dir", "./outputs"))
        story_files = self.find_story_files(story_dir)
        
        if not story_files:
            print("❌ 没有找到可用的故事文件")
            # 尝试扫描其他可能的目录
            other_dirs = [Path("."), Path("../outputs")]
            for other_dir in other_dirs:
                if other_dir != story_dir:
                    other_files = self.find_story_files(other_dir)
                    if other_files:
                        print(f"\n在 {other_dir} 找到以下文件:")
                        story_files = other_files
                        break
            
            # 如果仍然没有找到，提供手动输入选项
            if not story_files:
                print("\n📋 您可以选择手动输入文件路径")
                print("1. 手动输入文件路径")
                print("2. 返回")
                
                choice = input("请选择操作 (1/2): ").strip()
                if choice == '1':
                    json_path = input("请输入故事JSON文件路径: ").strip()
                    if json_path and Path(json_path).exists():
                        return json_path
                return None
        
        # 按文件夹分组显示文件
        # 首先获取所有唯一的文件夹路径
        folders = set()
        for file_info in story_files:
            # 从相对路径中提取文件夹部分
            rel_path = file_info.get('rel_path', '')
            if '/' in rel_path or '\\' in rel_path:
                # Windows和Unix路径分隔符兼容
                folder_part = rel_path.rsplit('/', 1)[0] if '/' in rel_path else rel_path.rsplit('\\', 1)[0]
                folders.add(folder_part)
            else:
                folders.add('.')  # 根目录下的文件
        
        # 排序文件夹，将根目录放在最前面
        sorted_folders = sorted(folders)
        if '.' in sorted_folders:
            sorted_folders.remove('.')
            sorted_folders.insert(0, '.')
        
        # 提供文件夹选择界面
        print("\n📂 可用的文件夹:")
        print("=" * 60)
        print(f"{'序号':<5} {'文件夹':<45} {'文件数':<10}")
        print("=" * 60)
        
        # 统计每个文件夹的文件数
        folder_file_counts = {}
        for folder in sorted_folders:
            if folder == '.':
                # 根目录下的文件
                count = sum(1 for f in story_files if '/' not in f.get('rel_path', '') and '\\' not in f.get('rel_path', ''))
            else:
                # 子文件夹中的文件
                count = sum(1 for f in story_files if f.get('rel_path', '').startswith(folder + '/') or f.get('rel_path', '').startswith(folder + '\\'))
            folder_file_counts[folder] = count
            display_name = '根目录' if folder == '.' else folder
            print(f"{sorted_folders.index(folder) + 1:<5} {display_name[:45]:<45} {count:<10}")
        
        print("=" * 60)
        print(f"{0:<5} {'返回':<45}")
        
        # 获取文件夹选择
        while True:
            try:
                folder_choice = input("\n请选择要浏览的文件夹序号: ").strip()
                if folder_choice == '0':
                    return None
                    
                folder_idx = int(folder_choice) - 1
                if 0 <= folder_idx < len(sorted_folders):
                    selected_folder = sorted_folders[folder_idx]
                    break
                else:
                    print(f"❌ 请输入1-{len(sorted_folders)}之间的数字")
            except ValueError:
                print("❌ 无效的输入，请输入数字")
        
        # 过滤出选定文件夹中的文件
        filtered_files = []
        for file_info in story_files:
            rel_path = file_info.get('rel_path', '')
            if selected_folder == '.':
                # 根目录下的文件（没有子文件夹）
                if '/' not in rel_path and '\\' not in rel_path:
                    filtered_files.append(file_info)
            else:
                # 子文件夹中的文件
                if rel_path.startswith(selected_folder + '/') or rel_path.startswith(selected_folder + '\\'):
                    filtered_files.append(file_info)
        
        # 显示选定文件夹中的文件列表
        print(f"\n📚 {selected_folder if selected_folder != '.' else '根目录'} 中的故事文件:")
        print("=" * 100)
        print(f"{'序号':<5} {'相对路径':<35} {'文件名':<25} {'页数':<10} {'预览':<20}")
        print("=" * 100)
        
        for i, file_info in enumerate(filtered_files, 1):
            pages = file_info.get('pages', '未知')
            preview = file_info.get('preview', '')
            rel_path = file_info.get('rel_path', file_info['name'])
            display_rel_path = rel_path[:35] + '...' if len(rel_path) > 35 else rel_path
            print(f"{i:<5} {display_rel_path:<35} {file_info['name'][:25]:<25} {str(pages):<10} {preview[:20]:<20}")
        
        print("=" * 100)
        print(f"{0:<5} {'返回文件夹选择':<35}")
        
        # 获取文件选择
        while True:
            try:
                file_choice = input("\n请选择要加载的故事序号: ").strip()
                if file_choice == '0':
                    # 返回重新选择文件夹
                    return self.select_story_file(config)
                    
                file_idx = int(file_choice) - 1
                if 0 <= file_idx < len(filtered_files):
                    selected_file = filtered_files[file_idx]
                    print(f"\n✅ 已选择文件: {selected_file['rel_path']}")
                    print(f"   文件路径: {selected_file['path']}")
                    return selected_file['path']
                else:
                    print(f"❌ 请输入1-{len(filtered_files)}之间的数字")
            except ValueError:
                print("❌ 无效的输入，请输入数字")
                
    def load_story_from_json(self, json_path: str = None, config = None) -> bool:
        """
        从JSON文件加载故事内容，支持子文件夹中的文件
        
        Args:
            json_path: JSON文件路径
            config: 配置信息，用于自动选择文件
            
        Returns:
            bool: 加载是否成功
        """
        # 如果没有提供文件路径，让用户选择
        if json_path is None and config is not None:
            json_path = self.select_story_file(config)
            if json_path is None:
                return False
        
        if json_path is None:
            print("❌ 未提供文件路径")
            return False
        
        try:
            # 转换为绝对路径，确保跨平台兼容性
            json_path = str(Path(json_path).absolute())
            print(f"🔄 正在从 {json_path} 加载故事...")
            
            # 保存当前状态用于回退
            self._save_to_history_stack()
            
            # 读取JSON文件，尝试多种编码格式
            encodings = ['utf-8', 'gbk', 'gb2312', 'cp936']
            data = None
            
            for encoding in encodings:
                try:
                    with open(json_path, 'r', encoding=encoding) as f:
                        data = json.load(f)
                    print(f"📄 成功使用 {encoding} 编码打开文件")
                    break
                except UnicodeDecodeError:
                    continue
                except json.JSONDecodeError:
                    continue
            
            if data is None:
                raise Exception(f"无法使用尝试的编码格式解析文件: {json_path}")
            
            # 提取故事页面信息
            if isinstance(data, list):
                # 如果JSON直接是页面列表
                pages = data
            elif isinstance(data, dict):
                # 如果JSON是包含页面的字典
                if "pages" in data:
                    pages = data["pages"]
                elif "story" in data:
                    pages = data["story"]
                else:
                    # 尝试作为单个页面处理
                    pages = [data]
            else:
                print("❌ JSON文件格式错误: 数据类型不支持")
                return False
            
            # 处理页面数据，确保都是字符串
            processed_pages = []
            for page in pages:
                if isinstance(page, str):
                    processed_pages.append(page.strip())
                elif isinstance(page, dict):
                    # 尝试提取内容
                    if 'content' in page:
                        processed_pages.append(str(page['content']).strip())
                    elif 'story' in page:
                        processed_pages.append(str(page['story']).strip())
                    else:
                        # 使用字典的第一个值
                        processed_pages.append(str(list(page.values())[0]).strip() if page else "")
                else:
                    processed_pages.append(str(page).strip())
            
            # 过滤空页面
            processed_pages = [page for page in processed_pages if page]
            
            if not processed_pages:
                print("❌ 加载的故事内容为空！")
                return False
            
            # 保存加载的页面
            self.pages = processed_pages
            
            # 保存当前加载文件的路径，用于后续修改和保存
            self.current_json_path = json_path
            
            # 提取并保存会话ID（从文件路径中提取，因为文件夹名就是会话ID）
            # 获取文件所在的目录
            file_dir = str(Path(json_path).parent)
            # 会话ID通常是时间戳格式的目录名
            session_id = file_dir.split(os.sep)[-1] if file_dir else ""
            
            # 检查会话ID是否是有效的时间戳格式
            import re
            timestamp_pattern = r'\d{8}_\d{6}'  # 格式：YYYYMMDD_HHMMSS
            if re.match(timestamp_pattern, session_id):
                self.current_session_id = session_id
                print(f"📝 已识别会话ID: {session_id}")
            else:
                # 如果不是标准时间戳格式，可能是根目录下的文件
                self.current_session_id = ""
                print(f"📝 文件位于: {file_dir}")
            
            print(f"✅ 成功加载了 {len(processed_pages)} 页故事内容！")
            
            # 显示故事预览
            print("\n📖 故事预览:")
            for i, page in enumerate(processed_pages[:3], 1):  # 只显示前3页预览
                preview = page[:100] + '...' if len(page) > 100 else page
                print(f"   第{i}页: {preview}")
            if len(processed_pages) > 3:
                print(f"   ... 还有 {len(processed_pages) - 3} 页内容")
            
            return True
            
        except FileNotFoundError:
            print(f"❌ 找不到文件: {json_path}")
            return False
        except json.JSONDecodeError:
            print(f"❌ JSON文件格式错误: {json_path}")
            return False
        except Exception as e:
            print(f"❌ 加载故事时发生错误: {str(e)}")
            traceback.print_exc()
            return False
            
    def _save_to_history_stack(self):
        """保存当前状态到历史栈，用于操作回退"""
        # 保存页面的深拷贝
        import copy
        if hasattr(self, 'pages'):
            self.history_stack.append({
                "pages": copy.deepcopy(self.pages),
                "progress": copy.deepcopy(getattr(self, 'progress', {}))
            })
        
        # 限制历史栈大小
        if len(self.history_stack) > 10:
            self.history_stack.pop(0)
            
    def undo_last_operation(self) -> bool:
        """
        回退到上一个操作状态
        
        Returns:
            bool: 回退是否成功
        """
        if not self.history_stack:
            print("❌ 没有可回退的操作历史")
            return False
        
        last_state = self.history_stack.pop()
        if hasattr(self, 'pages'):
            self.pages = last_state["pages"]
        if hasattr(self, 'progress'):
            self.progress = last_state["progress"]
        
        print("✅ 已成功回退到上一个状态")
        return True
    
    def generate_modality_assets(self, config, pages):
        # 导入datetime用于生成唯一标识符
        from datetime import datetime
        
        print(f"\n{'='*60}")
        print(f"🖼️  开始生成多模态资源")
        print(f"{'='*60}")
        
        # 显示故事概览
        print(f"\n📊 故事概览:")
        print(f"   总页数: {len(pages)}")
        
        # 询问用户是否继续生成多模态资源
        print("\n🔧 请选择操作:")
        print("1. 继续生成图像和语音")
        print("2. 仅生成图像")
        print("3. 仅生成语音")
        
        choice = self._get_user_choice()
        
        # 根据用户选择调整使用的模态
        selected_modalities = []
        if choice in ['1', '2']:
            selected_modalities.append("image")
        if choice in ['1', '3']:
            selected_modalities.append("speech")
        
        script_data = {"pages": [{"story": page} for page in pages]}
        story_dir = Path(config["story_dir"])
        
        # 优先使用已加载故事的会话ID（如果存在），否则生成新的会话ID
        if hasattr(self, 'current_session_id') and self.current_session_id:
            session_id = self.current_session_id
            print(f"📁 使用已加载故事的会话ID: {session_id}")
        else:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"💡 生成新会话ID: {session_id} - 确保资源不会被覆盖")
        
        # 创建基于会话ID的子目录
        session_dir = story_dir / session_id
        print(f"📁 资源将保存在: {session_dir}")

        # 创建必要的目录结构
        for sub_dir in selected_modalities:
            (session_dir / sub_dir).mkdir(exist_ok=True, parents=True)

        agents = {}
        params = {}
        for modality in selected_modalities:
            agents[modality] = init_tool_instance(config[modality + "_generation"])
            params[modality] = config[modality + "_generation"]["params"].copy()
            params[modality].update({
                "pages": pages,
                "save_path": session_dir / modality,
                "session_id": session_id  # 添加会话ID参数，用于生成唯一文件名
            })

        processes = []
        return_dict = mp.Manager().dict()

        # 创建并启动进程
        for modality in selected_modalities:
            print(f"\n{'🖼️' if modality == 'image' else '🔊'} 开始生成{'图像' if modality == 'image' else '语音'}...")
            p = mp.Process(
                target=self.call_modality_agent,
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
        for modality, result in return_dict.items():
            try:
                if modality == "image":
                    images = result["generation_results"]
                    for idx in range(len(pages)):
                        script_data["pages"][idx]["image_prompt"] = result["prompts"][idx]
            except Exception as e:
                print(f"Error occurred during generation: {e}")
        
        # 在script_data中添加会话ID，便于追踪
        script_data["session_id"] = session_id
        
        # 保存脚本数据到会话目录
        with open(session_dir / f"script_data.json", "w") as writer:
            json.dump(script_data, writer, ensure_ascii=False, indent=4)
        
        # 仍然保留原始的script_data.json作为最新版本的引用
        with open(story_dir / "script_data.json", "w") as writer:
            # 添加会话目录信息，便于后续加载
            script_data_with_ref = script_data.copy()
            script_data_with_ref["session_dir"] = str(session_dir)
            json.dump(script_data_with_ref, writer, ensure_ascii=False, indent=4)
        
        print(f"\n✅ 多模态资源生成完成")
        print(f"📚 会话ID: {session_id}")
        return images
    
    def compose_storytelling_video(self, config, pages):
        # 获取会话ID或生成新的
        story_dir = Path(config["story_dir"])
        session_id = None
        session_dir = None
        
        # 优先使用已加载故事的会话ID（如果存在）
        if hasattr(self, 'current_session_id') and self.current_session_id:
            session_id = self.current_session_id
            session_dir = story_dir / session_id
            print(f"📁 使用已加载故事的会话ID: {session_id}")
        else:
            # 尝试从script_data.json中获取会话目录信息
            try:
                # 尝试多种编码格式打开script_data.json
                encodings = ['utf-8', 'gbk', 'gb2312', 'cp936']
                script_data = {}
                data_loaded = False
                
                for encoding in encodings:
                    try:
                        with open(story_dir / "script_data.json", "r", encoding=encoding) as reader:
                            script_data = json.load(reader)
                            print(f"📄 成功使用 {encoding} 编码打开script_data.json")
                            data_loaded = True
                            break
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                
                if not data_loaded:
                    raise Exception("无法使用尝试的编码格式解析script_data.json文件")
                    if "session_dir" in script_data:
                        session_dir = Path(script_data["session_dir"])
                        # 从会话目录路径中提取会话ID
                        session_id = session_dir.name
            except Exception as e:
                print(f"警告: 无法读取会话信息 - {e}")
            
            # 如果没有找到会话目录，则创建新的
            if not session_dir or not session_dir.exists():
                from datetime import datetime
                session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                session_dir = story_dir / session_id
                session_dir.mkdir(exist_ok=True, parents=True)
                print(f"📁 创建新会话目录: {session_dir}")
        
        # 初始化视频合成代理
        video_compose_agent = init_tool_instance(config["video_compose"])
        params = config["video_compose"]["params"].copy()
        
        # 更新参数，指定正确的资源路径和输出路径
        params.update({
            "pages": pages,
            "image_dir": str(session_dir / "image") if (session_dir / "image").exists() else None,
            "speech_dir": str(session_dir / "speech") if (session_dir / "speech").exists() else None,
            "output_path": str(session_dir / "output.mp4"),  # 视频输出到会话目录
            "session_id": session_id  # 添加会话ID参数，确保使用正确的资源目录
        })
        
        # 显示会话和输出路径信息
        print(f"📁 会话目录: {session_dir}")
        print(f"🎬 视频将输出到: {session_dir / 'output.mp4'}")
        print(f"💡 每个会话都有独立目录，视频文件不会被覆盖")
        
        # 执行视频合成
        video_compose_agent.call(params)
        
        print(f"✅ 视频已合成并保存至: {session_dir / 'output.mp4'}")

    def _show_main_menu(self):
        """
        显示主菜单
        """
        print("\n📋 多媒体故事生成助手 - 主菜单")
        print("=" * 60)
        print("1. 📝 生成新故事")
        print("2. 📂 加载现有故事")
        print("3. ✏️ 编辑故事内容")
        print("4. 🖼️ 编辑图像提示词")
        print("5. 🎨 生成多媒体资源 (图像/语音)")
        print("6. 🎥 合成视频故事")
        print("7. ⏪ 撤销上一步操作")
        print("8. 💾 保存当前进度")
        print("9. 📊 查看项目状态")
        print("10. ❓ 查看帮助")
        print("0. 👋 退出")
        print("=" * 60)
    
    def _show_status(self):
        """
        显示当前项目状态
        """
        print("\n📊 项目状态")
        print("-" * 60)
        print(f"📝 故事页数: {len(self.pages) if hasattr(self, 'pages') and self.pages else '0'}")
        
        # 显示故事预览
        if hasattr(self, 'pages') and self.pages:
            print("📖 故事预览:")
            for i, page in enumerate(self.pages[:2], 1):
                preview = page[:80] + '...' if len(page) > 80 else page
                print(f"   第{i}页: {preview}")
        
        # 显示资产信息
        if hasattr(self, 'assets') and self.assets:
            print(f"🎨 已生成资源组数: {len(self.assets)}")
            # 显示最新的资源组信息
            latest_assets = self.assets[-1]
            print(f"   最新资源ID: {latest_assets.get('session_id', '未知')}")
            print(f"   生成时间: {latest_assets.get('timestamp', '未知')}")
            
            # 显示具体资源数量
            assets_data = latest_assets.get('assets', {})
            image_count = sum(1 for page_assets in assets_data.values() if 'image' in page_assets)
            speech_count = sum(1 for page_assets in assets_data.values() if 'speech' in page_assets)
            print(f"   图像数量: {image_count}")
            print(f"   语音数量: {speech_count}")
            
            if 'video' in latest_assets:
                print(f"   视频: 已生成")
        
        # 显示历史操作状态
        if hasattr(self, 'history_stack') and self.history_stack:
            print(f"⏪ 可撤销操作: {len(self.history_stack)}")
        
        print("-" * 60)
    
    def _show_help(self):
        """
        显示帮助信息
        """
        print("\n❓ 使用帮助")
        print("=" * 60)
        print("多媒体故事生成助手使用指南:")
        print("\n1. 生成新故事: 输入故事主题，系统会生成相应内容")
        print("2. 加载现有故事: 自动扫描并选择已保存的故事文件")
        print("3. 编辑故事内容: 修改故事的文本内容，可以编辑单个页面或所有页面")
        print("4. 编辑图像提示词: 为故事页面自定义图像生成提示词")
        print("5. 生成多媒体资源: 根据故事内容生成图像和语音")
        print("6. 合成视频故事: 将故事内容、图像和语音合成为视频")
        print("7. 撤销操作: 可以撤销上一步操作")
        print("8. 保存进度: 手动保存当前项目状态")
        print("9. 查看状态: 显示当前项目的详细信息")
        print("10. 查看帮助: 显示此帮助信息")
        print("\n快捷操作:")
        print("  - 在任何输入提示下，输入 'c' 或 'cancel' 可以取消当前操作")
        print("  - 输入 'h' 或 'help' 可以随时查看帮助信息")
        print("  - 输入 'q' 或 'exit' 可以随时退出程序")
        print("=" * 60)
    
    def _handle_user_input(self, prompt, default=None, allow_cancel=True):
        """
        处理用户输入，支持取消操作
        
        Args:
            prompt: 提示信息
            default: 默认值
            allow_cancel: 是否允许取消
            
        Returns:
            str or None: 用户输入或None（取消时）
        """
        while True:
            if default is not None:
                prompt_with_default = f"{prompt} (默认: {default}): "
            else:
                prompt_with_default = f"{prompt}: "
            
            user_input = input(prompt_with_default).strip()
            
            # 检查是否取消
            if allow_cancel and user_input.lower() in ['c', 'cancel']:
                print("❌ 操作已取消")
                return None
                
            # 使用默认值
            if not user_input and default is not None:
                return default
                
            # 非空输入
            if user_input:
                return user_input
                
            # 空输入且无默认值
            print("❌ 输入不能为空，请重试")
    
    def call(self, config):
        """
        主循环函数，处理用户输入和故事生成流程
        
        Args:
            config: 配置信息
            
        Returns:
            dict: 包含生成内容的结果字典
        """
        # 初始化配置
        if config is None:
            config = {}
            
        # 确保必要的配置存在
        config.setdefault("story_dir", "./outputs")
        config.setdefault("asset_dir", "./assets")
        
        # 创建必要的目录
        story_dir = Path(config["story_dir"])
        asset_dir = Path(config["asset_dir"])
        story_dir.mkdir(parents=True, exist_ok=True)
        asset_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化状态
        self.history_stack = []
        self.pages = []
        self.assets = []
        self.current_session = None
        self.progress = {"story_generated": False, "image_generated": False, "speech_generated": False, "video_composed": False}
        
        print("🎬 欢迎使用多媒体故事生成助手！")
        print("=" * 60)
        print("这是一个功能强大的工具，可以帮助您创建完整的多媒体故事。")
        print("您可以生成故事内容、图像、语音，并将它们合成为精彩的视频。")
        print("\n💡 提示: 输入 'h' 或 'help' 查看所有可用命令。")
        print("=" * 60)
        
        # 显示当前项目状态
        self._show_status()
        
        # 检查是否有保存的进度
        progress_file = story_dir / "progress_status.json"
        saved_progress = None
        
        if progress_file.exists():
            try:
                with open(progress_file, "r", encoding='utf-8') as f:
                    saved_progress = json.load(f)
                print("\n🔍 发现之前的进度记录")
                print(f"   上次完成阶段: {saved_progress.get('stage', '未知')}")
                print("\n🔄 是否继续之前的进度?")
                print("1. 继续之前的进度")
                print("2. 重新开始")
                print("3. 加载其他故事JSON文件")
                
                while True:
                    choice = input("请选择操作 (1/2/3): ").strip()
                    if choice == '1' and saved_progress.get('pages'):
                        self.pages = saved_progress['pages']
                        print("\n✅ 已加载之前的故事内容")
                        # 显示故事概览
                        print(f"\n📖 已加载的故事: {len(self.pages)} 页")
                        for i, page in enumerate(self.pages[:3], 1):
                            print(f"   第{i}页: {page[:80]}...")
                        # 更新进度状态
                        self.progress["story_generated"] = True
                        break
                    elif choice == '2':
                        break
                    elif choice == '3':
                        if self.load_story_from_json(None, config):
                            self.progress["story_generated"] = True
                        break
                    else:
                        print("❌ 请输入正确选项（1/2/3）")
            except Exception as e:
                print(f"❌ 加载进度失败: {e}")
        
        # 主循环
        while True:
            try:
                # 显示主菜单
                self._show_main_menu()
                
                # 获取用户选择
                choice = self._handle_user_input("请输入操作序号", None, False).strip()
                
                if choice == "1" or choice.lower() == "new":
                    # 生成新故事
                    print("\n💡 请输入故事主题或开始内容")
                    print("(提示: 越详细的主题描述会生成越精准的故事内容)")
                    
                    # 保存当前状态用于回退
                    self._save_to_history_stack()
                    
                    # 设置主题
                    config_with_topic = config.copy()
                    topic = self._handle_user_input("主题")
                    if topic:
                        config_with_topic['topic'] = topic
                        pages = self.write_story(config_with_topic)
                        # 保存进度
                        self.pages = pages
                        self.progress["story_generated"] = True
                        print("\n✅ 故事生成完成！")
                        
                        # 询问是否继续下一步
                        next_step = self._handle_user_input("是否要生成多媒体资源？(y/n)", "y")
                        if next_step and next_step.lower() == "y":
                            self.generate_modality_assets(config, self.pages)
                            self.progress["image_generated"] = True
                            self.progress["speech_generated"] = True
                    
                elif choice == "2" or choice.lower() == "load":
                    # 加载现有故事
                    print("\n📂 正在扫描可用的故事文件...")
                    
                    # 保存当前状态用于回退
                    self._save_to_history_stack()
                    
                    if self.load_story_from_json(None, config):
                        print("✅ 故事加载成功！")
                        self.progress["story_generated"] = True
                        # 显示状态并询问下一步
                        self._show_status()
                        next_step = self._handle_user_input("是否要编辑提示词或生成资源？(e=编辑/g=生成/n=返回)", "n")
                        if next_step:
                            if next_step.lower() == "e":
                                self.edit_image_prompts(config)
                            elif next_step.lower() == "g":
                                self.generate_modality_assets(config, self.pages)
                                self.progress["image_generated"] = True
                                self.progress["speech_generated"] = True
                    
                elif choice == "3" or choice.lower() == "edit":
                    # 编辑故事内容
                    if not self.pages:
                        print("❌ 请先生成或加载故事！")
                        # 提供快捷选项
                        print("\n📋 您可以:")
                        print("1. 生成新故事")
                        print("2. 加载现有故事")
                        quick_choice = self._handle_user_input("请选择", "2")
                        if quick_choice == "1":
                            print("\n💡 请输入故事主题")
                            config_with_topic = config.copy()
                            topic = self._handle_user_input("主题")
                            if topic:
                                config_with_topic['topic'] = topic
                                pages = self.write_story(config_with_topic)
                                self.pages = pages
                                self.progress["story_generated"] = True
                                if self.pages:
                                    self.edit_story_content(config)
                        elif quick_choice == "2":
                            if self.load_story_from_json(None, config) and self.pages:
                                self.progress["story_generated"] = True
                                self.edit_story_content(config)
                    else:
                        self.edit_story_content(config)
                
                elif choice == "4" or choice.lower() == "image":
                    # 编辑图像提示词
                    if not self.pages:
                        print("❌ 请先生成或加载故事！")
                        # 提供快捷选项
                        print("\n📋 您可以:")
                        print("1. 生成新故事")
                        print("2. 加载现有故事")
                        quick_choice = self._handle_user_input("请选择", "2")
                        if quick_choice == "1":
                            print("\n💡 请输入故事主题")
                            config_with_topic = config.copy()
                            topic = self._handle_user_input("主题")
                            if topic:
                                config_with_topic['topic'] = topic
                                pages = self.write_story(config_with_topic)
                                self.pages = pages
                                self.progress["story_generated"] = True
                                if self.pages:
                                    self.edit_image_prompts(config)
                        elif quick_choice == "2":
                            if self.load_story_from_json(None, config) and self.pages:
                                self.progress["story_generated"] = True
                                self.edit_image_prompts(config)
                    else:
                        self.edit_image_prompts(config)
                    
                elif choice == "5" or choice.lower() == "generate":
                    # 生成多媒体资源
                    if not self.pages:
                        print("❌ 请先生成或加载故事！")
                        # 提供快捷选项
                        print("\n📋 您可以:")
                        print("1. 生成新故事")
                        print("2. 加载现有故事")
                        quick_choice = self._handle_user_input("请选择", "2")
                        if quick_choice == "1":
                            print("\n💡 请输入故事主题")
                            config_with_topic = config.copy()
                            topic = self._handle_user_input("主题")
                            if topic:
                                config_with_topic['topic'] = topic
                                pages = self.write_story(config_with_topic)
                                self.pages = pages
                                self.progress["story_generated"] = True
                                if self.pages:
                                    images = self.generate_modality_assets(config, self.pages)
                                    self.progress["image_generated"] = True
                                    self.progress["speech_generated"] = True
                        elif quick_choice == "2":
                            if self.load_story_from_json(None, config) and self.pages:
                                self.progress["story_generated"] = True
                                images = self.generate_modality_assets(config, self.pages)
                                self.progress["image_generated"] = True
                                self.progress["speech_generated"] = True
                    else:
                        images = self.generate_modality_assets(config, self.pages)
                        self.progress["image_generated"] = True
                        self.progress["speech_generated"] = True
                        # 询问是否合成视频
                        confirm = self._handle_user_input("是否要合成视频？(y/n)", "y")
                        if confirm and confirm.lower() == "y":
                            self.compose_storytelling_video(config, self.pages)
                            self.progress["video_composed"] = True
                    
                elif choice == "6" or choice.lower() == "video":
                    # 合成视频
                    if not self.pages:
                        print("❌ 请先生成或加载故事！")
                    else:
                        print("\n🎥 开始合成视频...")
                        self.compose_storytelling_video(config, self.pages)
                        self.progress["video_composed"] = True
                        print("✅ 视频合成完成！")
                        print("\n🎉 恭喜！您的多媒体故事已全部制作完成！")
                    
                elif choice == "7" or choice.lower() == "undo":
                    # 撤销上一步操作
                    if self.undo_last_operation():
                        # 更新进度状态
                        self.progress["story_generated"] = bool(hasattr(self, 'pages') and self.pages)
                        print("✅ 已撤销上一步操作")
                    else:
                        print("❌ 没有可撤销的操作")
                    
                elif choice == "8" or choice.lower() == "save":
                    # 保存当前进度
                    if hasattr(self, 'pages') and self.pages:
                        self._save_progress(config, {"stage": "story_completed", "pages": self.pages})
                        print("✅ 进度已保存")
                    else:
                        print("❌ 没有可保存的内容")
                    
                elif choice == "9" or choice.lower() == "status":
                    # 查看项目状态
                    self._show_status()
                    
                elif choice == "10" or choice.lower() in ["h", "help"]:
                    # 显示帮助
                    self._show_help()
                    
                elif choice == "0" or choice.lower() in ["exit", "quit", "q"]:
                    # 退出前保存
                    if hasattr(self, 'pages') and self.pages:
                        confirm = self._handle_user_input("是否保存当前进度？(y/n)", "y")
                        if confirm and confirm.lower() == "y":
                            self._save_progress(config, {"stage": "story_completed", "pages": self.pages})
                    print("\n👋 感谢使用多媒体故事生成助手！再见！")
                    break
                    
                else:
                    print("❌ 无效的选择，请重试")
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  操作被中断")
                confirm = input("是否要退出？(y/n): ").strip().lower()
                if confirm == "y":
                    print("\n👋 感谢使用！再见！")
                    break
            except Exception as e:
                print(f"\n❌ 发生错误: {str(e)}")
                import traceback
                traceback.print_exc()
                print("\n💡 提示: 您可以尝试撤销操作或重新开始")
                
        # 返回最终结果
        return {
            "pages": self.pages,
            "assets": self.assets
        }
    
    def _save_progress(self, config, progress_data):
        """保存当前进度状态，支持保存到原始文件路径"""
        # 检查是否有当前加载的JSON文件路径
        if hasattr(self, 'current_json_path') and self.current_json_path:
            # 如果是从特定文件加载的，保存回原文件
            save_path = Path(self.current_json_path)
            print(f"\n💾 正在保存到原始文件: {save_path}")
        else:
            # 否则保存到默认位置
            story_dir = Path(config.get("story_dir", "./outputs"))
            story_dir.mkdir(exist_ok=True, parents=True)
            save_path = story_dir / "progress_status.json"
            print(f"\n💾 正在保存到默认位置: {save_path}")
        
        # 确保目标目录存在
        save_path.parent.mkdir(exist_ok=True, parents=True)
        
        # 添加时间戳和页面数量信息
        progress_data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        progress_data["total_pages"] = len(progress_data.get("pages", []))
        
        # 保存数据
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 进度已成功保存至: {save_path}")
            
            # 如果是加载的故事，同时更新会话目录中的其他资源关联
            if hasattr(self, 'current_session_id') and self.current_session_id:
                print(f"📝 会话ID: {self.current_session_id} 的内容已更新")
                
        except Exception as e:
            print(f"❌ 保存进度失败: {str(e)}")
            traceback.print_exc()
        
    def _generate_image_prompt(self, text_content):
        """
        为没有提示词的页面生成默认图像提示词
        """
        return f"生动的场景展示: {text_content[:100]}，高质量，细节丰富，明亮清晰"
        
    def edit_story_content(self, config):
        """
        修改故事文本内容功能
        """
        if not hasattr(self, 'pages') or not self.pages:
            print("❌ 没有可编辑的故事页面。请先生成或加载故事。")
            return
            
        # 保存当前状态到历史栈
        self._save_to_history_stack()
        
        print("\n✏️  故事内容编辑")
        print("=" * 60)
        print(f"当前故事共有 {len(self.pages)} 页")
        
        # 显示页面预览
        for i, page in enumerate(self.pages):
            if isinstance(page, dict):
                content_preview = page.get('content', '') or page.get('story', '')
            else:
                content_preview = str(page)
            
            if len(content_preview) > 50:
                content_preview = content_preview[:50] + '...'
            print(f"{i+1}. {content_preview}")
        
        # 获取用户选择的页面
        while True:
            try:
                page_input = input("\n请输入要修改的页面序号 (输入 'all' 编辑所有页面, 'c' 取消): ")
                if page_input.lower() == 'c':
                    print("✅ 已取消操作")
                    return
                elif page_input.lower() == 'all':
                    # 编辑所有页面
                    for i in range(len(self.pages)):
                        self._edit_single_page(i)
                    print("\n✅ 所有页面已编辑完成")
                    
                    # 保存修改
                    self._save_progress(config, {"stage": "story_completed", "pages": self.pages})
                    return
                
                page_index = int(page_input) - 1
                if 0 <= page_index < len(self.pages):
                    break
                else:
                    print(f"❌ 请输入1-{len(self.pages)}之间的数字")
            except ValueError:
                print("❌ 无效的输入，请输入数字")
        
        # 编辑选定页面
        self._edit_single_page(page_index)
        
        # 保存修改
        self._save_progress(config, {"stage": "story_completed", "pages": self.pages})
        
        # 提示用户可以继续编辑其他页面
        continue_edit = input("\n是否继续编辑其他页面？(y/n): ").lower()
        if continue_edit == 'y':
            self.edit_story_content(config)
    
    def _edit_single_page(self, page_index):
        """
        编辑单个页面内容
        """
        # 获取当前页面内容
        current_page = self.pages[page_index]
        if isinstance(current_page, dict):
            current_content = current_page.get('content', '') or current_page.get('story', '')
        else:
            current_content = str(current_page)
        
        print(f"\n页面 {page_index + 1} 当前内容:")
        print("-" * 60)
        print(current_content)
        print("-" * 60)
        print("请输入新的内容 (直接回车保持当前内容，输入 'c' 取消): ")
        
        # 获取新的内容
        new_content = input().strip()
        if new_content.lower() == 'c':
            print("✅ 已取消修改")
            return
        elif new_content:
            # 更新页面内容
            if isinstance(current_page, dict):
                if 'content' in current_page:
                    current_page['content'] = new_content
                elif 'story' in current_page:
                    current_page['story'] = new_content
                else:
                    current_page['content'] = new_content
            else:
                self.pages[page_index] = new_content
            
            print("✅ 页面内容已更新")
        else:
            print("✅ 内容保持不变")
    
    def edit_image_prompts(self, config):
        """
        修改图像提示词功能
        """
        if not hasattr(self, 'pages') or not self.pages:
            print("❌ 没有可编辑的故事页面。请先生成或加载故事。")
            return
            
        # 保存当前状态到历史栈
        self._save_to_history_stack()
        
        print("\n🖼️  图像提示词编辑")
        print("=" * 60)
        print(f"当前故事共有 {len(self.pages)} 页")
        
        # 显示页面预览
        for i, page in enumerate(self.pages):
            content_preview = page.strip() if isinstance(page, str) else str(page)[:50]
            if len(content_preview) > 50:
                content_preview = content_preview[:50] + '...'
            print(f"{i+1}. {content_preview}")
        
        # 获取用户选择的页面
        while True:
            try:
                page_input = input("\n请输入要修改的页面序号 (输入 'c' 取消): ")
                if page_input.lower() == 'c':
                    print("✅ 已取消操作")
                    return
                
                page_index = int(page_input) - 1
                if 0 <= page_index < len(self.pages):
                    break
                else:
                    print(f"❌ 请输入1-{len(self.pages)}之间的数字")
            except ValueError:
                print("❌ 无效的输入，请输入数字")
        
        # 确保故事目录存在
        story_dir = Path(config.get("story_dir", "./outputs"))
        story_dir.mkdir(parents=True, exist_ok=True)
        script_file = story_dir / "script_data.json"
        current_prompt = ""
        
        # 初始化script_data
        script_data = {"pages": []}
        
        # 尝试从script_data.json获取当前提示词
        if script_file.exists():
            try:
                # 尝试多种编码格式
                encodings = ['utf-8', 'gbk', 'gb2312', 'cp936']
                data_loaded = False
                
                for encoding in encodings:
                    try:
                        with open(script_file, 'r', encoding=encoding) as f:
                            temp_data = json.load(f)
                            if isinstance(temp_data, dict):
                                script_data = temp_data
                                print(f"📄 成功使用 {encoding} 编码打开文件")
                            else:
                                print("⚠️  提示词文件格式不正确，将创建新文件")
                            
                            # 确保pages字段存在
                            if 'pages' not in script_data or not isinstance(script_data['pages'], list):
                                script_data['pages'] = []
                            
                            # 获取当前页面的提示词
                            if len(script_data['pages']) > page_index:
                                current_prompt = script_data['pages'][page_index].get('image_prompt', '')
                            
                            data_loaded = True
                            break
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                        
                if not data_loaded:
                    raise Exception("无法使用尝试的编码格式解析文件")
            except Exception as e:
                print(f"⚠️  加载现有提示词时出错: {str(e)}")
        
        # 如果没有找到现有提示词，生成默认提示词
        if not current_prompt:
            page_content = self.pages[page_index]
            if isinstance(page_content, dict):
                text_to_use = page_content.get('content', '') or page_content.get('story', '')
            else:
                text_to_use = str(page_content)
            current_prompt = self._generate_image_prompt(text_to_use)
        
        print(f"\n当前提示词: {current_prompt}")
        print("请输入新的提示词 (直接回车保持当前提示词，输入 'c' 取消): ")
        
        # 获取新的提示词
        new_prompt = input().strip()
        if new_prompt.lower() == 'c':
            print("✅ 已取消操作")
            return
        elif new_prompt:
            # 确保pages数组长度足够
            while len(script_data["pages"]) <= page_index:
                # 为每个页面添加基础信息
                idx = len(script_data["pages"])
                page_content = self.pages[idx] if idx < len(self.pages) else ""
                script_data["pages"].append({
                    "story": page_content if isinstance(page_content, str) else str(page_content),
                    "image_prompt": ""
                })
            
            # 更新提示词
            script_data["pages"][page_index]["image_prompt"] = new_prompt
            
            # 保存更新后的script_data.json
            try:
                # 确保保存目录存在
                script_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(script_file, 'w', encoding='utf-8') as f:
                    json.dump(script_data, f, ensure_ascii=False, indent=4)
                print("✅ 提示词已更新并保存")
                print(f"📁 保存位置: {script_file}")
                
                # 验证保存是否成功
                try:
                    # 尝试多种编码格式验证保存结果
                    encodings = ['utf-8', 'gbk', 'gb2312', 'cp936']
                    verify_data = {}
                    data_loaded = False
                    
                    for encoding in encodings:
                        try:
                            with open(script_file, 'r', encoding=encoding) as f:
                                verify_data = json.load(f)
                                print(f"📄 成功使用 {encoding} 编码验证保存结果")
                                data_loaded = True
                                break
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                    
                    if not data_loaded:
                        print("⚠️  无法使用尝试的编码格式解析保存的文件")
                    else:
                        # 只有在成功加载数据后才进行验证
                        if len(verify_data.get('pages', [])) > page_index and \
                           'image_prompt' in verify_data['pages'][page_index] and \
                           verify_data['pages'][page_index]['image_prompt'] == new_prompt:
                            print("✅ 保存验证成功")
                        else:
                            print("⚠️  保存验证失败，请检查文件")
                except Exception:
                    print("⚠️  无法验证保存结果")
                
                # 询问是否重新生成该页面的图像
                regen_input = input("\n是否重新生成该页面的图像？(y/n): ").lower()
                if regen_input == 'y':
                    print("\n🔄 正在重新生成图像...")
                    try:
                        # 临时创建一个只包含当前页面的数组
                        single_page = [self.pages[page_index]]
                        
                        # 准备参数
                        image_agent = init_tool_instance(config["image_generation"])
                        image_params = config["image_generation"]["params"].copy()
                        image_params.update({
                            "pages": single_page,
                            "save_path": story_dir / "image",
                            "start_index": page_index  # 传递起始索引，确保生成正确的文件名
                        })
                        
                        # 重新生成图像
                        result = image_agent.call(image_params)
                        if result and (result.get("status") == "success" or "generation_results" in result):
                            print("✅ 图像已重新生成成功")
                        else:
                            print("⚠️  图像生成可能未成功完成")
                    except Exception as e:
                        print(f"❌ 重新生成图像时出错: {str(e)}")
                        traceback.print_exc()
            except Exception as e:
                print(f"❌ 保存提示词时出错: {str(e)}")
                traceback.print_exc()
        else:
            print("✅ 提示词保持不变")
            
        # 提示用户可以继续编辑其他页面
        continue_edit = input("\n是否继续编辑其他页面？(y/n): ").lower()
        if continue_edit == 'y':
            self.edit_image_prompts(config)
