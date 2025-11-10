# MM-StoryAgent 中文使用指南

## 🎯 项目概述

这是《数据分析与可视化实践》课程的大作业项目，基于 MM-StoryAgent 实现**端到端视频生成**。

### 已完成的修改

✅ **任务1**：去除背景音乐和音效模块  
✅ **任务2**：修改图像生成为API调用（使用通义万相）  
⏳ **任务3**：长文本处理扩展（待实现）

---

## 🚀 快速开始

### 1. 设置环境变量

```powershell
# PowerShell中运行
$env:DASHSCOPE_API_KEY="sk-c5574beb924a4d9fbedd573f681090b0"
$env:ALIYUN_APP_KEY="AzX8QcMX3u1809OI"

# ⚠️ 重要：还需要补充以下两个变量（参考：如何获取阿里云AccessKey.md）
$env:ALIYUN_ACCESS_KEY_ID="您的AccessKey ID"
$env:ALIYUN_ACCESS_KEY_SECRET="您的AccessKey Secret"
```

### 2. 运行程序

```bash
# 使用优化后的API配置
python run.py -c configs/mm_story_agent_api.yaml
```

### 3. 查看结果

生成的文件在 `generated_stories/example/` 目录：
```
generated_stories/example/
├── image/          # API生成的图像
├── speech/         # 语音文件
├── script_data.json  # 元数据
└── output.mp4      # 最终视频 ⭐
```

---

## 📚 文档导航

### 环境配置相关
- 📖 [环境变量配置说明.md](环境变量配置说明.md) - 所有环境变量的详细说明
- 🔑 [如何获取阿里云AccessKey.md](如何获取阿里云AccessKey.md) - 获取缺失密钥的步骤

### 任务完成说明
- ✅ [修改说明-去除音乐音效.md](修改说明-去除音乐音效.md) - 任务1的详细说明
- ✅ [任务2完成说明-API图像生成.md](任务2完成说明-API图像生成.md) - 任务2的详细说明

### 使用指南
- 🚀 [快速开始指南.md](快速开始指南.md) - 快速上手指南
- 📋 [下一步操作指南.md](下一步操作指南.md) - 后续工作指引

---

## 🛠️ 配置文件说明

### configs/mm_story_agent_api.yaml（推荐使用）
- ✅ 使用API生成图像（无需下载模型）
- ✅ 去除了音乐和音效
- ✅ 速度更快，资源占用更少

### configs/mm_story_agent_no_music.yaml
- ✅ 去除了音乐和音效
- ❌ 仍使用本地Stable Diffusion（需下载大模型）

### configs/mm_story_agent.yaml（原配置）
- ❌ 包含音乐和音效
- ❌ 使用本地模型
- ⚠️ 不推荐使用

---

## 📊 改进效果对比

| 项目 | 原版 | 优化后 |
|------|------|--------|
| **生成时间** | ~40分钟 | ~5-10分钟 |
| **模型下载** | 需要8GB+ | 无需下载 |
| **GPU要求** | 必需 | 不需要 |
| **网络连接** | HuggingFace | 国内API |
| **音频模态** | 4种（图/声/音乐/语音） | 2种（图/语音） |

---

## 💻 核心修改文件

### 1. mm_story_agent/mm_story_agent.py
```python
# 修改：去除音乐和音效模态
self.modalities = ["image", "speech"]  # 原来是4个
```

### 2. mm_story_agent/video_compose_agent.py
```python
# 修改：去除音乐和音效处理逻辑
audio_clip = speech_clip  # 只使用语音
```

### 3. mm_story_agent/modality_agents/image_agent.py
```python
# 新增：DashScopeImageAgent 类
@register_tool("dashscope_image_api")
class DashScopeImageAgent:
    # 使用通义万相API生成图像
    ...
```

### 4. mm_story_agent/base.py
```python
# 新增：注册API图像生成工具
'dashscope_image_api': 'DashScopeImageAgent',
```

---

## 🎯 待完成任务

### 任务3：长文本处理扩展

**要求**：支持输入长文本（如文档、论文）而非简短主题

**修改文件**：
- `mm_story_agent/modality_agents/story_agent.py`

**参考思路**：
```python
# 原流程：topic → outline → story pages
# 新流程：long_text → summarize → story pages
```

### 选做创新点（40分）

建议选择以下至少1项：
1. ⭐⭐⭐⭐⭐ 长文本处理（任务3）
2. ⭐⭐⭐⭐ 数据可视化（添加图表）
3. ⭐⭐⭐ Web交互界面（Gradio/Streamlit）
4. ⭐⭐⭐⭐ 性能优化（缓存、并发）
5. ⭐⭐⭐ 场景拓展（科普、新闻等）

---

## ⚠️ 常见问题

### Q: 运行报错 "SDK.InvalidCredential"
**A**: 缺少阿里云AccessKey，请参考 `如何获取阿里云AccessKey.md`

### Q: 图像生成失败
**A**: 检查 `DASHSCOPE_API_KEY` 是否设置正确

### Q: 没有GPU可以运行吗？
**A**: 可以！使用API配置文件完全不需要GPU

### Q: API会收费吗？
**A**: DashScope有免费额度，具体查看控制台

---

## 📞 获取帮助

1. **查看文档**：先阅读相关文档
2. **查看日志**：仔细阅读错误信息
3. **搜索问题**：使用搜索引擎查找解决方案
4. **请教老师**：必要时请教老师或助教

---

## 📅 项目时间线

- **Week 1-2**: 环境搭建 + 任务1&2（已完成✅）
- **Week 3-4**: 任务3 + 创新点
- **Week 5-6**: 测试优化 + 撰写报告
- **Week 7-8**: 制作PPT + 准备汇报
- **2025-11-23**: 汇报时间

---

## 🙏 致谢

- 原项目：[MM-StoryAgent](https://github.com/X-PLUG/MM_StoryAgent)
- API服务：阿里云DashScope、智能语音交互

---

**祝您项目顺利！** 🎉

如有问题，欢迎查看详细文档或寻求帮助。

