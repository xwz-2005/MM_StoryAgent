from pathlib import Path
from typing import List, Union
import random
import re
import os
from datetime import timedelta
import datetime

from tqdm import trange
import numpy as np
import librosa
import cv2
from zhon.hanzi import punctuation as zh_punc

# 配置ImageMagick路径（用于字幕生成）
#下面这个路径记得替换成自己的路径
IMAGEMAGICK_BINARY = r"D:\software\ImageMagick-7.1.2-Q16-HDRI\magick.exe"
os.environ['IMAGEMAGICK_BINARY'] = IMAGEMAGICK_BINARY

from moviepy.editor import ImageClip, AudioFileClip, CompositeAudioClip, \
    CompositeVideoClip, ColorClip, VideoFileClip, VideoClip, TextClip, concatenate_audioclips 
import moviepy.video.compositing.transitions as transfx
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.audio.fx.all import audio_loop
from moviepy.video.tools.subtitles import SubtitlesClip

from mm_story_agent.base import register_tool
from mm_story_agent.utils.visualization import VisualizationTool


def generate_srt(timestamps: List,
                 captions: List,
                 save_path: Union[str, Path],
                 max_single_length: int = 30):
    
    def format_time(seconds: float) -> str:
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        millis = int((td.total_seconds() - total_seconds) * 1000)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"
    
    srt_content = []
    num_caps = len(timestamps)

    for idx in range(num_caps):
        start_time, end_time = timestamps[idx]
        caption_chunks = split_caption(captions[idx], max_single_length).split("\n")
        num_chunks = len(caption_chunks)
        
        if num_chunks == 0:
            continue

        segment_duration = (end_time - start_time) / num_chunks

        for chunk_idx, chunk in enumerate(caption_chunks):
            chunk_start_time = start_time + segment_duration * chunk_idx
            chunk_end_time = start_time + segment_duration * (chunk_idx + 1)
            start_time_str = format_time(chunk_start_time)
            end_time_str = format_time(chunk_end_time)
            srt_content.append(f"{len(srt_content) // 2 + 1}\n{start_time_str} --> {end_time_str}\n{chunk}\n\n")

    with open(save_path, 'w') as srt_file:
        srt_file.writelines(srt_content)


def add_caption(captions: List,
                srt_path: Union[str, Path],
                timestamps: List,
                video_clip: VideoClip,
                max_single_length: int = 30,
                **caption_config):
    generate_srt(timestamps, captions, srt_path, max_single_length)

    generator = lambda txt: TextClip(txt, **caption_config)
    subtitles = SubtitlesClip(srt_path.__str__(), generator)
    captioned_clip = CompositeVideoClip([video_clip,
                                         subtitles.set_position(("center", "bottom"), relative=True)])
    return captioned_clip


def split_keep_separator(text, separator):
    pattern = f'([{re.escape(separator)}])'
    pieces = re.split(pattern, text)
    return pieces


def split_caption(caption, max_length=30):
    lines = []
    if ord(caption[0]) >= ord("a") and ord(caption[0]) <= ord("z") or ord(caption[0]) >= ord("A") and ord(caption[0]) <= ord("Z"):
        words = caption.split(" ")
        current_words = []
        for word in words:
            if len(" ".join(current_words + [word])) <= max_length:
                current_words += [word]
            else:
                if current_words:
                    lines.append(" ".join(current_words))
                    current_words = []

        if current_words:
            lines.append(" ".join(current_words))
    else:
        sentences = split_keep_separator(caption, zh_punc)
        current_line = ""
        for sentence in sentences:
            if len(current_line + sentence) <= max_length:
                current_line += sentence
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = ""
                if sentence.startswith(tuple(zh_punc)):
                    if lines:
                        lines[-1] += sentence[0]
                    current_line = sentence[1:]
                else:
                    current_line = sentence

        if current_line:
            lines.append(current_line.strip())

    return '\n'.join(lines)


def add_bottom_black_area(clip: VideoFileClip,
                          black_area_height: int = 64):
    """
    Add a black area at the bottom of the video clip (for captions).

    Args:
        clip (VideoFileClip): Video clip to be processed.
        black_area_height (int): Height of the black area.

    Returns:
        VideoFileClip: Processed video clip.
    """
    black_bar = ColorClip(size=(clip.w, black_area_height), color=(0, 0, 0), duration=clip.duration)
    extended_clip = CompositeVideoClip([clip, black_bar.set_position(("center", "bottom"))])
    return extended_clip


def add_zoom_effect(clip, speed=1.0, mode='in', position='center'):
    fps = clip.fps
    duration = clip.duration
    total_frames = int(duration * fps)
    def main(getframe, t):
        frame = getframe(t)
        h, w = frame.shape[: 2]
        i = t * fps
        if mode == 'out':
            i = total_frames - i
        zoom = 1 + (i * ((0.1 * speed) / total_frames))
        positions = {'center':  [(w - (w * zoom)) / 2,  (h - (h  *  zoom)) / 2],
                     'left': [0, (h - (h * zoom)) / 2],
                     'right': [(w - (w * zoom)), (h - (h * zoom)) / 2],
                     'top': [(w - (w * zoom)) / 2, 0],
                     'topleft': [0, 0],
                     'topright': [(w - (w * zoom)), 0],
                     'bottom': [(w - (w * zoom)) / 2, (h - (h * zoom))],
                     'bottomleft': [0, (h - (h * zoom))],
                     'bottomright': [(w - (w * zoom)), (h - (h * zoom))]}
        tx, ty = positions[position]
        M = np.array([[zoom, 0, tx], [0, zoom, ty]])
        frame = cv2.warpAffine(frame, M, (w, h))
        return frame
    return clip.fl(main)


def add_move_effect(clip, direction="left", move_raito=0.95):

    orig_width = clip.size[0]
    orig_height = clip.size[1]

    new_width = int(orig_width / move_raito)
    new_height = int(orig_height / move_raito)
    clip = clip.resize(width=new_width, height=new_height)

    if direction == "left":
        start_position = (0, 0)
        end_position = (orig_width - new_width, 0) 
    elif direction == "right":
        start_position = (orig_width - new_width, 0)
        end_position = (0, 0)

    duration = clip.duration
    moving_clip = clip.set_position(
        lambda t: (start_position[0] + (
            end_position[0] - start_position[0]) / duration * t, start_position[1])
    )

    final_clip = CompositeVideoClip([moving_clip], size=(orig_width, orig_height))

    return final_clip


def add_slide_effect(clips, slide_duration):
    ####### CAUTION: requires at least `slide_duration` of silence at the end of each clip #######
    durations = [clip.duration for clip in clips]
    first_clip = CompositeVideoClip(
        [clips[0].fx(transfx.slide_out, duration=slide_duration, side="left")]
    ).set_start(0)

    slide_out_sides = ["left"]
    videos = [first_clip]

    out_to_in_mapping = {"left": "right", "right": "left"}
    
    for idx, clip in enumerate(clips[1: -1], start=1):
        # For all other clips in the middle, we need them to slide in to the previous clip and out for the next one

        # determine `slide_in_side` according to the `slide_out_side` of the previous clip
        slide_in_side = out_to_in_mapping[slide_out_sides[-1]]
        
        slide_out_side = "left" if random.random() <= 0.5 else "right"
        slide_out_sides.append(slide_out_side)
                
        videos.append(
            (
                CompositeVideoClip(
                    [clip.fx(transfx.slide_in, duration=slide_duration, side=slide_in_side)]
                )
                .set_start(sum(durations[:idx]) - (slide_duration) * idx)
                .fx(transfx.slide_out, duration=slide_duration, side=slide_out_side)
            )
        )
    
    last_clip = CompositeVideoClip(
        [clips[-1].fx(transfx.slide_in, duration=slide_duration, side=out_to_in_mapping[slide_out_sides[-1]])]
    ).set_start(sum(durations[:-1]) - slide_duration * (len(clips) - 1))
    videos.append(last_clip)

    video = CompositeVideoClip(videos)
    return video


def compose_video(story_dir: Union[str, Path],
                  save_path: Union[str, Path],
                  captions: List,
                  num_pages: int,
                  fps: int = 10,
                  audio_sample_rate: int = 16000,
                  audio_codec: str = "mp3",
                  caption_config: dict = {},
                  fade_duration: float = 1.0,
                  slide_duration: float = 0.4,
                  zoom_speed: float = 0.5,
                  move_ratio: float = 0.95,
                  session_id: str = None,
                  compute_audio_energy: bool = False):
    # 任务要求：已移除 sound_volume, music_volume, bg_speech_ratio, music_path 参数
    if not isinstance(story_dir, Path):
        story_dir = Path(story_dir)

    # sound_dir = story_dir / "sound"  # 任务要求：去除音效模块
    # 智能检测文件目录结构，支持会话ID子目录
    image_dir = None
    speech_dir = None
    
    # 如果提供了会话ID，优先使用指定的会话目录
    if session_id:
        print(f"🎯 使用指定的会话目录: {session_id}")
        session_dir = story_dir / session_id
        
        if session_dir.exists() and session_dir.is_dir():
            if (session_dir / "image").exists():
                image_dir = session_dir / "image"
            if (session_dir / "speech").exists():
                speech_dir = session_dir / "speech"
    # 如果没有指定会话ID或指定的会话目录不存在，回退到自动检测
    elif session_id is None:
        # 检查是否存在会话ID子目录（如类似20251115_171253的格式）
        import re
        session_dirs = [d for d in story_dir.iterdir() if d.is_dir() and re.match(r'\d{8}_\d{6}', d.name)]
        
        if session_dirs:
            # 如果找到多个会话目录，使用最新的一个
            latest_session = sorted(session_dirs)[-1]
            print(f"🔍 检测到会话目录: {latest_session.name}")
            
            if (latest_session / "image").exists():
                image_dir = latest_session / "image"
            if (latest_session / "speech").exists():
                speech_dir = latest_session / "speech"
    
    # 如果没有找到会话子目录或子目录中没有所需文件，回退到直接在story_dir下查找
    if image_dir is None:
        image_dir = story_dir / "image"
    if speech_dir is None:
        speech_dir = story_dir / "speech"
    
    print(f"📁 使用图像目录: {image_dir}")
    print(f"🔊 使用语音目录: {speech_dir}")

    video_clips = []
    # audio_durations = []
    cur_duration = 0
    timestamps = []

    for page in trange(1, num_pages + 1):
        ##### speech track
        slide_silence = AudioArrayClip(np.zeros((int(audio_sample_rate * slide_duration), 2)), fps=audio_sample_rate)
        fade_silence = AudioArrayClip(np.zeros((int(audio_sample_rate * fade_duration), 2)), fps=audio_sample_rate)

        if (speech_dir / f"p{page}.wav").exists(): # single speech file
            single_utterance = True
            speech_file = str(speech_dir / f"p{page}.wav")  # 直接使用字符串路径，去掉./前缀
            speech_clip = AudioFileClip(speech_file, fps=audio_sample_rate)
            speech_clip = concatenate_audioclips([fade_silence, speech_clip, fade_silence])
            has_speech = True
        elif not (speech_dir / f"p{page}.wav").exists() and not list(speech_dir.glob(f"p{page}_*.wav")):
            # 既没有单文件也没有多文件的情况
            print(f"⚠️  第{page}页未找到语音文件，创建5秒静默音频")
            single_utterance = True
            # 创建静默音频作为替代
            speech_clip = AudioArrayClip(np.zeros((int(audio_sample_rate * 5), 2)), fps=audio_sample_rate)
            speech_clip = concatenate_audioclips([fade_silence, speech_clip, fade_silence])
            # 设置默认能量值，避免后续计算错误
            speech_rms = 0.01  # 一个很小的默认值
            has_speech = False
        else: # multiple speech files
            single_utterance = False
            speech_files = list(speech_dir.glob(f"p{page}_*.wav"))
            speech_files = sorted(speech_files, key=lambda x: int(x.stem.split("_")[-1]))
            
            # 检查是否找到语音文件
            if not speech_files:
                print(f"⚠️  第{page}页未找到语音文件，创建5秒静默音频")
                speech_clip = AudioArrayClip(np.zeros((int(audio_sample_rate * 5), 2)), fps=audio_sample_rate)
                speech_clip = concatenate_audioclips([fade_silence, speech_clip, fade_silence])
                speech_rms = 0.01  # 一个很小的默认值
                has_speech = False
            else:
                has_speech = True
            speech_clips = []
            for utt_idx, speech_file in enumerate(speech_files):
                speech_clip = AudioFileClip(speech_file.__str__(), fps=audio_sample_rate)
                # add multiple timestamps of the same speech clip
                if utt_idx == 0:
                    timestamps.append([cur_duration + fade_duration,
                                       cur_duration + fade_duration + speech_clip.duration])
                    cur_duration += speech_clip.duration + fade_duration
                elif utt_idx == len(speech_files) - 1:
                    timestamps.append([
                        cur_duration,
                        cur_duration + speech_clip.duration
                    ])
                    cur_duration += speech_clip.duration + fade_duration + slide_duration
                else:
                    timestamps.append([
                        cur_duration,
                        cur_duration + speech_clip.duration
                    ])
                    cur_duration += speech_clip.duration
                speech_clips.append(speech_clip)
            speech_clip = concatenate_audioclips([fade_silence] + speech_clips + [fade_silence])
            speech_file = speech_files[0] # for energy calculation
        
        # add slide silence
        if page == 1:
            speech_clip = concatenate_audioclips([speech_clip, slide_silence])
        else:
            speech_clip = concatenate_audioclips([slide_silence, speech_clip, slide_silence])
        
        # add the timestamp of the whole clip as a single element 
        if single_utterance:
            if page == 1:
                timestamps.append([cur_duration + fade_duration,
                                   cur_duration + speech_clip.duration - fade_duration - slide_duration])
                cur_duration += speech_clip.duration - slide_duration
            else:
                timestamps.append([cur_duration + fade_duration + slide_duration,
                                   cur_duration + speech_clip.duration - fade_duration - slide_duration])
                cur_duration += speech_clip.duration - slide_duration

        # 可选：计算语音能量（默认关闭以提速）
        if compute_audio_energy:
            if 'has_speech' in locals() and has_speech and ('speech_rms' not in locals() or speech_rms is None):
                try:
                    if 'speech_files' in locals() and isinstance(speech_files, list) and speech_files:
                        speech_file = speech_files[0]
                    if 'speech_file' in locals() and speech_file and isinstance(speech_file, (str, Path)) and Path(speech_file).exists():
                        speech_array, _ = librosa.core.load(str(speech_file), sr=None)
                        speech_rms = librosa.feature.rms(y=speech_array)[0].mean()
                        print(f"🔊 第{page}页语音能量: {speech_rms:.6f}")
                    else:
                        speech_rms = 0.01
                except Exception as e:
                    print(f"⚠️  计算语音能量时出错: {e}")
                    speech_rms = 0.01

        # set image as the main content, align the duration
        # 改进图像文件路径处理，移除./前缀
        image_file = (image_dir / f"p{page}.png").absolute()
        
        # ========== 新增：支持数据可视化图表 ==========
        # 优先检查是否存在带图表的图像
        chart_image_file = image_dir / f"p{page}_with_chart.png"
        if chart_image_file.exists():
            image_file = chart_image_file
            print(f"📊 第{page}页：使用带图表的图像: {chart_image_file.name}")
        else:
            # 检查是否存在动态图表（GIF格式）
            animated_chart_file = image_dir / f"p{page}_animated_chart.gif"
            if animated_chart_file.exists():
                image_file = animated_chart_file
                print(f"📊 第{page}页：使用动态图表: {animated_chart_file.name}")
            else:
                # 检查是否存在单独的图表文件
                chart_only_file = image_dir / f"p{page}_chart.png"
                if chart_only_file.exists():
                    # 如果存在原始图像，尝试组合；否则直接使用图表
                    if image_file.exists():
                        print(f"📊 第{page}页：检测到图表文件，将组合图像和图表")
                        # 使用可视化工具组合图像和图表
                        try:
                            viz_tool = VisualizationTool(output_dir=str(image_dir))
                            
                            # 创建组合图像
                            combined_path = image_dir / f"p{page}_combined.png"
                            viz_tool.combine_image_with_chart(
                                str(image_file),
                                str(chart_only_file),
                                str(combined_path),
                                layout="bottom"  # 可以根据配置调整
                            )
                            image_file = combined_path
                            print(f"✅ 第{page}页：图像和图表已组合")
                        except Exception as e:
                            print(f"⚠️  第{page}页：组合图像和图表失败: {e}，使用原始图像")
                    else:
                        # 如果没有原始图像，直接使用图表
                        image_file = chart_only_file
                        print(f"📊 第{page}页：使用图表文件作为图像")
        # ========== 数据可视化支持结束 ==========
        
        # 检查图像文件是否存在
        if not image_file.exists():
            # 尝试查找其他可能的图像格式或命名
            alternative_images = list(image_dir.glob(f"p{page}*.png"))
            # 也检查GIF格式（用于动态图表）
            alternative_images.extend(list(image_dir.glob(f"p{page}*.gif")))
            if alternative_images:
                image_file = alternative_images[0]
                print(f"🔄 找到替代图像文件: {image_file.name}")
            else:
                # 如果找不到图像文件，创建一个临时的占位图像
                print(f"⚠️  第{page}页未找到图像文件，创建占位图像")
                # 创建一个简单的占位图像
                placeholder_path = image_dir / f"p{page}_placeholder.png"
                from PIL import Image, ImageDraw, ImageFont
                try:
                    # 创建一个带文字的占位图像
                    img = Image.new('RGB', (800, 600), color=(73, 109, 137))
                    d = ImageDraw.Draw(img)
                    d.text((400, 300), f"Page {page}", fill=(255, 255, 255), anchor='mm')
                    img.save(placeholder_path)
                    image_file = placeholder_path
                except Exception as e:
                    print(f"❌ 创建占位图像失败: {e}")
                    # 如果创建占位图像也失败，使用默认的空白图像
                    image_file = image_dir / f"p{page}.png"  # 继续尝试原路径，让ImageClip抛出更明确的错误
        
        try:
            # 检查文件类型，GIF需要特殊处理
            if image_file.suffix.lower() == '.gif':
                # 对于GIF文件，使用VideoFileClip而不是ImageClip
                from moviepy.editor import VideoFileClip
                image_clip = VideoFileClip(str(image_file))
                # 调整GIF的时长以匹配语音时长
                if image_clip.duration < speech_clip.duration:
                    # 如果GIF时长较短，循环播放
                    loops_needed = int(speech_clip.duration / image_clip.duration) + 1
                    image_clip = concatenate_videoclips([image_clip] * loops_needed)
                    image_clip = image_clip.subclip(0, speech_clip.duration)
                elif image_clip.duration > speech_clip.duration:
                    # 如果GIF时长较长，截取到语音时长
                    image_clip = image_clip.subclip(0, speech_clip.duration)
            else:
                # 对于静态图像，使用原有的ImageClip逻辑
                image_clip = ImageClip(str(image_file))
        except Exception as e:
            print(f"❌ 加载图像失败 {image_file}: {e}")
            # 如果图像加载失败，创建一个简单的占位视频
            from moviepy.video.VideoClip import ColorClip
            # 需要从params获取width和height，这里使用默认值
            default_width = 1024
            default_height = 512
            image_clip = ColorClip(size=(default_width, default_height), color=(73, 109, 137), duration=speech_clip.duration)
            print("📹 创建了占位视频片段")
        
        # 设置图像时长和帧率
        image_clip = image_clip.set_duration(speech_clip.duration).set_fps(fps)
        
        # 对于静态图像，添加淡入淡出效果（GIF已经有动画，不需要额外效果）
        if image_file.suffix.lower() != '.gif':
            image_clip = image_clip.crossfadein(fade_duration).crossfadeout(fade_duration)

        # 添加缩放或移动效果（仅对静态图像，动态图表保持原样）
        if image_file.suffix.lower() != '.gif':
            if random.random() <= 0.5: # zoom in or zoom out
                if random.random() <= 0.5:
                    zoom_mode = "in"
                else:
                    zoom_mode = "out"
                image_clip = add_zoom_effect(image_clip, zoom_speed, zoom_mode)
            else: # move left or right
                if random.random() <= 0.5:
                    direction = "left"
                else:
                    direction = "right"
                image_clip = add_move_effect(image_clip, direction=direction, move_raito=move_ratio)

        # 任务要求：去除音效模块，只使用语音
        # sound track - 已移除
        audio_clip = speech_clip

        video_clip = image_clip.set_audio(audio_clip)        
        video_clips.append(video_clip)

        # audio_durations.append(audio_clip.duration)

    # final_clip = concatenate_videoclips(video_clips, method="compose")
    composite_clip = add_slide_effect(video_clips, slide_duration=slide_duration)
    composite_clip = add_bottom_black_area(composite_clip, black_area_height=caption_config["area_height"])
    del caption_config["area_height"]
    max_caption_length = caption_config["max_length"]
    del caption_config["max_length"]
    composite_clip = add_caption(
        captions,
        story_dir / "captions.srt",
        timestamps,
        composite_clip,
        max_caption_length,
        **caption_config
    )

    # 任务要求：去除背景音乐模块
    # add music track - 已移除
    # 直接使用原有音频（仅包含语音）
    
    composite_clip.write_videofile(save_path.__str__(),
                                   audio_fps=audio_sample_rate,
                                   audio_codec=audio_codec,)


@register_tool("slideshow_video_compose")
class SlideshowVideoComposeAgent:

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def adjust_caption_config(self, width, height):
        area_height = int(height * 0.06)
        fontsize = int((width + height) / 2 * 0.025)
        max_length = int(width / (fontsize * 0.6))  # 根据宽度和字体大小动态调整字幕最大长度
        return {
            "fontsize": fontsize,
            "area_height": area_height,
            "max_length": max_length
        }

    def call(self, params):
        try:
            height = params["height"]
            width = params["width"]
            pages = params["pages"]
            story_dir = Path(params["story_dir"])
            
            # 检查参数完整性
            required_params = ["height", "width", "pages", "story_dir", "fps", "audio_sample_rate", 
                             "audio_codec", "caption", "slideshow_effect"]
            for param in required_params:
                if param not in params:
                    raise ValueError(f"缺少必要参数: {param}")
            
            # 检查story_dir是否存在
            if not story_dir.exists():
                raise FileNotFoundError(f"故事目录不存在: {story_dir}")
            
            # 更新字幕配置，添加最大长度设置
            caption_config = self.adjust_caption_config(width, height)
            caption_config.update(params["caption"])
            params["caption"] = caption_config
            
            # 优先使用传入的output_path参数，如果没有则使用默认命名方式
            if "output_path" in params and params["output_path"]:
                save_path = Path(params["output_path"])
            else:
                # 生成带有时间戳的视频文件名，避免覆盖现有文件
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                video_filename = f"story_video_{timestamp}.mp4"
                save_path = story_dir / video_filename
            
            print(f"📽️  开始合成视频，将保存至: {save_path}")
            print(f"📐 视频尺寸: {width}x{height}, FPS: {params['fps']}")
            print(f"📄 页数: {len(pages)}")
            
            # 调用合成函数
            # 从params中获取session_id参数（如果存在）
            session_id = params.get("session_id")
            
            compose_video(
                story_dir=story_dir,
                save_path=save_path,
                captions=pages,
                num_pages=len(pages),
                fps=params["fps"],
                audio_sample_rate=params["audio_sample_rate"],
                audio_codec=params["audio_codec"],
                caption_config=params["caption"],
                **params["slideshow_effect"],
                session_id=session_id,  # 传递会话ID参数
                compute_audio_energy=params.get("compute_audio_energy", False)
            )
            
            print(f"✅ 视频合成完成，已保存至: {save_path}")
            return {"video_path": str(save_path)}
        except Exception as e:
            print(f"❌ 视频合成失败: {str(e)}")
            # 提供详细的错误信息
            import traceback
            print(f"详细错误信息:\n{traceback.format_exc()}")
            return {"error": str(e), "video_path": None}
