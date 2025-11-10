"""
创建最简单的图像幻灯片（如果字幕有问题，先用纯图像）
"""
from pathlib import Path
from moviepy.editor import ImageClip, concatenate_videoclips
import json

story_dir = Path("generated_stories/example")
image_dir = story_dir / "image"
output_path = story_dir / "slideshow_simple.mp4"

print("=" * 70)
print("  创建纯图像幻灯片视频")
print("=" * 70)
print()

# 读取故事文本用于打印
script_file = story_dir / "script_data.json"
if script_file.exists():
    with open(script_file, 'r', encoding='utf-8') as f:
        script_data = json.load(f)
    pages = [page['story'] for page in script_data['pages']]
    
    # 将故事文本保存到txt文件，方便查看
    with open(story_dir / "story_text.txt", 'w', encoding='utf-8') as f:
        for i, page in enumerate(pages, 1):
            f.write(f"=== 第 {i} 页 ===\n")
            f.write(f"{page}\n\n")
    print(f"✅ 故事文本已保存到: {story_dir / 'story_text.txt'}")
    print()

# 获取图像
image_files = sorted(image_dir.glob("p*.png"))

print(f"📊 找到 {len(image_files)} 张图像")
print()

clips = []
for idx, img_file in enumerate(image_files, 1):
    print(f"处理第 {idx}/{len(image_files)} 张: {img_file.name}")
    clip = ImageClip(str(img_file))
    clip = clip.set_duration(6)  # 每张6秒
    clip = clip.set_fps(24)
    clips.append(clip)

print(f"\n🎬 合成视频...")
final = concatenate_videoclips(clips)

print(f"💾 保存到: {output_path}")
final.write_videofile(str(output_path), fps=24, codec='libx264', audio=False)

print(f"\n✅ 完成！")
print(f"   视频: {output_path}")
print(f"   时长: {final.duration:.1f} 秒")
print(f"   故事文本: {story_dir / 'story_text.txt'}")
print()
print("💡 提示：故事文本已保存为单独的txt文件，")
print("   观看视频时可以对照阅读！")

