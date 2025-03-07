import os
import re
import cv2
import pandas as pd
import random


# 自定义解析函数
def parse_llc_file(content):
    """
    解析非标准格式的 .llc 文件
    """
    # 移除换行符和多余的空格
    content = re.sub(r'\s+', '', content)

    # 提取 version
    version_match = re.search(r'version:(\d+)', content)
    version = int(version_match.group(1)) if version_match else 1

    # 提取 mediaFileName
    media_file_name_match = re.search(r"mediaFileName:'([^']+)'", content)
    media_file_name = media_file_name_match.group(1) if media_file_name_match else ''

    # 提取 cutSegments
    cut_segments = []
    cut_segments_match = re.search(r'cutSegments:\[(.*?)\]', content)
    if cut_segments_match:
        segments_content = cut_segments_match.group(1)
        # 匹配每个 segment
        segment_matches = re.finditer(r'\{([^}]+)\}', segments_content)
        for segment_match in segment_matches:
            segment_content = segment_match.group(1)
            # 提取 start, end, name
            start_match = re.search(r'start:([\d.]+)', segment_content)
            end_match = re.search(r'end:([\d.]+)', segment_content)
            name_match = re.search(r"name:'([^']*)'", segment_content)
            if start_match and end_match:
                segment = {
                    'start': float(start_match.group(1)),
                    'end': float(end_match.group(1)),
                    'name': name_match.group(1) if name_match else ''
                }
                cut_segments.append(segment)

    return {
        'version': version,
        'mediaFileName': media_file_name,
        'cutSegments': cut_segments
    }


# 获取视频帧率
def get_frame_rate(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps


# 将时间转换为帧号
def time_to_frame(time, fps):
    return int(time * fps)


# 处理单个视频和标注文件
def process_video_and_annotation(video_path, annotation_path, output_folder, clip_counter):
    # 读取标注文件
    with open(annotation_path, 'r') as f:
        content = f.read()
        annotation = parse_llc_file(content)

    # 获取视频帧率
    fps = get_frame_rate(video_path)

    # 获取视频总帧数
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    # 将标注时间转换为帧号
    cut_segments = annotation.get('cutSegments', [])
    if len(cut_segments) == 1 and 'end' not in cut_segments[0]:
        # 如果没有任何动作标签
        cut_segments = []
    else:
        # 转换时间到帧号
        for segment in cut_segments:
            segment['start_frame'] = time_to_frame(segment['start'], fps)
            segment['end_frame'] = time_to_frame(segment['end'], fps)

    # 以动作为中心进行随机分割
    for segment in cut_segments:
        # 计算动作持续时间
        action_duration = segment['end'] - segment['start']

        # 随机选择起始点，确保动作完整包含在 30 秒片段中
        max_start = segment['start']  # 动作开始时间
        min_start = max(0, segment['end'] - 30)  # 确保片段不超出视频范围

        if min_start > max_start:
            # 如果动作持续时间超过 30 秒，直接以动作开始时间为起始点
            clip_start_time = segment['start']
        else:
            # 随机选择起始点
            clip_start_time = random.uniform(min_start, max_start)

        # 计算片段结束时间
        clip_end_time = clip_start_time + 30

        # 如果片段超出视频范围，调整起始时间
        if clip_end_time > total_frames / fps:
            clip_end_time = total_frames / fps
            clip_start_time = max(0, clip_end_time - 30)

        # 生成视频片段文件名
        clip_name = f"sanda_clipanno_{clip_counter:06d}.mp4"
        clip_path = os.path.join(output_folder, clip_name)

        # 裁剪视频
        video = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(clip_path, fourcc, fps, (int(video.get(3)), int(video.get(4))))

        start_frame = time_to_frame(clip_start_time, fps)
        end_frame = time_to_frame(clip_end_time, fps)

        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_num in range(start_frame, end_frame):
            ret, frame = video.read()
            if not ret:
                break
            out.write(frame)

        out.release()
        video.release()

        # 生成CSV文件
        csv_data = []
        for frame_num in range(start_frame, end_frame):
            # 新视频片段的帧号从0开始
            new_frame_num = frame_num - start_frame
            label = 0
            for seg in cut_segments:
                # 判断当前帧是否在标注的动作范围内
                if seg['start_frame'] <= frame_num <= seg['end_frame']:
                    label = 1
                    break
            csv_data.append({'video_name': clip_name, 'frame_number': new_frame_num, 'label': label})

        csv_path = os.path.join(output_folder, f"{clip_name.replace('.mp4', '.csv')}")
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)

        # 更新片段计数器
        clip_counter += 1

    return clip_counter


# 主函数
if __name__ == "__main__":
    # 初始化片段计数器
    clip_counter = 1
    # 定义路径
    input_folder = '/Users/zhaosilei/Downloads/sanda_action_anno'
    output_folder = '/Users/zhaosilei/PycharmProjects/Benz/Sanda_action_recognation/data'

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有视频和标注文件
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                annotation_path = os.path.join(root, file.replace('.mp4', '-proj.llc'))
                if os.path.exists(annotation_path):
                    clip_counter = process_video_and_annotation(video_path, annotation_path, output_folder,
                                                                clip_counter)
