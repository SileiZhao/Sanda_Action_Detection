#!/bin/bash
# 该脚本用于视频抽帧

# 定义输入和输出目录
input_dir="/home/zhaosilei/Projects/PycharmProjects/Sanda_Action_Detection/data/videos"
output_dir="/home/zhaosilei/Projects/PycharmProjects/Sanda_Action_Detection/data/rawframes"

# 创建输出目录（如果不存在）
mkdir -p "$output_dir"

# 遍历输入目录中的所有 .mp4 文件
for video_path in "$input_dir"/*.mp4; do
    # 获取视频文件名（不带扩展名）
    video_name=$(basename "$video_path" .mp4)
    
    # 创建与视频名相同的子目录
    frame_dir="$output_dir/$video_name"
    mkdir -p "$frame_dir"
    
    # 使用 ffmpeg 将视频裁剪成帧，编号从 0 开始
    ffmpeg -i "$video_path" -q:v 2 -start_number 0 "$frame_dir/img_%04d.jpg"
    
    echo "Processed $video_path -> $frame_dir"
done

echo "All videos have been processed."
