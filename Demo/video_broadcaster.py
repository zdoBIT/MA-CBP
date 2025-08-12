import cv2
import zmq
import base64
import time
import argparse
import os
import glob
from pathlib import Path

def get_video_files(directory, extensions=('*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv')):
    """获取目录下所有视频文件"""
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(directory, f'**/{ext}'), recursive=True))
    return sorted(video_files)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='视频广播器 - 从目录读取视频文件并通过ZMQ发布')
    parser.add_argument('--input', type=str, required=True, help='视频文件路径或包含视频文件的目录')
    parser.add_argument('--fps', type=float, default=30, help='目标帧率 (默认: 30，如果输入是视频文件，将使用视频的原始帧率)')
    parser.add_argument('--port', type=int, default=5560, help='ZMQ发布端口 (默认: 5560)')
    parser.add_argument('--loop', action='store_true', help='循环播放视频')
    parser.add_argument('--shuffle', action='store_true', help='随机播放视频 (需要--loop)')
    args = parser.parse_args()

    print("⏳ 程序将在5秒后开始...")
    time.sleep(5)

    # 初始化ZMQ
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind(f"tcp://*:{args.port}")
    print(f"🎥 视频广播器已启动，正在监听 0.0.0.0:{args.port}")

    # 获取视频文件列表
    if os.path.isfile(args.input):
        video_files = [args.input]
    elif os.path.isdir(args.input):
        video_files = get_video_files(args.input)
        if not video_files:
            print(f"❌ 在目录 {args.input} 中找不到视频文件")
            return
        print(f"📁 找到 {len(video_files)} 个视频文件")
    else:
        print(f"❌ 无效的输入路径: {args.input}")
        return

    try:
        video_index = 0
        frame_count = 0
        start_time = time.time()
        
        while True:
            if video_index >= len(video_files):
                if args.loop:
                    video_index = 0
                    if args.shuffle and len(video_files) > 1:
                        import random
                        random.shuffle(video_files)
                        print("🔄 随机顺序重新开始播放视频...")
                    else:
                        print("🔄 重新开始播放视频...")
                else:
                    print("✅ 所有视频播放完成")
                    break
            
            current_video = video_files[video_index]
            print(f"📽️ 正在播放: {os.path.basename(current_video)} ({video_index+1}/{len(video_files)})")
            
            cap = cv2.VideoCapture(current_video)
            if not cap.isOpened():
                print(f"❌ 无法打开视频文件: {current_video}")
                video_index += 1
                continue
                
            # 获取视频信息
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps > 0:
                frame_delay = 1.0 / video_fps
            else:
                frame_delay = 1.0 / args.fps
                
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 调整帧大小（可选）
                # frame = cv2.resize(frame, (640, 480))
                
                # 编码为JPEG
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                jpg_bytes = base64.b64encode(buffer).decode('utf-8')
                
                # 发布帧
                publisher.send_json({
                    "frame": jpg_bytes,
                    "timestamp": time.time(),
                    "frame_count": frame_count,
                    "video_file": os.path.basename(current_video),
                    "video_index": video_index,
                    "total_videos": len(video_files),
                    "frame_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "frame_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                })
                
                # 控制帧率
                frame_count += 1
                elapsed = time.time() - start_time
                
                # 计算下一帧的显示时间
                next_frame_time = frame_count * frame_delay
                sleep_time = next_frame_time - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # 每秒打印一次状态
                if frame_count % int(video_fps if video_fps > 0 else args.fps) == 0:
                    actual_fps = frame_count / (time.time() - start_time)
                    print(f"📊 状态: {os.path.basename(current_video)[:20]}... | 帧: {frame_count} | 实际帧率: {actual_fps:.1f}fps | 目标帧率: {video_fps:.1f}fps")
            
            # 释放当前视频
            cap.release()
            video_index += 1

    except KeyboardInterrupt:
        print("\n🛑 正在停止视频广播...")
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")
    finally:
        cap.release()
        publisher.close()
        context.term()
        print("✅ 视频广播器已安全停止")

if __name__ == "__main__":
    main()
