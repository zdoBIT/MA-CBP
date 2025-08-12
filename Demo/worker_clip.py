from PIL import Image
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Dict
import os
import random
import cv2
import json
from transformers import LlavaForConditionalGeneration, LlavaProcessor,AutoProcessor
import torch
from peft import peft_model,PeftModel
from stage3_data import LlavaDatasetStage3, TrainLlavaModelCollator, LlavaForConditionalGenerationClip
import matplotlib.pyplot as plt
import zmq
import base64
import numpy as np
import threading
import time
from collections import deque
import re
import hashlib
import json

raw_model_name_or_path = "model/stage1_model"
peft_model_name_or_path1 = "model/checkpoint-12625"
peft_model_name_or_path2 = "model/checkpoint-1482"
model = LlavaForConditionalGenerationClip.from_pretrained(raw_model_name_or_path,device_map="cuda:0", torch_dtype=torch.bfloat16)
model_with_lora1 = PeftModel.from_pretrained(model, peft_model_name_or_path1, adapter_name="peft_v1")
merged_model = model_with_lora1.merge_and_unload()
model_with_lora2 = PeftModel.from_pretrained(merged_model, peft_model_name_or_path2)

processor = LlavaProcessor.from_pretrained(raw_model_name_or_path)
model_with_lora2.eval()

data_root_dir = "dataset"
test_dataset = LlavaDatasetStage3(data_root_dir,"test")
timestamp = time.strftime("%m%d_%M%S")

def natural_sort_key(filename):
    # 提取 filename 中的数字部分用于排序
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', filename)]

def sample_clips(clip_path: Path, num_frames: int, split_num: int):
    images_path = [clip_path.joinpath(image_file) for image_file in sorted(os.listdir(clip_path),key=natural_sort_key) if image_file.endswith(".jpg")]
    total_frames = len(images_path[split_num:])
    actual_frames = min(total_frames, num_frames)
    # print(f"actual_frames:{actual_frames}")
    interval = total_frames // actual_frames
    sampled_frames = []
    for i in range(actual_frames):
        frame_index = i * interval
        raw_image = Image.open(images_path[split_num + frame_index])
        sampled_frames.append(raw_image) 

    return  sampled_frames, len(sampled_frames)



def show_sampled_frames(frame_list):
    num_frames = len(frame_list)
    cols = min(num_frames, 3)  # 每行最多5张
    rows = (num_frames + cols - 1) // cols

    plt.figure(figsize=(15, 3 * rows))

    for idx, frame in enumerate(frame_list):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(frame)
        plt.axis('off')
        plt.title(f"Frame {idx}")

    plt.tight_layout()
    plt.show()
    

def build_qaclip(model, processor: LlavaProcessor, q_text: str, frames: list, summary: str):
    frame_num = len(frames)
    frame_list = [Image.fromarray(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)) for raw_image in frames]
    print(f"history:",summary)
    additional_descriptions = "<history>" + summary + "</history>" + "<clip>" + "<image><sep>"*(frame_num - 1) + "<image>" + "</clip>"
    q_text = additional_descriptions + q_text
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": q_text},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    processor.patch_size = 14
    inputs = processor(frame_list,prompt, return_tensors="pt")
    for tk in inputs.keys():
        inputs[tk] = inputs[tk].to(model.device)
    generate_ids = model.generate(**inputs,max_new_tokens=2000)
    
    generate_ids = [
        oid[len(iids):] for oid, iids in zip(generate_ids, inputs.input_ids)
    ]
    
    gen_text = processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    return gen_text

############################## 实现逻辑 ##################################

# 初始化 ZeroMQ 上下文
ctx = zmq.Context()

# 初始化发布者
publisher = ctx.socket(zmq.PUB)
publisher.bind("tcp://*:5575")  # 发布到端口5575，与app.py中的配置匹配

# 订阅图像帧
frame_sub = ctx.socket(zmq.SUB)
frame_sub.connect("tcp://localhost:5560")
frame_sub.setsockopt(zmq.SUBSCRIBE, b"")

# 订阅摘要
summary_sub = ctx.socket(zmq.SUB)
summary_sub.connect("tcp://localhost:5563")  # 修改为与 worker_summary.py 中的发布端口一致
summary_sub.setsockopt(zmq.SUBSCRIBE, b"")
print("🔔 已订阅摘要端口 5563")

# 全局缓存
frame_buffer = deque(maxlen=8)  # 缓存最近 8 帧
frame_lock = threading.Lock()

latest_summary = None
summary_lock = threading.Lock()

# 接收图像帧的线程函数
def receive_frames():
    global frame_buffer, current_video_info
    print("🔍 开始接收帧数据...")
    while True:
        try:
            msg = frame_sub.recv_json()
            print("📥 收到帧数据")
            jpg = base64.b64decode(msg["frame"])
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            print(f"🖼️ 解码帧: {img.shape if img is not None else 'None'}")

            video_meta = {
                "filename": msg.get("video_file", "unknown_video"),
                "index": msg.get("video_index", 0),
                "total": msg.get("total_videos", 1),
                "timestamp": msg.get("timestamp", time.time()),
                "resolution": f"{msg.get('frame_width', 0)}x{msg.get('frame_height', 0)}"
            }

            with frame_lock:
                if img is not None:  # 确保图像有效
                    frame_buffer.append(img)  # 必须保留的帧缓存操作
                    current_video_info = video_meta
            time.sleep(0.5)
        except Exception as e:
            print(f"❌ 接收帧时出错: {e}")
            time.sleep(1)  # 出错时等待1秒再重试

# 接收摘要的线程函数
def receive_summary():
    global latest_summary
    print("🔍 开始接收摘要数据...")
    while True:
        try:
            msg = summary_sub.recv_json()
            print(f"📝 收到摘要 (原始数据): {msg}")  # 调试用
            
            # 处理 summary（兼容 str 和 list）
            if isinstance(msg["summary"], list):
                # 如果是列表，合并成字符串
                summary_text = " ".join(msg["summary"])
            else:
                # 如果是字符串，直接使用
                summary_text = msg["summary"]
            
            # 提取摘要内容（去掉可能的前缀）
            if ":" in summary_text:
                latest_summary = summary_text.partition(":")[2].strip()
            else:
                latest_summary = summary_text.strip()
            
            print(f"📝 处理后摘要: {latest_summary}")
        except Exception as e:
            print(f"❌ 接收摘要时出错: {e}")

def hash_frames(frames):
    md5 = hashlib.md5()
    for f in frames:
        md5.update(f.tobytes())
    return md5.hexdigest()

def hash_text(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


# 启动信息
print("🚀 CLIP Worker 启动成功！")
print(f"📡 监听端口: 5560 (帧), 5558 (摘要)")
print(f"📤 发布端口: 5575")
print("🔄 等待数据...")

# 启动接收线程
threading.Thread(target=receive_frames, daemon=True).start()
threading.Thread(target=receive_summary, daemon=True).start()
q_text = "Focus on the historical texts and image frames. Identify any abnormalities in the following content and provide a reason if any are found."

last_frame_hash = None
last_summary_hash = None

# 主处理循环
while True:
    time.sleep(0.1)  # 控制主循环速率，避免空跑占用 CPU

    with frame_lock:
        if len(frame_buffer) < 8:
            continue
        frame_list = [f.copy() for f in frame_buffer]  # 安全复制每帧图像
        current_video = current_video_info['filename']

    with summary_lock:
        summary = latest_summary

    if summary is None:
        continue
    
    
    current_frame_hash = hash_frames(frame_list)
    current_summary_hash = hash_text(summary)

    if current_frame_hash == last_frame_hash and current_summary_hash == last_summary_hash:
        continue  # 无更新，跳过处理

    # 更新记录
    last_frame_hash = current_frame_hash
    last_summary_hash = current_summary_hash

    print(f"当前处理视频: {current_video}")  # 检查文件名是否正确

    # 模型推理
    result = build_qaclip(model_with_lora2, processor, q_text, frame_list, summary)
    
    # 输出到控制台
    print("Result:", result)
    
    # 保存结果到文件
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 使用时间戳创建唯一的文件名
    output_file = output_dir / f"{current_video}_{timestamp}.txt"
    
    # 保存结果
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"time:{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"video_file: {current_video_info['filename']}\n")
        f.write(f"Summary: {summary}\n")
        f.write(f"\nAnalysis Result: {result}\n")
        f.write("-" * 50 + "\n\n")
    
    print(f"\n✅ 分析结果已保存到: {output_file}")
    
    
    try:
        publisher.send_json({
        "category": [result],  # 保持与网页端一致的格式
        "type": "clip",            # 添加类型标识
        "timestamp": time.time()    # 添加时间戳
    })
        print("✅ 分析结果已发布到 WebSocket 服务器")
    except Exception as e:
        print(f"❌ 发布分析结果时出错: {e}")
