import os
import sys
from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from IPython.display import display
import zmq
import time
import base64
import cv2
import numpy as np
import threading
import hashlib
from pathlib import Path  # 添加Path类导入

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构造 BLIP 模块所在的路径
blip_dir = "model/BLIP"
# 添加到 sys.path 中
if blip_dir not in sys.path:
    sys.path.insert(0, blip_dir)

from models.blip import blip_decoder

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def process_image(np_image, image_size, device):
    raw_image = Image.fromarray(cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)) 
    w, h = raw_image.size
    display(raw_image.resize((w // 5, h // 5)))

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                             (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image

def hash_frame(frame):
    return hashlib.md5(frame.tobytes()).hexdigest()

# 模型相关设置
image_size = 384
model_path = r'/home/bit118/data/modelDir/MA_CBP/model/BLIP/checkpoint_best.pth'

# 加载模型
model = blip_decoder(pretrained=model_path, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

# 初始化ZMQ
context = zmq.Context()
publisher = context.socket(zmq.PUB)
publisher.bind("tcp://*:5565")
print("BLIP worker started")

# 订阅视频帧
subscriber = context.socket(zmq.SUB)
subscriber.connect("tcp://localhost:5560")
subscriber.setsockopt_string(zmq.SUBSCRIBE, '')
# 设置高水位标记，防止内存溢出
subscriber.setsockopt(zmq.RCVHWM, 1)

# 帧缓存及线程锁
latest_frame = None
frame_lock = threading.Lock()
last_frame_hash = None  # 初始化帧哈希值
descriptions = []  # 初始化描述列表

def reset_worker_state():
    """重置工作进程状态"""
    global last_frame_hash, descriptions
    with frame_lock:
        last_frame_hash = None
        descriptions = []
    print("工作进程状态已重置")

# 主循环
while True:
    # 接收帧或控制消息
    frame_data = subscriber.recv_json()
    
    # 检查是否是控制消息
    if 'control' in frame_data and frame_data['control'] == 'reset':
        reset_worker_state()
        continue
        
    # 确保是帧数据
    if 'frame' not in frame_data:
        continue
    frame = base64.b64decode(frame_data['frame'])
    np_arr = np.frombuffer(frame, np.uint8)
    current_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    current_hash = hash_frame(current_frame)

    if current_hash == last_frame_hash:
        continue  # 帧未变，跳过
    last_frame_hash = current_hash

    # 处理新帧
    frame_tensor = process_image(current_frame, image_size, device)

    with torch.no_grad():
        caption = model.generate(frame_tensor, sample=False, num_beams=1, max_length=800, min_length=5)[0]
        print(caption)
        
        # 将输出保存到文件
        try:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True, parents=True, mode=0o755)
            output_file = output_dir / "blip_output.txt"
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {caption}\n")
        except Exception as e:
            print(f"保存输出时出错: {e}")
            # 尝试使用临时目录
            try:
                import tempfile
                temp_dir = Path(tempfile.gettempdir()) / "blip_output"
                temp_dir.mkdir(exist_ok=True, mode=0o755)
                output_file = temp_dir / "blip_output.txt"
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {caption}\n")
                print(f"输出已保存到临时目录: {output_file}")
            except Exception as e2:
                print(f"无法保存到临时目录: {e2}")

    descriptions.append(caption)
    if len(descriptions) > 100:
        descriptions.pop(0)

    publisher.send_json({
    "descriptions": [caption],  # 保持与网页端一致的格式
    "type": "blip",            # 添加类型标识
    "timestamp": time.time()    # 添加时间戳
})
