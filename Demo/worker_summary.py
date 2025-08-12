import os
import json
from pathlib import Path
import argparse
from collections import Counter
import os
import re
from openai import OpenAI
from tqdm import tqdm
import zmq
import time
import threading
import hashlib

def build_frame_json(descriptions):
    return {
        f"frame{i}": {
            "image_name": f"frame_{i:05d}.jpg",
            "image_caption": caption
        }
        for i, caption in enumerate(descriptions)
    }

# 统计所有人物昵称出现次数
def count_names(data):
    name_counter = Counter()
    name_list = []  # 存储所有人物昵称
    
    for frame in data.values():
        caption = frame["image_caption"].lower()
        # 假设我们关注以下可能的人物名称
        names = ["man", "woman", "boy", "girl","child","kid","skateboard","balloon","snake","bird","parrot","horse","cow","playing","kitchen","restaurant"]
        for name in names:
            if name in caption:
                name_counter[name] += 1
                name_list.append(name)
    
    return name_counter, name_list

def filter_captions(data, name_counter):
    # 按出现次数降序排序
    sorted_names = sorted(name_counter.items(), key=lambda x: x[1], reverse=True)
    
    # 确定保留的名称
    filtered_names = set()
    
    if not sorted_names:
        return {}  # 如果 name_counter 为空，直接返回空字典
    
    # 规则 I：1st <= 2.5 * 2nd
    if len(sorted_names) > 1:
        if sorted_names[0][1] > 2.5 * sorted_names[1][1]:
            filtered_names.add(sorted_names[0][0])  # 仅保留出现次数最多的那个
        else:
            filtered_names.add(sorted_names[0][0])  # 保留第一个
            filtered_names.add(sorted_names[1][0])  # 保留第二个
            print(len(sorted_names))
            # 规则 II：2nd <= 2 * 3rd, 3rd <= 2 * 4th, ...
            for i in range(2, len(sorted_names)):
                print(i)
                
                if sorted_names[i - 1][1] <= 2 * sorted_names[i][1]:
                    filtered_names.add(sorted_names[i][0])
                else:
                    break  # 一旦不满足规则，停止保留后续的名称
    elif len(sorted_names) == 1:
        filtered_names.add(sorted_names[0][0])
        
                
    print("filtered_names:", filtered_names)
    sorted_names_set = {name for name, _ in sorted_names}
    difference_set = sorted_names_set - filtered_names
    # 过滤数据
    filtered_data = {}
    for key, frame in data.items():
        caption = frame["image_caption"].lower()
        if any(name in caption for name in difference_set):
            pass
        else:
            filtered_data[key] = {"image_name": frame["image_name"], "image_caption": frame["image_caption"]}
    
    return filtered_data

def qwen_history_summary(descriptions,api_key,base_url):
    data = build_frame_json(descriptions)
    name_counts, _ = count_names(data)
    clean_data = filter_captions(data, name_counts)
    if not clean_data:
        clean_data = data

    datastr = "\n".join([f"{k}: {v['image_caption']}" for k, v in clean_data.items()])
    content_str = """
    The provided descriptions outline the content of each frame in the video. Your task is to summarize the event in the video based on these descriptions.Identify and analyze whether the same person or object is referred to by different names across frames, such as a 'boy,' 'man,' 'girl,' 'woman,' or 'person' due to possible inconsistencies or ambiguities in the frame descriptions. Do not use bold text, markdown, or bullet points.
    Attention: 1.Use only factual information provided in the frame descriptions. Do not infer, guess, or supplement missing details!
                2.Do not mention or refer to specific frame numbers!
                3.Keep the summary concise and clear. If multiple frames convey the same meaning, retain only the description that best expresses the event.You can also use original frame description!
                4.The most important:if a weapon such as a gun, knife or bat is mentioned, include only the description of the person holding or using the weapon, or use the original frame description that includes the weapon!
                5.The second most important:When multiple frames describe the same person or action, judge whether they refer to the same subject, and merge only if consistent. Do not treat different descriptions blindly as separate people!
                6.No line breaks.Don't use "\n". If many frames describe different things, identify the event that occurs most frequently and is most significant, and summarize that one only!
                7.Remove frames with incorrect or inconsistent descriptions based on the context of nearby frames!
                8.Ignore any frame with incorrect, unclear, or illogical descriptions. Use contextual judgment to discard them!
                9.Write in a coherent, natural sentence. Ensure all parts of the output are logically connected, not disjointed.If the events are too many, output the most important you think!
                10.Only describe the important and frequent event. Ignore all others. Be concise and write in a single coherent sentence.
    Here is the example:
    Event: ......
    """
    # 初始化messages列表
    completion = client.chat.completions.create(
        model="qwen-long",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'system', 'content': datastr},
            {'role': 'user', 'content': content_str}
        ],
        stream=True,
        stream_options={"include_usage": True}
    )
    full_content = ""
    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta.content:
            # 拼接输出内容
            full_content += chunk.choices[0].delta.content
            # print(chunk.model_dump())
    summary = re.sub(r'\n+', '\n', full_content)
    return summary

def compute_desc_hash(desc_list):
    joined = "\n".join(desc_list)
    return hashlib.md5(joined.encode('utf-8')).hexdigest()

############################## 实现逻辑 ##################################
import logging
from datetime import datetime

# 配置日志
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'worker_summary.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('worker_summary')

# 初始化 ZeroMQ 上下文
ctx = zmq.Context()

# 订阅描述
subscriber = ctx.socket(zmq.SUB)
subscriber.connect("tcp://localhost:5565")  # 从 worker_blip 接收描述
subscriber.setsockopt_string(zmq.SUBSCRIBE, '')
subscriber.setsockopt(zmq.RCVHWM, 1)  # 只保留最新的1条消息

# 发布摘要
publisher = ctx.socket(zmq.PUB)
publisher.bind("tcp://*:5563")  # 发布摘要到 worker_clip
publisher.setsockopt(zmq.SNDHWM, 1)  # 只保留最新的1条消息

# 全局变量
latest_descriptions = []
description_lock = threading.Lock()



# 添加帧跳过计数器
frame_skip_counter = 0
FRAME_SKIP_RATE = 2  # 每2帧处理1帧
LAST_PROCESSING_TIME = 0
MIN_PROCESSING_INTERVAL = 1.0  # 最小处理间隔（秒）

# API 配置
api_key = "your-api-key-here"  # 请替换为您的API密钥
base_url = "your-base-url-here"  # 请替换为您的API基础URL

client = OpenAI(
    api_key=api_key,  # 如果您没有配置环境变量，请在此处替换您的API-KEY
    base_url=base_url,  # 填写DashScope服务base_url
)
description_lock = threading.Lock()

def receive_descriptions():
    global latest_descriptions
    while True:
        try:
            # 接收描述数据
            data = subscriber.recv_json()
            logger.info(f"收到描述数据: {data.keys() if isinstance(data, dict) else 'invalid format'}")
            
            if isinstance(data, dict) and 'descriptions' in data:
                with description_lock:
                    latest_descriptions = data['descriptions']
                    logger.info(f"更新描述列表，当前长度: {len(latest_descriptions)}")
        except Exception as e:
            logger.error(f"接收描述时出错: {e}", exc_info=True)

# 启动接收线程
threading.Thread(target=receive_descriptions, daemon=True).start()

last_desc_hash = None

while True:
    # 接收描述
    try:
        with description_lock:
            if not latest_descriptions:
                logger.debug("没有新的描述数据，等待中...")
                time.sleep(0.1)
                continue
            desc = list(latest_descriptions)
        
        # 计算当前描述的哈希值
        current_hash = compute_desc_hash(desc)
        
        # 如果描述没变，跳过处理
        if current_hash == last_desc_hash:
            logger.debug("描述未变化，跳过处理")
            time.sleep(0.1)
            continue
        
        last_desc_hash = current_hash
        logger.info(f"处理新的描述数据，长度: {len(desc)}")
        
        # 生成摘要
        try:
            summary = qwen_history_summary(desc, api_key, base_url)
            if summary:
                logger.info(f"生成摘要: {summary[:100]}...")
                
                # 发布摘要
                publisher.send_json({
                    "summary": [summary],  # 保持与网页端一致的格式
                    "type": "summary",            # 添加类型标识
                    "timestamp": time.time()    # 添加时间戳
                })
                logger.info("摘要已发布")
                
                # 将摘要保存到日志文件
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True, parents=True)
                output_file = output_dir / "summary_output.txt"
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Summary:\n{summary}\n\n{'='*50}\n\n")
        except Exception as e:
            logger.error(f"生成摘要时出错: {e}", exc_info=True)
            
    except Exception as e:
        logger.error(f"处理描述时发生错误: {e}", exc_info=True)
        time.sleep(1)  # 出错时稍作等待