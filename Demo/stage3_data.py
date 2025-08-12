import json
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

import cv2
from transformers import LlavaForConditionalGeneration
from typing import Any
import re
import os

@dataclass
class QaClipOutput:
    a_input_ids: torch.Tensor
    q_input_ids: torch.Tensor
    pixel_values: torch.Tensor
    
class LlavaDatasetStage3(Dataset):
    def __init__(self, dataset_dir: str, data_type: str, question_text: str = None):
        super().__init__()
        # 设置默认问题文本
        self.default_question = "Focus on the historical texts and image frames. Identify any abnormalities in the following content and provide a reason if any are found."
        self.question_text = question_text if question_text is not None else self.default_question
        self.data_dict_list = self.build_dataset(dataset_dir,data_type)
    
    def build_dataset(self, dataset_dir, data_type) -> List:
        data_root_dir = Path(dataset_dir)
        sub_dirs = [data_root_dir.joinpath(sub_dir) for sub_dir in os.listdir(data_root_dir) if data_type in sub_dir]
        
        # 构建三个列表
        ground_truth_json_files = [
            sub_dir.joinpath("Description").joinpath(json_file) for sub_dir in sub_dirs 
            for json_file in os.listdir(Path(sub_dir).joinpath("Description")) 
            if json_file.endswith(".json")
        ]
        
        history_summary_json_files = [
            sub_dir.joinpath("Json").joinpath(json_file) for sub_dir in sub_dirs 
            for json_file in os.listdir(Path(sub_dir).joinpath("Json")) 
            if json_file.endswith(".json")
        ]
        
        clip_folder_list = [
            sub_dir.joinpath("Clips").joinpath(clip_folder) for sub_dir in sub_dirs 
            for clip_folder in os.listdir(Path(sub_dir).joinpath("Clips")) 
        ]

        # 建立快速匹配索引
        gt_dict = {p.stem: p for p in ground_truth_json_files}
        hs_dict = {p.stem: p for p in history_summary_json_files}
        clip_dict = {p.name: p for p in clip_folder_list}
        
        dataset = []
        
        # 遍历可以匹配到的事件
        for key in set(gt_dict.keys()) & set(hs_dict.keys()) & set(clip_dict.keys()):
            gt_path = gt_dict[key]
            hs_path = hs_dict[key]
            clip_path = clip_dict[key]
            
            # 读取description
            with open(gt_path, "r") as f:
                gt_data = json.load(f)
            descriptions = [gt_data[f"Responses_and_Feedbacks{i}"] for i in range(3) if f"Responses_and_Feedbacks{i}" in gt_data]
            # 读取history summary
            with open(hs_path, "r") as f:
                hs_data = json.load(f)
            summaries = [hs_data[f"summary{i}"].partition(":")[2].strip() for i in range(3) if f"summary{i}" in hs_data] # list
            splits = [hs_data[f"split{i}"] for i in range(3) if f"split{i}" in hs_data] # list
            
            # 判断数据质量并舍弃
            if len(descriptions) != len(summaries):
                continue
            
            for index,value in enumerate(splits):
                dataset.append(
                    {
                        "ground_truth": descriptions[index],
                        "clip_path": clip_path,
                        "summary": summaries[index],
                        "split_num": value,
                        "human_input": "Focus on the history texts and image frames, determine whether there is any abnormality in the following contents, and if so, give a reason."
                    }
                )
        return dataset
    
    def __len__(self) -> int:
        return len(self.data_dict_list)
    
    def __getitem__(self, index) -> Tuple[str, str, Path, list]:
        cur_sample = self.data_dict_list[index]
        human_text_input = self.question_text
        gpt_output = cur_sample["ground_truth"][0]["analyze"]
        clip_path = cur_sample["clip_path"]
        summary = cur_sample["summary"]
        split_num = cur_sample["split_num"]
        return human_text_input, gpt_output, clip_path, summary, split_num  
    

def natural_sort_key(filename):
    # 提取 filename 中的数字部分用于排序
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', filename)]

def sample_clips(clip_path: Path, num_frames: int, split_num: int):
    images_path = [clip_path.joinpath(image_file) for image_file in sorted(os.listdir(clip_path),key=natural_sort_key) if image_file.endswith(".jpg")]
    total_frames = len(images_path[split_num:])
    actual_frames = min(total_frames, num_frames)
    if actual_frames != 0:
        # print(f"actual_frames:{actual_frames}")
        interval = total_frames // actual_frames
        sampled_frames = []
        for i in range(actual_frames):
            frame_index = i * interval
            raw_image = Image.open(images_path[split_num + frame_index])
            sampled_frames.append(raw_image) 
    else:
        warning_msg = (
            f"[⚠️ Warning] No usable frames in clip: {clip_path}, "
            f"total_frames={total_frames}, split_num={split_num}\n"
        )
        print(warning_msg)
        with open("/home/bit118/ltx/MA_CBP_Demo/logs/stage3_data_log.txt", "a", encoding="utf-8") as f:
            f.write(warning_msg)

    return  sampled_frames, len(sampled_frames)

# human_text_input, gpt_output, clip_path, summary, split_num  
def build_qaclip(processor: AutoProcessor, q_text: str, a_text: str, clip_path: Path, summary: str, split_num: int, num_frames: int):
    frame_list, frame_num = sample_clips(clip_path, num_frames, split_num)
    additional_descriptions = "<history>" + summary + "</history>" + "<clip>" + "<image><sep>"*(frame_num - 1) + "<image>" + "</clip>"
    q_text = additional_descriptions + q_text
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": q_text},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(frame_list, prompt, return_tensors="pt")
    a_input_ids = processor.tokenizer(
        a_text, 
        return_tensors="pt",
        padding = "longest",
        truncation = True,
    )['input_ids']
    
    return QaClipOutput(
        q_input_ids = inputs['input_ids'],
        a_input_ids = a_input_ids,
        pixel_values = inputs['pixel_values']
    )
    
class TrainLlavaModelCollator:
    def __init__(self, processor: AutoProcessor, IGNORE_INDEX: int, frame_scale: int, num_frames: int) -> None:
        self.processor = processor
        self.processor.patch_size = 14
        self.ignore_index = IGNORE_INDEX
        self.frame_scale = frame_scale
        self.num_frames = num_frames
    def convert_one_piece(self,
                          q_input_ids: torch.Tensor,
                          a_input_ids: torch.Tensor) -> Tuple:
        intput_ids = torch.concat([
            q_input_ids,
            a_input_ids,
            torch.tensor(data=self.processor.tokenizer.eos_token_id).reshape(1, -1)
        ], axis=1)
        
        labels = torch.concat([
            torch.full_like(q_input_ids, fill_value=self.ignore_index),
            a_input_ids,
            torch.tensor(data=self.processor.tokenizer.eos_token_id).reshape(1, -1)
        ], axis=1)
        
        return intput_ids, labels
    
    def __call__(self, features: list) -> None:
        input_ids_list = []
        labels_list = []
        pixel_values_list = []
        max_input_len_list = []
        max_pixel_values_frames_list = []
        attention_mask_list = []
        
        for feature in features:
            # build_qaclip(processor: AutoProcessor, q_text: str, a_text: str, clip_path: Path, summary: str, split_num: int, num_frames: int)
            qaclip_output = build_qaclip(processor=self.processor,q_text = feature[0], a_text=feature[1], clip_path=feature[2], 
                                         summary = feature[3], split_num = feature[4], num_frames = self.num_frames)
            temp_input_ids, temp_labels = self.convert_one_piece(
                q_input_ids = qaclip_output.q_input_ids,
                a_input_ids = qaclip_output.a_input_ids    
            )
            
            max_input_len_list.append(temp_input_ids.shape[1])
            input_ids_list.append(temp_input_ids)
            labels_list.append(temp_labels)
            pixel_values_list.append(qaclip_output.pixel_values)
            max_pixel_values_frames_list.append(qaclip_output.pixel_values.shape[0])
            
        max_input_len = max(max_input_len_list)
        final_input_ids = torch.concat([
            torch.concat([
                torch.full(size=(1, max_input_len - max_input_len_list[index]),fill_value= self.processor.tokenizer.pad_token_id),
                value
            ], dim=1)
            for index, value in enumerate(input_ids_list)
        ], dim=0)
        
        final_labels = torch.concat([
            torch.concat([
                torch.full(size=(1, max_input_len - max_input_len_list[index]),fill_value= self.ignore_index),
                value
            ], dim=1)
            for index, value in enumerate(labels_list)
        ], dim=0)
        
        max_pixel_values_frame_nums = max(max_pixel_values_frames_list)
        final_pixel_values = torch.concat([
            torch.concat([
                torch.full(size=(1, max_pixel_values_frame_nums - max_pixel_values_frames_list[index], 3, self.frame_scale, self.frame_scale), 
                           fill_value = 0),
                value.unsqueeze(0)
            ], dim=1)
            for index, value in enumerate(pixel_values_list)
        ], dim=0)
        
        attention_mask_list = torch.ones_like(final_input_ids)
        attention_mask_list[final_input_ids == self.processor.tokenizer.pad_token_id] = 0
        
        return{
            "input_ids": final_input_ids, 
            "labels": final_labels,
            "pixel_values": final_pixel_values,
            "attention_mask": attention_mask_list
        } 
        
class LlavaForConditionalGenerationClip(LlavaForConditionalGeneration):
    def __init__(self,config):
        super().__init__(config)
    
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer:Union[int, List[int]],
        vision_feature_select_strategy: str,
        **kwargs,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)` or '(batch_size, num_frames, channels, height, width)')
               The tensors corresponding to the input images.
            vision_feature_layer (`Union[int, List[int]]`):
                The index of the layer to select the vision feature. If multiple indices are provided,
                the vision feature of the corresponding indices will be concatenated to form the
                vision features.
            vision_feature_select_strategy (`str`):
                The feature selection strategy used to select the vision feature from the vision backbone.
                Can be one of `"default"` or `"full"`
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        
        """
        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")
        
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if pixel_values.dim() == 5:
            batch_size, num_frames, channels, height, width = pixel_values.shape
            pixel_values = pixel_values.reshape(batch_size * num_frames, channels, height, width)
            
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True, **kwargs)
        # If we have one vision feature layer, return the corresponding hidden states,
        # otherwise, select the hidden states of each feature layer and concatenate them
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
        else:
            hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            # For default; crop CLS from each hidden state in the hidden state pool
            if vision_feature_select_strategy == "default":
                hs_pool = [hs[:, 1:] for hs in hs_pool]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features 

if __name__ == "__main__":
    data_dir = "/home/bit118/data/datasetDir/MA_CBP/dataset"
    llavadataset = LlavaDatasetStage3(data_dir,"train")
    print(len(llavadataset))
    print(llavadataset[100])