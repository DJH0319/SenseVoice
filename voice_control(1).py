import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import wave
import time
import re
import jieba
import os
from datetime import datetime
from opencc import OpenCC
from dataclasses import dataclass
from typing import Optional

os.environ["MODELSCOPE_CACHE"] = "D:/modelscope_cache"

import socket
import asyncio

@dataclass
class ACState:
    switch: Optional[bool] = None
    mode: Optional[str] = None
    temp_set: Optional[int] = 25
    temp_current: Optional[int] = 25
    level: Optional[str] = 2
    mode_lock: Optional[bool] = None
    temp_lock: Optional[bool] = None
    keyboard_lock: Optional[bool] = None

    def __repr__(self):
        fields = [
            f"{name}={getattr(self, name)!r}" for name in self.__dataclass_fields__]
        return f"StatusState({', '.join(fields)})"

    def print_state(self):
        print("StatusState:")
        for name in self.__dataclass_fields__:
            print(f"  {name}: {getattr(self, name)}")

    # str形式的字典转化成为bool形式


AC = ACState()


def devide(instruction):
    degree = 0
    if "窗簾" in instruction:
        device = "窗簾"
        for key in ["開", "關", "閂", "拉開", "拉下", "打開"]:
            if key in instruction:
                operation = key
                break
    if "窗紗" in instruction:
        device = "窗紗"
        for key in ["開", "關",  "拉開", "拉下"]:
            if key in instruction:
                operation = key
                break
    if "病床燈" in instruction:
        device = "病床燈"
        for key in ["開", "關", "熄", "亮", "暗"]:
            if key in instruction:
                operation = key
                break
    if "房間燈" in instruction:
        device = "房間燈"
        for key in ["開", "關", "閂",  "熄","亮", "暗"]:
            if key in instruction:
                operation = key
                break
    if "夜燈" in instruction:
        device = "夜燈"
        for key in ["開", "關",   "熄","亮", "暗"]:
            if key in instruction:
                operation = key
                break
    if "廁所燈" in instruction:
        device = "廁所燈"
        for key in ["開", "關", "閂",  "熄","亮", "暗"]:
            if key in instruction:
                operation = key
                break
    if "空調" in instruction:
        device = "空調"
        for key in ["開", "關", "凍啲", "熱啲", "開大啲", "開小啲",
                          "調高", "調低", "強", "弱"]:
            if key in instruction:
                operation = key
                break
    return operation, device,degree

def encoder(instruction, device_id=None):
    operation,device,degree = devide(instruction)
    print("encoder")
    device_id = "01"
    if device == "窗簾":
        print("窗簾")
        device_code = "13"
        operation_dict = {"開": "82", "關": "81", "閂": "00", "拉開": "00", "拉下": "00", "打開": "00"}
        operation_code = operation_dict[operation]
        key_dict = {"開": "31", "關": "33", "閂": "32", "拉開": "31", "拉下": "33", "打開": "31"}
        key_value = key_dict[operation]
        # Device type code + Function code + ID code + Data string + Checksum code
    elif device == "窗紗":
        print("窗紗")
        device_code = "13"
        operation_dict = {"開": "82", "關": "81", "拉開": "82", "拉下": "81"}
        operation_code = operation_dict[operation]
        key_dict = {"開": "34", "關": "36", "拉開": "34", "拉下": "36"}
        key_value = key_dict[operation]
        # Device type code + Function code + ID code + Data string + Checksum code
    elif device == "病床燈":
        print("病床燈")
        device_code = "11"
        operation_dict = {"開": "82", "關": "81", "熄": "00", "亮": "00", "暗": "00"}
        key_value = "11"
        operation_code = operation_dict[operation]
        # Device type code + Function code + ID code + Data string + Checksum code
    elif device == "房間燈":
        print("房間燈")
        device_code = "11"
        operation_dict = {"開": "82", "關": "81", "閂": "00", "熄": "00", "亮": "00", "暗": "00"}
        key_value = "12"
        operation_code = operation_dict[operation]
        # Device type code + Function code + ID code + Data string + Checksum code
    elif device == "夜燈":
        print("夜燈")
        device_code = "11"
        operation_dict = {"開": "82", "關": "81", "熄": "00", "亮": "00", "暗": "00"}
        key_value = "13"
        operation_code = operation_dict[operation]
        # Device type code + Function code + ID code + Data string + Checksum code
    elif device == "廁所燈":
        print("廁所燈")
        device_code = "11"
        operation_dict = {"開": "82", "關": "81", "閂": "00", "熄": "00", "亮": "00", "暗": "00"}
        key_value = "14"
        operation_code = operation_dict[operation]
        # Device type code + Function code + ID code + Data string + Checksum code
    elif device == "空調":
        device_id_hex = f"{device_id:02X}"
        command = ["01", "80", "00", device_id_hex]
        print("空調")
        device_code = "13"
        operation_dict = {"開": None, "關": None, "凍啲": '02', "熱啲": '01', "開大啲": None, "開小啲": None,
                          "調高": None, "調低": None, "強": None, "弱": None}

        operation_code = operation_dict[operation]
        device_id_hex = f"{device_id:02X}"

        command = ["01", "85", "00", f"{device_id:02X}"]
        if operation_code == "開":
            switch = True
            checksum = sum(int(byte, 16) for byte in command) & 0xFF
            command.append(f"{checksum:02X}")
            return " ".join(command)
        elif operation_code == "關":
            switch = False
            command = ["01", "81", "00", device_id_hex]
            checksum = sum(int(byte, 16) for byte in command) & 0xFF
            command.append(f"{checksum:02X}")
            return " ".join(command)
        # Only control the A/C mode (set to cold): 01 85 00 id 02 FF FF FF 86
        elif operation_code == "凍啲":
            AC.mode = '0'
            command.append(operation_dict[operation_code])
            command.append('FF')
            command.append('FF')
            command.append('FF')
            return " ".join(command)
        elif operation_code == "熱啲":
            AC.mode = '1'
            command.append(operation_dict[operation_code])
            command.append('FF')
            command.append('FF')
            command.append('FF')
            return " ".join(command)
        elif operation_code == "開大啲" or operation_code == "調高":
            AC.temp_set += 1
            command.append('FF')
            command.append('FF')
            command.append('FF')
            command.append(f"{AC.temp_set:02x}")
        elif operation_code == "開小啲" or operation_code == "調低":
            AC.temp_set -= 1
            command.append('FF')
            command.append('FF')
            command.append('FF')
            command.append(f"{AC.temp_set:02x}")
        elif operation_code == '強':
            if AC.level == 1 or AC.level == 2:
                AC.level += 1
            command.append('FF')
            command.append('FF')
            command.append(f"{AC.temp_set:02x}")
            command.append('FF')
        elif operation_code == '弱':
            if AC.level == 2 or AC.level == 3:
                AC.level -= 1
            command.append('FF')
            command.append('FF')
            command.append(f"{AC.temp_set:02x}")
            command.append('FF')

        checksum = sum(int(byte, 16) for byte in command) & 0xFF
        command.append(f"{checksum:02X}")
        return " ".join(command)

    # mode_map = {
    #     "關": "00", "熱啲": "01", "凍啲": "02"
    # }
    # fan_map = {"開小啲": "01", "開小啲": "02", "開大啲": "03", "auto": "04"}
    # return " ".join(command)

# print("This is AC")
# return

    else:
        print("not available device")
        return

    # Device type code + Function code + ID code + Data string + Checksum code
    command = [device_code]
    command.append(operation_code)
    command.append(device_id)
    command.append(key_value)
    checksum = sum(int(byte, 16) for byte in command) & 0xFF
    command.append(f"{checksum:02X}")
    return " ".join(command)


# 初始化OpenCC进行繁简转换
cc = OpenCC('s2t')


# 调试模式开关
DEBUG = False

# 粤语同音词库 - 扩展版本
cantonese_homophones = {
    # 操作命令同音词
    "開": ["開", "揩", "鎧"],
    "關": ["關", "觀", "官"],
    "熄": ["熄", "識", "式"],
    "亮": ["亮", "光", "諒", "量"],  # 添加"光"作为"亮"的同义词
    "暗": ["暗", "案", "岸"],
    "凍": ["凍", "棟", "動"],
    "熱": ["熱", "月", "日"],
    
    # 设备名称同音词
    "窗簾": ["窗簾", "床簾"],
    "窗紗": ["窗紗", "瘡痂"],
    "病床燈": ["病床燈", "病牀燈", "病牀登", "病床登"],  # 扩展病床燈的变体
    "房間燈": ["房間燈", "房間登"],
    "夜燈": ["夜燈", "腋燈"],
    "廁所燈": ["廁所燈", "屎所燈", "廁所登", "屎所登"],  # 扩展廁所燈的变体
    "空調": ["空調", "冷氣", "控調"]
}

# 特殊修饰词 - 扩展版本
instruction_set = {
    "设备类型": {
        "窗簾": {
            "操作命令": ["開", "關", "閂", "拉開", "拉下", "打開"],
            "示例指令": ["開窗簾", "關窗簾啦", "閂埋啲窗簾吖", "拉開窗簾", "拉下窗簾"],
            "参数支持": [],
            "homophones": cantonese_homophones.get("窗簾", [])
        },
        "窗紗": {
            "操作命令": ["開", "關", "拉開", "拉下"],
            "示例指令": ["開窗紗", "閂窗紗唔該", "拉開窗紗", "拉下窗紗"],
            "参数支持": [],
            "homophones": cantonese_homophones.get("窗紗", [])
        },
        "病床燈": {
            "操作命令": ["開", "關", "熄", "亮", "暗"],
            "示例指令": ["幫我開病床燈", "熄咗病床燈", "病床燈光啲", "病床燈暗啲"],
            "参数支持": ["亮度控制"],
            "homophones": cantonese_homophones.get("病床燈", [])
        },
        "房間燈": {
            "操作命令": ["開", "關", "閂", "熄", "亮", "暗"],
            "示例指令": ["開燈", "閂燈啊", "熄燈啦", "房間燈光啲", "房間燈暗啲"],
            "参数支持": ["亮度控制"],
            "homophones": cantonese_homophones.get("房間燈", [])
        },
        "夜燈": {
            "操作命令": ["開", "關", "熄", "亮", "暗"],
            "示例指令": ["開夜燈啦", "夜燈太光熄少少", "夜燈光啲", "夜燈暗啲"],
            "参数支持": ["亮度控制"],
            "homophones": cantonese_homophones.get("夜燈", [])
        },
        "廁所燈": {
            "操作命令": ["開", "關", "閂", "熄", "亮", "暗"],
            "示例指令": ["開廁所燈", "閂廁所燈好喇", "熄廁所燈吖", "廁所燈光啲"],
            "参数支持": [],
            "homophones": cantonese_homophones.get("廁所燈", [])
        },
        "空調": {
            "操作命令": ["開", "關", "凍啲", "熱啲", "開大啲", "開小啲", "調高", "調低", "強", "弱"],
            "示例指令": ["開空調", "關空調", "空調凍啲吖", "空調熱啲啦", "空調風開大啲", "空調風開細啲"],
            "参数支持": ["温度调节", "风力控制"],
            "homophones": cantonese_homophones.get("空調", [])
        }
    },
    "组合命令": {
        "示例": [
            "幫我閂窗簾開夜燈",
            "熄咗廁所燈開大啲冷氣",
            "閂燈開細啲冷氣"
        ],
        "支持设备": ["任意设备组合"],
        "语法规则": "操作命令 + 设备名称"
    },
    "特殊修饰词": {
        "少少": "轻微调整程度",
        "大啲": "增加强度",
        "細啲": "减少强度",
        "太光": "光线过强反馈",
        "好": "语气词(可忽略)",  # 添加"好"作为语气词
        "啦": "语气词(可忽略)",
        "吖": "语气词(可忽略)",
        "啊": "语气词(可忽略)",
        "嘅": "助词(可忽略)",
        "光啲": "亮度增加",  # 添加"光啲"作为特殊修饰词
        "暗啲": "亮度减少"
    }
}



# 唤醒词
wake_words = ["你好"]  # 替换为实际的唤醒词
model_dir = "iic/SenseVoiceSmall"
# 关键修改：如果上述方法仍有问题，可以尝试使用完整路径
# model_dir = "D:/modelscope_cache/hub/models/iic/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cuda:0")
m.eval()

# 确保目录存在
def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        if DEBUG:
            print(f"创建目录: {directory}")
        os.makedirs(directory)

# 全局变量用于存储可视化窗口
fig = None
ax = None
bar = None
window_closed = False

# 实时音频录制
def record_audio():
    global fig, ax, bar, window_closed
    
    if DEBUG:
        print("进入录音函数")
        
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print("* 开始监听唤醒词")
    
    frames = []
    wake_word_detected = False
    silence_start_time = None
    volume_threshold = 0.02  # 音量阈值，可根据实际情况调整
    
    # 音频目录
    audio_dir = "audio/test_WAV"
    ensure_dir_exists(audio_dir)
    
    # 检查窗口是否已关闭
    if fig is None or not plt.fignum_exists(fig.number):
        plt.ion()
        fig, ax = plt.subplots()
        bar = ax.bar(0, 0, width=0.5)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_ylabel('Volume')
        ax.set_title('Volume Level')
        
        def on_close(event):
            global window_closed
            if DEBUG:
                print("用户关闭了音量可视化窗口")
            window_closed = True
            
        fig.canvas.mpl_connect('close_event', on_close)
    
    try:
        while True:
            # 检查窗口是否已关闭
            if window_closed:
                if DEBUG:
                    print("可视化窗口已关闭，退出录音")
                break
                
            data = stream.read(CHUNK)
            
            # 计算音量
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.max(np.abs(audio_data)) / 32767  # 归一化音量
            
            # 更新图形
            bar[0].set_height(volume)
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            if volume > volume_threshold:
                if DEBUG:
                    print(f"检测到音量: {volume:.4f} (阈值: {volume_threshold})")
                frames.append(data)
                
                # 保存音频文件进行识别
                audio_path = os.path.join(audio_dir, "test_audio.wav")
                wf = wave.open(audio_path, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                text = recognize_audio(audio_path)
                text = cc.convert(text)  # 转换为繁体
                
                if DEBUG:
                    print(f"识别文本: {text}")
                
                for wake_word in wake_words:
                    if wake_word in text:
                        if DEBUG:
                            print(f"检测到唤醒词: {wake_word}")
                        wake_word_detected = True
                        print("我在，请讲")  # 提示用户可以说具体指令
                        frames = []  # 清空之前的音频数据
                        break
            else:
                if DEBUG and len(frames) > 0:
                    print(f"检测到静音: {volume:.1f} (阈值: {volume_threshold})")
            
            if wake_word_detected:
                if DEBUG:
                    print("开始监听用户指令")
                    
                silence_start_time = None
                
                while True:
                    # 检查窗口是否已关闭
                    if window_closed:
                        if DEBUG:
                            print("可视化窗口已关闭，退出录音")
                        break
                        
                    data = stream.read(CHUNK)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    volume = np.max(np.abs(audio_data)) / 32767  # 归一化音量
                    
                    # 更新图形
                    bar[0].set_height(volume)
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    
                    if volume > volume_threshold:
                        if DEBUG:
                            print(f"指令音量: {volume:.4f} (阈值: {volume_threshold})")
                        frames.append(data)
                        silence_start_time = None
                        
                        # 保存音频文件进行识别
                        audio_path = os.path.join(audio_dir, "test_command.wav")
                        wf = wave.open(audio_path, 'wb')
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(p.get_sample_size(FORMAT))
                        wf.setframerate(RATE)
                        wf.writeframes(b''.join(frames))
                        wf.close()
                        
                        command_text = recognize_audio(audio_path)
                        command_text = cc.convert(command_text)  # 转换为繁体
                        print(f"用户说的话: {command_text}")
                    else:
                        if silence_start_time is None:
                            silence_start_time = time.time()
                            if DEBUG:
                                print("开始检测静音时间")
                        else:
                            silence_duration = time.time() - silence_start_time
                            if DEBUG:
                                print(f"持续静音: {silence_duration:.1f}秒")
                            
                            if silence_duration > 2:  # 2秒静音
                                if DEBUG:
                                    print("检测到2秒静音，结束指令录音")
                                break
                    
                    # 如果窗口已关闭，退出循环
                    if window_closed:
                        break
                
                if DEBUG:
                    print("指令录音结束，保存音频文件")
                break
    
    except Exception as e:
        if DEBUG:
            print(f"录音过程中发生异常: {str(e)}")
        raise
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # 保存录制的音频，文件名使用时间戳
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        recorded_dir = "audio/recorded"  # 修正目录名
        ensure_dir_exists(recorded_dir)
        audio_path = os.path.join(recorded_dir, f"{timestamp}.wav")
        
        if DEBUG:
            print(f"保存音频文件: {audio_path}")
            
        wf = wave.open(audio_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        if DEBUG:
            print("录音函数退出")
            
    return audio_path, window_closed

# 语音识别
def recognize_audio(audio_path):
    if DEBUG:
        print(f"开始识别音频: {audio_path}")
        
    res = m.inference(
        data_in=audio_path,
        language="yue",  # 指定粤语识别
        use_itn=False,
        ban_emo_unk=False,
        **kwargs,
    )
    
    text = rich_transcription_postprocess(res[0][0]["text"])
    
    if DEBUG:
        print(f"识别结果: {text}")
        
    return text


# 语义理解和关键字匹配 - 改进版本
def parse_instruction(text):
    if DEBUG:
        print(f"开始解析指令: {text}")
        
    # 处理空字符串
    if not text.strip():
        return ["未匹配到指令"]
        
    # 移除语气词和干扰词
    clean_text = remove_modifiers(text)
    if DEBUG:
        print(f"处理后的文本: {clean_text}")
        
    words = jieba.lcut(clean_text)
    devices = instruction_set["设备类型"]
    final_instructions = []
    
    # 先尝试组合命令匹配
    combined_commands = parse_combined_command(clean_text)
    if combined_commands:
        final_instructions.extend(combined_commands)
        if DEBUG:
            print(f"匹配到组合指令: {', '.join(combined_commands)}")
        return final_instructions
    
    # 处理单个指令
    matched = False
    
    # 先查找设备名称
    for device_name in devices.keys():
        device_matched = False
        
        # 检查设备名称及其同音词
        device_terms = [device_name] + devices[device_name].get("homophones", [])
        for term in device_terms:
            if term in clean_text:
                device_matched = True
                break
                
        if not device_matched:
            continue
            
        # 查找匹配的操作命令
        for command in devices[device_name]["操作命令"]:
            # 检查命令及其同音词
            command_terms = [command] + get_homophones(command)
            for cmd_term in command_terms:
                if cmd_term in clean_text:
                    # 处理空调特殊情况
                    if device_name == "空調":
                        final_instructions.append(handle_aircon_command(clean_text, cmd_term))
                    else:
                        # 处理亮度调整命令
                        if "光啲" in clean_text or "暗啲" in clean_text:
                            if "光啲" in clean_text:
                                final_instructions.append(f"亮{device_name}")
                            else:
                                final_instructions.append(f"暗{device_name}")
                        else:
                            final_instructions.append(f"{command}{device_name}")
                    if DEBUG:
                        print(f"匹配到指令: {final_instructions[-1]}")
                    matched = True
                    break
                    
            if matched:
                break
                
        if matched:
            break
    
    if not matched:
        # 尝试直接匹配完整指令
        for device, info in devices.items():
            for command in info["操作命令"]:
                full_command = f"{command}{device}"
                if full_command in clean_text:
                    final_instructions.append(full_command)
                    if DEBUG:
                        print(f"直接匹配到完整指令: {full_command}")
                    return final_instructions
    
    if not final_instructions:
        final_instructions.append("未匹配到指令")
        if DEBUG:
            print("未匹配到任何指令")
            
    return final_instructions

# 移除语气词和干扰词
def remove_modifiers(text):
    modifiers = instruction_set["特殊修饰词"].keys()
    for modifier in modifiers:
        text = text.replace(modifier, "")
    return text

# 获取词语的同音词列表
def get_homophones(word):
    return cantonese_homophones.get(word, [])

# 处理空调特殊指令
def handle_aircon_command(text, command):
    # 检查是否有风速控制关键词
    if "風" in text or "風量" in text or "風速" in text:
        if "大" in text or "強" in text:
            return "空調風開大啲"
        elif "小" in text or "細" in text or "弱" in text:
            return "空調風開細啲"
    
    # 检查是否有温度调节
    if "凍" in text or "冷" in text or "低" in text:
        # 检查是否有具体度数
        degree_match = re.search(r'(\d+)\s*度', text)
        if degree_match:
            degree = degree_match.group(1)
            return f"空調調低{degree}度"
        else:
            return "空調調低啲"
    
    if "熱" in text or "暖" in text or "高" in text:
        # 检查是否有具体度数
        degree_match = re.search(r'(\d+)\s*度', text)
        if degree_match:
            degree = degree_match.group(1)
            return f"空調調高{degree}度"
        else:
            return "空調調高啲"
    
    # 默认返回原始命令
    return f"{command}空調"

# 查找操作命令
def find_command(word, devices):
    for device, info in devices.items():
        for command in info["操作命令"]:
            if command in word:
                return command
    return None

# 查找设备名称
def find_device(text, command, devices):
    # 查找命令后面的设备名称
    command_index = text.find(command)
    if command_index >= 0:
        remaining_text = text[command_index + len(command):]
        for device in devices.keys():
            if device in remaining_text:
                return device
    
    # 如果没找到，尝试全文查找
    for device in devices.keys():
        if device in text:
            return device
            
    return None

# 解析组合命令
def parse_combined_command(text):
    devices = instruction_set["设备类型"]
    commands = []
    
    # 尝试多次匹配命令-设备对
    remaining_text = text
    while True:
        # 查找操作命令
        command = None
        for dev, info in devices.items():
            for cmd in info["操作命令"]:
                if cmd in remaining_text:
                    command = cmd
                    break
            if command:
                break
                
        if not command:
            break
            
        # 查找对应的设备
        device = None
        cmd_index = remaining_text.find(command)
        cmd_end = cmd_index + len(command)
        remaining_after_cmd = remaining_text[cmd_end:]
        
        for dev in devices.keys():
            if dev in remaining_after_cmd:
                device = dev
                break
                
        if device:
            commands.append(f"{command}{device}")
            # 移除已匹配的部分
            dev_index = remaining_after_cmd.find(device)
            remaining_text = remaining_after_cmd[dev_index + len(device):]
        else:
            # 没找到匹配的设备，退出循环
            break
    
    return commands if len(commands) > 1 else []  # 只有多个命令才算组合命令

@dataclass
class StatusState:
    switch:        Optional[bool] = None
    mode:          Optional[str]  = None
    temp_set:      Optional[int]  = None
    temp_current:  Optional[int]  = None
    level:         Optional[str]  = None
    mode_lock:     Optional[bool] = None
    temp_lock:     Optional[bool] = None
    keyboard_lock: Optional[bool] = None

    def __repr__(self):
        fields = [f"{name}={getattr(self, name)!r}" for name in self.__dataclass_fields__]
        return f"StatusState({', '.join(fields)})"

    def print_state(self):
        print("StatusState:")
        for name in self.__dataclass_fields__:
            print(f"  {name}: {getattr(self, name)}")

    def update(self, decoded: dict, class_id: int):
        # top-level fields
        self.switch       = decoded["Mode"] != "Off"
        self.mode         = decoded["Mode"].lower()
        self.temp_set     = decoded["Set Temperature"]
        self.temp_current = decoded["Current Temperature"]
        self.level        = decoded["Fan Speed"].lower()

        # nested locks
        locks = decoded["Lock Status"]
        self.keyboard_lock = locks["Keyboard Lock"]
        self.temp_lock     = locks["Temperature Lock"]
        self.mode_lock     = locks["Mode Lock"]
        
        mode_mapping = {"cool": 0, "heat": 1}
        level_mapping = {"low": 0, "middle": 1, "high": 2, "auto": 3}
        
        # Convert mode and level to numeric values
        current_mode = mode_mapping.get(self.mode, -1)  # Default to -1 if value not found
        current_level = level_mapping.get(self.level, -1)  # Default to -1 if value not found

        
        attrs = [
                    (f"Switch_{class_id}",       self.switch ),
                    (f"Mode_{class_id}",         current_mode),
                    (f"Temp_set_{class_id}",     self.temp_set),
                    (f"Level_{class_id}",        current_level),
                    (f"Mode_lock_{class_id}",    self.mode_lock),
                    (f"Temp_lock_{class_id}",    self.temp_lock),
                    (f"Keyboard_lock_{class_id}", self.keyboard_lock)
                ]
        return attrs

@dataclass
class CmdState:
    switch:        Optional[bool] = None
    mode:          Optional[str]  = None
    temp_set:      Optional[int]  = None
    level:         Optional[str]  = None
    mode_lock:     Optional[bool] = None
    temp_lock:     Optional[bool] = None
    keyboard_lock: Optional[bool] = None

    def __repr__(self):
        fields = [f"{name}={getattr(self, name)!r}" for name in self.__dataclass_fields__]
        return f"CmdState({', '.join(fields)})"

    def print_state(self):
        print("CmdState:")
        for name in self.__dataclass_fields__:
            print(f"  {name}: {getattr(self, name)}")

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.__dataclass_fields__:
                setattr(self, key, value)

@dataclass
class Thermostat:
    ip: str
    device_id: int
    class_id: int #from 1 to n
    conn: Optional[socket.socket] = None  # <-- NEW

    temp_current: StatusState = field(default_factory=StatusState)
    temp_cmd: CmdState = field(default_factory=CmdState)

    def __post_init__(self):
        # Initialize all status and command values to None (BACnet untouched)
        self.temp_current = StatusState()  # All fields default to None
        self.temp_cmd = CmdState()         # All fields default to None

    # Method1: Print out status
    def print_status(self):
        print("=== Thermostat Status ===")
        self.temp_current.print_state()

    # Method2: Print out command values
    def print_cmd(self):
        print("=== Thermostat Command ===")
        self.temp_cmd.print_state()

    # Method3: Assign status value
    def set_status(self, status_dict: dict):
        print("Updating status...")
        self.temp_current.update(**status_dict)
        print("Updated status")

    # Method4: Assign command value
    def set_cmd(self, cmd_dict: dict):
        print("Updating command...")
        self.temp_cmd.update(**cmd_dict)
        print("Updated command")
    
    # Method5: Decode Thermostat Respons
    def decode_thermostat_response(self,response_bytes: bytes):
        if len(response_bytes) < 11:
            return "Invalid response format"

        response_hex = response_bytes.hex().upper()
        response_list = [response_hex[i:i+2] for i in range(0, len(response_hex), 2)]

        device_id = response_list[3]
        mode = response_list[4]
        lock_status_raw = int(response_list[5], 16)
        fan_speed = response_list[6]
        set_temp = int(response_list[7], 16)
        current_temp = int(response_list[8], 16)
        valve_status = response_list[9]

        mode_map = {
            "00": "Off", "01": "heat", "02": "cool", "03": "Auto",
            "04": "Floor Heating", "05": "Rapid Heat", "06": "Ventilation"
        }
        fan_map = {"01": "low", "02": "medium", "03": "high", "04": "uto"}
        valve_map = {"00": "Closed", "01": "Open", "10": "Stopped"}

        keyboard_lock = bool(lock_status_raw & 0b00000001)
        temp_lock = bool(lock_status_raw & 0b00000010)
        mode_lock = bool(lock_status_raw & 0b00000100)

        lock_status_decoded = {
            "Keyboard Lock": keyboard_lock,
            "Temperature Lock": temp_lock,
            "Mode Lock": mode_lock
        }

        return {
            "Device ID": device_id,
            "Mode": mode_map.get(mode, "Unknown"),
            "Lock Status": lock_status_decoded,
            "Fan Speed": fan_map.get(fan_speed, "Unknown"),
            "Set Temperature": set_temp,
            "Current Temperature": current_temp,
            "Valve Status": valve_map.get(valve_status, "Unknown")
        }

    # Method6: Encode Command to Thermostat
    def create_thermostat_command(self, device_id=None, switch=None, current_mode1=None, mode=None, keyboard_lock=None, temp_lock=None, mode_lock=None, fan_speed=None, temperature=None):
        device_id_hex = f"{device_id:02X}"

        if not switch:
            # Switch is OFF → return 01 80 00 ID CHECKSUM
            command = ["01", "80", "00", device_id_hex]
            checksum = sum(int(byte, 16) for byte in command) & 0xFF
            command.append(f"{checksum:02X}")
            return " ".join(command)
        
        if current_mode1 == "off":
            # Mode is "off" → return 01 81 00 ID CHECKSUM
            command = ["01", "81", "00", device_id_hex]
            checksum = sum(int(byte, 16) for byte in command) & 0xFF
            command.append(f"{checksum:02X}")
            return " ".join(command)

        
        command = ["01", "85", "00", f"{device_id:02X}"]

        mode_map = {
            "off": "00", "heat": "01", "cool": "02", "auto": "03",
            "floor_heating": "04", "rapid_heat": "05", "ventilation": "06"
        }
        fan_map = {"low": "01", "medium": "02", "high": "03", "auto": "04"}

        command.append(mode_map.get(mode, "FF"))

        lock_byte = 0
        if keyboard_lock:
            lock_byte |= 0b00000001
        if temp_lock:
            lock_byte |= 0b00000010
        if mode_lock:
            lock_byte |= 0b00000100
        command.append(f"{lock_byte:02X}")

        command.append(fan_map.get(fan_speed, "FF"))
        command.append(f"{temperature:02X}" if temperature is not None else "FF")

        checksum = sum(int(byte, 16) for byte in command) & 0xFF
        command.append(f"{checksum:02X}")

        return " ".join(command)

    # Method7: Handling Initial Connection & Initial Thermostat Status
    def initial_status(self):
        print("Handling Connection with thermostat:", self.ip)
        buffer = b''
        while len(buffer) < 20:
            data = self.conn.recv(1024)
            if not data:
                continue
            print("Initial handshake data skipped:", data.hex())
            buffer += data
        print("Initial handshake data skipped:", buffer.hex())

        #init_command = bytes.fromhex("01 45 00 01 47")
        init_command = bytes([
            0x01,
            0x45,
            0x00,
            self.device_id,
            (0x01 + 0x45 + 0x00 + self.device_id) & 0xFF
        ])
        self.conn.sendall(init_command)
        print("Sent initialization command to conn1:", init_command.hex())
        # Step 3: Wait for "01 85 00" response
        matching_message = None
        get_msg = True
        temp_buffer = b''
        while get_msg:
            data = self.conn.recv(1024)
            print("data received:", data.hex(), "length: ", len(data))
            if not data:
                print("Connection closed while waiting for response.")
                continue
            temp_buffer = data

            while len(temp_buffer) >= 11:
                chunk = temp_buffer[:11]
                temp_buffer = temp_buffer[11:]

                if chunk.startswith(b'\x01\xc5\x00'):
                    get_msg = False
                    matching_message = chunk
                    print("Received target message:", matching_message.hex())

                    decoded = self.decode_thermostat_response(matching_message)
                    print("Decoded thermostat data:", decoded)

                    # Update current variables
                    self.temp_current.update(decoded, self.class_id)
                    print(self.temp_current)
                    break
        print("Finished Inialization of Thermostat Status")

    # Method8: Update Thermostat Status
    def update_status(self):
        print("Handling Connection with thermostat:", self.ip)
        
        data = self.conn.recv(1024)
        #init_command = bytes.fromhex("01 45 00 01 47")
        init_command = bytes([
            0x01,
            0x45,
            0x00,
            self.device_id,
            (0x01 + 0x45 + 0x00 + self.device_id) & 0xFF
        ])
        self.conn.sendall(init_command)
        print("Sent initialization command to conn1:", init_command.hex())
        # Step 3: Wait for "01 85 00" response
        matching_message = None
        get_msg = True
        temp_buffer = b''
        while get_msg:
            data = self.conn.recv(1024)
            print("data received:", data.hex(), "length: ", len(data))
            if not data:
                print("Connection closed while waiting for response.")
                continue
            temp_buffer = data

            while len(temp_buffer) >= 11:
                chunk = temp_buffer[:11]
                temp_buffer = temp_buffer[11:]

                if chunk.startswith(b'\x01\xc5\x00'):
                    get_msg = False
                    matching_message = chunk
                    print("Received target message:", matching_message.hex())

                    decoded = self.decode_thermostat_response(matching_message)
                    print("Decoded thermostat data:", decoded)

                    # Update current variables
                    attrs = self.temp_current.update(decoded,self.class_id)
                    print(self.temp_current)
                    
                    return attrs
                    #break
        print("Finished Inialization of Thermostat Status")

    # Method9: Update and Send Command to Thermostat    
    def send_update_command(self, *, switch, mode, temp_set, level, keyboard_lock, temp_lock, mode_lock):
        mode_mapping_back = {0: "cool", 1:"heat"}
        level_mapping_back = {0: "low", 1: "middle", 2: "high", 3: "auto"}
        boolean_mapping_back = {"inactive": False, "active": True}
        cmd_mode = mode_mapping_back.get(int(mode), -1)  # Default to -1 if value not found
        cmd_level = level_mapping_back.get(int(level), -1)  # Default to -1 if value not found
        cmd_switch = boolean_mapping_back.get(str(switch), True)  # Default to -1 if value not found
        cmd_mode_lock = boolean_mapping_back.get(str(mode_lock), True)  # Default to -1 if value not found
        cmd_temp_lock = boolean_mapping_back.get(str(temp_lock), True)  # Default to -1 if value not found
        cmd_keyboard_lock = boolean_mapping_back.get(str(keyboard_lock), True)  # Default to -1 if value not found
        cmd_temp_set = int(temp_set)
        
        print("thermostat command value after value mapping")
        print(f"get switch: '{cmd_switch}'", type(cmd_switch))
        print(f"get mode: '{cmd_mode}'", type(cmd_mode))
        print(f"get temp_set: '{cmd_temp_set}'", type(cmd_temp_set))
        print(f"get level: '{cmd_level}'",  type(cmd_level))
        print(f"get cmd_mode_lock: '{cmd_mode_lock}'",  type(cmd_mode_lock))
        print(f"get cmd_temp_lock: '{cmd_temp_lock}'",  type(cmd_temp_lock))
        print(f"get cmd_keyboard_lock: '{cmd_keyboard_lock}'",  type(cmd_keyboard_lock))
        
        print("Generate Command for Thermostat")
        send_mode = "off" if not cmd_switch else cmd_mode
        thermo_cmd_str = self.create_thermostat_command(
            device_id=int("01", 16),
            switch = cmd_switch,
            current_mode1=self.temp_current.mode,
            mode=send_mode,
            keyboard_lock=cmd_keyboard_lock,
            temp_lock=cmd_temp_lock,
            mode_lock=cmd_mode_lock,
            fan_speed=cmd_level,
            temperature=cmd_temp_set
        )
        self.conn.sendall(bytes.fromhex(thermo_cmd_str))
        print("Sent command to 1st thermostat:", thermo_cmd_str)
        print("\nCommand send to Thermostat:")
        print(f"  cmd_switch = {cmd_switch}")
        print(f"  cmd_mode = {cmd_mode}")
        print(f"  cmd_temp_set = {cmd_temp_set}")
        print(f"  cmd_level = {cmd_level}")
        print(f"  cmd_mode_lock = {cmd_mode_lock}")
        print(f"  cmd_temp_lock = {cmd_temp_lock}")
        print(f"  cmd_keyboard_lock = {cmd_keyboard_lock}")
        
        self.temp_cmd.update(
            switch=cmd_switch,
            mode=cmd_mode,
            temp_set=cmd_temp_set,
            level=cmd_level,
            keyboard_lock=cmd_keyboard_lock,
            temp_lock=cmd_temp_lock,
            mode_lock=cmd_mode_lock
        )


def main():
    thermostat_list = [
        Thermostat(ip="192.168.12.206", device_id=1, class_id=1)
    ]
    therm = thermostat_list[0]
    number_ther = 1

    ### Please Write Program for Easy Scalling of Thermostat List ###

    """ V3 TCP Connection """

    # 1. Host PC information
    HOST = '192.168.12.195'
    PORT = 8082

    # 2. TCP Socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(100)  # Adjust max queue size for more devices
    print(f"Listening for connections on {HOST}:{PORT}...")

    # 3. Thermostat dictionary for fast IP-to-thermostat lookup & Connection
    thermostat_dict = {t.ip: t for t in thermostat_list}
    ALLOWED_IPS = set(thermostat_dict.keys())  # Faster lookup than list

     # 4. TCP Connection with All Target Thermostat 
    cnt = 0
    while cnt < number_ther: #continue to finding new connection untill all target thermostat are connected
        conn, addr = server_socket.accept()
        client_ip = addr[0]
        if client_ip in ALLOWED_IPS:
            print(f"Accepted connection from allowed device: {client_ip}")
            cnt+=1
            thermostat = thermostat_dict[client_ip]
            thermostat.conn = conn
            print(f"Stored connection in Thermostat {thermostat.device_id} ({client_ip})")
        else:
            print(f"Ignored unauthorized connection from: {client_ip}")
            conn.close()
    
    # 5. Get the thermostat status
    for therm in thermostat_list:
        therm.initial_status()


    """ V6 BACnet Command Getting """
    ###Initial Cmd to thermostat
    # 7. Get Command From DDC
    print(f"for Commnad of thermostat Class ID: 1")
    #1) Read all cmd values into a dict
    attrs = [
        ("switch",    f"Switch_1"),
        ("mode",      f"Mode_1"),
        ("temp_set",  f"Temp_set_1"),
        ("level",     f"Level_1"),
        ("mode_lock", f"Mode_lock_1"),
        ("temp_lock", f"Temp_lock_1"),
        ("keyboard_lock", f"Keyboard_lock_1")
    ]

    cmds = {}
    for key, prop in attrs:
        val = "value"
        print(f"get {key}: ", val)
        cmds[key] = val

    # 8. Send Command to Thermostat
    therm = thermostat_list[0]  # Assuming single thermostat for simplicity
    therm.send_update_command(**cmds)
    attrs2 = therm.update_status()
    
    print("send current status to Insight Status Point")
    # 8. Get Current Status from Thermotstat

    print("==== 语音控制系统启动 ====")
    
    try:
        while True:
            if DEBUG:
                print("\n==== 等待唤醒词 ====")
                
            # 修改这里：接收两个返回值
            audio_path, window_closed = record_audio()
            
            # 如果窗口被用户关闭，退出主循环
            if window_closed:
                print("用户关闭了可视化窗口，程序退出")
                break
                
            # 只传递音频路径给识别函数
            text = recognize_audio(audio_path)
            text = cc.convert(text)  # 转换为繁体
            instructions = parse_instruction(text)
            
            print(f"\n原语音文本文字: {text}")
            print(f"匹配实际指令: {', '.join(instructions)}")
            
            while "未匹配到指令" in instructions:
                print("抱歉，我没听清，请再说一次")
                # 修改这里：接收两个返回值
                audio_path, window_closed = record_audio()
                
                # 如果窗口被用户关闭，退出主循环
                if window_closed:
                    print("用户关闭了可视化窗口，程序退出")
                    return
                    
                # 只传递音频路径给识别函数
                text = recognize_audio(audio_path)
                text = cc.convert(text)  # 转换为繁体
                instructions = parse_instruction(text)
                print(f"\n原语音文本文字: {text}")
                print(f"匹配实际指令: {', '.join(instructions)}")
            
            for instruction in instructions:
                if instruction != "未匹配到指令":
                    print(f"好的，现在为您{instruction}")
                    print(encoder(instruction))

            if DEBUG:
                print("\n==== 指令处理完成，回到等待唤醒词状态 ====")
                
    except KeyboardInterrupt:
        print("\n用户手动终止程序")
    except Exception as e:
        print(f"程序发生错误: {str(e)}")
    finally:
        # 只在程序真正退出时关闭窗口
        if fig is not None and plt.fignum_exists(fig.number):
            plt.ioff()
            plt.close()
        print("==== 语音控制系统已停止 ====")

if __name__ == "__main__":
    main()