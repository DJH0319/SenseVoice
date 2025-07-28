import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import wave
import time
import re
import jieba
import os
import socket
import noisereduce as nr  # 用于降噪
import webrtcvad  # 用于精准VAD
import collections  # 用于音频缓冲区
from datetime import datetime
from opencc import OpenCC
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from instruction_sending import Thermostat, CmdState, StatusState  


# 关键修改：设置模型缓存路径到不含中文和特殊字符的目录
# 请确保该路径存在或有权限创建
os.environ["MODELSCOPE_CACHE"] = "D:/modelscope_cache"

# 初始化OpenCC进行繁简转换
cc = OpenCC('s2t')

# 调试模式开关
DEBUG = False

# 降噪和VAD参数
VAD_AGGRESSIVENESS = 2  # VAD灵敏度 1-3，3最灵敏
NOISE_REDUCE_PROP = 0.7  # 降噪强度 0-1
VAD_WINDOW_DURATION = 20  # VAD窗口时长(ms)，必须是10, 20, 30
SILENCE_THRESHOLD = 2.5  # 静音阈值(秒)，超过此时长认为语音结束

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
    "窗簾": ["窗簾", "床簾", "窗"],
    "窗紗": ["窗紗", "瘡痂"],
    "病床燈": ["病床燈", "病牀燈", "病牀登", "病床登"],  # 扩展病床燈的变体
    "房間燈": ["房間燈", "房間登"],
    "夜燈": ["夜燈", "腋燈"],
    "廁所燈": ["廁所燈", "屎所燈", "廁所登", "屎所登"],  # 扩展廁所燈的变体
    "空調": ["空調", "冷氣", "控調"],
    "左窗簾": ["左窗簾", "左床簾", "左窗廉"],
    "右窗簾": ["右窗簾", "右床簾", "右窗廉"]
}

# 特殊修饰词 - 扩展版本
instruction_set = {
    "设备类型": {
        "窗簾": {
            "操作命令": ["開", "關", "閂", "拉開", "拉下", "打開", "拉"],
            "示例指令": ["開窗簾", "關窗簾啦", "閂埋啲窗簾吖", "拉開窗簾", "拉下窗簾", "拉窗簾"],
            "参数支持": [],
            "homophones": cantonese_homophones.get("窗簾", [])
        },
        "左窗簾": {
            "操作命令": ["開", "關", "拉開", "拉下", "打開", "拉"],
            "示例指令": ["開左窗簾", "關左窗簾", "拉開左窗簾", "拉左窗簾"],
            "参数支持": [],
            "homophones": cantonese_homophones.get("左窗簾", [])
        },
        "右窗簾": {
            "操作命令": ["開", "關", "拉開", "拉下", "打開", "拉"],
            "示例指令": ["開右窗簾", "關右窗簾", "拉下右窗簾", "拉右窗簾"],
            "参数支持": [],
            "homophones": cantonese_homophones.get("右窗簾", [])
        },
        # "窗紗": {
        #     "操作命令": ["開", "關", "拉開", "拉下"],
        #     "示例指令": ["開窗紗", "閂窗紗唔該", "拉開窗紗", "拉下窗紗"],
        #     "参数支持": [],
        #     "homophones": cantonese_homophones.get("窗紗", [])
        # },
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
            "示例指令": ["開廁所燈", "閂廁所燈好喇", "熄廁所燈吖", "廁所燈光啲", "開廁燈"],
            "参数支持": [],
            "homophones": cantonese_homophones.get("廁所燈", [])
        },
        "空調": {
            "操作命令": ["開", "關", "凍啲", "熱啲", "開大啲", "開小啲", "調高", "調低", "強", "弱", "到"],
            "示例指令": [
                "開空調", "關空調", "空調凍啲吖", "空調熱啲啦",
                "開冷氣", "關冷氣", "冷氣風開大啲", "冷氣風開細啲",
                "空調開到26度", "冷氣調高2度", "空調開小3度",  # 新增带度数示例
                "冷氣調至24度", "空調溫度開大2度",
                "空調開到十六度", "冷氣調高五度",  # 汉字数字示例
                "空調開小三度", "冷氣調至廿二度",  # 包含粤语简写“廿”
                "空調溫度設為卅度"  # 包含粤语简写“卅”
            ],
            "参数支持": ["温度调节", "风力控制", "具体度数设置"],  # 新增参数说明
            "homophones": ["空調", "冷氣", "控調"]
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

def chinese_to_num(chinese_num):
    """
    将汉字数字（如一到九十九）转换为阿拉伯数字
    支持：一至十、十一至九十九、廿(20)、卅(30)等粤语常用表达
    """
    # 基础数字映射
    num_map = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
        '十': 10, '廿': 20, '卅': 30, '百': 100  # 廿、卅是粤语中20、30的简写
    }
    
    # 处理特殊情况（纯“十”对应10）
    if chinese_num == '十':
        return 10
    
    total = 0
    current = 0
    
    for char in chinese_num:
        if char == '十':
            # “十”可单独使用（如“十”=10）或组合（如“十六”=10+6）
            current = current * 10 if current != 0 else 10
        else:
            num = num_map[char]
            if num == 100:  # 处理“百”（实际温度场景中很少用到，但保留扩展性）
                total += current * num
                current = 0
            else:
                current += num
    
    total += current
    return total

# 唤醒词
wake_words = ["你好港怡"]  # 替换为实际的唤醒词
model_dir = "iic/SenseVoiceSmall"

# model_dir = "D:/modelscope_cache/hub/models/iic/SenseVoiceSmall"

m, kwargs = SenseVoiceSmall.from_pretrained(
    model=model_dir, 
    device="cuda:0", 
    hot_words=[
        # 完整唤醒词（整体强化）
        "你好港怡",
        # “港怡”核心词的全量变体（覆盖发音/字形偏差）
        "港怡", "港儀", "港易", "港二",  # “怡”的同音字（粤语发音相同）
        "岡怡", "剛怡", "崗怡",          # “港”的近音字（粤语“港”“岡”“剛”发音接近）
        "講怡", "港宜", "港義",          # 新增常见误识别变体
        # 前半部分“你好”的强化（避免漏检前缀）
    ],
    hot_word_weight=50.0  # 提高权重（从20→30，增强模型对热词的关注度）
)
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

# 降噪处理
def denoise_audio(audio_data, rate):
    """对音频数据进行降噪处理，增加窗口大小控制和长度检查"""
    if DEBUG:
        print("进行音频降噪处理")
    
    # 转换为float32格式
    audio_float32 = audio_data.astype(np.float32) / 32767.0
    
    # 最小有效音频长度（至少为窗口大小的2倍，避免窗口无效）
    min_window = 1024  # 手动指定最小窗口大小（2的幂，如512、1024、2048）
    min_length = min_window * 2  # 音频长度至少为窗口的2倍
    if len(audio_float32) < min_length:
        if DEBUG:
            print(f"音频过短（{len(audio_float32)}样本），跳过降噪")
        return audio_data  # 返回原始数据，避免错误
    
    # 提取噪声样本（取前0.5秒或音频的1/4，取较小值）
    noise_sample_length = min(int(rate * 0.5), len(audio_float32) // 4)
    noise_sample = audio_float32[:noise_sample_length]
    
    try:
        # 关键修复：手动指定n_fft（窗口大小），确保为2的幂且不超过音频长度
        n_fft = min(min_window, len(audio_float32) // 2)  # 窗口大小不超过音频长度的一半
        # 确保n_fft是2的幂（noisereduce要求）
        n_fft = 2 ** int(np.log2(n_fft)) if n_fft > 0 else 1024
        
        reduced_noise = nr.reduce_noise(
            y=audio_float32,
            y_noise=noise_sample,
            prop_decrease=NOISE_REDUCE_PROP,
            sr=rate,
            n_fft=n_fft,  # 显式指定窗口大小
            hop_length=n_fft // 2  # 步长设为窗口的一半（推荐值）
        )
        return (reduced_noise * 32767).astype(np.int16)
    except ValueError as e:
        if "window size" in str(e):
            if DEBUG:
                print(f"降噪窗口大小错误：{e}，使用原始音频")
            return audio_data
        else:
            raise


# 初始化VAD
def init_vad():
    """初始化VAD（语音活动检测器）"""
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    return vad

# 检查音频片段是否包含语音
def is_speech(vad, audio_segment, sample_rate):
    """检查音频片段是否包含语音，自动调整帧长度至有效范围"""
    # 计算目标帧长度（根据VAD_WINDOW_DURATION，确保是10/20/30ms）
    target_frame_length = int(sample_rate * VAD_WINDOW_DURATION / 1000)
    
    # 核心修复：调整音频片段长度至目标长度（不足则补0，过长则截断）
    if len(audio_segment) < target_frame_length:
        # 填充0至目标长度
        audio_segment = np.pad(audio_segment, (0, target_frame_length - len(audio_segment)), mode='constant')
    else:
        # 截断至目标长度
        audio_segment = audio_segment[:target_frame_length]
    
    # 转换为16位PCM并检查语音
    pcm_data = audio_segment.tobytes()
    return vad.is_speech(pcm_data, sample_rate)

# 实时音频录制（带降噪和VAD）
def record_audio():
    global fig, ax, bar, window_closed
    
    if DEBUG:
        print("进入录音函数")
        
    # 音频参数
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000  # WebRTC VAD推荐16000Hz
    
    # 初始化VAD
    vad = init_vad()
    
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
    volume_threshold = 0.03  # 音量阈值，可根据实际情况调整

    last_wake_check_time = time.time()
    
    # 音频目录
    audio_dir = "audio/test_WAV"
    ensure_dir_exists(audio_dir)
    
    # 用于VAD的音频缓冲区
    vad_buffer = collections.deque(maxlen=int(RATE * 1 / CHUNK))  # 0.5秒的缓冲区
    
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
            
            # 添加到VAD缓冲区
            vad_buffer.append(audio_data)
            
            # 检查是否有语音活动
            speech_detected = False
            if len(vad_buffer) == vad_buffer.maxlen:
                # 合并缓冲区数据用于VAD检测
                vad_audio = np.concatenate(list(vad_buffer))
                # 分割为30ms的片段进行VAD检测
                frame_length = int(RATE * VAD_WINDOW_DURATION / 1000)
                for i in range(0, len(vad_audio), frame_length):
                    frame = vad_audio[i:i+frame_length]
                    if len(frame) < frame_length:
                        break
                    if is_speech(vad, frame, RATE):
                        speech_detected = True
                        break
            
            # 结合音量和VAD判断是否有有效音频
            if volume > volume_threshold and speech_detected:
                if DEBUG:
                    print(f"检测到语音: 音量={volume:.4f}, VAD={speech_detected}")
                frames.append(data)
                
                # 保存音频文件进行识别
                audio_path = os.path.join(audio_dir, "test_audio.wav")
                wf = wave.open(audio_path, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                # 识别文本并转换为繁体
                text = recognize_audio(audio_path)
                text = cc.convert(text)
                
                # 关键新增：打印系统实际解析到的完整文本（用于调试）
                
                print(f"【唤醒词解析】系统识别到的内容：{text}")
                
                if DEBUG:
                    print(f"识别文本: {text}")
                
                # 检测唤醒词
                # 在record_audio函数的唤醒词检测部分修改
                # 构建唤醒词的模糊匹配模式（考虑常见发音偏差）
                wake_word = wake_words[0]
                wake_word_patterns = [
                    re.escape(wake_word),  # 精确匹配：你好港怡
                    re.sub(r'港怡', r'港[儀易二宜義]', wake_word),  # “怡”的更多变体
                    re.sub(r'港怡', r'[講岡剛扛]怡', wake_word),     # “港”的更多变体
                    re.sub(r'港怡', r'[講岡] [儀易]', wake_word),    # 允许中间有空格（如“講 儀”）
                    re.sub(r'港怡', r'港儀', wake_word),
                    re.sub(r'港怡', r'講', wake_word),
                    re.sub(r'港怡', r'粵', wake_word),
                    re.sub(r'港怡', r'[/s]*[港岡剛崗講][怡儀易二宜義]', wake_word),
                    re.sub(r'港怡', r'講[怡儀]?', wake_word),
                    re.sub(r'港怡', r'你好', wake_word)
                ]

                wake_word_matched = False
                for pattern in wake_word_patterns:
                    if re.search(pattern, text):
                        # 额外验证：如果是模糊匹配，打印修正提示
                        if pattern != re.escape(wake_words[0]):
                            print(f"【唤醒词修正】识别到'{text}'，修正为'{wake_words[0]}'")
                        wake_word_detected = True
                        wake_word_matched = True
                        print("我在，请讲")
                        frames = []
                        break
                
                # 关键新增：未匹配到唤醒词时，显示具体原因
                # if not wake_word_matched and len(text.strip()) > 0:
                #     print(f"【唤醒词未匹配】识别到内容：'{text}'，未包含唤醒词'{wake_words[0]}'")
                # elif not text.strip():
                #     print("【唤醒词未匹配】未识别到有效语音")
                current_time = time.time()
                if not wake_word_matched:
                    # 仅当距离上次提示超过1.5秒时才输出
                    if current_time - last_wake_check_time > 1.5:
                        if len(text.strip()) > 0:
                            print(f"【唤醒词未匹配】识别到内容：'{text}'，未包含唤醒词'{wake_words[0]}'")
                        else:
                            print("【唤醒词未匹配】未识别到有效语音")
                        # 更新上次检查时间
                        last_wake_check_time = current_time

            else:
                if DEBUG and len(frames) > 0:
                    print(f"检测到静音: 音量={volume:.1f}, VAD={speech_detected}")
            
            if wake_word_detected:
                if DEBUG:
                    print("开始监听用户指令")
                    
                silence_start_time = None
                command_frames = []
                speech_active = False
                
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
                    
                    # 检查语音活动
                    vad_audio = audio_data[:int(RATE * VAD_WINDOW_DURATION / 1000)]
                    current_speech = is_speech(vad, vad_audio, RATE) if len(vad_audio) > 0 else False
                    
                    if volume > volume_threshold and current_speech:
                        if DEBUG:
                            print(f"指令语音: 音量={volume:.4f}, VAD={current_speech}")
                        command_frames.append(data)
                        silence_start_time = None
                        speech_active = True
                    else:
                        if speech_active:  # 只在检测到语音活动后才开始计时静音
                            if silence_start_time is None:
                                silence_start_time = time.time()
                                if DEBUG:
                                    print("开始检测静音时间")
                            else:
                                silence_duration = time.time() - silence_start_time
                                if DEBUG:
                                    print(f"持续静音: {silence_duration:.1f}秒")
                                
                                if silence_duration > SILENCE_THRESHOLD:  # 超过阈值
                                    if DEBUG:
                                        print(f"检测到{SILENCE_THRESHOLD}秒静音，结束指令录音")
                                    
                                    # ===== 插入修改：指令录音后的处理逻辑 =====
                                    # 核心修复：检查指令音频是否有效
                                    if not command_frames:  # 为空
                                        if DEBUG:
                                            print("未检测到有效指令音频，重新监听唤醒词")
                                        wake_word_detected = False  # 重置状态，重新监听
                                        frames = []
                                        break  # 跳出指令录音循环
                                    
                                    # 检查音频长度是否足够（至少0.1秒）
                                    combined_data = b''.join(command_frames)
                                    audio_np = np.frombuffer(combined_data, dtype=np.int16)
                                    audio_duration = len(audio_np) / RATE  # 计算音频时长（秒）
                                    if audio_duration < 0.1:
                                        if DEBUG:
                                            print(f"指令音频过短（{audio_duration:.2f}秒），重新监听唤醒词")
                                        wake_word_detected = False
                                        frames = []
                                        break  # 跳出指令录音循环
                                    
                                    # 正常进行降噪处理
                                    denoised_np = denoise_audio(audio_np, RATE)
                                    frames = [denoised_np.tobytes()]
                                    # ==========================================
                                    
                                    break  # 结束指令录音
                    
                    # 如果窗口已关闭，退出循环
                    if window_closed:
                        break
                
                if DEBUG:
                    print("指令录音结束，保存音频文件")
                break  # 跳出唤醒词监听循环
    
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
        recorded_dir = "audio/recorded"
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


# 语音识别（增强抗噪声）
def recognize_audio(audio_path):
    if DEBUG:
        print(f"开始识别音频: {audio_path}")
    
    # 读取并预处理音频（再次降噪确保质量）
    with wave.open(audio_path, 'rb') as wf:
        rate = wf.getframerate()
        frames = wf.readframes(-1)
        audio_data = np.frombuffer(frames, dtype=np.int16)
    
    # 应用降噪
    denoised_data = denoise_audio(audio_data, rate)
    
    # 保存降噪后的音频用于识别
    denoised_path = audio_path.replace(".wav", "_denoised.wav")
    with wave.open(denoised_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(denoised_data.tobytes())
    
    # 调整参数增强抗噪声识别
    res = m.inference(
        data_in=denoised_path,
        language="yue",  # 指定粤语识别
        use_itn=False,
        ban_emo_unk=False,
        beam_size=10,  # 增大beam size提高噪声环境下的识别率
        temperature=0.8,  # 调整温度参数
        **kwargs,
    )
    
    text = rich_transcription_postprocess(res[0][0]["text"])

    confusion_map = {
        r'講易': '港怡',
        r'講二': '港怡',
        r'講儀': '港怡',
        r'岡易': '港怡',
        r'港二': '港怡',
        r'岡儀': '港怡',    # 新增：岡儀 → 港怡
        r'剛怡': '港怡',    # 新增：剛怡 → 港怡
        r'港宜': '港怡',    # 新增：港宜 → 港怡
        r'講 怡': '港怡',   # 新增：带空格的情况
        r'港 儀': '港怡'    # 新增：带空格的情况
    }
    
    for pattern, replacement in confusion_map.items():
        text = re.sub(pattern, replacement, text)
    
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
        
    # 保留数字和“度”字（避免被分词拆分）
    text = re.sub(r'(\d+)\s*度', lambda m: m.group(1)+"度", text)
    
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
    text = text.replace("冷氣", "空調")  # 统一为“空调”
    
    # 匹配汉字/阿拉伯数字温度
    chinese_num_pattern = r'([一二三四五六七八九十廿卅百]+)\s*度'
    arabic_num_pattern = r'(\d+)\s*度'
    degree_match = re.search(chinese_num_pattern, text) or re.search(arabic_num_pattern, text)
    
    if not degree_match:
        # 无温度时返回基础指令
        if "關" in text or "熄" in text:
            return "關空調"
        elif "凍" in text or "冷" in text:
            return "空調凍啲"
        elif "熱" in text or "暖" in text:
            return "空調熱啲"
        elif "大" in text or "強" in text:
            return "空調風開大啲"
        elif "小" in text or "弱" in text:
            return "空調風開小啲"
        else:
            return "開空調"
    
    # 提取并转换温度
    degree_str = degree_match.group(1)
    try:
        # 区分汉字/阿拉伯数字
        if degree_str in ["一","二","三","四","五","六","七","八","九","十","廿","卅","百"]:
            degree = chinese_to_num(degree_str)
        else:
            degree = int(degree_str)
        
        if not (16 <= degree <= 30):
            return "溫度範圍不支持，請設置16-30度"
    except:
        return "未能識別溫度，請重新輸入"
    
    # 生成带温度的自然语言指令
    if "到" in text:
        return f"空調開到{degree}度"
    elif "高" in text or "大" in text:
        return f"空調開高{degree}度"
    elif "低" in text or "小" in text:
        return f"空調開低{degree}度"
    else:
        return f"空調調至{degree}度"

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

def instruction_sending(instructions, therm1):
    operation = instructions
    if operation == "開空調":
        init_command = bytes([
                0x01,
                0x81,
                0x00,
                0x01,
                0x83
            ])# turn off
        therm1.conn.sendall(init_command)
    elif operation == "關空調":
        init_command = bytes([
                0x01,
                0x80,
                0x00,
                0x01,
                0x82
            ])# turn on
        therm1.conn.sendall(init_command)
    elif operation == "thermostat temp set to 25":
        temperature = 25 #0x19
        temp = hex(temperature)
        #01 85 00 01 FF FF FF 14 98 
        init_command = bytes([
                0x01,
                0x85,
                0x01,
                0xFF,
                0xFF,
                0xFF,
                temp,
                (0x01 + 0x85 + 0x01 + 0xFF + 0xFF + 0xFF + 0x19) & 0xFF
            ])# set temp
        therm1.conn.sendall(init_command)
        
    elif operation == "開房間燈":
        init_command = bytes([
                    0x12, #device type code
                    0x82, #function code
                    0x03, #ID
                    0x22, #device key value
                    (0x12 + 0x82 + 0x03 + 0x22) & 0xFF  #sum
            ])# turn on
        therm1.conn.sendall(init_command)
    elif operation == "關房間燈":
        init_command = bytes([
                    0x12, #device type code
                    0x81, #function code
                    0x03, #ID
                    0x22, #device key value
                    (0x12 + 0x81 + 0x03 + 0x22) & 0xFF #sum
                ])# turn on
        therm1.conn.sendall(init_command)
        
    elif operation == "開廁所燈":
        init_command = bytes([
                    0x12, #device type code
                    0x82, #function code
                    0x03, #ID
                    0x28, #device key value
                    (0x12 + 0x82 + 0x03 + 0x28) & 0xFF #sum
                ])# turn on
        therm1.conn.sendall(init_command)
    elif operation == "關廁所燈":
        init_command = bytes([
                    0x12, #device type code
                    0x81, #function code
                    0x03, #ID
                    0x28, #device key value
                    (0x12 + 0x81 + 0x03 + 0x28) & 0xFF #sum
                ])# turn on
        therm1.conn.sendall(init_command)
        
    elif operation == "開夜燈":
        init_command = bytes([
                    0x12, #device type code
                    0x82, #function code
                    0x03, #ID
                    0x2B, #device key value
                    (0x12 + 0x82 + 0x03 + 0x2B) & 0xFF #sum
                ])# turn on
        therm1.conn.sendall(init_command)
    elif operation == "關夜燈":
        init_command = bytes([
                    0x12, #device type code
                    0x81, #function code
                    0x03, #ID
                    0x2B, #device key value
                    (0x12 + 0x81 + 0x03 + 0x2B) & 0xFF #sum
                ])# turn on
        therm1.conn.sendall(init_command)
        
    elif operation == "開病床燈":
        init_command = bytes([
                    0x12, #device type code
                    0x82, #function code
                    0x03, #ID
                    0x25, #device key value
                    (0x12 + 0x82 + 0x03 + 0x25) & 0xFF #sum
                ])# turn on
        therm1.conn.sendall(init_command)
    elif operation == "關病床燈":
        init_command = bytes([
                    0x12, #device type code
                    0x81, #function code
                    0x03, #ID
                    0x25, #device key value
                    (0x12 + 0x81 + 0x03 + 0x25) & 0xFF #sum
                ])# turn on
        therm1.conn.sendall(init_command)
        
    elif operation == "開窗簾":
        print(operation)
        init_command = bytes([
                    0x13, #device type code
                    0x82, #function code
                    0x04, #ID
                    0x31, #device key value
                    (0x13 + 0x82 + 0x04 + 0x31) & 0xFF #sum
                ])# turn on
        therm1.conn.sendall(init_command)
        init_command = bytes([
                    0x13, #device type code
                    0x82, #function code
                    0x04, #ID
                    0x34, #device key value
                    (0x13 + 0x82 + 0x04 + 0x34) & 0xFF #sum
                ])# turn on
        therm1.conn.sendall(init_command)
    elif operation == "關窗簾":
        print(operation)
        init_command = bytes([
                    0x13, #device type code
                    0x81, #function code
                    0x04, #ID
                    0x33, #device key value
                    (0x13 + 0x81 + 0x04 + 0x33) & 0xFF #sum
                ])# turn on
        therm1.conn.sendall(init_command)
        init_command = bytes([
                    0x13, #device type code
                    0x81, #function code
                    0x04, #ID
                    0x36, #device key value
                    (0x13 + 0x81 + 0x04 + 0x36) & 0xFF #sum
                ])# turn on
        therm1.conn.sendall(init_command)
        
    else:
        print("no recognized operation")
    

def main():
        #Target Thermostat
    thermostat_list = [
        Thermostat(ip="192.168.137.82", device_id=1, class_id=1)
    ]
    number_ther = 1
    therm = thermostat_list[0]

    ### Please Write Program for Easy Scalling of Thermostat List ###

    """ V3 TCP Connection """

    # 1. Host PC information
    HOST = '192.168.137.209'
    PORT = 8900

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

    """ V4 Thermostat Initialization """

    # 5. Get the thermostat status
    for therm in thermostat_list:
        therm.initial_status()

    """ V5 BACnet Side Setup """
    therm1 = thermostat_list[0]
    operation = "turn on"


    print("==== 语音控制系统启动 ====")
    
    try:
        while True:
            if DEBUG:
                print("\n==== 等待唤醒词 ====")
                
            audio_path, window_closed = record_audio()
            if window_closed:
                print("用户关闭了可视化窗口，程序退出")
                break
                
            text = recognize_audio(audio_path)
            text = cc.convert(text)
            instructions = parse_instruction(text)
            
            print(f"\n原语音文本文字: {text}")
            print(f"匹配实际指令: {', '.join(instructions)}")
            
            # 处理未匹配的情况（保持不变）
            MAX_RETRY = 3
            retry_count = 0
            while "未匹配到指令" in instructions and retry_count < MAX_RETRY:
                retry_count += 1
                print(f"抱歉，我没听清，请再说一次（{retry_count}/{MAX_RETRY}）")
                audio_path, window_closed = record_audio()
                if window_closed:
                    print("用户关闭了可视化窗口，程序退出")
                    return
                text = recognize_audio(audio_path)
                text = cc.convert(text)
                instructions = parse_instruction(text)
                print(f"\n原语音文本文字: {text}")
                print(f"匹配实际指令: {', '.join(instructions)}")
            
            # 生成带完整指令的回应
            for instruction in instructions:
                if instruction != "未匹配到指令":
                    # 优化回应话术，包含完整指令
                    print(f"好的，现在为您{instruction}")
                    instruction_sending(instruction, therm1)
            
            if DEBUG:
                print("\n==== 指令处理完成，回到等待唤醒词状态 ====")
                
    except KeyboardInterrupt:
        print("\n用户手动终止程序")
    except Exception as e:
        print(f"程序发生错误: {str(e)}")
    finally:
        if fig is not None and plt.fignum_exists(fig.number):
            plt.ioff()
            plt.close()
        print("==== 语音控制系统已停止 ====")

if __name__ == "__main__":
    main()