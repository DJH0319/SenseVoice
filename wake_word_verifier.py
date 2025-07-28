import os
import librosa
import numpy as np
from dtw import dtw
from scipy.spatial.distance import euclidean
import wave
import pyaudio

# 配置参数
WAKE_WORD_TEMPLATES = "wake_word_templates"  # 模板存储目录
SAMPLE_RATE = 16000  # 采样率需与语音采集一致
N_MELS = 40  # 梅尔频谱特征维度
THRESHOLD = 800  # 比对阈值（可根据实际测试调整）


class WakeWordVerifier:
    def __init__(self):
        self.templates = []
        self._ensure_template_dir()
        self._load_templates()
        
        # 首次运行时自动录制模板
        if not self.templates:
            print("未检测到唤醒词模板，将引导录制...")
            self.record_templates(num_samples=3)
            self._load_templates()

    def _ensure_template_dir(self):
        """确保模板目录存在"""
        if not os.path.exists(WAKE_WORD_TEMPLATES):
            os.makedirs(WAKE_WORD_TEMPLATES)

    def _extract_mel_features(self, audio_path):
        """提取音频的梅尔频谱特征"""
        y, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS)
        return librosa.power_to_db(mel, ref=np.max)  # 转换为分贝值

    def _load_templates(self):
        """加载所有预录模板的特征"""
        self.templates = []
        for filename in os.listdir(WAKE_WORD_TEMPLATES):
            if filename.endswith(".wav"):
                path = os.path.join(WAKE_WORD_TEMPLATES, filename)
                self.templates.append(self._extract_mel_features(path))
        print(f"已加载 {len(self.templates)} 个唤醒词模板")

    def record_templates(self, num_samples=3):
        """录制唤醒词样本作为模板"""
        print(f"请录制 {num_samples} 次唤醒词：你好港怡")
        
        # 音频参数（需与voice_control保持一致）
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = SAMPLE_RATE

        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        for i in range(num_samples):
            input(f"按回车开始录制第 {i+1}/{num_samples} 次（说完后等待3秒自动结束）")
            print("开始录制...")
            frames = []
            for _ in range(int(RATE / CHUNK * 3)):  # 录制3秒
                data = stream.read(CHUNK)
                frames.append(data)
            print("录制完成")

            # 保存样本
            sample_path = os.path.join(WAKE_WORD_TEMPLATES, f"template_{i}.wav")
            with wave.open(sample_path, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))

        stream.stop_stream()
        stream.close()
        p.terminate()
        print(f"模板已保存至 {WAKE_WORD_TEMPLATES}")

    def verify(self, audio_path):
        """比对音频与模板的相似度，返回是否匹配"""
        if not self.templates:
            print("未找到唤醒词模板，请先录制")
            return False

        test_feature = self._extract_mel_features(audio_path)
        min_distance = float('inf')

        # 与每个模板进行DTW比对
        for template in self.templates:
            distance, _, _, _ = dtw(
                test_feature.T,  # 转置为 [时间步, 特征数]
                template.T,
                dist=euclidean
            )
            min_distance = min(min_distance, distance)

        print(f"声学特征比对距离：{min_distance}（阈值：{THRESHOLD}）")
        return min_distance < THRESHOLD