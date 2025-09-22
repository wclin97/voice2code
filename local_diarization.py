"""
本地说话人分离模块 (Local Speaker Diarization)
使用基于音频特征聚类的方法进行完全本地化的说话人分离
"""

import numpy as np
import librosa
import soundfile as sf
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class LocalDiarization:
    """本地说话人分离类"""

    def __init__(self,
                 window_length: float = 2.0,  # 窗口长度（秒）
                 hop_length: float = 1.0,  # 跳跃长度（秒）
                 n_mfcc: int = 13,  # MFCC特征数量
                 n_speakers: Optional[int] = None):  # 说话人数量，None表示自动估计
        """
        初始化本地说话人分离器

        Args:
            window_length: 分析窗口长度（秒）
            hop_length: 窗口间跳跃长度（秒）
            n_mfcc: MFCC特征维度
            n_speakers: 说话人数量，None表示自动估计
        """
        self.window_length = window_length
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.n_speakers = n_speakers
        self.scaler = StandardScaler()

    def extract_features(self, audio_file: str) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        提取音频特征

        Args:
            audio_file: 音频文件路径

        Returns:
            特征矩阵和对应的时间窗口
        """
        try:
            # 加载音频
            y, sr = librosa.load(audio_file, sr=None)

            # 计算窗口大小（样本点数）
            window_samples = int(self.window_length * sr)
            hop_samples = int(self.hop_length * sr)

            features = []
            time_windows = []

            # 滑动窗口提取特征
            for start in range(0, len(y) - window_samples + 1, hop_samples):
                end = start + window_samples
                window_audio = y[start:end]

                # 检查音频活动性（简单的能量阈值）
                if self._is_speech(window_audio):
                    # 提取MFCC特征
                    mfcc = librosa.feature.mfcc(
                        y=window_audio,
                        sr=sr,
                        n_mfcc=self.n_mfcc
                    )

                    # 使用统计特征（均值和标准差）
                    mfcc_mean = np.mean(mfcc, axis=1)
                    mfcc_std = np.std(mfcc, axis=1)
                    feature_vector = np.concatenate([mfcc_mean, mfcc_std])

                    features.append(feature_vector)

                    # 记录时间窗口
                    start_time = start / sr
                    end_time = end / sr
                    time_windows.append((start_time, end_time))

            return np.array(features), time_windows

        except Exception as e:
            print(f"特征提取出错: {str(e)}")
            return np.array([]), []

    def _is_speech(self, audio_segment: np.ndarray, threshold: float = 0.01) -> bool:
        """
        简单的语音活动检测

        Args:
            audio_segment: 音频片段
            threshold: 能量阈值

        Returns:
            是否包含语音
        """
        energy = np.mean(audio_segment ** 2)
        return energy > threshold

    def estimate_speakers(self, features: np.ndarray, max_speakers: int = 6) -> int:
        """
        估计说话人数量

        Args:
            features: 特征矩阵
            max_speakers: 最大说话人数

        Returns:
            估计的说话人数量
        """
        if len(features) < 2:
            return 1

        inertias = []
        k_range = range(1, min(max_speakers + 1, len(features) + 1))

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)

        # 使用肘部法则估计最优聚类数
        if len(inertias) < 2:
            return 1

        # 计算二阶差分
        diff2 = np.diff(inertias, 2)
        if len(diff2) > 0:
            optimal_k = np.argmax(diff2) + 2  # +2因为差分操作导致的索引偏移
            return min(optimal_k, max_speakers)

        return 2  # 默认返回2个说话人

    def cluster_speakers(self, features: np.ndarray, n_speakers: Optional[int] = None) -> np.ndarray:
        """
        对特征进行聚类以识别说话人

        Args:
            features: 特征矩阵
            n_speakers: 说话人数量

        Returns:
            聚类标签
        """
        if len(features) == 0:
            return np.array([])

        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)

        # 降维（如果特征维度过高）
        if features_scaled.shape[1] > 50:
            pca = PCA(n_components=20)
            features_scaled = pca.fit_transform(features_scaled)

        # 估计说话人数量
        if n_speakers is None:
            n_speakers = self.n_speakers or self.estimate_speakers(features_scaled)

        print(f"使用 {n_speakers} 个说话人进行聚类")

        # 使用层次聚类（通常比K-means更稳定）
        clustering = AgglomerativeClustering(
            n_clusters=n_speakers,
            linkage='ward'
        )

        labels = clustering.fit_predict(features_scaled)
        return labels

    def diarize(self, audio_file: str) -> Dict:
        """
        执行说话人分离

        Args:
            audio_file: 音频文件路径

        Returns:
            说话人分离结果
        """
        print("开始提取音频特征...")
        features, time_windows = self.extract_features(audio_file)

        if len(features) == 0:
            return {
                "segments": [],
                "speakers": [],
                "error": "无法提取音频特征或未检测到语音"
            }

        print(f"提取到 {len(features)} 个特征窗口")

        print("开始说话人聚类...")
        labels = self.cluster_speakers(features)

        # 格式化结果
        segments = []
        for i, (start_time, end_time) in enumerate(time_windows):
            segments.append({
                "start": start_time,
                "end": end_time,
                "speaker": f"Speaker_{labels[i]}"
            })

        # 合并相邻的同一说话人片段
        merged_segments = self._merge_consecutive_segments(segments)

        return {
            "segments": merged_segments,
            "speakers": list(set([seg["speaker"] for seg in merged_segments])),
            "total_speakers": len(set(labels))
        }

    def _merge_consecutive_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        合并相邻的同一说话人片段

        Args:
            segments: 原始分段列表

        Returns:
            合并后的分段列表
        """
        if not segments:
            return []

        merged = [segments[0].copy()]

        for seg in segments[1:]:
            last_seg = merged[-1]

            # 如果是同一说话人且时间连续，则合并
            if (seg["speaker"] == last_seg["speaker"] and
                abs(seg["start"] - last_seg["end"]) < self.hop_length * 1.5):
                last_seg["end"] = seg["end"]
            else:
                merged.append(seg.copy())

        return merged

    def visualize_diarization(self, diarization_result: Dict, output_file: str = "diarization_plot.png"):
        """
        可视化说话人分离结果

        Args:
            diarization_result: 分离结果
            output_file: 输出图片文件名
        """
        try:
            segments = diarization_result["segments"]
            speakers = diarization_result["speakers"]

            fig, ax = plt.subplots(figsize=(12, 4))

            # 为每个说话人分配颜色
            colors = plt.cm.Set3(np.linspace(0, 1, len(speakers)))
            speaker_colors = {speaker: colors[i] for i, speaker in enumerate(speakers)}

            # 绘制时间线
            for seg in segments:
                start, end, speaker = seg["start"], seg["end"], seg["speaker"]
                ax.barh(0, end - start, left=start, height=0.5,
                        color=speaker_colors[speaker], alpha=0.7,
                        label=speaker if speaker not in ax.get_legend_handles_labels()[1] else "")

            ax.set_xlabel("时间 (秒)")
            ax.set_ylabel("说话人")
            ax.set_title("说话人分离结果")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"可视化结果已保存到: {output_file}")

        except Exception as e:
            print(f"可视化出错: {str(e)}")


if __name__ == "__main__":
    # 测试代码
    diarizer = LocalDiarization(
        window_length=2.0,
        hop_length=1.0,
        n_speakers=None  # 自动估计
    )

    # 示例用法
    # result = diarizer.diarize("test_audio.wav")
    # print("说话人分离结果:")
    # print(f"检测到 {result['total_speakers']} 个说话人")
    # print("分段信息:")
    # for seg in result['segments']:
    #     print(f" {seg['start']:.2f}s - {seg['end']:.2f}s: {seg['speaker']}")