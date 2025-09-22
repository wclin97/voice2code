"""
本地语音转文字模块 (Local ASR)
使用 mlx-whisper 进行完全本地化的语音识别 (仅支持 Apple Silicon)
"""

import platform
import numpy as np
from typing import Dict, List, Tuple
import soundfile as sf

# 只在 Apple Silicon 上导入 mlx_whisper
try:
    import mlx_whisper
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mlx_whisper = None


class LocalASR:
    """本地语音识别类"""

    def __init__(self, model_path: str = "mlx-community/whisper-large-v3-turbo"):
        """
        初始化本地ASR模型

        Args:
            model_path: 模型路径，使用本地MLX格式的whisper模型
        """
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX Whisper 不可用。请确保在 Apple Silicon Mac 上运行，并已安装 mlx-whisper")

        self.model_path = model_path
        print(f"正在加载本地ASR模型: {model_path}")

    def transcribe_with_timestamps(self, audio_file: str, language: str = None) -> Dict:
        """
        转录音频并返回带时间戳的结果

        Args:
            audio_file: 音频文件路径
            language: 语言代码，None表示自动检测

        Returns:
            包含转录文本和时间戳信息的字典
        """
        try:
            print(f"开始转录音频: {audio_file}")
            print(f"使用模型: {self.model_path}")

            # 使用mlx-whisper进行转录，启用词级时间戳
            result = mlx_whisper.transcribe(
                audio_file,
                path_or_hf_repo=self.model_path,
                word_timestamps=True,  # 启用词级时间戳
                language=language,  # None表示自动检测语言
                verbose=True  # 显示进度信息
            )

            print(f"转录完成，检测到语言: {result.get('language', 'unknown')}")
            return self._format_transcription_result(result)

        except Exception as e:
            print(f"转录过程出错: {str(e)}")
            return {"segments": [], "text": "", "error": str(e)}

    def _format_transcription_result(self, raw_result: Dict) -> Dict:
        """
        格式化转录结果，提取关键信息

        Args:
            raw_result: mlx-whisper的原始输出

        Returns:
            格式化后的结果字典
        """
        formatted_segments = []

        if "segments" in raw_result:
            for segment in raw_result["segments"]:
                formatted_segment = {
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": segment.get("text", "").strip(),
                    "words": []
                }

                # 如果有词级时间戳
                if "words" in segment:
                    for word in segment["words"]:
                        formatted_segment["words"].append({
                            "word": word.get("word", "").strip(),
                            "start": word.get("start", 0),
                            "end": word.get("end", 0)
                        })

                formatted_segments.append(formatted_segment)

        return {
            "segments": formatted_segments,
            "text": raw_result.get("text", ""),
            "language": raw_result.get("language", "unknown")
        }

    def get_audio_duration(self, audio_file: str) -> float:
        """
        获取音频文件时长

        Args:
            audio_file: 音频文件路径

        Returns:
            音频时长（秒）
        """
        try:
            data, sample_rate = sf.read(audio_file)
            return len(data) / sample_rate
        except Exception as e:
            print(f"获取音频时长出错: {str(e)}")
            return 0.0


if __name__ == "__main__":
    # 测试代码
    asr = LocalASR()

    # 示例用法
    # result = asr.transcribe_with_timestamps("test_audio.wav")
    # print("转录结果:")
    # print(f"文本: {result['text']}")
    # print(f"语言: {result['language']}")
    # print("分段信息:")
    # for i, segment in enumerate(result['segments']):
    #     print(f" 段落{i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s")
    #     print(f" 文本: {segment['text']}")