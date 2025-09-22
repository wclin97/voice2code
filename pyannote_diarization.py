"""
PyAnnote 说话人分离模块
使用 pyannote/speaker-diarization-3.1 进行高质量说话人分离
需要 HuggingFace token 首次下载，之后完全离线
"""

import os
from typing import Dict, List, Optional
import warnings

# 抑制一些不必要的警告
warnings.filterwarnings("ignore", category=UserWarning)


class PyAnnoteDiarization:
    """PyAnnote 说话人分离类"""

    def __init__(self, token: str = None):
        """
        初始化 PyAnnote 说话人分离器

        Args:
            token: HuggingFace token，None表示使用已登录的token
        """
        self.token = token
        self.pipeline = None
        self._check_and_load_model()

    def _check_and_load_model(self):
        """检查并加载 PyAnnote 模型"""
        try:
            from pyannote.audio import Pipeline

            print("正在加载 PyAnnote 说话人分离模型...")
            print("注意: 首次使用需要 HuggingFace token，之后完全离线")

            # 确定使用的token
            auth_token = self.token if self.token else True

            # 尝试加载模型
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=auth_token
            )

            print(" PyAnnote 模型加载成功!")

        except ImportError:
            raise ImportError(
                "未安装 pyannote.audio。请运行: pip install pyannote.audio"
            )
        except Exception as e:
            error_msg = str(e)
            if "authenticate" in error_msg.lower() or "token" in error_msg.lower():
                # 如果没有提供token，尝试提示用户输入
                if not self.token:
                    print(" 需要 HuggingFace token")
                    print("\n有3种方式提供token:")
                    print("1. 运行 huggingface-cli login 登录")
                    print("2. 直接传入token: PyAnnoteDiarization(token='your_token')")
                    print("3. 设置环境变量: export HUGGINGFACE_HUB_TOKEN='your_token'")
                    print("\n获取token: https://huggingface.co/settings/tokens")

                    # 尝试从环境变量获取
                    import os
                    env_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
                    if env_token:
                        print(" 找到环境变量中的token，重试加载...")
                        self.token = env_token
                        self._check_and_load_model()
                        return

                    # 提示用户手动输入token
                    token_input = input("\n请输入你的 HuggingFace token (或按Enter跳过): ").strip()
                    if token_input:
                        print(" 使用提供的token，重试加载...")
                        self.token = token_input
                        self._check_and_load_model()
                        return

                raise RuntimeError(
                    "需要 HuggingFace token。请使用以上任意一种方式提供token"
                )
            else:
                raise RuntimeError(f"加载 PyAnnote 模型失败: {error_msg}")

    def diarize(self, audio_file: str, max_duration: float = None) -> Dict:
        """
        执行说话人分离

        Args:
            audio_file: 音频文件路径
            max_duration: 最大处理时长（秒），超过则分段处理

        Returns:
            说话人分离结果
        """
        if self.pipeline is None:
            raise RuntimeError("PyAnnote 模型未正确加载")

        try:
            # 检查音频时长
            duration = self._get_audio_duration(audio_file)
            print(f"开始 PyAnnote 说话人分离... (音频时长: {duration:.1f}s)")

            # 如果音频过长，给出提示
            if duration > 300:  # 5分钟
                print(" 音频较长，PyAnnote 处理可能需要较长时间...")
                print(" 建议: 长音频可以考虑使用本地聚类方法获得更快速度")

            # 如果设置了最大时长且音频超长，进行分段处理
            if max_duration and duration > max_duration:
                print(f" 音频超过 {max_duration}s，启用分段处理...")
                return self._diarize_chunked(audio_file, max_duration)

            # 显示进度提示
            import time
            start_time = time.time()

            # 预处理音频以避免张量尺寸问题
            preprocessed_audio = self._preprocess_audio(audio_file)

            try:
                # 执行说话人分离
                diarization_result = self.pipeline(preprocessed_audio)

                elapsed = time.time() - start_time
                print(f" PyAnnote 处理完成，耗时: {elapsed:.1f}s")

                # 转换结果格式
                segments = []
                speakers = set()

                for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                    segments.append({
                        "start": turn.start,
                        "end": turn.end,
                        "speaker": f"Speaker_{speaker}"
                    })
                    speakers.add(f"Speaker_{speaker}")

                # 按时间排序
                segments.sort(key=lambda x: x["start"])

                print(f" PyAnnote 分离完成，检测到 {len(speakers)} 个说话人")

                return {
                    "segments": segments,
                    "speakers": list(speakers),
                    "total_speakers": len(speakers),
                    "method": "pyannote",
                    "processing_time": elapsed
                }

            finally:
                # 清理临时文件
                self._cleanup_temp_file(preprocessed_audio, audio_file)

        except Exception as e:
            print(f" PyAnnote 说话人分离失败: {str(e)}")
            return {
                "segments": [],
                "speakers": [],
                "total_speakers": 0,
                "error": str(e),
                "method": "pyannote"
            }

    def _get_audio_duration(self, audio_file: str) -> float:
        """获取音频时长"""
        try:
            import librosa
            y, sr = librosa.load(audio_file, sr=None)
            return len(y) / sr
        except Exception:
            return 0.0

    def _preprocess_audio(self, audio_file: str) -> str:
        """
        预处理音频以解决 PyAnnote 张量尺寸问题

        Args:
            audio_file: 原始音频文件路径

        Returns:
            预处理后的音频文件路径
        """
        try:
            import librosa
            import soundfile as sf
            import tempfile
            import os

            # 加载音频
            y, sr = librosa.load(audio_file, sr=16000)  # PyAnnote 推荐 16kHz

            # 确保音频长度是合适的
            # PyAnnote 期望的最小长度约为 0.5 秒
            min_samples = int(0.5 * sr)
            if len(y) < min_samples:
                # 填充短音频
                y = librosa.util.fix_length(y, size=min_samples)

            # 标准化音频
            y = librosa.util.normalize(y)

            # 创建临时文件
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"pyannote_preprocessed_{os.getpid()}.wav")

            # 保存预处理后的音频
            sf.write(temp_file, y, sr)

            return temp_file

        except Exception as e:
            print(f"音频预处理失败，使用原始文件: {str(e)}")
            return audio_file

    def _cleanup_temp_file(self, temp_file: str, original_file: str):
        """清理临时文件"""
        try:
            import os
            if temp_file != original_file and os.path.exists(temp_file):
                os.unlink(temp_file)
        except Exception:
            pass  # 忽略清理错误

    def _diarize_chunked(self, audio_file: str, chunk_duration: float) -> Dict:
        """分段处理长音频"""
        try:
            import librosa
            import soundfile as sf
            import tempfile
            import os

            # 加载音频
            y, sr = librosa.load(audio_file, sr=None)
            total_duration = len(y) / sr
            chunk_samples = int(chunk_duration * sr)

            print(f" 分成 {int(total_duration / chunk_duration) + 1} 段处理...")

            all_segments = []
            chunk_count = 0

            for start_sample in range(0, len(y), chunk_samples):
                end_sample = min(start_sample + chunk_samples, len(y))
                chunk_audio = y[start_sample:end_sample]
                start_time = start_sample / sr

                # 保存临时音频文件
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    sf.write(tmp_file.name, chunk_audio, sr)

                print(f"处理第 {chunk_count + 1} 段...")

                # 处理这一段
                chunk_result = self.pipeline(tmp_file.name)

                # 调整时间戳并添加到总结果
                for turn, _, speaker in chunk_result.itertracks(yield_label=True):
                    all_segments.append({
                        "start": turn.start + start_time,
                        "end": turn.end + start_time,
                        "speaker": f"Speaker_{speaker}_chunk{chunk_count}"
                    })

                # 清理临时文件
                os.unlink(tmp_file.name)

                chunk_count += 1

            # 合并相邻的同说话人片段
            merged_segments = self._merge_speaker_segments(all_segments)
            speakers = list(set(seg["speaker"] for seg in merged_segments))

            return {
                "segments": merged_segments,
                "speakers": speakers,
                "total_speakers": len(speakers),
                "method": "pyannote_chunked"
            }

        except Exception as e:
            print(f" 分段处理失败: {e}")
            # 回退到普通处理
            return self.diarize(audio_file, max_duration=None)

    def _merge_speaker_segments(self, segments: List[Dict]) -> List[Dict]:
        """合并相邻的同说话人片段"""
        if not segments:
            return []

        # 按时间排序
        segments.sort(key=lambda x: x["start"])

        merged = [segments[0].copy()]

        for seg in segments[1:]:
            last_seg = merged[-1]

            # 如果是同一说话人且时间接近，合并
            if (seg["speaker"].split("_chunk")[0] == last_seg["speaker"].split("_chunk")[0] and
                seg["start"] - last_seg["end"] < 2.0):  # 2秒内的间隔合并
                last_seg["end"] = seg["end"]
                # 统一说话人标签
                base_speaker = last_seg["speaker"].split("_chunk")[0]
                last_seg["speaker"] = base_speaker
            else:
                # 清理chunk标记
                seg_copy = seg.copy()
                seg_copy["speaker"] = seg["speaker"].split("_chunk")[0]
                merged.append(seg_copy)

        return merged

    def is_available(self) -> bool:
        """检查 PyAnnote 是否可用"""
        try:
            import pyannote.audio
            return self.pipeline is not None
        except ImportError:
            return False

    @staticmethod
    def check_requirements() -> Dict[str, bool]:
        """检查 PyAnnote 相关依赖"""
        requirements = {
            "pyannote.audio": False,
            "torch": False,
            "huggingface_hub": False
        }

        try:
            import pyannote.audio
            requirements["pyannote.audio"] = True
        except ImportError:
            pass

        try:
            import torch
            requirements["torch"] = True
        except ImportError:
            pass

        try:
            import huggingface_hub
            requirements["huggingface_hub"] = True
        except ImportError:
            pass

        return requirements


def setup_pyannote():
    """设置 PyAnnote 的辅助函数"""
    print(" PyAnnote 设置指南")
    print("=" * 40)

    # 检查依赖
    requirements = PyAnnoteDiarization.check_requirements()

    print("依赖检查:")
    for package, installed in requirements.items():
        status = "✓" if installed else "✗"
        print(f" {status} {package}")

    if not all(requirements.values()):
        print("\n安装缺失的依赖:")
        if not requirements["pyannote.audio"]:
            print(" pip install pyannote.audio")
        if not requirements["torch"]:
            print(" pip install torch")
        if not requirements["huggingface_hub"]:
            print(" pip install huggingface_hub")

    print("\n Token 设置:")
    print("1. 访问 https://huggingface.co/settings/tokens")
    print("2. 创建新的 Access Token")
    print("3. 运行: huggingface-cli login")
    print("4. 输入你的 token")

    print("\n 同意模型使用条款:")
    print("访问以下链接并同意使用条款:")
    print("- https://huggingface.co/pyannote/speaker-diarization-3.1")
    print("- https://huggingface.co/pyannote/segmentation-3.0")

    print("\n 设置完成后，模型将被缓存到本地，后续使用完全离线!")


if __name__ == "__main__":
    # 运行设置向导
    setup_pyannote()

    # 测试 PyAnnote
    try:
        diarizer = PyAnnoteDiarization()
        print("\n PyAnnote 设置成功!")
    except Exception as e:
        print(f"\n PyAnnote 设置失败: {str(e)}")
        print("请按照上述指南完成设置")