"""
混合说话人分离器
支持 PyAnnote (高质量) 和本地聚类 (无需token) 两种方法
"""

from typing import Dict, Optional
from local_diarization import LocalDiarization


class HybridDiarization:
    """混合说话人分离器"""

    def __init__(self,
                 method: str = "auto",  # "pyannote", "local", "auto"
                 window_length: float = 2.0,
                 hop_length: float = 1.0,
                 n_speakers: Optional[int] = None):
        """
        初始化混合说话人分离器

        Args:
            method: 分离方法 ("pyannote", "local", "auto")
            window_length: 本地方法的窗口长度
            hop_length: 本地方法的跳跃长度
            n_speakers: 说话人数量
        """
        self.method = method
        self.window_length = window_length
        self.hop_length = hop_length
        self.n_speakers = n_speakers

        # 初始化本地分离器
        self.local_diarizer = LocalDiarization(
            window_length=window_length,
            hop_length=hop_length,
            n_speakers=n_speakers
        )

        # 尝试初始化 PyAnnote 分离器
        self.pyannote_diarizer = None
        self._init_pyannote()

    def _init_pyannote(self):
        """初始化 PyAnnote 分离器"""
        try:
            from pyannote_diarization import PyAnnoteDiarization
            self.pyannote_diarizer = PyAnnoteDiarization()
            if self.pyannote_diarizer.is_available():
                print(" PyAnnote 分离器可用")
            else:
                self.pyannote_diarizer = None
        except Exception as e:
            print(f" PyAnnote 不可用: {str(e)}")
            self.pyannote_diarizer = None

    def diarize(self, audio_file: str, force_method: str = None) -> Dict:
        """
        执行说话人分离

        Args:
            audio_file: 音频文件路径
            force_method: 强制使用的方法 ("pyannote", "pyannote_fast", "local")

        Returns:
            说话人分离结果
        """
        # 确定使用的方法
        chosen_method = force_method or self.method

        if chosen_method == "auto":
            chosen_method = "pyannote" if self.pyannote_diarizer else "local"

        print(f"使用说话人分离方法: {chosen_method}")

        # 执行分离
        if chosen_method in ["pyannote", "pyannote_fast"] and self.pyannote_diarizer:
            return self._diarize_with_pyannote(audio_file, chosen_method)
        else:
            return self._diarize_with_local(audio_file)

    def _diarize_with_pyannote(self, audio_file: str, method: str = "pyannote") -> Dict:
        """使用 PyAnnote 进行分离"""
        try:
            if method == "pyannote_fast":
                # 快速模式：自动分段处理长音频
                result = self.pyannote_diarizer.diarize(audio_file, max_duration=300)  # 5分钟分段
                result["method_used"] = "pyannote_fast"
            else:
                # 标准模式
                result = self.pyannote_diarizer.diarize(audio_file)
                result["method_used"] = "pyannote"
            return result
        except Exception as e:
            print(f"PyAnnote 分离失败，回退到本地方法: {str(e)}")
            return self._diarize_with_local(audio_file)

    def _diarize_with_local(self, audio_file: str) -> Dict:
        """使用本地聚类方法进行分离"""
        result = self.local_diarizer.diarize(audio_file)
        result["method_used"] = "local_clustering"
        return result

    def get_available_methods(self) -> Dict[str, bool]:
        """获取可用的分离方法"""
        return {
            "pyannote": self.pyannote_diarizer is not None,
            "local": True  # 本地方法总是可用
        }

    def get_method_info(self) -> Dict:
        """获取方法信息"""
        return {
            "pyannote": {
                "available": self.pyannote_diarizer is not None,
                "description": "高质量说话人分离，需要首次下载模型",
                "pros": ["准确度高", "鲁棒性好", "支持复杂场景"],
                "cons": ["需要token首次下载", "模型较大", "依赖更多"]
            },
            "local": {
                "available": True,
                "description": "基于音频特征聚类的本地方法",
                "pros": ["完全离线", "无需token", "轻量级"],
                "cons": ["准确度一般", "复杂场景效果差", "需要调参"]
            }
        }

    def visualize_diarization(self, diarization_result: Dict, output_file: str = "diarization_plot.png"):
        """
        可视化说话人分离结果

        Args:
            diarization_result: 分离结果
            output_file: 输出图片文件名
        """
        # 使用本地分离器的可视化功能
        self.local_diarizer.visualize_diarization(diarization_result, output_file)

    def benchmark_methods(self, audio_file: str) -> Dict:
        """
        对比两种方法的效果 (如果都可用)

        Args:
            audio_file: 音频文件路径

        Returns:
            对比结果
        """
        results = {}

        # 测试本地方法
        print(" 测试本地聚类方法...")
        local_result = self._diarize_with_local(audio_file)
        results["local"] = {
            "speakers": local_result.get("total_speakers", 0),
            "segments": len(local_result.get("segments", [])),
            "method": "local_clustering"
        }

        # 测试 PyAnnote 方法 (如果可用)
        if self.pyannote_diarizer:
            print(" 测试 PyAnnote 方法...")
            try:
                pyannote_result = self._diarize_with_pyannote(audio_file)
                results["pyannote"] = {
                    "speakers": pyannote_result.get("total_speakers", 0),
                    "segments": len(pyannote_result.get("segments", [])),
                    "method": "pyannote"
                }
            except Exception as e:
                results["pyannote"] = {
                    "error": str(e),
                    "method": "pyannote"
                }

        return results


if __name__ == "__main__":
    # 测试混合分离器
    diarizer = HybridDiarization(method="auto")

    print(" 可用方法:")
    methods = diarizer.get_available_methods()
    for method, available in methods.items():
        status = "✓" if available else "✗"
        print(f" {status} {method}")

    print("\n 方法详情:")
    info = diarizer.get_method_info()
    for method, details in info.items():
        if details["available"]:
            print(f"\n{method.upper()}:")
            print(f" 描述: {details['description']}")
            print(f" 优点: {', '.join(details['pros'])}")
            print(f" 缺点: {', '.join(details['cons'])}")