"""
跨平台语音转文字模块
支持 Apple Silicon (MLX) 和其他平台 (OpenAI Whisper)
支持配置文件管理不同模型
"""

import platform
import sys
from typing import Dict, Optional
from .config_manager import ConfigManager


def get_platform_info():
    """获取平台信息"""
    system = platform.system()
    machine = platform.machine()

    # 检查是否为 Apple Silicon
    is_apple_silicon = (
        system == "Darwin" and
        machine in ["arm64", "aarch64"]
    )

    return {
        "system": system,
        "machine": machine,
        "is_apple_silicon": is_apple_silicon,
        "is_windows": system == "Windows",
        "is_linux": system == "Linux",
        "is_intel_mac": system == "Darwin" and machine == "x86_64"
    }


class CrossPlatformASR:
    """跨平台语音识别类"""

    def __init__(self, model_size: str = "auto", interactive: bool = False):
        """
        初始化跨平台ASR

        Args:
            model_size: 模型大小 ("auto" 使用配置, "interactive" 交互选择, 或具体模型名)
            interactive: 是否启用交互式模型选择
        """
        self.config_manager = ConfigManager()
        self.platform_info = get_platform_info()
        self.asr_engine = None
        self.model_size = model_size
        self.interactive = interactive
        self._initialize_engine()

    def _initialize_engine(self):
        """根据平台初始化对应的ASR引擎"""
        print(f"检测到平台: {self.platform_info['system']} {self.platform_info['machine']}")

        # 确定使用的引擎
        if self.platform_info["is_apple_silicon"]:
            engine = "mlx"
            self._init_mlx_whisper()
        else:
            engine = "openai"
            self._init_openai_whisper()

        # 设置引擎类型以便后续使用
        self.engine_type = engine

    def _init_mlx_whisper(self):
        """初始化 MLX Whisper (Apple Silicon)"""
        try:
            import mlx_whisper
        except ImportError:
            raise RuntimeError("MLX Whisper 不可用。请确保在 Apple Silicon Mac 上运行，并已安装 mlx-whisper")

        try:
            self.asr_engine = "mlx"

            # 根据模式选择模型
            if self.model_size == "interactive" or self.interactive:
                model = self.config_manager.list_models_interactive("mlx")
                if not model:
                    model = self.config_manager.get_default_whisper_model("mlx")
            elif self.model_size == "auto":
                model = self.config_manager.get_default_whisper_model("mlx")
            else:
                # 使用指定的模型
                model = self.model_size

            self.model_path = model
            model_desc = self.config_manager.get_model_description("mlx", model)
            print(f" 使用 MLX Whisper: {model}")
            print(f" 模型描述: {model_desc}")

        except ImportError:
            print(" MLX Whisper 不可用，回退到 OpenAI Whisper")
            self.engine_type = "openai"
            self._init_openai_whisper()

    def _init_openai_whisper(self):
        """初始化 OpenAI Whisper (跨平台)"""
        try:
            import whisper
            self.asr_engine = "openai"

            # 根据模式选择模型
            if self.model_size == "interactive" or self.interactive:
                model = self.config_manager.list_models_interactive("openai")
                if not model:
                    model = self.config_manager.get_default_whisper_model("openai")
            elif self.model_size == "auto":
                model = self.config_manager.get_default_whisper_model("openai")
            else:
                # 使用指定的模型
                model = self.model_size

            print(f" 正在加载 OpenAI Whisper 模型: {model}")
            model_desc = self.config_manager.get_model_description("openai", model)
            print(f" 模型描述: {model_desc}")

            self.model = whisper.load_model(model)
            self.model_name = model
            print(f" OpenAI Whisper 模型加载完成")

        except ImportError:
            raise ImportError(
                "需要安装 openai-whisper。请运行: pip install openai-whisper"
            )

    def transcribe_with_timestamps(self, audio_file: str, language: str = None) -> Dict:
        """
        转录音频并返回带时间戳的结果

        Args:
            audio_file: 音频文件路径
            language: 语言代码

        Returns:
            转录结果字典
        """
        print(f"开始转录音频: {audio_file}")
        print(f"使用引擎: {self.asr_engine}")

        try:
            if self.asr_engine == "mlx":
                return self._transcribe_with_mlx(audio_file, language)
            else:
                return self._transcribe_with_openai(audio_file, language)
        except Exception as e:
            print(f"转录失败: {str(e)}")
            return {"segments": [], "text": "", "error": str(e)}

    def _transcribe_with_mlx(self, audio_file: str, language: str = None) -> Dict:
        """使用 MLX Whisper 转录"""
        try:
            import mlx_whisper
        except ImportError:
            raise RuntimeError("MLX Whisper 不可用。请确保在 Apple Silicon Mac 上运行，并已安装 mlx-whisper")

        result = mlx_whisper.transcribe(
            audio_file,
            path_or_hf_repo=self.model_path,
            word_timestamps=True,
            language=language,
            verbose=True
        )

        return self._format_result(result, "mlx")

    def _transcribe_with_openai(self, audio_file: str, language: str = None) -> Dict:
        """使用 OpenAI Whisper 转录"""
        options = {
            "word_timestamps": True,
            "verbose": True
        }

        if language:
            options["language"] = language

        result = self.model.transcribe(audio_file, **options)
        return self._format_result(result, "openai")

    def _format_result(self, raw_result: Dict, engine: str) -> Dict:
        """格式化转录结果"""
        formatted_segments = []

        segments = raw_result.get("segments", [])
        for segment in segments:
            formatted_segment = {
                "start": segment.get("start", 0),
                "end": segment.get("end", 0),
                "text": segment.get("text", "").strip(),
                "words": []
            }

            # 处理词级时间戳
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
            "language": raw_result.get("language", "unknown"),
            "engine": engine
        }

    def get_engine_info(self) -> Dict:
        """获取当前引擎信息"""
        current_model = getattr(self, 'model_path', getattr(self, 'model_name', self.model_size))
        return {
            "platform": self.platform_info,
            "engine": self.asr_engine,
            "model": current_model,
            "model_description": self.config_manager.get_model_description(self.asr_engine, current_model),
            "optimized": self.asr_engine == "mlx",
            "available_models": self.config_manager.get_available_whisper_models(self.asr_engine)
        }

    def list_available_models(self):
        """列出当前引擎的可用模型"""
        print(f"\n{self.asr_engine.upper()} Whisper 可用模型:")
        print("-" * 50)

        models = self.config_manager.get_available_whisper_models(self.asr_engine)
        current_model = getattr(self, 'model_path', getattr(self, 'model_name', self.model_size))

        for model in models:
            description = self.config_manager.get_model_description(self.asr_engine, model)
            is_current = model == current_model
            current_mark = " [当前使用]" if is_current else ""
            print(f"  • {model}{current_mark}")
            print(f"    {description}")
            print()

    def switch_model_interactive(self) -> bool:
        """交互式切换模型"""
        print(f"\n当前使用: {self.asr_engine.upper()} Whisper")
        selected_model = self.config_manager.list_models_interactive(self.asr_engine)

        if selected_model:
            print(f"\n正在切换到模型: {selected_model}")
            try:
                if self.asr_engine == "mlx":
                    self.model_path = selected_model
                    print(f" MLX 模型路径已更新: {selected_model}")
                else:
                    import whisper
                    print(f" 正在重新加载 OpenAI Whisper 模型...")
                    self.model = whisper.load_model(selected_model)
                    self.model_name = selected_model
                    print(f" OpenAI Whisper 模型重新加载完成")

                print(f" 模型切换成功!")
                return True

            except Exception as e:
                print(f" 模型切换失败: {e}")
                return False
        else:
            print(" 取消模型切换")
            return False

    def get_model_recommendations(self, audio_duration: float = None) -> Dict:
        """获取模型推荐"""
        current_model = getattr(self, 'model_path', getattr(self, 'model_name', self.model_size))
        recommended = self.config_manager.recommend_model(self.asr_engine, audio_duration)

        return {
            "current_model": current_model,
            "recommended_model": recommended,
            "is_current_optimal": current_model == recommended,
            "recommendation_reason": self._get_recommendation_reason(recommended, audio_duration)
        }

    def _get_recommendation_reason(self, model: str, audio_duration: float = None) -> str:
        """获取推荐理由"""
        if audio_duration and audio_duration > 3600:
            return "长音频建议使用较小模型以节省处理时间"
        elif "turbo" in model.lower():
            return "Turbo 模型在速度和准确度之间提供良好平衡"
        elif "large" in model.lower():
            return "大型模型提供最佳准确度"
        elif "medium" in model.lower():
            return "中型模型在性能和资源使用之间平衡"
        else:
            return "基于当前配置的推荐"


def check_platform_compatibility():
    """检查平台兼容性"""
    info = get_platform_info()

    print(" 平台兼容性检查")
    print("=" * 30)
    print(f"操作系统: {info['system']}")
    print(f"处理器架构: {info['machine']}")

    if info["is_apple_silicon"]:
        print(" 支持 MLX Whisper (最优性能)")
        print(" 支持 OpenAI Whisper")
    elif info["is_intel_mac"]:
        print(" 不支持 MLX Whisper")
        print(" 支持 OpenAI Whisper")
    elif info["is_windows"]:
        print(" 不支持 MLX Whisper")
        print(" 支持 OpenAI Whisper")
        print(" 建议: 可考虑使用 faster-whisper 获得更好性能")
    elif info["is_linux"]:
        print(" 不支持 MLX Whisper")
        print(" 支持 OpenAI Whisper")
        print(" 建议: 如有 NVIDIA GPU，可使用 faster-whisper")

    return info


if __name__ == "__main__":
    # 检查兼容性
    platform_info = check_platform_compatibility()

    print("\n 测试 ASR 引擎...")
    try:
        asr = CrossPlatformASR()
        engine_info = asr.get_engine_info()

        print(f"\n 引擎信息:")
        print(f" 使用引擎: {engine_info['engine']}")
        print(f" 模型: {engine_info['model']}")
        print(f" 硬件优化: {'是' if engine_info['optimized'] else '否'}")

    except Exception as e:
        print(f" 引擎初始化失败: {e}")
        print("\n安装建议:")
        if platform_info["is_apple_silicon"]:
            print(" pip install mlx-whisper")
        print(" pip install openai-whisper")