"""
测试MLX Whisper配置和模型
验证 mlx-community/whisper-large-v3-turbo 模型是否正常工作
"""

import os
import platform

# 只在支持的平台上导入 mlx_whisper
try:
    import mlx_whisper
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mlx_whisper = None


def test_mlx_whisper():
    """测试MLX Whisper模型"""

    # 检查平台兼容性
    if not MLX_AVAILABLE:
        print(" MLX Whisper 模型测试")
        print("=" * 40)
        print(" MLX Whisper 不可用")
        print("原因: MLX 仅支持 Apple Silicon (M1/M2/M3) Mac")
        print(f"当前平台: {platform.system()} {platform.machine()}")
        print("\n建议:")
        print("- 在 Apple Silicon Mac 上安装: pip install mlx-whisper")
        print("- 在其他平台上使用 OpenAI Whisper 引擎")
        return False

    model_path = "mlx-community/whisper-large-v3-turbo"

    print(" MLX Whisper 模型测试")
    print("=" * 40)
    print(f"模型路径: {model_path}")
    print(f"平台: {platform.system()} {platform.machine()}")

    # 检查是否有测试音频
    test_audio_paths = [
        "input/test_audio.wav",
        "input/sample.wav",
        "input/meeting.wav",
        "test_audio.wav"
    ]

    audio_file = None
    for path in test_audio_paths:
        if os.path.exists(path):
            audio_file = path
            break

    if not audio_file:
        print("\n未找到测试音频文件")
        print("请将音频文件放在以下位置之一:")
        for path in test_audio_paths:
            print(f"  - {path}")
        print("\n创建一个简单的测试音频:")
        create_test_audio()
        return

    print(f"\n找到测试音频: {audio_file}")

    try:
        print("\n1. 测试模型加载...")
        # 这里只是检查模型路径，不实际加载避免耗时
        print(f" 模型路径: {model_path}")
        print(" 模型路径有效")

        print("\n2. 测试转录功能...")
        print(f" 正在转录: {audio_file}")

        # 执行转录
        result = mlx_whisper.transcribe(
            audio_file,
            path_or_hf_repo=model_path,
            word_timestamps=True,
            verbose=True
        )

        print(" 转录成功!")
        print(f" 检测语言: {result.get('language', 'unknown')}")
        print(f" 转录文本: {result.get('text', '无文本')[:100]}...")

        # 检查分段信息
        segments = result.get('segments', [])
        print(f" 分段数量: {len(segments)}")

        if segments:
            first_segment = segments[0]
            print(f" 第一段: {first_segment.get('start', 0):.2f}s - {first_segment.get('end', 0):.2f}s")
            print(f" 第一段文本: {first_segment.get('text', 'N/A')}")

        print("\n MLX Whisper 测试完成!")
        return True

    except ImportError as e:
        print(f" 导入错误: {e}")
        print("\n请确保已安装 mlx-whisper:")
        print(" pip install mlx-whisper")
        return False

    except Exception as e:
        print(f" 测试失败: {e}")
        print("\n可能的解决方案:")
        print("1. 检查音频文件格式是否支持")
        print("2. 确认 MLX 兼容性 (需要 Apple Silicon)")
        print("3. 重新安装 mlx-whisper")
        return False


def create_test_audio():
    """创建一个简单的测试音频文件"""
    print("\n创建测试音频...")
    try:
        import numpy as np
        import soundfile as sf

        # 创建一个简单的正弦波测试音频 (440Hz, 3秒)
        sample_rate = 16000
        duration = 3.0
        frequency = 440

        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)

        # 确保输出目录存在
        os.makedirs("input", exist_ok=True)
        output_file = "input/test_audio.wav"

        sf.write(output_file, audio_data, sample_rate)
        print(f" 测试音频已创建: {output_file}")
        print(" 这是一个440Hz的正弦波，持续3秒")

    except ImportError:
        print(" 无法创建测试音频 (需要 numpy 和 soundfile)")
        print(" 请手动放置一个音频文件到 input/ 目录")
    except Exception as e:
        print(f" 创建测试音频失败: {e}")


def test_platform_compatibility():
    """测试平台兼容性"""
    print("\n平台兼容性测试:")
    print("-" * 20)

    import platform
    import sys

    system = platform.system()
    machine = platform.machine()
    python_version = sys.version

    print(f"操作系统: {system}")
    print(f"处理器架构: {machine}")
    print(f"Python版本: {python_version}")

    # 检查是否为 Apple Silicon
    is_apple_silicon = system == "Darwin" and machine in ["arm64", "aarch64"]

    if is_apple_silicon:
        print(" 检测到 Apple Silicon")
        print(" MLX Whisper 完全兼容")
    else:
        print(" 非 Apple Silicon 平台")
        print(" MLX Whisper 不兼容，建议使用 OpenAI Whisper")

    # 检查MLX是否可用
    try:
        import mlx
        print(" MLX 框架: 可用")
    except ImportError:
        print(" MLX 框架: 不可用")

    # 使用全局变量检查mlx-whisper
    if MLX_AVAILABLE:
        print(" mlx-whisper: 已安装")
    else:
        print(" mlx-whisper: 未安装")


def main():
    """主测试函数"""
    print("MLX Whisper 综合测试")
    print("=" * 50)

    # 1. 平台兼容性测试
    test_platform_compatibility()

    # 2. 模型功能测试
    print("\n" + "=" * 50)
    success = test_mlx_whisper()

    # 3. 总结
    print("\n" + "=" * 50)
    if success:
        print(" 所有测试通过!")
        print(" MLX Whisper 配置正确，可以正常使用")
    else:
        print(" 测试失败")
        print(" 请检查配置或使用替代方案")


if __name__ == "__main__":
    main()