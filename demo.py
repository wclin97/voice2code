"""
会议转录系统 - 主入口
提供完整的菜单系统，可以访问所有功能模块
"""

import os
import sys
import subprocess
from meeting_transcriber import MeetingTranscriber
from cross_platform_asr import check_platform_compatibility


def clear_screen():
    """清屏函数"""
    os.system('cls' if os.name == 'nt' else 'clear')


def show_main_menu():
    """显示主菜单"""
    clear_screen()
    print("=" * 60)
    print("           语音转录系统")
    print("=" * 60)
    print()
    print("主菜单:")
    print()
    print("  核心功能")
    print("    1. 开始会议转录")
    print("    2. 平台兼容性检查")
    print()
    print("  配置管理")
    print("    3. Whisper 模型管理")
    print("    4. PyAnnote 设置向导")
    print("    5. HuggingFace Token 助手")
    print()
    print("  测试工具")
    print("    6. 测试 MLX Whisper")
    print("    7. 系统配置概览")
    print()
    print("  帮助与信息")
    print("    8. 查看使用指南")
    print("    9. 关于本程序")
    print()
    print("    0. 退出程序")
    print()


def start_meeting_transcription():
    """开始会议转录"""
    clear_screen()
    print("=" * 50)
    print("         会议转录")
    print("=" * 50)

    # 显示平台兼容性信息
    check_platform_compatibility()
    print()

    # 询问Whisper模型选择
    print("选择 Whisper 模型:")
    print("1. 自动选择 (使用配置文件默认)")
    print("2. 交互式选择 (显示所有可用模型)")
    print("3. 使用当前配置")

    while True:
        try:
            choice = input("\n请选择模型方式 (1-3): ").strip()
            if choice == "1":
                whisper_model = "auto"
                break
            elif choice == "2":
                whisper_model = "interactive"
                break
            elif choice == "3":
                whisper_model = "auto"
                break
            else:
                print("请输入 1、2 或 3")
        except KeyboardInterrupt:
            print("\n返回主菜单")
            return

    # 询问说话人分离方法
    print("\n选择说话人分离方法:")
    print("1. 自动选择 (PyAnnote优先，回退到本地)")
    print("2. PyAnnote (高质量，需要token)")
    print("3. PyAnnote 快速模式 (长音频分段处理)")
    print("4. 本地聚类 (无需token，效果一般，速度快)")

    while True:
        try:
            choice = input("\n请选择方法 (1-4): ").strip()
            if choice == "1":
                diarization_method = "auto"
                break
            elif choice == "2":
                diarization_method = "pyannote"
                break
            elif choice == "3":
                diarization_method = "pyannote_fast"
                break
            elif choice == "4":
                diarization_method = "local"
                break
            else:
                print("请输入 1、2、3 或 4")
        except KeyboardInterrupt:
            print("\n返回主菜单")
            return

    # 初始化转录器
    print("\n正在初始化转录器...")
    try:
        transcriber = MeetingTranscriber(
            whisper_model=whisper_model,
            diarization_method=diarization_method,
            window_length=2.0,
            hop_length=1.0
        )
    except Exception as e:
        print(f"初始化失败: {e}")
        input("\n按回车键返回主菜单...")
        return

    # 处理音频文件
    _process_audio_files(transcriber)


def _process_audio_files(transcriber):
    """处理音频文件"""
    input_dir = "input"

    # 创建input文件夹（如果不存在）
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"创建了 {input_dir} 文件夹")

    # 获取input文件夹中的所有音频文件
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    audio_files = []

    if os.path.exists(input_dir):
        for file in os.listdir(input_dir):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(input_dir, file))

    if not audio_files:
        print(f"\n未在 {input_dir} 文件夹中找到音频文件")
        print("请将音频文件放在 input/ 文件夹下，支持的格式:")
        print("  - WAV (.wav)")
        print("  - MP3 (.mp3)")
        print("  - M4A (.m4a)")
        print("  - FLAC (.flac)")
        print("  - OGG (.ogg)")
        input("\n按回车键返回主菜单...")
        return

    # 选择音频文件
    if len(audio_files) == 1:
        audio_file = audio_files[0]
    else:
        print(f"\n在 {input_dir} 文件夹中找到 {len(audio_files)} 个音频文件:")
        for i, file in enumerate(audio_files, 1):
            filename = os.path.basename(file)
            print(f"  {i}. {filename}")

        while True:
            try:
                choice = input(f"\n请选择要转录的文件 (1-{len(audio_files)}): ").strip()
                if choice.lower() in ['q', 'quit', 'exit']:
                    return

                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(audio_files):
                    audio_file = audio_files[choice_idx]
                    break
                else:
                    print(f"请输入 1-{len(audio_files)} 之间的数字")
            except ValueError:
                print("请输入有效的数字")
            except KeyboardInterrupt:
                print("\n返回主菜单")
                return

    # 执行转录
    _run_transcription(transcriber, audio_file)


def _run_transcription(transcriber, audio_file):
    """执行转录过程"""
    try:
        import time
        demo_start_time = time.time()

        print(f"\n开始处理音频文件: {os.path.basename(audio_file)}")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        result = transcriber.transcribe_meeting(audio_file)

        if "error" in result:
            print(f"\n转录失败: {result['error']}")
            input("\n按回车键返回主菜单...")
            return

        # 显示转录结果摘要
        _print_summary(result)

        # 保存结果
        _save_results(result, audio_file)

        # 显示运行时间统计
        demo_total_time = time.time() - demo_start_time
        _print_timing_stats(result, demo_start_time, demo_total_time)

        print("\n转录完成！")
        input("\n按回车键返回主菜单...")

    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        print("请检查:")
        print("1. 是否已安装所有依赖: pip install -r requirements.txt")
        print("2. 音频文件格式是否支持")
        print("3. 模型是否可用")
        input("\n按回车键返回主菜单...")


def _save_results(result, audio_file):
    """保存转录结果"""
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = os.path.splitext(os.path.basename(audio_file))[0]

    # 保存为文本格式
    txt_file = os.path.join(output_dir, f"{base_name}_transcript.txt")
    from meeting_transcriber import MeetingTranscriber
    transcriber = MeetingTranscriber()
    transcriber.export_to_txt(result, txt_file)

    # 保存为JSON格式
    json_file = os.path.join(output_dir, f"{base_name}_transcript.json")
    transcriber.export_to_json(result, json_file)

    # 保存为CSV格式
    csv_file = os.path.join(output_dir, f"{base_name}_transcript.csv")
    transcriber.export_to_csv(result, csv_file)

    print(f"\n结果已保存到 {output_dir}/ 文件夹:")
    print(f"  文本格式: {txt_file}")
    print(f"  JSON格式: {json_file}")
    print(f"  CSV格式: {csv_file}")


def _print_summary(result):
    """打印转录结果摘要"""
    info = result["meeting_info"]

    print(f"\n会议转录摘要:")
    print("-" * 30)
    print(f"总时长: {_format_time(info['total_duration'])}")
    print(f"说话人数: {info['total_speakers']}")
    print(f"语言: {info['language']}")
    print(f"转录片段数: {len(result['timeline'])}")

    if "processing_time" in result:
        proc_time = result["processing_time"]
        print(f"处理耗时: {proc_time['total']:.1f}秒")
        efficiency_ratio = info['total_duration'] / proc_time['total']
        print(f"处理效率: {efficiency_ratio:.1f}x 实时")

    print(f"\n说话人统计:")
    for speaker, stats in result["speaker_statistics"].items():
        duration_str = _format_time(stats["total_duration"])
        percentage = (stats["total_duration"] / info["total_duration"]) * 100
        print(f"  {speaker}: {duration_str} ({percentage:.1f}%)")

    print(f"\n部分转录内容预览:")
    for item in result["timeline"][:3]:
        print(f"  [{item['timestamp']}] {item['speaker']}: {item['text'][:50]}...")


def _print_timing_stats(result, demo_start_time, demo_total_time):
    """打印时间统计"""
    import time

    print(f"\n运行时长统计")
    print("=" * 30)
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(demo_start_time))}")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总运行时长: {demo_total_time:.1f}秒 ({demo_total_time/60:.1f}分钟)")

    if "processing_time" in result:
        proc_time = result["processing_time"]
        print(f"\n处理时间详情:")
        print(f"语音转文字: {proc_time['asr']:.1f}秒")
        print(f"说话人分离: {proc_time['diarization']:.1f}秒")
        print(f"对齐处理: {proc_time['alignment']:.1f}秒")
        print(f"生成纪要: {proc_time['summary']:.1f}秒")

        audio_duration = result["meeting_info"]["total_duration"]
        efficiency_ratio = audio_duration / proc_time['total']
        print(f"\n处理效率:")
        print(f"音频时长: {_format_time(audio_duration)}")
        print(f"处理时间: {proc_time['total']:.1f}秒")
        print(f"效率比: {efficiency_ratio:.1f}x (处理速度是实时的{efficiency_ratio:.1f}倍)")


def _format_time(seconds):
    """格式化时间显示"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def run_script(script_name, description):
    """运行指定的Python脚本"""
    clear_screen()
    print(f"启动 {description}...")
    print("-" * 40)

    if not os.path.exists(script_name):
        print(f"脚本文件 {script_name} 不存在")
        input("\n按回车键返回主菜单...")
        return

    try:
        # 使用当前Python解释器运行脚本
        result = subprocess.run([sys.executable, script_name],
                              capture_output=False,
                              text=True)

        if result.returncode != 0:
            print(f"\n脚本执行完成，返回码: {result.returncode}")

    except KeyboardInterrupt:
        print("\n\n脚本被用户中断")
    except Exception as e:
        print(f"\n执行脚本时出错: {e}")

    input("\n按回车键返回主菜单...")


def show_platform_compatibility():
    """显示平台兼容性检查"""
    clear_screen()
    print("=" * 50)
    print("         平台兼容性检查")
    print("=" * 50)

    check_platform_compatibility()

    input("\n按回车键返回主菜单...")


def show_usage_guide():
    """显示使用指南"""
    guide_file = "MODEL_CONFIG_GUIDE.md"

    clear_screen()
    print("=" * 50)
    print("         使用指南")
    print("=" * 50)

    if os.path.exists(guide_file):
        print(f"详细指南请查看: {guide_file}")
        print("\n快速指南:")
        print()
        print("1. 基本使用:")
        print("   - 将音频文件放入 input/ 文件夹")
        print("   - 选择菜单选项 1 开始转录")
        print("   - 结果保存在 output/ 文件夹")
        print()
        print("2. 配置管理:")
        print("   - 选择菜单选项 3 管理 Whisper 模型")
        print("     > 选择模型管理器中的选项 5 添加自定义模型")
        print("   - 选择菜单选项 4 设置 PyAnnote")
        print("   - 选择菜单选项 5 配置 HuggingFace Token")
        print()
        print("3. 测试工具:")
        print("   - 选择菜单选项 6 测试 MLX Whisper")
        print("   - 选择菜单选项 2 检查平台兼容性")
        print()
        print("4. 文件结构:")
        print("   - input/     - 放置音频文件")
        print("   - output/    - 转录结果输出")
        print("   - models_config.json - 模型配置")
        print("   - user_config.json   - 用户偏好")
    else:
        print(f"指南文件 {guide_file} 不存在")

    input("\n按回车键返回主菜单...")


def show_about():
    """显示关于信息"""
    clear_screen()
    print("=" * 50)
    print("         关于本程序")
    print("=" * 50)
    print()
    print("语音转录系统")
    print("版本: 2.0")
    print()
    print("主要功能:")
    print("  • 本地化语音转录 (支持 Whisper)")
    print("  • 智能说话人分离 (PyAnnote + 本地聚类)")
    print("  • 跨平台支持 (Apple Silicon 优化)")
    print("  • 灵活的配置管理")
    print("  • 详细的处理统计")
    print()
    print("支持的模型:")
    print("  • MLX Whisper (Apple Silicon)")
    print("  • OpenAI Whisper (跨平台)")
    print("  • PyAnnote 说话人分离")
    print("  • 本地聚类算法")
    print()
    print("支持的格式:")
    print("  • 输入: WAV, MP3, M4A, FLAC, OGG")
    print("  • 输出: TXT, JSON, CSV")
    print()
    print("相关技术:")
    print("  • MLX (Apple 机器学习框架)")
    print("  • OpenAI Whisper")
    print("  • PyAnnote Audio")
    print("  • scikit-learn")
    print("  • HuggingFace Transformers")

    input("\n按回车键返回主菜单...")


def show_system_overview():
    """显示系统配置概览"""
    clear_screen()
    print("=" * 50)
    print("         系统配置概览")
    print("=" * 50)

    # 导入配置管理器
    try:
        from config_manager import ConfigManager
        config = ConfigManager()
        config.show_config_summary()
    except Exception as e:
        print(f"无法加载配置: {e}")

    print()

    # 显示平台信息
    check_platform_compatibility()

    input("\n按回车键返回主菜单...")


def main():
    """主函数"""
    try:
        while True:
            show_main_menu()

            try:
                choice = input("请选择操作 (0-9): ").strip()

                if choice == "0":
                    print("\n感谢使用语音转录系统！")
                    break

                elif choice == "1":
                    start_meeting_transcription()

                elif choice == "2":
                    show_platform_compatibility()

                elif choice == "3":
                    run_script("model_manager.py", "Whisper 模型管理")

                elif choice == "4":
                    run_script("setup_pyannote.py", "PyAnnote 设置向导")

                elif choice == "5":
                    run_script("token_helper.py", "HuggingFace Token 助手")

                elif choice == "6":
                    run_script("test_mlx_whisper.py", "MLX Whisper 测试")

                elif choice == "7":
                    show_system_overview()

                elif choice == "8":
                    show_usage_guide()

                elif choice == "9":
                    show_about()

                else:
                    print("无效选择，请输入 0-9")
                    input("\n按回车键继续...")

            except KeyboardInterrupt:
                choice = input("\n\n确定要退出吗? (y/n): ").strip().lower()
                if choice in ['y', 'yes', '是']:
                    break

    except KeyboardInterrupt:
        print("\n\n再见！")


if __name__ == "__main__":
    main()