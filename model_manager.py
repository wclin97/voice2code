#!/usr/bin/env python3
"""
模型管理工具
用于管理 Whisper 模型配置和偏好设置
"""

import sys
import os
from config_manager import ConfigManager
from cross_platform_asr import CrossPlatformASR, check_platform_compatibility


def clear_screen():
    """清屏函数"""
    os.system('cls' if os.name == 'nt' else 'clear')


def show_main_menu():
    """显示主菜单"""
    clear_screen()
    print("=" * 50)
    print("         Whisper 模型管理工具")
    print("=" * 50)
    print()
    print("选择操作:")
    print("1. 查看当前配置")
    print("2. 列出可用模型")
    print("3. 测试ASR引擎")
    print("4. 切换默认模型")
    print("5. 添加自定义模型")
    print("6. 移除自定义模型")
    print("7. 平台兼容性检查")
    print("8. 配置管理")
    print("0. 退出")
    print()


def show_current_config():
    """显示当前配置"""
    config = ConfigManager()
    clear_screen()
    print("=" * 50)
    print("         当前配置信息")
    print("=" * 50)

    config.show_config_summary()

    # 显示平台信息
    platform_info = check_platform_compatibility()
    print()


def list_available_models():
    """列出可用模型"""
    config = ConfigManager()
    clear_screen()
    print("=" * 50)
    print("         可用模型列表")
    print("=" * 50)

    for engine in ["mlx", "openai"]:
        print(f"\n{engine.upper()} Whisper 模型:")
        print("-" * 30)

        models = config.get_available_whisper_models(engine)
        default_model = config.get_default_whisper_model(engine)

        for i, model in enumerate(models, 1):
            description = config.get_model_description(engine, model)
            is_default = model == default_model
            default_mark = " [默认]" if is_default else ""
            print(f"{i}. {model}{default_mark}")
            print(f"   {description}")
            print()


def test_asr_engine():
    """测试ASR引擎"""
    clear_screen()
    print("=" * 50)
    print("         测试 ASR 引擎")
    print("=" * 50)

    print("\n选择测试模式:")
    print("1. 使用默认配置")
    print("2. 交互式选择模型")
    print("3. 返回主菜单")

    while True:
        try:
            choice = input("\n请选择 (1-3): ").strip()

            if choice == "1":
                print("\n正在使用默认配置初始化 ASR 引擎...")
                try:
                    asr = CrossPlatformASR("auto")
                    _show_engine_info(asr)
                    return
                except Exception as e:
                    print(f"初始化失败: {e}")
                    return

            elif choice == "2":
                print("\n正在启动交互式模型选择...")
                try:
                    asr = CrossPlatformASR("interactive")
                    _show_engine_info(asr)
                    return
                except Exception as e:
                    print(f"初始化失败: {e}")
                    return

            elif choice == "3":
                return

            else:
                print("请输入 1、2 或 3")

        except KeyboardInterrupt:
            print("\n返回主菜单")
            return


def _show_engine_info(asr):
    """显示引擎信息"""
    info = asr.get_engine_info()

    print("\n" + "-" * 40)
    print("ASR 引擎信息:")
    print("-" * 40)
    print(f"平台: {info['platform']['system']} {info['platform']['machine']}")
    print(f"引擎: {info['engine'].upper()}")
    print(f"当前模型: {info['model']}")
    print(f"模型描述: {info['model_description']}")
    print(f"硬件优化: {'是' if info['optimized'] else '否'}")
    print(f"可用模型数量: {len(info['available_models'])}")

    # 询问是否查看或切换模型
    print("\n其他操作:")
    print("1. 查看所有可用模型")
    print("2. 切换模型")
    print("3. 返回")

    while True:
        try:
            choice = input("\n请选择 (1-3): ").strip()

            if choice == "1":
                asr.list_available_models()
                break

            elif choice == "2":
                success = asr.switch_model_interactive()
                if success:
                    print("\n更新后的引擎信息:")
                    _show_engine_info(asr)
                break

            elif choice == "3":
                break

            else:
                print("请输入 1、2 或 3")

        except KeyboardInterrupt:
            break


def switch_default_model():
    """切换默认模型"""
    config = ConfigManager()
    clear_screen()
    print("=" * 50)
    print("         切换默认模型")
    print("=" * 50)

    print("\n选择引擎:")
    print("1. MLX Whisper (Apple Silicon)")
    print("2. OpenAI Whisper (跨平台)")
    print("3. 返回主菜单")

    while True:
        try:
            choice = input("\n请选择引擎 (1-3): ").strip()

            if choice == "1":
                selected = config.list_models_interactive("mlx")
                if selected:
                    print(f"\nMLX 引擎默认模型已设置为: {selected}")
                break

            elif choice == "2":
                selected = config.list_models_interactive("openai")
                if selected:
                    print(f"\nOpenAI 引擎默认模型已设置为: {selected}")
                break

            elif choice == "3":
                break

            else:
                print("请输入 1、2 或 3")

        except KeyboardInterrupt:
            print("\n返回主菜单")
            break


def add_custom_model():
    """添加自定义模型"""
    config = ConfigManager()
    clear_screen()
    print("=" * 50)
    print("         添加自定义模型")
    print("=" * 50)

    print("\n选择引擎类型:")
    print("1. MLX Whisper (Apple Silicon)")
    print("2. OpenAI Whisper (跨平台)")
    print("3. 返回主菜单")

    while True:
        try:
            choice = input("\n请选择引擎 (1-3): ").strip()

            if choice == "1":
                _add_custom_mlx_model(config)
                break
            elif choice == "2":
                _add_custom_openai_model(config)
                break
            elif choice == "3":
                break
            else:
                print("请输入 1、2 或 3")

        except KeyboardInterrupt:
            print("\n返回主菜单")
            break


def _add_custom_mlx_model(config):
    """添加自定义 MLX 模型"""
    print("\n添加自定义 MLX Whisper 模型")
    print("-" * 40)
    print("\nMLX 模型通常格式为: mlx-community/whisper-xxx")
    print("或 HuggingFace 上的其他 MLX 格式模型")
    print("\n示例:")
    print("  mlx-community/whisper-tiny")
    print("  mlx-community/whisper-base")
    print("  your-username/custom-mlx-whisper")

    while True:
        try:
            model_name = input("\n请输入 MLX 模型名称: ").strip()

            if not model_name:
                print("模型名称不能为空")
                continue

            description = input("请输入模型描述 (可选): ").strip()
            if not description:
                description = f"用户自定义模型 ({model_name})"

            # 添加到配置
            success = _add_model_to_config(config, "mlx", model_name, description)

            if success:
                print(f"\n自定义 MLX 模型已添加: {model_name}")

                # 询问是否设为默认
                set_default = input("是否将此模型设为默认? (y/n): ").strip().lower()
                if set_default in ['y', 'yes', '是']:
                    config.set_preferred_model("mlx", model_name)

                # 询问是否测试
                test_model = input("是否立即测试此模型? (y/n): ").strip().lower()
                if test_model in ['y', 'yes', '是']:
                    _test_custom_model("mlx", model_name)

            break

        except KeyboardInterrupt:
            print("\n取消添加模型")
            break


def _add_custom_openai_model(config):
    """添加自定义 OpenAI 模型"""
    print("\n添加自定义 OpenAI Whisper 模型")
    print("-" * 40)
    print("\nOpenAI Whisper 支持的模型:")
    print("  tiny, base, small, medium, large, large-v2, large-v3")
    print("  或本地模型文件路径")

    while True:
        try:
            model_name = input("\n请输入 OpenAI 模型名称: ").strip()

            if not model_name:
                print("模型名称不能为空")
                continue

            description = input("请输入模型描述 (可选): ").strip()
            if not description:
                if model_name in ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]:
                    size_map = {
                        "tiny": "最小模型 (~40MB)",
                        "base": "基础模型 (~150MB)",
                        "small": "小型模型 (~250MB)",
                        "medium": "中型模型 (~770MB)",
                        "large": "大型模型 (~1.5GB)",
                        "large-v2": "大型模型v2 (~1.5GB)",
                        "large-v3": "大型模型v3 (~1.5GB)"
                    }
                    description = size_map.get(model_name, f"OpenAI Whisper {model_name} 模型")
                else:
                    description = f"用户自定义模型 ({model_name})"

            # 添加到配置
            success = _add_model_to_config(config, "openai", model_name, description)

            if success:
                print(f"\n自定义 OpenAI 模型已添加: {model_name}")

                # 询问是否设为默认
                set_default = input("是否将此模型设为默认? (y/n): ").strip().lower()
                if set_default in ['y', 'yes', '是']:
                    config.set_preferred_model("openai", model_name)

                # 询问是否测试
                test_model = input("是否立即测试此模型? (y/n): ").strip().lower()
                if test_model in ['y', 'yes', '是']:
                    _test_custom_model("openai", model_name)

            break

        except KeyboardInterrupt:
            print("\n取消添加模型")
            break


def _add_model_to_config(config, engine, model_name, description):
    """将模型添加到配置文件"""
    try:
        import json

        # 读取当前配置
        if hasattr(config, 'config'):
            current_config = config.config
        else:
            with open(config.config_file, 'r', encoding='utf-8') as f:
                current_config = json.load(f)

        # 检查模型是否已存在
        existing_models = current_config.get("whisper_models", {}).get(engine, {}).get("available_models", [])
        if model_name in existing_models:
            print(f"模型 {model_name} 已存在")
            return True

        # 添加到可用模型列表
        if engine not in current_config.get("whisper_models", {}):
            current_config["whisper_models"][engine] = {"available_models": [], "descriptions": {}}

        current_config["whisper_models"][engine]["available_models"].append(model_name)
        current_config["whisper_models"][engine]["descriptions"][model_name] = description

        # 保存配置文件
        with open(config.config_file, 'w', encoding='utf-8') as f:
            json.dump(current_config, f, ensure_ascii=False, indent=2)

        # 重新加载配置
        config.config = current_config

        return True

    except Exception as e:
        print(f"添加模型失败: {e}")
        return False


def _test_custom_model(engine, model_name):
    """测试自定义模型"""
    print(f"\n测试模型: {engine} - {model_name}")
    print("-" * 40)

    try:
        from cross_platform_asr import CrossPlatformASR

        print("正在初始化 ASR 引擎...")
        asr = CrossPlatformASR(model_name)

        info = asr.get_engine_info()
        print(f"\n引擎信息:")
        print(f"  平台: {info['platform']['system']} {info['platform']['machine']}")
        print(f"  引擎: {info['engine'].upper()}")
        print(f"  模型: {info['model']}")
        print(f"  描述: {info.get('model_description', '无描述')}")
        print(f"  硬件优化: {'是' if info['optimized'] else '否'}")

        print(f"\n模型测试成功!")

    except Exception as e:
        print(f"模型测试失败: {e}")
        print("\n可能的原因:")
        print("1. 模型名称不正确")
        print("2. 网络连接问题 (首次下载)")
        print("3. 平台不兼容 (MLX 仅支持 Apple Silicon)")
        print("4. 模型不存在或权限问题")


def remove_custom_model():
    """移除自定义模型"""
    config = ConfigManager()
    clear_screen()
    print("=" * 50)
    print("         移除自定义模型")
    print("=" * 50)

    print("\n选择引擎类型:")
    print("1. MLX Whisper (Apple Silicon)")
    print("2. OpenAI Whisper (跨平台)")
    print("3. 返回主菜单")

    while True:
        try:
            choice = input("\n请选择引擎 (1-3): ").strip()

            if choice == "1":
                _remove_engine_models(config, "mlx")
                break
            elif choice == "2":
                _remove_engine_models(config, "openai")
                break
            elif choice == "3":
                break
            else:
                print("请输入 1、2 或 3")

        except KeyboardInterrupt:
            print("\n返回主菜单")
            break


def _remove_engine_models(config, engine):
    """移除指定引擎的模型"""
    models = config.get_available_whisper_models(engine)
    default_model = config.get_default_whisper_model(engine)

    if not models:
        print(f"\n{engine.upper()} 引擎没有可用模型")
        return

    print(f"\n{engine.upper()} Whisper 可用模型:")
    print("-" * 40)

    for i, model in enumerate(models, 1):
        description = config.get_model_description(engine, model)
        is_default = model == default_model
        default_mark = " [默认]" if is_default else ""
        print(f"{i}. {model}{default_mark}")
        print(f"   {description}")
        print()

    while True:
        try:
            choice = input(f"\n请选择要移除的模型 (1-{len(models)}, 或按回车返回): ").strip()

            if not choice:
                return

            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(models):
                    model_to_remove = models[choice_idx]

                    # 检查是否为默认模型
                    if model_to_remove == default_model:
                        print(f"\n警告: {model_to_remove} 是当前默认模型")
                        confirm = input("确定要移除吗? 这可能会影响系统正常运行 (y/n): ").strip().lower()
                        if confirm not in ['y', 'yes', '是']:
                            print("取消移除")
                            continue

                    # 确认移除
                    confirm = input(f"\n确定要移除模型 {model_to_remove} 吗? (y/n): ").strip().lower()
                    if confirm in ['y', 'yes', '是']:
                        success = config.remove_custom_model(engine, model_to_remove)
                        if success:
                            print(f"\n模型 {model_to_remove} 已移除")

                            # 如果移除的是默认模型，提示设置新的默认模型
                            if model_to_remove == default_model:
                                remaining_models = config.get_available_whisper_models(engine)
                                if remaining_models:
                                    print(f"\n请选择新的默认模型:")
                                    for i, model in enumerate(remaining_models, 1):
                                        print(f"{i}. {model}")

                                    new_choice = input(f"\n请选择 (1-{len(remaining_models)}): ").strip()
                                    try:
                                        new_idx = int(new_choice) - 1
                                        if 0 <= new_idx < len(remaining_models):
                                            new_default = remaining_models[new_idx]
                                            config.set_preferred_model(engine, new_default)
                                    except ValueError:
                                        print("无效选择，请手动设置默认模型")
                        else:
                            print(f"\n移除模型失败")
                    else:
                        print("取消移除")

                    break
                else:
                    print(f"请输入 1-{len(models)} 之间的数字")
            except ValueError:
                print("请输入有效的数字")

        except KeyboardInterrupt:
            print("\n取消操作")
            break


def show_config_management():
    """配置管理子菜单"""
    clear_screen()
    print("=" * 50)
    print("         配置管理")
    print("=" * 50)

    print("\n选择操作:")
    print("1. 查看配置文件位置")
    print("2. 重置用户配置")
    print("3. 导出当前配置")
    print("4. 返回主菜单")

    while True:
        try:
            choice = input("\n请选择 (1-4): ").strip()

            if choice == "1":
                _show_config_files()
                break

            elif choice == "2":
                _reset_user_config()
                break

            elif choice == "3":
                _export_config()
                break

            elif choice == "4":
                break

            else:
                print("请输入 1、2、3 或 4")

        except KeyboardInterrupt:
            print("\n返回主菜单")
            break


def _show_config_files():
    """显示配置文件位置"""
    print("\n配置文件位置:")
    print("-" * 30)
    print("主配置文件: models_config.json")
    print("用户配置文件: user_config.json")
    print("说明: 用户配置会覆盖主配置中的相同设置")


def _reset_user_config():
    """重置用户配置"""
    import os
    user_config_file = "user_config.json"

    if os.path.exists(user_config_file):
        confirm = input(f"\n确定要删除用户配置文件 {user_config_file} 吗? (y/n): ").strip().lower()
        if confirm in ['y', 'yes', '是']:
            try:
                os.remove(user_config_file)
                print(f" 用户配置文件已删除，将使用默认配置")
            except Exception as e:
                print(f" 删除失败: {e}")
        else:
            print(" 取消重置")
    else:
        print(f" 用户配置文件 {user_config_file} 不存在")


def _export_config():
    """导出当前配置"""
    config = ConfigManager()
    import json
    from datetime import datetime

    export_data = {
        "export_time": datetime.now().isoformat(),
        "platform": check_platform_compatibility(),
        "whisper_models": {
            "mlx": {
                "default": config.get_default_whisper_model("mlx"),
                "available": config.get_available_whisper_models("mlx")
            },
            "openai": {
                "default": config.get_default_whisper_model("openai"),
                "available": config.get_available_whisper_models("openai")
            }
        },
        "user_preferences": config.get_user_preferences()
    }

    filename = f"config_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        print(f"\n 配置已导出到: {filename}")
    except Exception as e:
        print(f" 导出失败: {e}")


def main():
    """主函数"""
    try:
        while True:
            show_main_menu()

            try:
                choice = input("请选择操作 (0-6): ").strip()

                if choice == "0":
                    print("\n感谢使用模型管理工具!")
                    break

                elif choice == "1":
                    show_current_config()

                elif choice == "2":
                    list_available_models()

                elif choice == "3":
                    test_asr_engine()

                elif choice == "4":
                    switch_default_model()

                elif choice == "5":
                    add_custom_model()

                elif choice == "6":
                    remove_custom_model()

                elif choice == "7":
                    check_platform_compatibility()

                elif choice == "8":
                    show_config_management()

                else:
                    print("无效选择，请输入 0-8")

            except KeyboardInterrupt:
                choice = input("\n\n确定要退出吗? (y/n): ").strip().lower()
                if choice in ['y', 'yes', '是']:
                    break

    except KeyboardInterrupt:
        print("\n\n再见!")


if __name__ == "__main__":
    main()