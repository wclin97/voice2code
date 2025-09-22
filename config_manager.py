"""
配置管理模块
管理模型配置、用户偏好设置等
"""

import json
import os
from typing import Dict, List, Optional, Any


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_file: str = "models_config.json"):
        """
        初始化配置管理器

        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = self._load_config()
        self.user_config_file = "user_config.json"
        self.user_config = self._load_user_config()

    def _load_config(self) -> Dict:
        """加载主配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print(f"配置文件 {self.config_file} 不存在，使用默认配置")
                return self._get_default_config()
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return self._get_default_config()

    def _load_user_config(self) -> Dict:
        """加载用户自定义配置"""
        try:
            if os.path.exists(self.user_config_file):
                with open(self.user_config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"加载用户配置失败: {e}")
        return {}

    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "whisper_models": {
                "mlx": {
                    "available_models": ["mlx-community/whisper-large-v3-turbo"],
                    "default": "mlx-community/whisper-large-v3-turbo"
                },
                "openai": {
                    "available_models": ["large-v3"],
                    "default": "large-v3"
                }
            }
        }

    def get_available_whisper_models(self, engine: str) -> List[str]:
        """
        获取指定引擎的可用模型列表

        Args:
            engine: 'mlx' 或 'openai'

        Returns:
            模型列表
        """
        return self.config.get("whisper_models", {}).get(engine, {}).get("available_models", [])

    def get_default_whisper_model(self, engine: str) -> str:
        """
        获取指定引擎的默认模型

        Args:
            engine: 'mlx' 或 'openai'

        Returns:
            默认模型名称
        """
        # 检查用户配置
        user_model = self.user_config.get("preferred_models", {}).get(engine)
        if user_model:
            return user_model

        # 返回系统默认
        return self.config.get("whisper_models", {}).get(engine, {}).get("default", "")

    def get_model_description(self, engine: str, model: str) -> str:
        """
        获取模型描述

        Args:
            engine: 'mlx' 或 'openai'
            model: 模型名称

        Returns:
            模型描述
        """
        descriptions = self.config.get("whisper_models", {}).get(engine, {}).get("descriptions", {})
        return descriptions.get(model, "无描述")

    def set_preferred_model(self, engine: str, model: str):
        """
        设置用户偏好模型

        Args:
            engine: 'mlx' 或 'openai'
            model: 模型名称
        """
        if "preferred_models" not in self.user_config:
            self.user_config["preferred_models"] = {}

        self.user_config["preferred_models"][engine] = model
        self._save_user_config()
        print(f"已设置 {engine} 引擎的偏好模型为: {model}")

    def remove_custom_model(self, engine: str, model: str) -> bool:
        """
        移除自定义模型

        Args:
            engine: 'mlx' 或 'openai'
            model: 模型名称

        Returns:
            是否成功移除
        """
        try:
            import json

            # 读取当前配置
            with open(self.config_file, 'r', encoding='utf-8') as f:
                current_config = json.load(f)

            # 检查模型是否存在
            models = current_config.get("whisper_models", {}).get(engine, {}).get("available_models", [])
            if model not in models:
                return False

            # 移除模型
            current_config["whisper_models"][engine]["available_models"].remove(model)

            # 移除描述
            if model in current_config["whisper_models"][engine].get("descriptions", {}):
                del current_config["whisper_models"][engine]["descriptions"][model]

            # 保存配置文件
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(current_config, f, ensure_ascii=False, indent=2)

            # 重新加载配置
            self.config = current_config

            return True

        except Exception as e:
            print(f"移除模型失败: {e}")
            return False

    def _save_user_config(self):
        """保存用户配置"""
        try:
            with open(self.user_config_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存用户配置失败: {e}")

    def list_models_interactive(self, engine: str) -> Optional[str]:
        """
        交互式选择模型

        Args:
            engine: 'mlx' 或 'openai'

        Returns:
            选择的模型名称
        """
        models = self.get_available_whisper_models(engine)
        if not models:
            print(f"没有找到 {engine} 引擎的可用模型")
            return None

        print(f"\n{engine.upper()} Whisper 可用模型:")
        print("-" * 50)

        for i, model in enumerate(models, 1):
            description = self.get_model_description(engine, model)
            is_default = model == self.get_default_whisper_model(engine)
            default_mark = " [当前默认]" if is_default else ""
            print(f"{i}. {model}{default_mark}")
            print(f"   {description}")
            print()

        try:
            choice = input(f"请选择模型 (1-{len(models)}, 回车使用当前默认): ").strip()

            if not choice:
                return self.get_default_whisper_model(engine)

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(models):
                selected_model = models[choice_idx]

                # 询问是否设为默认
                set_default = input("是否将此模型设为默认? (y/n): ").strip().lower()
                if set_default in ['y', 'yes', '是']:
                    self.set_preferred_model(engine, selected_model)

                return selected_model
            else:
                print("选择无效")
                return None

        except (ValueError, KeyboardInterrupt):
            print("选择取消")
            return None

    def get_diarization_models(self) -> Dict[str, str]:
        """获取说话人分离模型配置"""
        return self.config.get("diarization_models", {}).get("pyannote", {})

    def get_user_preferences(self) -> Dict[str, Any]:
        """获取用户偏好设置"""
        default_prefs = self.config.get("user_preferences", {})
        user_prefs = self.user_config.get("user_preferences", {})

        # 合并配置，用户配置优先
        merged_prefs = default_prefs.copy()
        merged_prefs.update(user_prefs)
        return merged_prefs

    def should_auto_select_model(self) -> bool:
        """是否自动选择最佳模型"""
        return self.get_user_preferences().get("auto_select_best_model", True)

    def recommend_model(self, engine: str, audio_duration: float = None) -> str:
        """
        根据条件推荐合适的模型

        Args:
            engine: 'mlx' 或 'openai'
            audio_duration: 音频时长（秒）

        Returns:
            推荐的模型名称
        """
        prefs = self.get_user_preferences()
        models = self.get_available_whisper_models(engine)

        if not models:
            return self.get_default_whisper_model(engine)

        # 根据偏好选择
        if prefs.get("prefer_speed_over_accuracy", False):
            # 优先速度，选择较小模型
            for model in models:
                if any(size in model for size in ["tiny", "base", "small"]):
                    return model

        # 根据音频时长推荐
        if audio_duration:
            if audio_duration > 3600:  # 超过1小时
                print("检测到长音频，推荐使用较小模型以节省时间")
                for model in models:
                    if any(size in model for size in ["small", "medium"]):
                        return model

        # 默认返回最佳平衡的模型
        return self.get_default_whisper_model(engine)

    def show_config_summary(self):
        """显示配置摘要"""
        print("\n当前模型配置:")
        print("=" * 50)

        for engine in ["mlx", "openai"]:
            default_model = self.get_default_whisper_model(engine)
            models_count = len(self.get_available_whisper_models(engine))
            print(f"{engine.upper()} Whisper:")
            print(f"  默认模型: {default_model}")
            print(f"  可用模型: {models_count} 个")
            print()

        prefs = self.get_user_preferences()
        print("用户偏好:")
        print(f"  自动选择模型: {prefs.get('auto_select_best_model', True)}")
        print(f"  偏好速度: {prefs.get('prefer_speed_over_accuracy', False)}")
        print(f"  最大模型大小: {prefs.get('max_model_size_gb', 2.0)} GB")


def main():
    """配置管理主函数"""
    config = ConfigManager()

    print("模型配置管理")
    print("=" * 30)

    while True:
        print("\n选择操作:")
        print("1. 查看当前配置")
        print("2. 选择 MLX 模型")
        print("3. 选择 OpenAI 模型")
        print("4. 退出")

        try:
            choice = input("\n请选择 (1-4): ").strip()

            if choice == "1":
                config.show_config_summary()

            elif choice == "2":
                selected = config.list_models_interactive("mlx")
                if selected:
                    print(f"选择了 MLX 模型: {selected}")

            elif choice == "3":
                selected = config.list_models_interactive("openai")
                if selected:
                    print(f"选择了 OpenAI 模型: {selected}")

            elif choice == "4":
                print("退出配置管理")
                break

            else:
                print("无效选择")

        except KeyboardInterrupt:
            print("\n\n退出配置管理")
            break


if __name__ == "__main__":
    main()