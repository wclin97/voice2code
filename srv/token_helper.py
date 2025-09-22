"""
HuggingFace Token 助手
帮助用户管理和使用 HuggingFace token
"""

import os
import sys


def get_token_instructions():
    """获取token的详细说明"""
    instructions = """
HuggingFace Token 获取和使用指南

1. 获取 Token:
   • 访问: https://huggingface.co/settings/tokens
   • 点击 "New token"
   • 类型选择 "Read"
   • 复制生成的 token

2. 使用 Token (选择任一方式):

   方式1: 命令行登录 (推荐)
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   # 输入你的 token
   ```

   方式2: 环境变量
   ```bash
   export HUGGINGFACE_HUB_TOKEN="your_token_here"
   ```

   方式3: 代码中直接传入
   ```python
   from pyannote_diarization import PyAnnoteDiarization
   diarizer = PyAnnoteDiarization(token="your_token_here")
   ```

3. 同意模型使用条款:
   • https://huggingface.co/pyannote/speaker-diarization-3.1
   • https://huggingface.co/pyannote/segmentation-3.0

4. 完成后模型会缓存到本地，后续使用完全离线!
"""
    return instructions


def check_token_status():
    """检查token状态"""
    print("检查 HuggingFace Token 状态...")

    # 检查是否已登录
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami()
        if user_info:
            print(f" 已登录用户: {user_info.get('name', 'Unknown')}")
            return True
    except Exception:
        pass

    # 检查环境变量
    env_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
    if env_token:
        print(" 找到环境变量中的 token")
        return True

    print(" 未找到有效的 token")
    return False


def setup_token_interactive():
    """交互式token设置"""
    print(" 交互式 Token 设置")
    print("-" * 30)

    if check_token_status():
        print("Token 已配置，可以直接使用 PyAnnote!")
        return True

    print("\n" + get_token_instructions())

    choice = input("\n选择设置方式 (1: 命令行登录, 2: 环境变量, 3: 跳过): ").strip()

    if choice == "1":
        print("\n运行以下命令:")
        print("huggingface-cli login")
        print("然后重新运行程序")
        return False

    elif choice == "2":
        token = input("\n请输入你的 HuggingFace token: ").strip()
        if token:
            # 设置环境变量 (当前会话)
            os.environ['HUGGINGFACE_HUB_TOKEN'] = token
            print(" Token 已设置为环境变量")

            # 提示永久设置
            shell = os.getenv('SHELL', '/bin/bash')
            if 'zsh' in shell:
                config_file = "~/.zshrc"
            else:
                config_file = "~/.bashrc"

            print(f"\n 要永久保存，请添加到 {config_file}:")
            print(f'export HUGGINGFACE_HUB_TOKEN="{token}"')
            return True
        else:
            print(" 未提供 token")
            return False

    elif choice == "3":
        print("跳过 token 设置，将使用本地聚类方法")
        return False

    else:
        print(" 无效选择")
        return False


def save_token_to_file(token: str, file_path: str = ".env"):
    """保存token到文件"""
    try:
        with open(file_path, 'w') as f:
            f.write(f"HUGGINGFACE_HUB_TOKEN={token}\n")
        print(f" Token 已保存到 {file_path}")
        return True
    except Exception as e:
        print(f" 保存失败: {e}")
        return False


def load_token_from_file(file_path: str = ".env"):
    """从文件加载token"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('HUGGINGFACE_HUB_TOKEN='):
                        token = line.split('=', 1)[1].strip()
                        os.environ['HUGGINGFACE_HUB_TOKEN'] = token
                        return token
    except Exception as e:
        print(f" 加载失败: {e}")
    return None


if __name__ == "__main__":
    print(" HuggingFace Token 助手")
    print("=" * 40)

    # 检查当前状态
    if check_token_status():
        print("\n Token 已配置，可以使用 PyAnnote!")
    else:
        print("\n需要设置 HuggingFace Token")
        setup_token_interactive()