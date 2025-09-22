"""
PyAnnote 设置向导
帮助用户设置 PyAnnote 说话人分离
"""

import subprocess
import sys
import os


def check_package_installed(package_name):
    """检查包是否已安装"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def install_package(package_name):
    """安装 Python 包"""
    try:
        print(f"正在安装 {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f" {package_name} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f" {package_name} 安装失败: {e}")
        return False


def check_huggingface_login():
    """检查是否已登录 HuggingFace"""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami()
        return user_info is not None
    except Exception:
        return False


def setup_pyannote():
    """PyAnnote 完整设置流程"""
    print(" PyAnnote 说话人分离设置向导")
    print("=" * 50)

    # 步骤1: 检查和安装依赖
    print("1. 检查依赖包...")

    required_packages = {
        "pyannote.audio": "pyannote.audio",
        "torch": "torch",
        "huggingface_hub": "huggingface_hub"
    }

    all_installed = True
    for import_name, package_name in required_packages.items():
        if check_package_installed(import_name):
            print(f" {package_name} 已安装")
        else:
            print(f" {package_name} 未安装")
            all_installed = False

    if not all_installed:
        print("\n安装缺失的依赖包...")
        for import_name, package_name in required_packages.items():
            if not check_package_installed(import_name):
                success = install_package(package_name)
                if not success:
                    print(f" 无法安装 {package_name}，请手动安装")
                    return False

    # 步骤2: HuggingFace 登录
    print("\n2. HuggingFace 认证...")

    if check_huggingface_login():
        print(" 已登录 HuggingFace")
    else:
        print(" 未登录 HuggingFace")
        print("\n请按照以下步骤登录:")
        print("1. 访问 https://huggingface.co/settings/tokens")
        print("2. 创建新的 Access Token (选择 'Read' 权限)")
        print("3. 复制 token")

        token = input("\n请输入你的 HuggingFace token: ").strip()

        if token:
            try:
                from huggingface_hub import login
                login(token=token)
                print(" HuggingFace 登录成功")
            except Exception as e:
                print(f" 登录失败: {e}")
                return False
        else:
            print(" 未提供 token")
            return False

    # 步骤3: 同意模型使用条款
    print("\n3. 模型使用条款...")
    print("请在浏览器中访问以下链接并同意使用条款:")
    print(" https://huggingface.co/pyannote/speaker-diarization-3.1")
    print(" https://huggingface.co/pyannote/segmentation-3.0")

    agreed = input("\n是否已同意使用条款? (y/n): ").strip().lower()
    if agreed not in ['y', 'yes', '是']:
        print(" 需要同意使用条款才能使用模型")
        return False

    # 步骤4: 测试模型加载
    print("\n4. 测试模型加载...")
    try:
        from pyannote_diarization import PyAnnoteDiarization
        diarizer = PyAnnoteDiarization()
        print(" PyAnnote 模型加载成功!")
        return True
    except Exception as e:
        print(f" 模型加载失败: {e}")
        print("\n可能的解决方案:")
        print("1. 确认已同意所有模型的使用条款")
        print("2. 检查网络连接")
        print("3. 重试设置过程")
        return False


def main():
    """主函数"""
    success = setup_pyannote()

    if success:
        print("\n PyAnnote 设置完成!")
        print("现在可以使用高质量的说话人分离功能了")
        print("模型已缓存到本地，后续使用完全离线")

        # 询问是否运行测试
        test = input("\n是否运行测试? (y/n): ").strip().lower()
        if test in ['y', 'yes', '是']:
            print("\n运行测试:")
            os.system("python test_mlx_whisper.py")
    else:
        print("\n PyAnnote 设置失败")
        print("可以继续使用本地聚类方法进行说话人分离")


if __name__ == "__main__":
    main()