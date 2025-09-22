# Whisper 模型配置指南

## 概述

现在所有的 Whisper 模型都通过配置文件进行管理，您可以轻松切换不同的模型而无需修改代码。

## 🚀 快速开始

### 1. 使用模型管理工具
```bash
python model_manager.py
```

这是最简单的方式，提供图形化菜单来：
- 查看当前配置
- 列出所有可用模型
- 测试不同的模型
- 切换默认模型

### 2. 在程序中使用

#### 自动模式（推荐）
```python
from meeting_transcriber import MeetingTranscriber

# 使用配置文件中的默认模型
transcriber = MeetingTranscriber(whisper_model="auto")
```

#### 交互式选择
```python
# 程序启动时让用户选择模型
transcriber = MeetingTranscriber(whisper_model="interactive")
```

#### 指定具体模型
```python
# MLX 模型（Apple Silicon）
transcriber = MeetingTranscriber(whisper_model="mlx-community/whisper-medium")

# OpenAI 模型（跨平台）
transcriber = MeetingTranscriber(whisper_model="medium")
```

## 📋 可用模型

### MLX Whisper（Apple Silicon 优化）
- `mlx-community/whisper-tiny` - 最小模型 (~40MB, 速度最快)
- `mlx-community/whisper-base` - 基础模型 (~150MB)
- `mlx-community/whisper-small` - 小型模型 (~250MB)
- `mlx-community/whisper-medium` - 中型模型 (~770MB)
- `mlx-community/whisper-large-v2` - 大型模型v2 (~1.5GB)
- `mlx-community/whisper-large-v3` - 大型模型v3 (~1.5GB)
- `mlx-community/whisper-large-v3-turbo` - 大型turbo模型 (~1.5GB) **[默认]**

### OpenAI Whisper（跨平台）
- `tiny` - 最小模型 (~40MB)
- `base` - 基础模型 (~150MB)
- `small` - 小型模型 (~250MB)
- `medium` - 中型模型 (~770MB)
- `large-v2` - 大型模型v2 (~1.5GB)
- `large-v3` - 大型模型v3 (~1.5GB) **[默认]**

## ⚙️ 配置文件

### 主配置文件：`models_config.json`
包含所有可用模型和默认设置，通常不需要手动修改。

### 用户配置文件：`user_config.json`
存储您的个人偏好，会自动创建：

```json
{
  "preferred_models": {
    "mlx": "mlx-community/whisper-medium",
    "openai": "medium"
  },
  "user_preferences": {
    "auto_select_best_model": true,
    "prefer_speed_over_accuracy": false,
    "max_model_size_gb": 2.0
  }
}
```

## 🔧 高级功能

### 1. 程序化配置管理

```python
from config_manager import ConfigManager

config = ConfigManager()

# 查看可用模型
models = config.get_available_whisper_models("mlx")
print(models)

# 设置偏好模型
config.set_preferred_model("mlx", "mlx-community/whisper-medium")

# 获取推荐模型
recommended = config.recommend_model("mlx", audio_duration=3600)
```

### 2. 运行时切换模型

```python
from cross_platform_asr import CrossPlatformASR

asr = CrossPlatformASR("auto")

# 列出可用模型
asr.list_available_models()

# 交互式切换模型
asr.switch_model_interactive()

# 获取模型推荐
recommendations = asr.get_model_recommendations(audio_duration=1800)
```

## 💡 使用建议

### 模型选择策略

**短音频（< 10分钟）**
- 追求准确度：使用 `large-v3` 或 `large-v3-turbo`
- 追求速度：使用 `medium` 或 `small`

**长音频（> 1小时）**
- 建议使用 `medium` 或 `small` 以节省时间
- 系统会自动推荐合适的模型

**不同语言**
- 中文：推荐 `large-v3` 获得最佳效果
- 英文：`medium` 通常就足够好

### 平台优化

**Apple Silicon (M1/M2/M3)**
- 优先使用 MLX 模型获得最佳性能
- `mlx-community/whisper-large-v3-turbo` 是默认推荐

**Intel/AMD/Windows/Linux**
- 使用 OpenAI Whisper 模型
- 考虑使用 `faster-whisper` 获得更好性能（未来支持）

## 🔄 迁移说明

如果您之前使用的是硬编码模型名称，现在只需要：

1. **将代码中的模型参数改为 "auto"**
2. **使用 `python model_manager.py` 设置您的偏好模型**
3. **享受配置化的便利！**

原来的代码：
```python
transcriber = MeetingTranscriber(whisper_model="large-v3")
```

现在的代码：
```python
transcriber = MeetingTranscriber(whisper_model="auto")  # 使用配置
```

## ❓ 常见问题

**Q: 如何重置配置到默认状态？**
A: 删除 `user_config.json` 文件，或使用模型管理工具的重置功能。

**Q: 配置文件在哪里？**
A: 在项目根目录下的 `models_config.json` 和 `user_config.json`。

**Q: 如何添加新的模型？**
A: 编辑 `models_config.json` 文件，在相应引擎的 `available_models` 列表中添加。

**Q: 模型会自动下载吗？**
A: 是的，首次使用时会自动从 Hugging Face 下载并缓存到本地。

---

## 🎯 总结

新的配置系统让您能够：
- ✅ 轻松切换不同的 Whisper 模型
- ✅ 无需修改代码即可更改模型
- ✅ 根据音频特性自动推荐合适模型
- ✅ 保存个人偏好设置
- ✅ 支持交互式模型选择

享受更灵活的语音转录体验！🚀