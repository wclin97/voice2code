# 平台安装指南

## 自动平台检测

本项目会自动检测您的平台并选择最佳的语音识别引擎：

- **Apple Silicon (M1/M2/M3) Mac**: 自动使用 MLX Whisper (最佳性能)
- **Windows/Linux/Intel Mac**: 自动使用 OpenAI Whisper (跨平台兼容)

## 安装步骤

### 1. 安装基础依赖

```bash
pip install -r requirements.txt
```

### 2. 根据平台安装语音识别引擎

#### Apple Silicon Mac (推荐)

```bash
# 安装 MLX Whisper (最佳性能)
pip install mlx-whisper
```

#### Windows/Linux/Intel Mac

```bash
# 安装 OpenAI Whisper (跨平台)
pip install openai-whisper
```

#### 可选: 更快的推理引擎 (所有平台)

```bash
# 可以替代 OpenAI Whisper，推理速度更快
pip install faster-whisper
```

### 3. 可选组件

#### PyAnnote 说话人分离 (高质量，需要 HuggingFace Token)

```bash
pip install pyannote.audio
```

然后运行设置向导：
```bash
python srv/setup_pyannote.py
```

#### 仅使用本地聚类 (无需 Token，速度快)

无需额外安装，系统会自动回退到本地聚类方法。

## 验证安装

### 测试整体功能
```bash
python demo.py
```

### 测试 MLX Whisper (仅 Apple Silicon)
```bash
python srv/test_mlx_whisper.py
```

### 检查平台兼容性
```bash
python demo.py
# 选择 "2. 平台兼容性检查"
```

## 常见问题

### Q: Windows 上出现 MLX 导入错误
**A**: 这是正常的。系统会自动检测到您不在 Apple Silicon 平台上，并使用 OpenAI Whisper。

### Q: 如何选择不同的语音识别引擎？
**A**: 系统会自动选择最佳引擎。如果需要手动选择，可以在模型管理中切换。

### Q: PyAnnote 需要网络吗？
**A**: 只有首次下载模型时需要网络和 HuggingFace Token，之后完全离线运行。

### Q: 本地聚类效果如何？
**A**: 速度快10倍以上，准确率稍低，但完全离线无需 Token。

## 性能对比

| 平台 | 引擎 | 速度 | 质量 | 离线 |
|------|------|------|------|------|
| Apple Silicon | MLX Whisper | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ |
| Windows/Linux | OpenAI Whisper | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ |
| 所有平台 | Faster Whisper | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ |

## 说话人分离对比

| 方法 | 速度 | 质量 | 要求 |
|------|------|------|------|
| PyAnnote | ⭐⭐ | ⭐⭐⭐⭐⭐ | HuggingFace Token |
| 本地聚类 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 无 |