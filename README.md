# 语音转录系统

一个完全本地化的会议转录解决方案，支持跨平台语音转文字和智能说话人分离，提供完整的交互式菜单系统。

## 特性

- **完全本地化**: 无需联网，保护隐私
- **跨平台支持**: Apple Silicon (MLX) + 其他平台 (OpenAI Whisper)
- **配置化模型管理**: 灵活的模型配置系统，轻松切换不同 Whisper 模型
- **智能说话人分离**:
  - **PyAnnote**: 高质量分离 (需token，下载后离线)
  - **本地聚类**: 无需token的快速方案
  - **混合模式**: 自动选择最佳方法
- **交互式菜单**: 完整的命令行界面，无需记忆命令
- **多格式导出**: 支持 TXT、JSON、CSV 格式
- **可视化**: 生成说话人时间轴图表
- **实时统计**: 详细的处理时间和效率统计
- **智能回退**: 自动选择最佳可用方法

## 安装

1. **克隆项目**

```bash
git clone <your-repo>
cd voice2code
```

2. **安装依赖**

```bash
pip install -r requirements.txt
```

3. **选择说话人分离方法**

**方案A: 仅使用本地聚类 (无需token)**

```bash
# 已包含在 requirements.txt 中，无需额外操作
```

**方案B: 使用 PyAnnote (推荐，效果更好)**

```bash
# 运行设置向导
python setup_pyannote.py
```

或手动设置：

```bash
pip install pyannote.audio
huggingface-cli login  # 输入你的 HuggingFace token
```

**可选工具**：

```bash
python token_helper.py       # Token 管理助手
python test_mlx_whisper.py   # 测试 MLX Whisper 配置
```

## 快速开始

### 主菜单系统 (推荐)

```bash
python demo.py
```

启动完整的交互式菜单系统，包含所有功能：

```
语音转录系统
============================================================

主菜单:

  核心功能
    1. 开始会议转录
    2. 平台兼容性检查

  配置管理
    3. Whisper 模型管理
    4. PyAnnote 设置向导
    5. HuggingFace Token 助手

  测试工具
    6. 测试 MLX Whisper
    7. 系统配置概览

  帮助与信息
    8. 查看使用指南
    9. 关于本程序

    0. 退出程序
```

### 独立工具使用

```bash
# 模型管理
python model_manager.py

# PyAnnote 设置
python setup_pyannote.py

# Token 管理
python token_helper.py

# MLX Whisper 测试
python test_mlx_whisper.py
```

### 程序化使用

```python
from meeting_transcriber import MeetingTranscriber

# 使用配置文件默认设置
transcriber = MeetingTranscriber(whisper_model="auto")

# 交互式选择模型
transcriber = MeetingTranscriber(whisper_model="interactive")

# 指定具体模型
transcriber = MeetingTranscriber(
    whisper_model="mlx-community/whisper-medium",
    diarization_method="auto"
)

# 转录会议音频
result = transcriber.transcribe_meeting("meeting.wav")

# 导出结果
transcriber.export_to_txt(result, "transcript.txt")
transcriber.export_to_json(result, "transcript.json")
transcriber.export_to_csv(result, "transcript.csv")
```

## 项目结构

```
voice2code/
├── input/                    # 音频文件输入目录
├── output/                   # 转录结果输出目录
│
├── demo.py                   # 主菜单系统 (入口程序)
├── meeting_transcriber.py    # 主转录器
│
├── config_manager.py         # 配置管理器
├── model_manager.py          # 模型管理工具
├── models_config.json        # 模型配置文件
├── user_config.json          # 用户配置文件 (自动生成)
│
├── cross_platform_asr.py     # 跨平台语音转文字
├── local_asr.py              # 本地语音转文字 (MLX)
├── hybrid_diarization.py     # 混合说话人分离器
├── local_diarization.py      # 本地说话人分离
├── pyannote_diarization.py   # PyAnnote 说话人分离
│
├── setup_pyannote.py         # PyAnnote 设置向导
├── token_helper.py           # HuggingFace Token 助手
├── test_mlx_whisper.py       # MLX Whisper 测试工具
│
├── requirements.txt          # 依赖列表
├── MODEL_CONFIG_GUIDE.md     # 配置指南
└── README.md                 # 说明文档
```

## 配置系统

### 模型配置文件

系统使用 JSON 配置文件管理所有模型设置：

- `models_config.json` - 主配置文件，包含所有可用模型
- `user_config.json` - 用户偏好设置，自动生成

### Whisper 模型选择

```python
# 自动选择 (使用配置文件默认)
transcriber = MeetingTranscriber(whisper_model="auto")

# 交互式选择
transcriber = MeetingTranscriber(whisper_model="interactive")

# 指定具体模型
transcriber = MeetingTranscriber(whisper_model="mlx-community/whisper-medium")
```

### 可用的 Whisper 模型

**MLX Whisper (Apple Silicon)**

- `mlx-community/whisper-tiny` - 最小模型 (~40MB)
- `mlx-community/whisper-base` - 基础模型 (~150MB)
- `mlx-community/whisper-small` - 小型模型 (~250MB)
- `mlx-community/whisper-medium` - 中型模型 (~770MB)
- `mlx-community/whisper-large-v3` - 大型模型 (~1.5GB)
- `mlx-community/whisper-large-v3-turbo` - 默认推荐 (~1.5GB)

**OpenAI Whisper (跨平台)**

- `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`

### 说话人分离配置

```python
transcriber = MeetingTranscriber(
    whisper_model="auto",
    diarization_method="auto",     # 自动选择最佳方法
    window_length=2.0,             # 分析窗口长度
    hop_length=1.0                 # 跳跃长度
)
```

**可选的分离方法:**

- `"auto"` - 自动选择 (PyAnnote 优先，回退本地)
- `"pyannote"` - 高质量 PyAnnote (需 token)
- `"pyannote_fast"` - PyAnnote 快速模式 (长音频分段)
- `"local"` - 本地聚类 (无需 token，速度快)

## 输出格式

### 文本格式 (.txt)

```
=============================================================
会议转录结果
=============================================================

总时长: 05:23
说话人数: 2
语言: zh

说话人统计:
------------------------------
Speaker_0: 03:12 (15 个片段)
Speaker_1: 02:11 (12 个片段)

转录时间轴:
------------------------------
[00:05] Speaker_0: 大家好，今天我们讨论一下项目进展
[00:12] Speaker_1: 好的，我先汇报一下我这边的情况
...
```

### JSON 格式 (.json)

```json
{
  "meeting_info": {
    "total_duration": 323.5,
    "total_speakers": 2,
    "language": "zh",
    "transcription_date": "2024-01-15T10:30:00"
  },
  "speaker_statistics": {
    "Speaker_0": {
      "total_duration": 192.3,
      "segment_count": 15
    }
  },
  "timeline": [
    {
      "timestamp": "00:05",
      "speaker": "Speaker_0",
      "text": "大家好，今天我们讨论一下项目进展",
      "start_seconds": 5.2,
      "end_seconds": 8.7
    }
  ]
}
```

## 工作流程

### 1. 音频处理流程

```
音频文件 → 平台检测 → 模型选择 → 并行处理
                                    ├── 语音转文字 (ASR)
                                    └── 说话人分离 (Diarization)
                                           ↓
                              结果对齐 → 时间轴生成 → 多格式导出
```

### 2. 平台适配

- **Apple Silicon**: 优先使用 MLX Whisper 获得最佳性能
- **其他平台**: 自动回退到 OpenAI Whisper
- **配置系统**: 用户可以随时切换模型和方法

### 3. 智能说话人分离

**PyAnnote 模式:**

- 使用预训练神经网络模型
- 支持长音频分段处理
- 首次需要 HuggingFace token

**本地聚类模式:**

- MFCC 特征提取 + 层次聚类
- 无需网络，完全离线
- 处理速度快 10 倍以上

### 4. 实时统计

系统会自动记录和显示：

- 各步骤处理时间
- 音频时长 vs 处理时间效率比
- 内存使用情况
- 模型加载时间

## 性能优化

### PyAnnote 速度优化策略

PyAnnote 说话人分离虽然质量高，但对长音频处理较慢。我们提供了多种优化方案：

#### 1. 分段处理 (Chunking)

- **核心思路**: 将长音频分成小段处理，然后智能合并结果
- **适用场景**: 10分钟以上的长音频
- **性能提升**: 对30分钟音频可提升50%以上的速度
- **实现**: 选择"PyAnnote 快速模式"

#### 2. 处理时间对比

| 音频长度 | PyAnnote 标准模式 | PyAnnote 快速模式 | 本地聚类 |
| -------- | ----------------- | ----------------- | -------- |
| 5分钟    | 2-3分钟           | 2-3分钟           | 30秒     |
| 15分钟   | 10-15分钟         | 6-8分钟           | 1-2分钟  |
| 30分钟   | 25-40分钟         | 12-18分钟         | 3-5分钟  |
| 60分钟   | 60-120分钟        | 25-35分钟         | 8-12分钟 |

#### 3. 方法选择建议

- **短音频 (<5分钟)**: PyAnnote 标准模式，最佳质量
- **中等音频 (5-15分钟)**: PyAnnote 快速模式，平衡质量与速度
- **长音频 (15-30分钟)**: 根据需求选择，建议本地聚类
- **超长音频 (>30分钟)**: 强烈推荐本地聚类方法

#### 4. 优化技术细节

**分段处理算法**:

```
1. 将音频按5分钟切分
2. 并行处理各段 (可扩展)
3. 智能合并相邻的同说话人片段
4. 统一说话人标签
```

**用户体验优化**:

- 实时显示处理进度和耗时
- 长音频自动提示建议方法
- 智能回退机制防止处理失败

## 故障排除

### 使用主菜单诊断

建议先使用主菜单系统进行诊断：

```bash
python demo.py
# 选择 2: 平台兼容性检查
# 选择 6: 测试 MLX Whisper
# 选择 7: 系统配置概览
```

### 常见问题

1. **模型加载失败**

   ```bash
   python model_manager.py  # 检查和切换模型
   ```
2. **PyAnnote Token 问题**

   ```bash
   python token_helper.py   # Token 助手
   python setup_pyannote.py # 设置向导
   ```
3. **平台兼容性问题**

   - Apple Silicon: 自动使用 MLX Whisper
   - 其他平台: 自动使用 OpenAI Whisper
   - 检查: 主菜单 → 选项 2
4. **音频格式支持**

   - 支持: WAV, MP3, M4A, FLAC, OGG
   - 推荐: WAV 格式，16kHz 采样率
5. **处理速度优化**

   - 短音频 (<5min): PyAnnote 标准模式
   - 中等音频 (5-15min): PyAnnote 快速模式
   - 长音频 (>15min): 本地聚类模式

### 配置文件问题

```bash
# 重置配置
rm user_config.json

# 重新配置模型
python model_manager.py
```

### 性能调优建议

- **Apple Silicon**: 使用 MLX 模型获得最佳性能
- **长音频**: 选择本地聚类或 PyAnnote 快速模式
- **高质量需求**: 使用 PyAnnote 标准模式
- **离线使用**: 首次下载模型后完全离线运行
