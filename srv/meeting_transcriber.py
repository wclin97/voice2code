"""
会议转录器 - 整合ASR和说话人分离
完全本地化的会议转录解决方案
"""

import json
import platform
from typing import Dict, List
from .hybrid_diarization import HybridDiarization
import pandas as pd
from datetime import datetime


class MeetingTranscriber:
    """会议转录器主类"""

    def __init__(self,
                 whisper_model: str = "auto",
                 diarization_method: str = "auto",
                 window_length: float = 2.0,
                 hop_length: float = 1.0):
        """
        初始化会议转录器

        Args:
            whisper_model: Whisper模型 ("auto" 自动选择最佳模型)
            diarization_method: 说话人分离方法 ("auto", "pyannote", "local")
            window_length: 说话人分离窗口长度
            hop_length: 说话人分离跳跃长度
        """
        print("初始化语音转录器...")
        self.asr = self._init_asr(whisper_model)
        self.diarizer = HybridDiarization(
            method=diarization_method,
            window_length=window_length,
            hop_length=hop_length
        )

    def _init_asr(self, model_config):
        """动态初始化ASR引擎"""
        from .cross_platform_asr import CrossPlatformASR

        print("初始化语音识别引擎...")

        # 确定是否使用交互式选择
        interactive = model_config == "interactive"

        # 使用统一的跨平台ASR，它会自动处理配置
        if model_config in ["auto", "interactive"]:
            return CrossPlatformASR(model_config, interactive=interactive)
        else:
            # 使用指定的模型
            return CrossPlatformASR(model_config)

    def transcribe_meeting(self, audio_file: str) -> Dict:
        """
        转录会议音频

        Args:
            audio_file: 音频文件路径

        Returns:
            完整的转录结果
        """
        import time

        total_start_time = time.time()
        print("=" * 50)
        print("开始会议转录...")
        print("=" * 50)

        # 步骤1: 语音转文字
        print("\n1. 执行语音转文字...")
        step1_start = time.time()
        asr_result = self.asr.transcribe_with_timestamps(audio_file)
        step1_time = time.time() - step1_start
        print(f"   语音转文字完成，耗时: {step1_time:.1f}秒")

        if "error" in asr_result:
            return {"error": f"ASR失败: {asr_result['error']}"}

        # 步骤2: 说话人分离
        print("\n2. 执行说话人分离...")
        step2_start = time.time()
        diarization_result = self.diarizer.diarize(audio_file)
        step2_time = time.time() - step2_start
        print(f"   说话人分离完成，耗时: {step2_time:.1f}秒")

        if "error" in diarization_result:
            return {"error": f"说话人分离失败: {diarization_result['error']}"}

        # 步骤3: 对齐结果
        print("\n3. 对齐转录文本和说话人...")
        step3_start = time.time()
        aligned_result = self._align_transcription_and_speakers(
            asr_result, diarization_result
        )
        step3_time = time.time() - step3_start
        print(f"   对齐处理完成，耗时: {step3_time:.1f}秒")

        # 步骤4: 生成最终报告
        print("\n4. 生成会议纪要...")
        step4_start = time.time()
        final_result = self._generate_meeting_summary(aligned_result)
        step4_time = time.time() - step4_start
        print(f"   会议纪要生成完成，耗时: {step4_time:.1f}秒")

        total_time = time.time() - total_start_time
        print(f"\n 会议转录完成! 总耗时: {total_time:.1f}秒")
        print(f"   - 语音转文字: {step1_time:.1f}秒")
        print(f"   - 说话人分离: {step2_time:.1f}秒")
        print(f"   - 对齐处理: {step3_time:.1f}秒")
        print(f"   - 生成纪要: {step4_time:.1f}秒")

        # 添加处理时间到结果中
        final_result["processing_time"] = {
            "total": total_time,
            "asr": step1_time,
            "diarization": step2_time,
            "alignment": step3_time,
            "summary": step4_time
        }

        return final_result

    def _align_transcription_and_speakers(self,
                                        asr_result: Dict,
                                        diarization_result: Dict) -> Dict:
        """
        对齐ASR结果和说话人分离结果

        Args:
            asr_result: ASR转录结果
            diarization_result: 说话人分离结果

        Returns:
            对齐后的结果
        """
        aligned_segments = []
        asr_segments = asr_result.get("segments", [])
        speaker_segments = diarization_result.get("segments", [])

        for asr_seg in asr_segments:
            asr_start = asr_seg["start"]
            asr_end = asr_seg["end"]
            asr_text = asr_seg["text"]

            # 找到与ASR时间段重叠最多的说话人
            best_speaker = self._find_best_speaker_overlap(
                asr_start, asr_end, speaker_segments
            )

            aligned_segments.append({
                "start": asr_start,
                "end": asr_end,
                "text": asr_text,
                "speaker": best_speaker,
                "duration": asr_end - asr_start
            })

        return {
            "aligned_segments": aligned_segments,
            "total_speakers": diarization_result.get("total_speakers", 0),
            "speakers": diarization_result.get("speakers", []),
            "language": asr_result.get("language", "unknown")
        }

    def _find_best_speaker_overlap(self,
                                 asr_start: float,
                                 asr_end: float,
                                 speaker_segments: List[Dict]) -> str:
        """
        找到与ASR时间段重叠最多的说话人

        Args:
            asr_start: ASR片段开始时间
            asr_end: ASR片段结束时间
            speaker_segments: 说话人分离片段列表

        Returns:
            最匹配的说话人标签
        """
        max_overlap = 0
        best_speaker = "Unknown"

        for speaker_seg in speaker_segments:
            speaker_start = speaker_seg["start"]
            speaker_end = speaker_seg["end"]

            # 计算重叠时间
            overlap_start = max(asr_start, speaker_start)
            overlap_end = min(asr_end, speaker_end)
            overlap_duration = max(0, overlap_end - overlap_start)

            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_speaker = speaker_seg["speaker"]

        return best_speaker

    def _generate_meeting_summary(self, aligned_result: Dict) -> Dict:
        """
        生成会议摘要

        Args:
            aligned_result: 对齐后的结果

        Returns:
            完整的会议摘要
        """
        segments = aligned_result["aligned_segments"]

        # 按说话人统计
        speaker_stats = {}
        speaker_content = {}

        for seg in segments:
            speaker = seg["speaker"]
            duration = seg["duration"]
            text = seg["text"]

            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    "total_duration": 0,
                    "segment_count": 0
                }
                speaker_content[speaker] = []

            speaker_stats[speaker]["total_duration"] += duration
            speaker_stats[speaker]["segment_count"] += 1
            speaker_content[speaker].append(text)

        # 生成时间轴格式的转录
        timeline = []
        for seg in segments:
            timeline.append({
                "timestamp": self._format_timestamp(seg["start"]),
                "speaker": seg["speaker"],
                "text": seg["text"],
                "start_seconds": seg["start"],
                "end_seconds": seg["end"]
            })

        # 计算总时长
        total_duration = max([seg["end"] for seg in segments]) if segments else 0

        return {
            "meeting_info": {
                "total_duration": total_duration,
                "total_speakers": aligned_result["total_speakers"],
                "language": aligned_result["language"],
                "transcription_date": datetime.now().isoformat()
            },
            "speaker_statistics": speaker_stats,
            "timeline": timeline,
            "speaker_content": speaker_content,
            "segments": segments
        }

    def _format_timestamp(self, seconds: float) -> str:
        """
        格式化时间戳

        Args:
            seconds: 秒数

        Returns:
            格式化的时间字符串 (MM:SS)
        """
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def export_to_json(self, result: Dict, output_file: str):
        """
        导出结果为JSON格式

        Args:
            result: 转录结果
            output_file: 输出文件路径
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f" JSON结果已保存到: {output_file}")
        except Exception as e:
            print(f" 保存JSON失败: {str(e)}")

    def export_to_txt(self, result: Dict, output_file: str):
        """
        导出结果为可读的文本格式

        Args:
            result: 转录结果
            output_file: 输出文件路径
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # 写入会议信息
                info = result["meeting_info"]
                f.write("=" * 60 + "\n")
                f.write("会议转录结果\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"总时长: {self._format_timestamp(info['total_duration'])}\n")
                f.write(f"说话人数: {info['total_speakers']}\n")
                f.write(f"语言: {info['language']}\n")
                f.write(f"转录时间: {info['transcription_date']}\n\n")

                # 写入说话人统计
                f.write("说话人统计:\n")
                f.write("-" * 30 + "\n")
                for speaker, stats in result["speaker_statistics"].items():
                    duration_str = self._format_timestamp(stats["total_duration"])
                    f.write(f"{speaker}: {duration_str} ({stats['segment_count']} 个片段)\n")
                f.write("\n")

                # 写入时间轴
                f.write("转录时间轴:\n")
                f.write("-" * 30 + "\n")
                for item in result["timeline"]:
                    f.write(f"[{item['timestamp']}] {item['speaker']}: {item['text']}\n")

            print(f" 文本结果已保存到: {output_file}")
        except Exception as e:
            print(f" 保存文本失败: {str(e)}")

    def export_to_csv(self, result: Dict, output_file: str):
        """
        导出结果为CSV格式

        Args:
            result: 转录结果
            output_file: 输出文件路径
        """
        try:
            # 创建DataFrame
            data = []
            for item in result["timeline"]:
                data.append({
                    "时间戳": item["timestamp"],
                    "开始时间(秒)": item["start_seconds"],
                    "结束时间(秒)": item["end_seconds"],
                    "说话人": item["speaker"],
                    "文本内容": item["text"]
                })

            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f" CSV结果已保存到: {output_file}")
        except Exception as e:
            print(f" 保存CSV失败: {str(e)}")


if __name__ == "__main__":
    # 使用示例
    transcriber = MeetingTranscriber()

    # 转录会议
    # audio_file = "meeting.wav"
    # result = transcriber.transcribe_meeting(audio_file)

    # 导出结果
    # transcriber.export_to_json(result, "meeting_transcript.json")
    # transcriber.export_to_txt(result, "meeting_transcript.txt")
    # transcriber.export_to_csv(result, "meeting_transcript.csv")

    print("会议转录器已准备就绪!")
    print("使用方法:")
    print("1. transcriber = MeetingTranscriber()")
    print("2. result = transcriber.transcribe_meeting('your_audio.wav')")
    print("3. transcriber.export_to_txt(result, 'output.txt')")