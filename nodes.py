import io
import os
import uuid
from typing import Any, Dict, List

import folder_paths
import torchaudio
from audiocraft.models import AudioGen
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

def generate_audiogen_audio(
    descriptions: List[str],
    duration: int = 5,
    model_path: str = "facebook/audiogen-medium",
) -> bytes:
    """
    使用 AudioGen 模型生成音频

    参数:
        descriptions (list[str]): 文本提示列表
        duration (int): 生成音频的时长（秒）
        model_path (str): 预训练模型路径

    返回:
        bytes: wav 文件的二进制数据
    """
    if not isinstance(descriptions, list) or len(descriptions) == 0:
        raise ValueError("descriptions 必须是非空列表")

    # 加载模型
    try:
        model = AudioGen.get_pretrained(model_path)
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {e}")

    # 设置生成参数
    model.set_generation_params(duration=duration)

    # 生成音频
    wav = model.generate(descriptions)
    one_wav = wav[0].cpu()

    # 保存到 BytesIO
    buffer = io.BytesIO()
    torchaudio.save(buffer, one_wav, model.sample_rate, format="wav")
    buffer.seek(0)

    return buffer.getvalue()


class AudioGenGenerateNode:
    """
    ComfyUI 节点：使用 Meta AudioGen 模型根据文本提示生成音频。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "A calm lo-fi beat with gentle piano and vinyl crackle.", "multiline": True}),
                "duration": ("INT", {"default": 5, "min": 1, "max": 60}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/generation"

    def _prepare_descriptions(self, prompt: str) -> List[str]:
        # 支持多行提示，每行视为一个 prompt
        if not prompt or not prompt.strip():
            raise ValueError("prompt 不能为空")

        descriptions = [line.strip() for line in prompt.splitlines() if line.strip()]
        if not descriptions:
            raise ValueError("未找到有效的 prompt，请提供至少一条文本提示")
        return descriptions

    def _bytes_to_audio(self, wav_bytes: bytes) -> Dict[str, Any]:
        """
        将生成的 wav 字节流转换成 ComfyUI AUDIO 结构。
        """
        buffer = io.BytesIO(wav_bytes)
        waveform, sample_rate = torchaudio.load(buffer)
        # ComfyUI AUDIO 要求形状为 [Batch, Channels, Frames]
        waveform = waveform.unsqueeze(0)
        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
        }

    def generate(self, prompt: str, duration: int, model_path: str = "facebook/audiogen-medium"):
        descriptions = self._prepare_descriptions(prompt)
        duration = max(int(duration), 1)

        wav_bytes = generate_audiogen_audio(
            descriptions=descriptions,
            duration=duration,
            model_path=model_path,
        )

        audio = self._bytes_to_audio(wav_bytes)
        return (audio,)


class SenseVoiceTranscribeNode:
    """
    ComfyUI 节点：使用 SenseVoiceSmall 模型将音频转写为文本。
    """

    _model = None
    _model_config: Dict[str, Any] = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "device": ("STRING", {"default": "cuda:0"}),
                "language": ("STRING", {"default": "auto"}),
                "use_itn": ("BOOLEAN", {"default": True}),
                "batch_size_s": ("INT", {"default": 60, "min": 1, "max": 600}),
                "merge_vad": ("BOOLEAN", {"default": True}),
                "merge_length_s": ("INT", {"default": 15, "min": 1, "max": 600}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "transcribe"
    CATEGORY = "audio/transcription"

    @classmethod
    def _load_model(cls, model_id: str, device: str):
        config_key = {"model_id": model_id, "device": device}
        if cls._model is not None and cls._model_config == config_key:
            return cls._model

        cls._model = AutoModel(
            model=model_id,
            trust_remote_code=True,
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
            hub="hf",
        )
        cls._model_config = config_key
        return cls._model

    def _write_temp_audio(self, audio: Dict[str, Any]) -> str:
        """
        保存输入音频到临时文件，返回路径。
        """
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        if waveform.dim() == 3:
            waveform = waveform[0]

        waveform = waveform.cpu()
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"sensevoice_{uuid.uuid4().hex}.wav")
        torchaudio.save(temp_path, waveform, sample_rate)
        return temp_path

    def transcribe(
        self,
        audio: Dict[str, Any],
        device: str = "cuda:0",
        language: str = "auto",
        use_itn: bool = True,
        batch_size_s: int = 60,
        merge_vad: bool = True,
        merge_length_s: int = 15,
    ):
        temp_path = None
        model_id = os.path.join(folder_paths.models_dir, "SenseVoiceSmall")
        try:
            temp_path = self._write_temp_audio(audio)
            model = self._load_model(model_id, device)
            result = model.generate(
                input=temp_path,
                cache={},
                language=language,
                use_itn=use_itn,
                batch_size_s=batch_size_s,
                merge_vad=merge_vad,
                merge_length_s=merge_length_s,
            )
            text = result[0]["text"] if result else ""
            processed_text = rich_transcription_postprocess(text)
            return (processed_text,)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


NODE_CLASS_MAPPINGS = {
    "AudioGenGenerateNode": AudioGenGenerateNode,
    "SenseVoiceTranscribeNode": SenseVoiceTranscribeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioGenGenerateNode": "🔊 AudioGen Generate",
    "SenseVoiceTranscribeNode": "🗣️ SenseVoice Transcribe",
}
