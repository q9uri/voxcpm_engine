"""TTSEngine のモック"""

import copy
from typing import Final

import numpy as np
from numpy.typing import NDArray
#from pyopenjtalk import tts

from ...metas.metas import StyleId
from ...model import AudioQuery
from ...tts_pipeline.audio_postprocessing import raw_wave_to_output_wave
from ...tts_pipeline.tts_engine import (
    TTSEngine,
    to_flatten_moras,
)
from ..core.mock import MockCoreWrapper

import json
import voxcpm
from voxcpm.model.voxcpm import LoRAConfig
from ...utility.path_utility import engine_root, transcripts_path, resource_root
from ...tts_pipeline.kana_converter import create_kana

class MockTTSEngine(TTSEngine):
    """製品版コア無しに音声合成が可能なモック版TTSEngine"""

    def __init__(self) -> None:
        super().__init__(MockCoreWrapper())

        # Load config from checkpoint

        model_dir = resource_root() / "checkpoints/WariHima__voxcpm-1.5-resized-large"
        lora_ckpt_dir = resource_root() / "checkpoints/1.5-large-ja-rev-a"

        with open(f"{lora_ckpt_dir}/lora_config.json") as f:
            lora_info = json.load(f)

        base_model = lora_info["base_model"]
        lora_cfg = LoRAConfig(**lora_info["lora_config"])

        self.voxcpm_model = voxcpm.VoxCPM(voxcpm_model_path=model_dir, lora_config=lora_cfg,
                                          lora_weights_path=lora_ckpt_dir, optimize=False)

    def synthesize_wave(
        self,
        query: AudioQuery,
        style_id: StyleId,
        enable_interrogative_upspeak: bool,
    ) -> NDArray[np.float32]:

        inference_timesteps_input = 10
        cfg_value_input = 2.0
        do_normalize = True
        sr_raw_wave = self.voxcpm_model.tts_model.sample_rate

        with open(transcripts_path(), "r", encoding="utf-8") as f:
            wav_file_lists = json.load(f)

        wav_file_lists = wav_file_lists["file_lists"]
        for i in wav_file_lists:
            if i["id"] == style_id:
                filename = i["filename"]
                prompt_wav_path = resource_root() / filename
                prompt_text = i["transcript"]
                print(prompt_text, prompt_wav_path)


        kana_text = create_kana(query.accent_phrases)

        raw_wave = self.voxcpm_model.generate(
            text=kana_text,
            prompt_text=prompt_text,
            prompt_wav_path=prompt_wav_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=int(inference_timesteps_input),
            normalize=do_normalize,
            denoise=False,
        )
        wave = raw_wave_to_output_wave(query, raw_wave, sr_raw_wave)
        return wave

        """音声合成用のクエリに含まれる読み仮名に基づいてOpenJTalkで音声波形を生成する。モーラごとの調整は反映されない。"""
        # モーフィング時などに同一参照のqueryで複数回呼ばれる可能性があるので、元の引数のqueryに破壊的変更を行わない
        query = copy.deepcopy(query)

        # recall text in katakana
        flatten_moras = to_flatten_moras(query.accent_phrases)
        kana_text = "".join([mora.text for mora in flatten_moras])

        raw_wave, sr_raw_wave = self.forward(kana_text)
        wave = raw_wave_to_output_wave(query, raw_wave, sr_raw_wave)
        return wave

    def forward(self, text: str) -> tuple[NDArray[np.float32], int]:
        """文字列から pyopenjtalk を用いて音声を合成する。"""
        return
        OJT_SAMPLING_RATE: Final = 48000
        OJT_AMPLITUDE_MAX: Final = 2 ** (16 - 1)
        raw_wave: NDArray[np.float64] = tts(text)[0]
        raw_wave /= OJT_AMPLITUDE_MAX
        return raw_wave.astype(np.float32), OJT_SAMPLING_RATE
