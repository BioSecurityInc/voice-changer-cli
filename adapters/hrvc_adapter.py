"""
hRVC Modular Engine Adapter — Refined SOLA + Intelligent Gate
===========================================================
"""

import os
import sys
import json
import torch
import numpy as np
import traceback
from scipy.signal import resample_poly, lfilter
from scipy.special import erf
from adapters.base_adapter import BaseVoiceAdapter

class HRVCVoiceAdapter(BaseVoiceAdapter):
    """
    Implementation of the hRVC (Halim-RVC) engine adapter.
    Features:
    - Refined SOLA (Phase-aligned stitching)
    - Sine Crossfade (Soft transition)
    - Intelligent Gate (Prevents breathing noise boost)
    - Auto-Gain (RMS volume matching)
    """

    def __init__(self, model_path: str, engine_path: str = None, device: str = None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        super().__init__(model_path, engine_path, device)
        
        self.engine = None
        self.device_obj = torch.device(device)
        self.model_config = {}

        # Session Buffers
        self.prev_output_tail = None
        self.prev_input_tail = None
        
        # Audio Parameters
        self.output_sr = 48000
        self.overlap_len = 4800   # 100ms
        self.context_len = 24000  # 500ms
        
        # Internal State
        self.current_gain = -1.0 

    def _inject_engine_paths(self):
        if not self.engine_path:
            script_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.engine_path = os.path.join(script_root, "model_source", "HalimGSSv4.2.228")
        if os.path.exists(self.engine_path) and self.engine_path not in sys.path:
            sys.path.insert(0, self.engine_path)

    def load(self):
        self._inject_engine_paths()
        config_path = os.path.join(self.model_path, "model.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing 'model.json' in {self.model_path}")
        with open(config_path, "r") as f:
            self.model_config = json.load(f)

        from HalimGSS.TimbreNode.Inference.TimbreInference import TimbreInference
        
        class EngineConfig:
            def __init__(self, dev):
                self.device = dev
                self.is_half = False
                self.x_pad = 0.4
                self.hop_length_tgt = 480 
                self.use_torch_hpf = False
                self.hpf_cutoff_hz = 60.0

        self.engine = TimbreInference(tgt_sr=self.output_sr, config=EngineConfig(self.device_obj))
        
        # Patch the engine's internal glue
        self.engine.ola_glue = self._sola_sine_glue

        # Load sub-modules
        from HalimGSS.TimbreNode.Phonetic.LoadHubert import HubertLoader
        from HalimGSS.TimbreNode.Phonetic.LoadContentVec import ContentVecLoader
        assets = os.path.join(self.engine_path, "assets")
        hubert = HubertLoader(model_path=os.path.join(assets, "mHuBERT-147"), is_half=False)
        contentvec = ContentVecLoader(model_path=os.path.join(assets, "contentvec700", "checkpoint_best_legacy_500.pt"), is_half=False)
        self.engine.SetPhoneticModel(hubert_model=hubert, contentvec_model=contentvec)

        from HalimGSS.TimbreNode.Models.models import SynthesizerTrnMs768NSFsid
        weight_path = os.path.join(self.model_path, "TimbreNode", "G_best_EMA_by_MCD.pth")
        if not os.path.exists(weight_path):
            weight_path = os.path.join(self.model_path, "TimbreNode", "G_Final_EMA.pth")
        
        m_cfg = self.model_config["model"]
        net_g = SynthesizerTrnMs768NSFsid(
            spec_channels=self.model_config["data"]["filter_length"] // 2 + 1,
            segment_size=self.model_config["train"]["segment_size"] // self.model_config["data"]["hop_length"],
            sr=self.model_config["sample_rate"], is_half=False, **m_cfg
        )
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
        net_g.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        net_g.eval().to(self.device_obj)
        self.engine.SetAcousticModel(net_g)

        self.engine.index_rate_fused = self.params.get("index", 0.7)
        self.engine.f0_up_key = self.params.get("pitch", 0.0)
        self.engine.protect = 0.33 # Balanced protection

        from HalimGSS.TimbreNode.Predictors.RMVPEF0Predictor import RMVPEF0Predictor
        rmvpe_p = os.path.join(self.engine_path, "assets", "rmvpe", "rmvpe_big.pt")
        if os.path.exists(rmvpe_p):
            self.engine.model_rmvpe = RMVPEF0Predictor(model_path=rmvpe_p, device=self.device_obj)

    def _sola_sine_glue(self, prev_tail, cur, overlap_samples):
        cur = np.asarray(cur, dtype=np.float32).reshape(-1)
        if prev_tail is None or len(prev_tail) == 0:
            step_len = max(0, len(cur) - int(overlap_samples))
            return cur[:step_len], cur[step_len:].copy()

        L = int(overlap_samples)
        search_range = int(0.02 * 48000) 
        
        trim_prev = prev_tail[-L:]
        search_region = cur[:L + search_range]
        
        if len(trim_prev) >= L and len(search_region) >= L:
            corr = np.correlate(search_region, trim_prev, mode='valid')
            best_lag = np.argmax(corr)
            cur = cur[best_lag:]
            L_actual = min(L, len(prev_tail), len(cur))
            if L_actual > 0:
                phase = np.linspace(0, np.pi / 2, L_actual, dtype=np.float32)
                fade_in, fade_out = np.sin(phase), np.cos(phase)
                cur[:L_actual] = (prev_tail[-L_actual:] * fade_out) + (cur[:L_actual] * fade_in)
        
        step_len = max(0, len(cur) - L)
        return cur[:step_len], cur[step_len:].copy()

    def change_voice(self, audio_chunk_i16: np.ndarray) -> np.ndarray:
        if self.engine is None: return audio_chunk_i16
        
        f32_in = audio_chunk_i16.astype(np.float32) / 32767.0
        # DC removal
        f32_in = f32_in - np.mean(f32_in)
        in_rms = np.sqrt(np.mean(f32_in**2)) if len(f32_in) > 0 else 0
        
        # Context management
        if self.prev_input_tail is not None:
            x_input = np.concatenate([self.prev_input_tail[-self.context_len:], f32_in])
        else:
            x_input = f32_in
        self.prev_input_tail = f32_in.copy()

        # Resample to 16k
        audio_16k = resample_poly(x_input, 1, 3).astype(np.float32)
        
        # SMART NORMALIZATION (Mumble-Fix with Noise Gate)
        peak = np.max(np.abs(audio_16k))
        if peak > 0.01: # Signal threshold (approx -40dB)
            # Only normalize if there's actual signal
            norm_factor = 0.5 / (peak + 1e-6)
            audio_16k_norm = audio_16k * min(norm_factor, 10.0)
        else:
            # If quiet (breath/silence), don't boost noise
            audio_16k_norm = audio_16k

        # Infer
        processed_f32 = self.engine.infer(audio_16k=audio_16k_norm)

        # Trim context
        if len(x_input) > len(f32_in):
            processed_f32 = processed_f32[self.context_len:]

        # Restore volume
        out_rms = np.sqrt(np.mean(processed_f32**2)) if len(processed_f32) > 0 else 0
        if out_rms > 0.001:
            target_gain = in_rms / out_rms
            if self.current_gain < 0:
                self.current_gain = target_gain
            else:
                self.current_gain = 0.8 * self.current_gain + 0.2 * target_gain
            processed_f32 *= self.current_gain

        # Stitching
        step, self.prev_output_tail = self._sola_sine_glue(self.prev_output_tail, processed_f32, self.overlap_len)
        
        return np.clip(step * 32767.0, -32768, 32767).astype(np.int16)

    def reset_state(self):
        self.prev_output_tail = None
        self.prev_input_tail = None
        self.current_gain = -1.0

    @classmethod
    def discover_models(cls, model_root: str) -> dict:
        found = {}
        if not os.path.exists(model_root): return found
        for item in sorted(os.listdir(model_root)):
            path = os.path.join(model_root, item)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "model.json")):
                try:
                    with open(os.path.join(path, "model.json"), "r") as f: 
                        data = json.load(f)
                    found[item] = {"name": f"⭐ {data.get('name', item)} (hRVC)", "path": path, "type": "hrvc"}
                except Exception: pass
        return found
