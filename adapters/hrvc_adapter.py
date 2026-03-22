"""
hRVC Modular Engine Adapter — Academic Submission Edition
========================================================
This module implements the bridge between the Universal CLI and the 
Halim-GSS / hRVC neural processing engine.

Key Responsibilities:
1. Orchestrating the loading of Phonetic Encoders (mHuBERT/ContentVec).
2. Initializing FiLM (Feature-wise Linear Modulation) conditioning.
3. Managing Acoustic Model weights (EMA-optimized).
4. Providing real-time F0 (pitch) estimation via librosa.pyin.
"""

import os
import sys
import json
import torch
import numpy as np
import librosa
import traceback
from adapters.base_adapter import BaseVoiceAdapter

# ── Pitch Estimation Engine ──────────────────────────────────────────────────
class LibrosaF0Predictor:
    """
    High-performance fundamental frequency (F0) estimation.
    Uses Probabilistic Yin (pYIN) algorithm for stable pitch tracking in speech.
    """
    def __init__(self, sample_rate=16000, hop_length=160):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f0_bin = 256
        self.f0_mel_min = 1127.0 * np.log(1.0 + 50.0 / 700.0)
        self.f0_mel_max = 1127.0 * np.log(1.0 + 1100.0 / 700.0)
    
    def compute_f0(self, wav, p_len=None):
        """Calculates F0 trajectory for the input waveform."""
        wav_f32 = np.asarray(wav, dtype=np.float32).flatten()
        
        # pYIN is robust to noise and background artifacts
        f0, _, _ = librosa.pyin(
            wav_f32, fmin=85.0, fmax=1100.0,
            sr=self.sample_rate, frame_length=1024,
            hop_length=self.hop_length, fill_na=0.0
        )
        
        # Calculate RMS energy to apply a silence gate
        rms = librosa.feature.rms(y=wav_f32, frame_length=1024, hop_length=self.hop_length)[0]
        length = min(len(f0), len(rms))
        f0, rms = f0[:length], rms[:length]
        
        # Threshold: any signal below -55dB is treated as silence (f0 = 0)
        f0[rms < 10**(-55.0/20.0)] = 0.0 

        # Apply median filtering to smooth out pitch transitions
        from scipy.signal import medfilt
        if np.count_nonzero(f0) > 3:
            f0 = np.where(f0 > 0, medfilt(f0, kernel_size=7), 0.0)
        
        # Handle sequence padding for batch processing compatibility
        if p_len:
            f0 = np.pad(f0, (0, max(0, p_len - len(f0))))[:p_len]
        return f0.astype(np.float32)

    def coarse_f0(self, f0):
        """Converts raw Hz frequencies into discretized Mel bins for the engine."""
        f0 = np.where(np.isfinite(f0), f0, 0.0)
        f0_mel = 1127.0 * np.log(1.0 + (f0 / 700.0))
        pos = f0_mel > 0.0
        if np.any(pos):
            f0_mel[pos] = (f0_mel[pos] - self.f0_mel_min) * (self.f0_bin - 2) / (self.f0_mel_max - self.f0_mel_min) + 1.0
        return np.clip(f0_mel, 1.0, self.f0_bin - 1).astype(np.float32)

# ── hRVC Main Adapter ──────────────────────────────────────────────────────────
class HRVCVoiceAdapter(BaseVoiceAdapter):
    """
    Implementation of the hRVC (Halim-RVC) engine adapter.
    Handles complex dependency injection and multi-stage neural pipeline loading.
    """

    def __init__(self, model_path: str, engine_path: str = None, device: str = None):
        # Auto-detect Apple Silicon (MPS) or fallback to CPU
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        super().__init__(model_path, engine_path, device)
        
        self.engine = None
        self.device_obj = torch.device(device)
        self.model_config = {}

    def _inject_engine_paths(self):
        """Dynamically prepares Python sys.path for the deep neural engine core."""
        if not self.engine_path:
            script_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.engine_path = os.path.join(script_root, "model_source", "HalimGSSv4.2.228")
        
        if os.path.exists(self.engine_path) and self.engine_path not in sys.path:
            sys.path.insert(0, self.engine_path)

    def load(self):
        """
        Main entry point for loading all neural components.
        Iteratively initializes encoders, conditioning, and acoustic models.
        """
        self._inject_engine_paths()
        
        # Load the model metadata (contains sampling rates, hidden dimensions, etc.)
        config_path = os.path.join(self.model_path, "model.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing mandatory 'model.json' in {self.model_path}")
        with open(config_path, "r") as f:
            self.model_config = json.load(f)

        # Initialize the core Inference Orchestrator
        from HalimGSS.TimbreNode.Inference.TimbreInference import TimbreInference
        
        class EngineConfig:
            """Internal configuration capsule for the inference engine."""
            def __init__(self, dev):
                self.device = dev
                self.is_half = False  # Keep FP32 for CPU stability
                self.x_pad = 0.3      # Audio padding for boundary artifacts
                self.hop_length_tgt = 480 
                self.use_torch_hpf = False
                self.hpf_cutoff_hz = 60.0

        self.engine = TimbreInference(tgt_sr=48000, config=EngineConfig(self.device_obj))
        
        # Assign pitch predictor
        self.engine.model_rmvpe = LibrosaF0Predictor() 
        
        # Execute consecutive loading pipeline
        try:
            self._load_phonetic_encoders()
            self._load_film_conditioning()
            self._load_acoustic_model()
            self._load_faiss_index()
            
            # Sync runtime settings
            self.engine.index_rate_fused = self.params.get("index", 0.7)
            self.engine.f0_up_key = self.params.get("pitch", 0.0)
            self.engine.protect = self.params.get("protect", 0.5)
            
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"hRVC Load Pipeline Failed: {e}")

    def _load_phonetic_encoders(self):
        """Loads mHuBERT and ContentVec encoders for feature extraction."""
        from HalimGSS.TimbreNode.Phonetic.LoadHubert import HubertLoader
        from HalimGSS.TimbreNode.Phonetic.LoadContentVec import ContentVecLoader
        
        assets = os.path.join(self.engine_path, "assets")
        hubert = HubertLoader(model_path=os.path.join(assets, "mHuBERT-147"), is_half=False)
        contentvec = ContentVecLoader(model_path=os.path.join(assets, "contentvec700", "checkpoint_best_legacy_500.pt"), is_half=False)
        
        self.engine.SetPhoneticModel(hubert_model=hubert, contentvec_model=contentvec)

    def _load_film_conditioning(self):
        """Handles FiLM (Feature-wise Linear Modulation) layer initialization."""
        try:
            from HalimGSS.TimbreNode.Models.FiLMManager import FiLMManager
            from HalimGSS.TimbreNode.Inference.SearchCheckpoint import resolve_checkpoint_dir
            
            ckpt_dir = resolve_checkpoint_dir(self.model_path)
            film_path = os.path.join(ckpt_dir, "FiLM_latest.pth")
            
            if os.path.exists(film_path):
                ck = torch.load(film_path, map_location="cpu")
                blob = ck.get("model", ck) if isinstance(ck, dict) else ck
                
                # Determine which FiLM components are available in the checkpoint
                mgr = FiLMManager(
                    d_model=768, style_dim=256, device=str(self.device_obj),
                    enable_energy=("energy_film" in blob or any(k.startswith("gamma_mlp") for k in blob.keys())),
                    enable_speaker=("speaker_film" in blob),
                    enable_style=("style_film" in blob),
                    enable_prosody=("prosody_film" in blob or "prosody_encoder" in blob)
                )
                mgr.load_film_state_dict(blob, strict=False)
                
                # Inject FiLM components into the inference engine
                self.engine.SetFilmStyleModel(mgr.style_film)
                self.engine.SetFilmSpeakerModel(mgr.spk_film)
                self.engine.SetProsodyModels(mgr.prosody_encoder, mgr.prosody_film)
                self.engine.SetFilmEnergyModel(film_enabled=(mgr.energy_film is not None), energy_film=mgr.energy_film)
                
                # Load pre-calculated style vector if available
                style_vec_p = os.path.join(self.model_path, "style", "style_vec.npy")
                if os.path.exists(style_vec_p): 
                    self.engine.SetStyleEmbedding(np.load(style_vec_p).astype(np.float32))
        except Exception as e: 
            print(f"  [Optional Module Skip] FiLM: {e}")

    def _load_acoustic_model(self):
        """Loads the core VAE/GAN acoustic synthesizer."""
        from HalimGSS.TimbreNode.Models.models import SynthesizerTrnMs768NSFsid
        
        # Find the best available EMA (Exponential Moving Average) weights
        ema_p = os.path.join(self.model_path, "TimbreNode", "G_best_EMA_by_MCD.pth")
        final_p = os.path.join(self.model_path, "TimbreNode", "G_Final_EMA.pth")
        weight_path = ema_p if os.path.exists(ema_p) else final_p
        
        m_cfg = self.model_config["model"]
        net_g = SynthesizerTrnMs768NSFsid(
            spec_channels=self.model_config["data"]["filter_length"] // 2 + 1,
            segment_size=self.model_config["train"]["segment_size"] // self.model_config["data"]["hop_length"],
            sr=self.model_config["sample_rate"], is_half=False, **m_cfg
        )
        
        checkpoint = torch.load(weight_path, map_location="cpu")
        net_g.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
        net_g.eval().to(self.device_obj)
        self.engine.SetAcousticModel(net_g)

    def _load_faiss_index(self):
        """Loads FAISS (Facebook AI Similarity Search) index for feature retrieval."""
        faiss_dir = os.path.join(self.model_path, "faiss_index")
        if not os.path.isdir(faiss_dir): 
            return
        
        import faiss
        for file in os.listdir(faiss_dir):
            if file.endswith(".index") and ("fused" in file or "added" in file):
                index = faiss.read_index(os.path.join(faiss_dir, file))
                npy_path = os.path.join(faiss_dir, "total_fused_features.npy")
                if os.path.exists(npy_path):
                    self.engine.set_faiss_index_settings(
                        file_index_fused=index, big_npy_fused=np.load(npy_path),
                        index_rate_fused=1.0, faiss_k=int(self.model_config.get("faiss_k", 8))
                    )
                    break

    def change_voice(self, audio_chunk_i16: np.ndarray) -> np.ndarray:
        """Processes 16-bit PCM chunk through the neural pipeline."""
        if self.engine is None: 
            raise RuntimeError("Engine not ready. Call load() first.")
        
        # Normalization and internal resampling
        f32 = audio_chunk_i16.astype(np.float32) / 32767.0
        audio_16k = librosa.resample(f32, orig_sr=48000, target_sr=16000)
        
        # Perform Inference
        processed_f32 = self.engine.infer(audio_16k=audio_16k)
        
        # Safe clip and back to 16-bit PCM
        return np.clip(processed_f32 * 32767.0, -32768, 32767).astype(np.int16)

    @classmethod
    def discover_models(cls, model_root: str) -> dict:
        """Utility to scan a directory for compatible hRVC voice models."""
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
