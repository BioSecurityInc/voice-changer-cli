import os
import sys
import json
import torch
import numpy as np
import librosa
import traceback
from adapters.base_adapter import BaseVoiceAdapter

# ── Librosa Pitch Predictor ────────────────────────────────────────────────────
class LibrosaF0Predictor:
    """High-performance F0 estimation using librosa's pyin."""
    def __init__(self, sample_rate=16000, hop_length=160):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f0_bin = 256
        self.f0_mel_min = 1127.0 * np.log(1.0 + 50.0 / 700.0)
        self.f0_mel_max = 1127.0 * np.log(1.0 + 1100.0 / 700.0)
    
    def compute_f0(self, wav, p_len=None):
        wav_f32 = np.asarray(wav, dtype=np.float32).flatten()
        f0, _, _ = librosa.pyin(
            wav_f32, fmin=85.0, fmax=1100.0,
            sr=self.sample_rate, frame_length=1024,
            hop_length=self.hop_length, fill_na=0.0
        )
        rms = librosa.feature.rms(y=wav_f32, frame_length=1024, hop_length=self.hop_length)[0]
        L = min(len(f0), len(rms))
        f0, rms = f0[:L], rms[:L]
        f0[rms < 10**(-55.0/20.0)] = 0.0 # -55dBFS Threshold

        from scipy.signal import medfilt
        if np.count_nonzero(f0) > 3:
            f0 = np.where(f0 > 0, medfilt(f0, kernel_size=7), 0.0)
        
        if p_len:
            f0 = np.pad(f0, (0, max(0, p_len - len(f0))))[:p_len]
        return f0.astype(np.float32)

    def coarse_f0(self, f0):
        f0 = np.where(np.isfinite(f0), f0, 0.0)
        f0_mel = 1127.0 * np.log(1.0 + (f0 / 700.0))
        pos = f0_mel > 0.0
        if np.any(pos):
            f0_mel[pos] = (f0_mel[pos] - self.f0_mel_min) * (self.f0_bin - 2) / (self.f0_mel_max - self.f0_mel_min) + 1.0
        return np.clip(f0_mel, 1.0, self.f0_bin - 1).astype(np.float32)

# ── hRVC Adapter ───────────────────────────────────────────────────────────────

class HRVCVoiceAdapter(BaseVoiceAdapter):
    """
    Adapter for hRVC (Halim-RVC) Neural Engine.
    Decoupled and optimized for modern voice conversion workloads.
    """

    def __init__(self, model_path: str, engine_path: str = None, device: str = None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        super().__init__(model_path, engine_path, device)
        
        self.engine = None
        self.device_obj = torch.device(device)
        self.model_config = {}

    def _inject_engine_paths(self):
        """Set up environment for engine-specific imports."""
        if not self.engine_path:
            # Fallback path logic
            script_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.engine_path = os.path.join(script_root, "model_source", "HalimGSSv4.2.228")
        
        if os.path.exists(self.engine_path) and self.engine_path not in sys.path:
            sys.path.insert(0, self.engine_path)

    def load(self):
        """Sequential loading of engine components."""
        self._inject_engine_paths()
        
        # 1. Load Model Config
        config_path = os.path.join(self.model_path, "model.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config /model.json/ missing in {self.model_path}")
        with open(config_path, "r") as f:
            self.model_config = json.load(f)

        # 2. Initialize Engine & Predictors
        from HalimGSS.TimbreNode.Inference.TimbreInference import TimbreInference
        
        class EngConfig:
            def __init__(self, dev):
                self.device = dev
                self.is_half, self.x_pad, self.hop_length_tgt, self.use_torch_hpf, self.hpf_cutoff_hz = False, 0.3, 480, False, 60.0

        self.engine = TimbreInference(tgt_sr=48000, config=EngConfig(self.device_obj))
        self.engine.model_rmvpe = LibrosaF0Predictor() 
        
        # 3. Component Loading
        try:
            self._load_phonetic_encoders()
            self._load_film_conditioning()
            self._load_acoustic_model()
            self._load_faiss_index()
            
            # Sync settings from adapter params
            self.engine.index_rate_fused = self.params.get("index", 0.7)
            self.engine.f0_up_key = self.params.get("pitch", 0.0)
            self.engine.protect = self.params.get("protect", 0.5)
            
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"hRVC Loading Pipeline Failed: {e}")

    def _load_phonetic_encoders(self):
        from HalimGSS.TimbreNode.Phonetic.LoadHubert import HubertLoader
        from HalimGSS.TimbreNode.Phonetic.LoadContentVec import ContentVecLoader
        
        assets = os.path.join(self.engine_path, "assets")
        hubert = HubertLoader(model_path=os.path.join(assets, "mHuBERT-147"), is_half=False)
        contentvec = ContentVecLoader(model_path=os.path.join(assets, "contentvec700", "checkpoint_best_legacy_500.pt"), is_half=False)
        
        self.engine.SetPhoneticModel(hubert_model=hubert, contentvec_model=contentvec)

    def _load_film_conditioning(self):
        try:
            from HalimGSS.TimbreNode.Models.FiLMManager import FiLMManager
            from HalimGSS.TimbreNode.Inference.SearchCheckpoint import resolve_checkpoint_dir
            
            ckpt_dir = resolve_checkpoint_dir(self.model_path)
            film_path = os.path.join(ckpt_dir, "FiLM_latest.pth")
            
            if os.path.exists(film_path):
                ck = torch.load(film_path, map_location="cpu")
                blob = ck.get("model", ck) if isinstance(ck, dict) else ck
                
                mgr = FiLMManager(
                    d_model=768, style_dim=256, device=str(self.device_obj),
                    enable_energy=("energy_film" in blob or any(k.startswith("gamma_mlp") for k in blob.keys())),
                    enable_speaker=("speaker_film" in blob),
                    enable_style=("style_film" in blob),
                    enable_prosody=("prosody_film" in blob or "prosody_encoder" in blob)
                )
                mgr.load_film_state_dict(blob, strict=False)
                
                self.engine.SetFilmStyleModel(mgr.style_film)
                self.engine.SetFilmSpeakerModel(mgr.spk_film)
                self.engine.SetProsodyModels(mgr.prosody_encoder, mgr.prosody_film)
                self.engine.SetFilmEnergyModel(film_enabled=(mgr.energy_film is not None), energy_film=mgr.energy_film)
                
                st_vec = os.path.join(self.model_path, "style", "style_vec.npy")
                if os.path.exists(st_vec): 
                    self.engine.SetStyleEmbedding(np.load(st_vec).astype(np.float32))
        except Exception as e: print(f"  [Warn] FiLM: {e}")

    def _load_acoustic_model(self):
        from HalimGSS.TimbreNode.Models.models import SynthesizerTrnMs768NSFsid
        
        # Determine weight path (EMA priority)
        ema_p = os.path.join(self.model_path, "TimbreNode", "G_best_EMA_by_MCD.pth")
        final_p = os.path.join(self.model_path, "TimbreNode", "G_Final_EMA.pth")
        w_path = ema_p if os.path.exists(ema_p) else final_p
        
        m_cfg = self.model_config["model"]
        net_g = SynthesizerTrnMs768NSFsid(
            spec_channels=self.model_config["data"]["filter_length"] // 2 + 1,
            segment_size=self.model_config["train"]["segment_size"] // self.model_config["data"]["hop_length"],
            sr=self.model_config["sample_rate"], is_half=False, **m_cfg
        )
        
        ckpt = torch.load(w_path, map_location="cpu")
        net_g.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        net_g.eval().to(self.device_obj)
        self.engine.SetAcousticModel(net_g)

    def _load_faiss_index(self):
        faiss_p = os.path.join(self.model_path, "faiss_index")
        if not os.path.isdir(faiss_p): return
        
        import faiss
        for f in os.listdir(faiss_p):
            if f.endswith(".index") and ("fused" in f or "added" in f):
                idx = faiss.read_index(os.path.join(faiss_p, f))
                npy_p = os.path.join(faiss_p, "total_fused_features.npy")
                if os.path.exists(npy_p):
                    self.engine.set_faiss_index_settings(
                        file_index_fused=idx, big_npy_fused=np.load(npy_p),
                        index_rate_fused=1.0, faiss_k=int(self.model_config.get("faiss_k", 8))
                    )
                    break

    def change_voice(self, audio_chunk_i16: np.ndarray) -> np.ndarray:
        if self.engine is None: raise RuntimeError("hRVC Engine not initialized.")
        
        # Internal pipeline: resample to 16k -> infer -> resample back happens if needed, 
        # but original engine expects 16k for phonetic loaders.
        f32 = audio_chunk_i16.astype(np.float32) / 32767.0
        f16k = librosa.resample(f32, orig_sr=48000, target_sr=16000)
        out_f32 = self.engine.infer(audio_16k=f16k)
        return np.clip(out_f32 * 32767.0, -32768, 32767).astype(np.int16)

    @classmethod
    def discover_models(cls, model_root: str) -> dict:
        models = {}
        if not os.path.exists(model_root): return models
        for item in sorted(os.listdir(model_root)):
            path = os.path.join(model_root, item)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "model.json")):
                try:
                    with open(os.path.join(path, "model.json"), "r") as f: data = json.load(f)
                    models[item] = {"name": f"⭐ {data.get('name', item)} (hRVC)", "path": path, "type": "hrvc"}
                except Exception: pass
        return models
