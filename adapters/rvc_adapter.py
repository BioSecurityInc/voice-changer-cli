import os
import sys
import json
import numpy as np
from adapters.base_adapter import BaseVoiceAdapter

class RVCVoiceAdapter(BaseVoiceAdapter):
    """Adapter for Standard RVC (W-Okada Engine)."""

    def __init__(self, model_path: str, device: str = None):
        super().__init__(model_path, device)
        self.manager = None
        self.slot_index = None

    def _inject_paths(self):
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        server_dir = os.path.join(script_dir, "server")
        if server_dir not in sys.path:
            sys.path.insert(0, server_dir)

    def load(self):
        self._inject_paths()
        from voice_changer.VoiceChangerManager import VoiceChangerManager
        from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
        
        # In RVC slot mode, the model_path is the directory of the slot
        # We need to find the slot index from params.json
        params_path = os.path.join(self.model_path, "params.json")
        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                data = json.load(f)
                self.slot_index = data.get("slotIndex")

        hubert_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server", "model_dir_static", "hubert_base.pt")
        params = VoiceChangerParams(model_dir=self.model_path, hubert_base=hubert_path)
        self.manager = VoiceChangerManager.get_instance(params)
        
        if self.slot_index is not None:
            self.manager.update_settings("modelSlotIndex", self.slot_index)
        
        self.manager.update_settings("f0Detector", self.params.get("f0_detector", "rmvpe"))
        self.manager.update_settings("tran", self.params.get("pitch", 0))

    def change_voice(self, audio_chunk_i16: np.ndarray) -> np.ndarray:
        if self.manager is None:
            raise RuntimeError("Manager not loaded. Call load() first.")
        
        # VoiceChangerManager.changeVoice usually returns (audio, metadata)
        out_i16, _ = self.manager.changeVoice(audio_chunk_i16)
        return out_i16

    @classmethod
    def discover_models(cls, model_root: str) -> dict:
        models = {}
        if not os.path.exists(model_root): return models
        for item in sorted(os.listdir(model_root)):
            path = os.path.join(model_root, item)
            if not os.path.isdir(path): continue
            params_path = os.path.join(path, "params.json")
            if os.path.exists(params_path):
                try:
                    with open(params_path, "r") as f: data = json.load(f)
                    slot_idx = data.get("slotIndex", item)
                    name     = data.get("name", f"Slot {slot_idx}")
                    models[slot_idx] = {"name": name, "path": path, "type": "rvc"}
                except Exception: pass
        return models
