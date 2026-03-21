import abc
import numpy as np

class BaseVoiceAdapter(abc.ABC):
    """
    Abstract Base Class for Voice Changer Adapters.
    Each engine (RVC, HalimGSS, etc.) should implement this.
    """

    def __init__(self, model_path: str, engine_path: str = None, device: str = "cpu"):
        self.model_path = model_path
        self.engine_path = engine_path
        self.device = device
        self.params = {}

    @abc.abstractmethod
    def load(self):
        """Load models, weights, and initialize the engine."""
        pass

    @abc.abstractmethod
    def change_voice(self, audio_chunk_i16: np.ndarray) -> np.ndarray:
        """
        Convert input audio chunk (int16) through the engine.
        Returns: converted audio chunk (int16).
        """
        pass

    def update_settings(self, **kwargs):
        """Update runtime settings like pitch, index rate, etc."""
        self.params.update(kwargs)

    @classmethod
    @abc.abstractmethod
    def discover_models(cls, model_root: str) -> dict:
        """
        Scan a directory and return a dict of models this adapter can handle.
        Format: { "id": { "name": "...", "path": "...", "type": "..." } }
        """
        pass
