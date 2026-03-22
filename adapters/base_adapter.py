"""
Abstract Base Module for Universal Voice Adapters
================================================
This module defines the standard interface (blueprint) that all 
voice conversion engine adapters must follow. 

Ensures a consistent API for the Universal CLI, allowing decoupled 
switching between different engines (RVC, HalimGSS, etc.).
"""

import abc
import numpy as np

class BaseVoiceAdapter(abc.ABC):
    """
    Abstract Base Class for Voice Changer Adapters.
    Encapsulates core requirements: loading models, performing inference, 
    and model discovery.
    """

    def __init__(self, model_path: str, engine_path: str = None, device: str = "cpu"):
        """
        Initializes the adapter with mandatory local paths.
        
        Args:
            model_path (str): Absolute path to the trained voice model files.
            engine_path (str, optional): Absolute path to the neural core (source code).
            device (str): Computation device, e.g., 'cpu', 'cuda', or 'mps'.
        """
        self.model_path = model_path
        self.engine_path = engine_path
        self.device = device
        self.params = {}

    @abc.abstractmethod
    def load(self):
        """
        Orchestrates the loading of the neural engine, weights, and indexes. 
        Must be implemented by subclasses.
        """
        pass

    @abc.abstractmethod
    def change_voice(self, audio_chunk_i16: np.ndarray) -> np.ndarray:
        """
        The core neural conversion method.
        
        Args:
            audio_chunk_i16 (np.ndarray): Input 16-bit PCM waveform array.
            
        Returns:
            np.ndarray: Processed 16-bit PCM waveform array.
        """
        pass

    def update_settings(self, **kwargs):
        """
        Updates runtime parameters like pitch shift, feature index, 
        or protector flags without reloading the core model.
        """
        self.params.update(kwargs)

    @classmethod
    @abc.abstractmethod
    def discover_models(cls, model_root: str) -> dict:
        """
        Scans a file-system directory and returns a dictionary of models 
        compatible with this specific adapter type.
        
        Format: { "model_id": { "name": "...", "path": "...", "type": "..." } }
        """
        pass
