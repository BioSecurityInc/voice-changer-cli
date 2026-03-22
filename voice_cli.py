#!/usr/bin/env python3
"""
🎙️ UNIVERSAL VOICE CHANGER CLI — Academic Research Edition
=========================================================
A professional, modular command-line interface for conducting 
neural voice conversion experiments. Optimized for Apple Silicon (MPS).

Features:
- Real-time dB monitoring during recording.
- Dynamic plugin/adapter loading via importlib.
- Device state persistence (remembers your last I/O choice).
- Cross-platform dependency patching.
"""

import sys
import os
import argparse
import time
import importlib.util
import warnings
import torch
import numpy as np
import sounddevice as sd
import librosa
import json
import traceback
import threading
from typing import Optional, Type

# ── Stability & Compatibility Patches ──────────────────────────────────────────
def apply_stability_patches():
    """
    Ensures the system runs smoothly across different versions of PyTorch 
    and sound engines, specifically handling Mac ARM64 edge cases.
    """
    # Fix for SpeechBrain/Torchaudio version mismatch
    if not hasattr(torch, "list_audio_backends"):
        try:
            import torchaudio
            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: []
        except ImportError: pass

    # Fix for PyTorch 2.6+ to allow loading older model weights (RVC v2 compatible)
    orig_load = torch.load
    def patched_load(*args, **kwargs):
        if "weights_only" not in kwargs: 
            kwargs["weights_only"] = False
        return orig_load(*args, **kwargs)
    torch.load = patched_load
    
    # Suppress non-critical library warnings for a cleaner CLI experience
    warnings.filterwarnings("ignore")

apply_stability_patches()

# ── Visual Components & Theme ───────────────────────────────────────────────────
class Theme:
    """ANSI color codes for consistent and professional UI styling."""
    BOLD   = "\033[1m"
    GREEN  = "\033[32m"
    CYAN   = "\033[36m"
    BLUE   = "\033[34m"
    YELLOW = "\033[33m"
    RED    = "\033[31m"
    RESET  = "\033[0m"
    DIM    = "\033[2m"
    
    @staticmethod
    def header(text): 
        return f"\n{Theme.CYAN}{Theme.BOLD}── {text} ────────────────────────────────────────{Theme.RESET}"
    
    @staticmethod
    def success(text): 
        return f"{Theme.GREEN}✔ {text}{Theme.RESET}"
    
    @staticmethod
    def info(text): 
        return f"{Theme.BLUE}ℹ {Theme.RESET}{text}"
    
    @staticmethod
    def error(text): 
        return f"{Theme.RED}✘ {text}{Theme.RESET}"
    
    @staticmethod
    def warn(text): 
        return f"{Theme.YELLOW}⚠ {text}{Theme.RESET}"

# ── User Interface Layer ────────────────────────────────────────────────────────
class UI:
    """Handles CLI visualization, banners, and keyboard input processing."""
    
    @staticmethod
    def banner():
        """Prints the professional application header."""
        title = "🎙  UNIVERSAL VOICE CHANGER CLI"
        subtitle = "Modular Neural Engine Interface"
        width = 58
        print(f"\n{Theme.CYAN}{Theme.BOLD}╔" + "═"*width + "╗")
        print(f"║{Theme.RESET}{Theme.BOLD}{title:^{width}}{Theme.CYAN}{Theme.BOLD}║")
        print(f"║{Theme.RESET}{Theme.DIM}{subtitle:^{width}}{Theme.CYAN}{Theme.BOLD}║")
        print(f"╚" + "═"*width + f"╝{Theme.RESET}")

    @staticmethod
    def get_char():
        """Reads a single keypress without requiring Enter (Unix-specific)."""
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally: 
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    @staticmethod
    def select_item(prompt, items, default_id=None):
        """Standardized menu for selecting audio devices or engines."""
        print(Theme.header(prompt))
        for i, (idx, name) in enumerate(items):
            is_def = f" {Theme.YELLOW}{Theme.BOLD}[DEFAULT]{Theme.RESET}" if idx == default_id else ""
            print(f"  {Theme.GREEN}[{i}]{Theme.RESET}  {name}  {Theme.DIM}(ID: {idx}){Theme.RESET}{is_def}")
        
        prompt_text = f"\n{Theme.BOLD}Choice index"
        if default_id is not None:
            prompt_text += f" {Theme.DIM}(Enter for default){Theme.RESET}"
        prompt_text += f":{Theme.RESET} "
        
        while True:
            try:
                raw = input(prompt_text).strip()
                if raw == "" and default_id is not None:
                    return default_id
                choice = int(raw)
                if 0 <= choice < len(items): 
                    return items[choice][0]
            except (ValueError, KeyboardInterrupt): pass
            print(Theme.error("Invalid input. Please choose a valid index."))

# ── Audio Processing Layer ──────────────────────────────────────────────────────
class AudioHandler:
    """Manages audio hardware interaction: listing devices, recording, and playback."""
    
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate

    def list_devices(self):
        """Scans the system for all available input and output audio endpoints."""
        all_devs = sd.query_devices()
        inputs = [(i, d["name"]) for i, d in enumerate(all_devs) if d["max_input_channels"] > 0]
        outputs = [(i, d["name"]) for i, d in enumerate(all_devs) if d["max_output_channels"] > 0]
        return inputs, outputs

    def record_until_space(self, input_id):
        """Interactive recording logic with a real-time RMS visualizer."""
        print(f"\n{Theme.info('Press ')}{Theme.BOLD}SPACE{Theme.RESET}{Theme.info(' to start recording…')}")
        while UI.get_char() != " ": pass
        
        print(f"  {Theme.RED}{Theme.BOLD}⏺  RECORDING — Press SPACE to stop{Theme.RESET}")
        
        recorded_data = []
        stop_event = threading.Event()

        def callback(indata, frames, time, status):
            """SoundDevice stream callback: captures raw audio and updates the level meter."""
            if not stop_event.is_set():
                recorded_data.append(indata.copy())
                # Live Level Meter Calculation (dB RMS)
                db = 20 * np.log10(np.sqrt(np.mean(indata**2))) if np.any(indata) else -200
                meter = '█' * int(max(0, (db + 60) / 60 * 30))
                sys.stdout.write(f"\r  🎤 [{meter:<30}] {db:6.1f} dB")
                sys.stdout.flush()

        # Start recording stream
        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=callback, device=input_id):
            while UI.get_char() != " ": pass
            stop_event.set()
        
        final_array = np.concatenate(recorded_data).flatten()
        print(f"\n{Theme.success('Captured ' + str(round(len(final_array)/self.sample_rate, 2)) + 's')}")
        
        # Convert to 16-bit PCM for the inference engine
        return (final_array * 32767.0).astype(np.int16)

    def play(self, data_i16, output_id):
        """Outputs processed 16-bit PCM audio back to the selected hardware target."""
        print(f"{Theme.info('Playing back through hardware…')}")
        sd.play(data_i16.astype(np.float32) / 32767.0, self.sample_rate, device=output_id)
        sd.wait()

# ── Plugin / Adapter System ─────────────────────────────────────────────────────
class AdapterManager:
    """Implements dynamic loading of engine-specific adapters at runtime."""
    
    @staticmethod
    def load(adapt_input):
        """Discovers and instantiates a VoiceAdapter class from a file."""
        from adapters.base_adapter import BaseVoiceAdapter
        file_path = None
        
        # Ensure correct file extension
        name = adapt_input if adapt_input.endswith(".py") else f"{adapt_input}.py"
        
        # Search priority: 1. Absolute Path / Current Dir, 2. adapters/ subdirectory
        if os.path.exists(name):
            file_path = os.path.abspath(name)
        else:
            candidate = os.path.join(os.path.dirname(__file__), "adapters", name)
            if os.path.exists(candidate):
                file_path = candidate
        
        if not file_path:
            raise FileNotFoundError(f"Adapter error: '{name}' not found.")

        # Perform the dynamic import
        spec = importlib.util.spec_from_file_location("adapter_mod", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Scan module for any subclass of BaseVoiceAdapter
        for name in dir(module):
            cls = getattr(module, name)
            if isinstance(cls, type) and issubclass(cls, BaseVoiceAdapter) and cls is not BaseVoiceAdapter:
                print(Theme.success(f"Plugin Initialized: {name}"))
                return cls
        return None

# ── Main Application Workflow ────────────────────────────────────────────────────
class VoiceChangerApp:
    """Core application logic: setup, persistence, and inference cycles."""
    
    def __init__(self, args):
        self.args = args
        self.audio = AudioHandler()
        self.adapter = None
        self.config_path = os.path.join(os.path.dirname(__file__), "last_devices.json")

    def load_last_devices(self):
        """Loads previously used audio devices to speed up repeated runs."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    return json.load(f)
            except: pass
        return {}

    def save_devices(self, input_id, output_id):
        """Persists chosen devices to a local configuration file."""
        try:
            with open(self.config_path, "w") as f:
                json.dump({"input_id": input_id, "output_id": output_id}, f)
        except: pass

    def setup(self):
        """Initializes hardware, persists settings, and loads the neural engine."""
        UI.banner()
        
        # Audio Configuration
        last = self.load_last_devices()
        inputs, outputs = self.audio.list_devices()
        
        # Resolving Input/Output (priority: CLI args -> persistence -> user menu)
        self.input_id = self.args.input_device if self.args.input_device is not None else \
                        UI.select_item("Select Input Source", inputs, default_id=last.get("input_id"))

        self.output_id = self.args.output_device if self.args.output_device is not None else \
                         UI.select_item("Select Output Target", outputs, default_id=last.get("output_id"))

        self.save_devices(self.input_id, self.output_id)

        # Engine Validation
        if not self.args.adapt or not self.args.model:
            print(Theme.error("Required parameters missing: -adapt and -model must be provided."))
            sys.exit(1)

        adapter_cls = AdapterManager.load(self.args.adapt)
        if not adapter_cls:
            print(Theme.error("Failed to load a valid adapter class."))
            sys.exit(1)

        model_path = os.path.abspath(self.args.model)
        if not os.path.exists(model_path):
            print(Theme.error(f"Inaccessible model path: {model_path}"))
            sys.exit(1)

        # Neural Engine Initialization
        print(Theme.header("Initializing Engine"))
        self.adapter = adapter_cls(model_path=model_path, engine_path=self.args.engine)
        self.adapter.update_settings(pitch=self.args.pitch, index=self.args.index, protect=self.args.protect)
        
        t0 = time.time()
        self.adapter.load()
        print(Theme.success(f"Engine Ready in {time.time()-t0:.2f}s"))

    def run_inference_cycle(self):
        """Executes a single capture -> convert -> playback cycle."""
        # Step 1: Record user voice
        audio_in = self.audio.record_until_space(self.input_id)
        
        # Step 2: Perform Neural Conversion
        print(f"\n{Theme.header('Neural Processing')}")
        t1 = time.time()
        try:
            audio_out = self.adapter.change_voice(audio_in)
            print(Theme.success(f"Inference complete in {time.time()-t1:.2f}s"))
            
            # Step 3: Audio Playback
            self.audio.play(audio_out, self.output_id)
        except Exception as e:
            print(Theme.error(f"Inference processing failed: {e}"))
            traceback.print_exc()

# ── Entry Point ──────────────────────────────────────────────────────────────
def build_parser():
    """Defines command-line arguments for research flexibility."""
    p = argparse.ArgumentParser(description="Professional Universal Voice Changer CLI")
    p.add_argument("-model", help="Path to the trained model directory")
    p.add_argument("-adapt", help="Adapter module name or full path")
    p.add_argument("-engine", help="Path to the core engine (HalimGSS, etc.)")
    p.add_argument("--input_device", type=int, help="Force specific input device ID")
    p.add_argument("--output_device", type=int, help="Force specific output device ID")
    p.add_argument("--pitch", type=int, default=0, help="Semitone pitch shift (-12 to 12)")
    p.add_argument("--index", type=float, default=0.7, help="Feature index ratio (0.0 to 1.0)")
    p.add_argument("--protect", type=float, default=0.5, help="Voiceless protection threshold")
    return p

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    
    app = VoiceChangerApp(args)
    try:
        app.setup()
        app.run_inference_cycle()
        print(f"\n{Theme.GREEN}{Theme.BOLD}✔ ALL OPERATIONS COMPLETE{Theme.RESET}\n")
    except KeyboardInterrupt:
        print(f"\n{Theme.warn('Execution terminated by user.')}")
    except Exception as e:
        print(Theme.error(f"Critical System Failure: {e}"))
        traceback.print_exc()
