#!/usr/bin/env python3
"""
Universal Voice Changer CLI — Refactored Edition
================================================
A premium, modular CLI for neural voice conversion.
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

import torch.serialization

# ── Stability & Compatibility ──────────────────────────────────────────────────
def apply_stability_patches():
    # Fix torchaudio compatibility with speechbrain
    if not hasattr(torch, "list_audio_backends"):
        try:
            import torchaudio
            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: []
        except ImportError: pass

    # Fix PyTorch 2.6+ weights_only loading issue
    orig_load = torch.load
    def patched_load(*args, **kwargs):
        if "weights_only" not in kwargs: kwargs["weights_only"] = False
        return orig_load(*args, **kwargs)
    torch.load = patched_load
    warnings.filterwarnings("ignore")

apply_stability_patches()

# ── Design System ──────────────────────────────────────────────────────────────
class Theme:
    BOLD   = "\033[1m"
    GREEN  = "\033[32m"
    CYAN   = "\033[36m"
    BLUE   = "\033[34m"
    YELLOW = "\033[33m"
    RED    = "\033[31m"
    RESET  = "\033[0m"
    DIM    = "\033[2m"
    
    @staticmethod
    def header(text): return f"\n{Theme.CYAN}{Theme.BOLD}── {text} ────────────────────────────────────────{Theme.RESET}"
    @staticmethod
    def success(text): return f"{Theme.GREEN}✔ {text}{Theme.RESET}"
    @staticmethod
    def info(text): return f"{Theme.BLUE}ℹ {Theme.RESET}{text}"
    @staticmethod
    def error(text): return f"{Theme.RED}✘ {text}{Theme.RESET}"
    @staticmethod
    def warn(text): return f"{Theme.YELLOW}⚠ {text}{Theme.RESET}"

# ── Core Components ────────────────────────────────────────────────────────────

class UI:
    @staticmethod
    def banner():
        title = "🎙  UNIVERSAL VOICE CHANGER CLI"
        subtitle = "Modular Neural Engine Interface"
        width = 58
        print(f"\n{Theme.CYAN}{Theme.BOLD}╔" + "═"*width + "╗")
        print(f"║{Theme.RESET}{Theme.BOLD}{title:^{width}}{Theme.CYAN}{Theme.BOLD}║")
        print(f"║{Theme.RESET}{Theme.DIM}{subtitle:^{width}}{Theme.CYAN}{Theme.BOLD}║")
        print(f"╚" + "═"*width + f"╝{Theme.RESET}")

    @staticmethod
    def get_char():
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally: termios.tcsetattr(fd, termios.TCSADRAIN, old)

    @staticmethod
    def select_item(prompt, items, default_id=None):
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
                if 0 <= choice < len(items): return items[choice][0]
            except (ValueError, KeyboardInterrupt): pass
            print(Theme.error("Invalid input. Try again."))

class AudioHandler:
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate

    def list_devices(self):
        all_devs = sd.query_devices()
        inputs = [(i, d["name"]) for i, d in enumerate(all_devs) if d["max_input_channels"] > 0]
        outputs = [(i, d["name"]) for i, d in enumerate(all_devs) if d["max_output_channels"] > 0]
        return inputs, outputs

    def record_until_space(self, input_id):
        print(f"\n{Theme.info('Press ')}{Theme.BOLD}SPACE{Theme.RESET}{Theme.info(' to start recording…')}")
        while UI.get_char() != " ": pass
        
        print(f"  {Theme.RED}{Theme.BOLD}⏺  RECORDING — Press SPACE to stop{Theme.RESET}")
        
        recorded_data = []
        stop_event = threading.Event()

        def callback(indata, frames, time, status):
            if not stop_event.is_set():
                recorded_data.append(indata.copy())
                # Live Level Meter
                db = 20 * np.log10(np.sqrt(np.mean(indata**2))) if np.any(indata) else -100
                meter = '█' * int((db + 60) / 60 * 30)
                sys.stdout.write(f"\r  🎤 [{meter:<30}] {db:6.1f} dB")
                sys.stdout.flush()

        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=callback, device=input_id):
            while UI.get_char() != " ": pass
            stop_event.set()
        
        print(f"\n{Theme.success('Captured ' + str(round(len(np.concatenate(recorded_data))/self.sample_rate, 2)) + 's')}")
        return (np.concatenate(recorded_data).flatten() * 32767.0).astype(np.int16)

    def play(self, data_i16, output_id):
        print(f"{Theme.info('Playing back through hardware…')}")
        sd.play(data_i16.astype(np.float32) / 32767.0, self.sample_rate, device=output_id)
        sd.wait()

class AdapterManager:
    @staticmethod
    def load(adapt_input):
        from adapters.base_adapter import BaseVoiceAdapter
        file_path = None
        
        # Ensure .py extension
        name = adapt_input if adapt_input.endswith(".py") else f"{adapt_input}.py"
        
        # 1. Search in current path
        if os.path.exists(name):
            file_path = os.path.abspath(name)
        # 2. Search in adapters/ folder
        else:
            candidate = os.path.join(os.path.dirname(__file__), "adapters", name)
            if os.path.exists(candidate):
                file_path = candidate
        
        if not file_path:
            raise FileNotFoundError(f"Adapter not found: {name} (tried current dir and adapters/ folder)")

        spec = importlib.util.spec_from_file_location("adapter_mod", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        for name in dir(module):
            cls = getattr(module, name)
            if isinstance(cls, type) and issubclass(cls, BaseVoiceAdapter) and cls is not BaseVoiceAdapter:
                print(Theme.success(f"Plugin Initialized: {name}"))
                return cls
        return None

# ══════════════════════════════════════════════════════════════════════════════
# Main Application Class
# ══════════════════════════════════════════════════════════════════════════════

class VoiceChangerApp:
    def __init__(self, args):
        self.args = args
        self.audio = AudioHandler()
        self.adapter = None
        self.config_path = os.path.join(os.path.dirname(__file__), "last_devices.json")

    def load_last_devices(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    return json.load(f)
            except: pass
        return {}

    def save_devices(self, input_id, output_id):
        with open(self.config_path, "w") as f:
            json.dump({"input_id": input_id, "output_id": output_id}, f)

    def setup(self):
        UI.banner()
        
        # 1. Device Selection (with persistence)
        last = self.load_last_devices()
        inputs, outputs = self.audio.list_devices()
        
        # Determine Input
        if self.args.input_device is not None:
            self.input_id = self.args.input_device
        else:
            self.input_id = UI.select_item("Select Input Source", inputs, default_id=last.get("input_id"))

        # Determine Output
        if self.args.output_device is not None:
            self.output_id = self.args.output_device
        else:
            self.output_id = UI.select_item("Select Output Target", outputs, default_id=last.get("output_id"))

        # Save for next time
        self.save_devices(self.input_id, self.output_id)

        # 2. Engine Loading
        if not self.args.adapt or not self.args.model:
            print(Theme.error("Missing -adapt or -model parameters."))
            sys.exit(1)

        adapter_cls = AdapterManager.load(self.args.adapt)
        if not adapter_cls:
            print(Theme.error("Could not find a valid adapter class in the specified file."))
            sys.exit(1)

        model_path = os.path.abspath(self.args.model)
        if not os.path.exists(model_path):
            print(Theme.error(f"Model path does not exist: {model_path}"))
            sys.exit(1)

        # 3. Initialization
        print(Theme.header("Initializing Engine"))
        self.adapter = adapter_cls(model_path=model_path, engine_path=self.args.engine)
        self.adapter.update_settings(pitch=self.args.pitch, index=self.args.index, protect=self.args.protect)
        
        start_init = time.time()
        self.adapter.load()
        print(Theme.success(f"Engine Ready in {time.time()-start_init:.2f}s"))

    def run_inference_cycle(self):
        # Record
        audio_in = self.audio.record_until_space(self.input_id)
        
        # Process
        print(f"\n{Theme.header('Neural Processing')}")
        start_p = time.time()
        try:
            audio_out = self.adapter.change_voice(audio_in)
            print(Theme.success(f"Inference complete in {time.time()-start_p:.2f}s"))
            
            # Play
            self.audio.play(audio_out, self.output_id)
        except Exception as e:
            print(Theme.error(f"Inference failed: {e}"))
            traceback.print_exc()

def build_parser():
    p = argparse.ArgumentParser(description="Universal Voice Changer CLI — Refactored")
    p.add_argument("-model", help="Path to model folder")
    p.add_argument("-adapt", help="Adapter name/path")
    p.add_argument("-engine", help="Path to engine source")
    p.add_argument("--input_device", type=int)
    p.add_argument("--output_device", type=int)
    p.add_argument("-s", "--select", action="store_true", help="Force device selection")
    p.add_argument("--pitch", type=int, default=0)
    p.add_argument("--index", type=float, default=0.7)
    p.add_argument("--protect", type=float, default=0.5)
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
        print(f"\n{Theme.warn('Session interrupted by user.')}")
    except Exception as e:
        print(Theme.error(f"Critical System Error: {e}"))
        traceback.print_exc()
