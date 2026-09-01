import math
import os
import platform
import struct
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path


def generate_beep_wav(filepath, frequency=1000, duration_sec=0.4, sample_rate=44100, volume=0.8):
    """
    Generate a clean sine-wave WAV file using Python's standard library (zero external dependencies).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    num_samples = int(sample_rate * duration_sec)
    max_amplitude = 32767.0 * max(0.0, min(1.0, volume))

    with wave.open(str(filepath), "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        frames = bytearray()
        for i in range(num_samples):
            # Sine wave sample with smooth fade-in and fade-out to prevent audio pops
            t = float(i) / sample_rate
            envelope = 1.0
            fade_samples = int(sample_rate * 0.02)  # 20ms fade
            if i < fade_samples:
                envelope = float(i) / fade_samples
            elif i > num_samples - fade_samples:
                envelope = float(num_samples - i) / fade_samples

            sample_val = int(envelope * max_amplitude * math.sin(2.0 * math.pi * frequency * t))
            frames.extend(struct.pack("<h", sample_val))

        wav_file.writeframes(frames)


class Alarm:
    """
    Universal Cross-Platform Drowsiness Alarm Manager.
    
    Supports:
    - Windows: winsound.Beep / winsound.PlaySound
    - macOS: afplay (native CLI) / pygame / sounddevice
    - Linux & Raspberry Pi: aplay / paplay / pygame
    - Universal Fallback: System bell (\a) & synthesized WAV playback
    """

    def __init__(self, frequency=1000, duration_ms=400, interval_ms=200, sound_file=None):
        self.frequency = frequency
        self.duration_ms = duration_ms
        self.interval_sec = interval_ms / 1000.0
        self.system = platform.system().lower()

        # Path to cached alarm WAV file
        if sound_file is None:
            sound_dir = Path(__file__).resolve().parent.parent / "output"
            self.sound_file = sound_dir / "alarm_beep.wav"
        else:
            self.sound_file = Path(sound_file)

        # Threading controls
        self.stop_event = threading.Event()
        self.thread = None
        self._winsound = None
        self._pygame_sound = None

        self._detect_backend()

    def _detect_backend(self):
        """Identify available audio playback backend on the host OS."""
        if "windows" in self.system:
            try:
                import winsound
                self._winsound = winsound
                self.backend = "winsound"
                return
            except ImportError:
                pass

        # Check for Pygame mixer
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            if not self.sound_file.exists():
                generate_beep_wav(self.sound_file, self.frequency, duration_sec=self.duration_ms / 1000.0)
            self._pygame_sound = pygame.mixer.Sound(str(self.sound_file))
            self.backend = "pygame"
            return
        except Exception:
            pass

        # Check for native CLI audio tools on Linux/macOS
        if "darwin" in self.system:  # macOS
            if not self.sound_file.exists():
                generate_beep_wav(self.sound_file, self.frequency, duration_sec=self.duration_ms / 1000.0)
            self.backend = "afplay"
        elif "linux" in self.system:  # Linux / Raspberry Pi
            if not self.sound_file.exists():
                generate_beep_wav(self.sound_file, self.frequency, duration_sec=self.duration_ms / 1000.0)
            if self._has_command("aplay"):
                self.backend = "aplay"
            elif self._has_command("paplay"):
                self.backend = "paplay"
            else:
                self.backend = "bell"
        else:
            self.backend = "bell"

    @staticmethod
    def _has_command(cmd):
        """Check if a command exists in system PATH."""
        try:
            subprocess.run(["which", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except Exception:
            return False

    def _play_single_beep(self):
        """Play one warning beep using the detected backend."""
        try:
            if self.backend == "winsound" and self._winsound is not None:
                self._winsound.Beep(self.frequency, self.duration_ms)
            elif self.backend == "pygame" and self._pygame_sound is not None:
                self._pygame_sound.play()
                time.sleep(self.duration_ms / 1000.0)
            elif self.backend == "afplay":
                subprocess.run(["afplay", str(self.sound_file)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif self.backend in ("aplay", "paplay"):
                subprocess.run([self.backend, "-q", str(self.sound_file)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                # Terminal Bell fallback
                sys.stdout.write("\a")
                sys.stdout.flush()
                time.sleep(self.duration_ms / 1000.0)
        except Exception:
            sys.stdout.write("\a")
            sys.stdout.flush()

    def _alarm_worker(self):
        """Background loop continuously playing alarm beeps until stopped."""
        while not self.stop_event.is_set():
            self._play_single_beep()
            if self.stop_event.wait(self.interval_sec):
                break

    def start(self):
        """Start the alarm loop asynchronously without blocking the video stream."""
        if self.thread is not None and self.thread.is_alive():
            return

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._alarm_worker, name="AlarmThread", daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the alarm immediately."""
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.8)

    def is_active(self):
        """Check if alarm is currently ringing."""
        return self.thread is not None and self.thread.is_alive() and not self.stop_event.is_set()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
