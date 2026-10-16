from __future__ import annotations

import importlib.util
import io
import json
import os
import threading
import wave
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


MAX_VOICE_AUDIO_BYTES = 5_000_000
MAX_VOICE_SECONDS = 30
MAX_SPEECH_TEXT_LENGTH = 2_000
MAX_SYNTHESIZED_AUDIO_BYTES = 8_000_000
SAMPLE_RATE = 16_000


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


class VoiceInputError(ValueError):
    pass


class VoiceUnavailable(RuntimeError):
    pass


class LocalVoiceService:
    """Local Vosk recognition and loopback-only Piper speech synthesis."""

    def __init__(
        self,
        asr_model_path: str | None = None,
        piper_url: str | None = None,
    ) -> None:
        configured_model = asr_model_path or os.environ.get("ELIA_VOSK_MODEL")
        self.asr_model_path = Path(configured_model).expanduser() if configured_model else None
        self.piper_url = piper_url or os.environ.get(
            "ELIA_PIPER_URL",
            "http://127.0.0.1:5000/synthesize",
        )
        self._model = None
        self._model_lock = threading.Lock()
        self._synthesis_lock = threading.Lock()
        self._validate_piper_url()

    def _validate_piper_url(self) -> None:
        parsed = urlsplit(self.piper_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path != "/synthesize"
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "ELIA_PIPER_URL must be a loopback HTTP /synthesize URL"
            )

    def status(self) -> dict[str, object]:
        return {
            "speech_to_text": {
                "engine": "Vosk",
                "configured": self.asr_model_path is not None,
                "model_available": bool(
                    self.asr_model_path and self.asr_model_path.is_dir()
                ),
                "dependency_available": importlib.util.find_spec("vosk") is not None,
                "input": "mono 16 kHz signed 16-bit PCM WAV, up to 30 seconds",
            },
            "text_to_speech": {
                "engine": "Piper",
                "configured_url": "loopback Piper HTTP service",
                "url": self.piper_url,
                "output": "WAV",
            },
            "external_api_fallback": False,
        }

    @staticmethod
    def _validate_wav(audio: bytes) -> tuple[bytes, float]:
        if not audio or len(audio) > MAX_VOICE_AUDIO_BYTES:
            raise VoiceInputError(
                f"Audio must be between 1 and {MAX_VOICE_AUDIO_BYTES} bytes"
            )
        try:
            with wave.open(io.BytesIO(audio), "rb") as source:
                channels = source.getnchannels()
                width = source.getsampwidth()
                rate = source.getframerate()
                frames = source.getnframes()
                if source.getcomptype() != "NONE":
                    raise VoiceInputError("Audio WAV must use uncompressed PCM")
                if channels != 1 or width != 2 or rate != SAMPLE_RATE:
                    raise VoiceInputError(
                        "Audio WAV must be mono, 16-bit PCM at 16000 Hz"
                    )
                if frames <= 0 or frames > SAMPLE_RATE * MAX_VOICE_SECONDS:
                    raise VoiceInputError(
                        f"Audio duration must be between 0 and {MAX_VOICE_SECONDS} seconds"
                    )
                pcm = source.readframes(frames)
        except (wave.Error, EOFError) as exc:
            raise VoiceInputError("Audio must be a valid PCM WAV file") from exc
        return pcm, frames / rate

    def _load_model(self):
        if self.asr_model_path is None or not self.asr_model_path.is_dir():
            raise VoiceUnavailable(
                "Local speech recognition is not configured. Set ELIA_VOSK_MODEL "
                "to a downloaded Vosk model directory."
            )
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            try:
                from vosk import Model
            except ImportError as exc:
                raise VoiceUnavailable(
                    "Vosk is not installed. Install ELIA's optional voice dependency."
                ) from exc
            try:
                self._model = Model(str(self.asr_model_path))
            except Exception as exc:
                raise VoiceUnavailable(
                    "Vosk could not load the configured local model."
                ) from exc
        return self._model

    def transcribe_wav(self, audio: bytes) -> dict[str, object]:
        pcm, duration = self._validate_wav(audio)
        model = self._load_model()
        try:
            from vosk import KaldiRecognizer

            recognizer = KaldiRecognizer(model, SAMPLE_RATE)
            for offset in range(0, len(pcm), 8_000):
                recognizer.AcceptWaveform(pcm[offset:offset + 8_000])
            result = json.loads(recognizer.FinalResult())
        except (ImportError, json.JSONDecodeError, TypeError) as exc:
            raise VoiceUnavailable(
                "Local speech recognition returned an invalid result."
            ) from exc
        except Exception as exc:
            raise VoiceUnavailable("Local speech recognition failed.") from exc
        transcript = result.get("text")
        if not isinstance(transcript, str):
            raise VoiceUnavailable("Local speech recognition returned no transcript.")
        return {
            "text": transcript.strip(),
            "duration_seconds": round(duration, 3),
            "engine": "vosk_local",
        }

    @staticmethod
    def _validate_synthesized_wav(audio: bytes) -> None:
        if not audio or len(audio) > MAX_SYNTHESIZED_AUDIO_BYTES:
            raise VoiceUnavailable("Piper returned an empty or oversized audio file.")
        try:
            with wave.open(io.BytesIO(audio), "rb") as source:
                if (
                    source.getcomptype() != "NONE"
                    or source.getnchannels() not in (1, 2)
                    or source.getsampwidth() not in (1, 2, 3, 4)
                    or source.getnframes() <= 0
                ):
                    raise VoiceUnavailable("Piper returned an unsupported WAV file.")
                if source.getnframes() / source.getframerate() > 60:
                    raise VoiceUnavailable("Piper response exceeds 60 seconds.")
        except (wave.Error, EOFError, ZeroDivisionError) as exc:
            raise VoiceUnavailable("Piper returned an invalid WAV file.") from exc

    def synthesize(self, text: object) -> bytes:
        if not isinstance(text, str) or not text.strip():
            raise VoiceInputError("text must be a non-empty string")
        if len(text) > MAX_SPEECH_TEXT_LENGTH:
            raise VoiceInputError(
                f"text exceeds {MAX_SPEECH_TEXT_LENGTH} characters"
            )
        request = Request(
            self.piper_url,
            data=json.dumps({"text": text.strip()}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "audio/wav"},
            method="POST",
        )
        try:
            with self._synthesis_lock:
                opener = build_opener(_NoRedirectHandler())
                with opener.open(request, timeout=20) as response:
                    content_type = response.headers.get_content_type()
                    if content_type not in {"audio/wav", "audio/x-wav"}:
                        raise VoiceUnavailable(
                            "Piper returned an unexpected content type."
                        )
                    audio = response.read(MAX_SYNTHESIZED_AUDIO_BYTES + 1)
        except HTTPError as exc:
            raise VoiceUnavailable(
                f"Local Piper service rejected the request (HTTP {exc.code})."
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise VoiceUnavailable(
                "Local Piper service is unavailable. Start the configured loopback Piper server."
            ) from exc
        self._validate_synthesized_wav(audio)
        return audio
