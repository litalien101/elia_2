from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import types
import unittest
import wave
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from elia.app import make_server
from elia.voice import (
    LocalVoiceService,
    VoiceInputError,
    VoiceUnavailable,
    _NoRedirectHandler,
)


def make_wav(
    sample_rate: int = 16_000,
    channels: int = 1,
    sample_width: int = 2,
    frames: int = 160,
) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(sample_width)
        target.setframerate(sample_rate)
        target.writeframes(b"\0" * frames * channels * sample_width)
    return output.getvalue()


class LocalVoiceServiceTests(unittest.TestCase):
    def test_accepts_bounded_mono_16khz_pcm_and_reports_duration(self):
        pcm, duration = LocalVoiceService._validate_wav(make_wav(frames=8000))
        self.assertEqual(len(pcm), 16_000)
        self.assertEqual(duration, 0.5)

    def test_rejects_unsupported_wav_formats_and_oversized_audio(self):
        for audio in (
            b"not wav",
            make_wav(sample_rate=48_000),
            make_wav(channels=2),
            make_wav(sample_width=1),
        ):
            with self.subTest(audio_length=len(audio)):
                with self.assertRaises(VoiceInputError):
                    LocalVoiceService._validate_wav(audio)
        with self.assertRaises(VoiceInputError):
            LocalVoiceService._validate_wav(b"x" * 5_000_001)

    def test_reports_missing_local_asr_configuration(self):
        service = LocalVoiceService(asr_model_path=None)
        with self.assertRaisesRegex(VoiceUnavailable, "ELIA_VOSK_MODEL"):
            service.transcribe_wav(make_wav())

    def test_rejects_non_loopback_piper_url(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            LocalVoiceService(piper_url="http://example.com/synthesize")

    def test_validates_synthesized_wav(self):
        LocalVoiceService._validate_synthesized_wav(make_wav(frames=100))
        with self.assertRaises(VoiceUnavailable):
            LocalVoiceService._validate_synthesized_wav(b"invalid")

    def test_synthesis_rejects_empty_or_oversized_text(self):
        service = LocalVoiceService()
        for text in ("", "  ", "x" * 2001):
            with self.subTest(length=len(text)):
                with self.assertRaises(VoiceInputError):
                    service.synthesize(text)


class VoiceHttpApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.server = make_server(Path(self.temp.name) / "voice.sqlite3", port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, method: str, path: str, body: bytes | None = None, content_type: str | None = None):
        headers = {"Content-Type": content_type} if content_type else {}
        request = Request(self.base + path, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=5) as response:
                payload = response.read()
                if response.headers.get_content_type() == "application/json":
                    payload = json.loads(payload)
                return response.status, payload
        except HTTPError as error:
            return error.code, json.loads(error.read())

    def test_voice_status_exposes_local_only_runtime(self):
        status, body = self.request("GET", "/api/v1/voice/status")
        self.assertEqual(status, 200)
        self.assertFalse(body["external_api_fallback"])
        self.assertEqual(body["speech_to_text"]["engine"], "Vosk")
        self.assertEqual(body["text_to_speech"]["engine"], "Piper")

    def test_transcription_rejects_wrong_content_type_and_missing_model(self):
        status, body = self.request(
            "POST",
            "/api/v1/voice/transcribe",
            make_wav(),
            "application/octet-stream",
        )
        self.assertEqual(status, 415)
        status, body = self.request(
            "POST",
            "/api/v1/voice/transcribe",
            make_wav(),
            "audio/wav",
        )
        self.assertEqual(status, 503)
        self.assertIn("ELIA_VOSK_MODEL", body["error"])

    def test_transcription_uses_local_vosk_and_returns_transcript(self):
        model_path = Path(self.temp.name) / "model"
        model_path.mkdir()
        fake_vosk = types.ModuleType("vosk")

        class Model:
            def __init__(self, path):
                self.path = path

        class Recognizer:
            def __init__(self, model, sample_rate):
                self.model = model
                self.sample_rate = sample_rate
                self.audio = bytearray()

            def AcceptWaveform(self, audio):
                self.audio.extend(audio)
                return False

            def FinalResult(self):
                return json.dumps({"text": "recognized words"})

        fake_vosk.Model = Model
        fake_vosk.KaldiRecognizer = Recognizer
        self.server.voice = LocalVoiceService(asr_model_path=str(model_path))
        with patch.dict(sys.modules, {"vosk": fake_vosk}):
            status, body = self.request(
                "POST",
                "/api/v1/voice/transcribe",
                make_wav(),
                "audio/wav",
            )
        self.assertEqual(status, 200)
        self.assertEqual(body["text"], "recognized words")
        self.assertEqual(body["engine"], "vosk_local")

    def test_synthesis_validates_request_without_contacting_provider(self):
        status, body = self.request(
            "POST",
            "/api/v1/voice/synthesize",
            json.dumps({"text": ""}).encode(),
            "application/json",
        )
        self.assertEqual(status, 400)
        self.assertIn("non-empty", body["error"])

    def test_synthesis_uses_loopback_piper_and_validates_wav_response(self):
        response = MagicMock()
        response.headers = Message()
        response.headers["Content-Type"] = "audio/wav"
        response.__enter__.return_value = response
        response.read.return_value = make_wav(frames=100)
        opener = MagicMock()
        opener.open.return_value = response
        service = LocalVoiceService()

        with patch("elia.voice.build_opener", return_value=opener):
            audio = service.synthesize("Hello there")

        self.assertEqual(audio, make_wav(frames=100))
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:5000/synthesize")
        self.assertEqual(json.loads(request.data), {"text": "Hello there"})
        self.assertIsNone(
            _NoRedirectHandler().redirect_request(
                None, None, 302, "Found", {}, "http://example.com/synthesize"
            )
        )


if __name__ == "__main__":
    unittest.main()
