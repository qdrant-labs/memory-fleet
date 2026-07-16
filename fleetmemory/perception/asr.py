"""On-device speech: whisper-base via onnx-asr (CPU), same model as the course
robot. Teach by voice ("this is my mug") and ask by voice ("where did I leave
my keys") — the browser records 16 kHz mono WAV, the server transcribes here.
"""

import re
import wave
from functools import lru_cache

WHISPER_MODEL = "whisper-base"


@lru_cache(maxsize=1)
def _model():
    import onnx_asr

    return onnx_asr.load_model(WHISPER_MODEL, providers=["CPUExecutionProvider"])


def load():
    """Warm the model off the request path."""
    _model()


def transcribe(wav_path: str) -> str:
    """WAV file -> text. Silence returns '' (whisper hallucinates 'Thanks for
    watching!' on quiet audio, so a dead mic must fail visibly, not teach junk).
    An empty payload also means the browser tab has no mic permission."""
    if _wav_rms(wav_path) < SILENCE_RMS:
        return ""
    return _model().recognize(wav_path).strip()


SILENCE_RMS = 120  # int16 RMS below this is a failed capture, not speech

# ASR drops the punctuation that would end the naming clause, so the label also
# stops at words that start a new clause ("...my mug I bought it" -> "my mug").
_CLAUSE_WORDS = {
    "i", "it", "and", "that", "which", "because", "she", "he", "they", "we",
    "you", "made", "bought", "got", "from", "at", "about", "in", "on",
}  # fmt: skip


def parse_label(transcript: str) -> str:
    """'This is my mug — Maria made it.' -> 'my mug'. Free-form fallback."""
    m = re.search(r"this is (?:an? )?(.+?)(?:\s*[,.;!?—–-]|$)", transcript, re.IGNORECASE)
    phrase = m.group(1) if m else transcript
    words = []
    for w in phrase.split():
        if w.lower().strip(".,!?") in _CLAUSE_WORDS:
            break
        words.append(w)
    return " ".join(words[:5]).rstrip(".,!?") or "unnamed"


def _wav_rms(path: str) -> float:
    import numpy as np

    with wave.open(str(path)) as w:
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if not len(samples):
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
