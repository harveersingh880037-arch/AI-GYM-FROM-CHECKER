"""
Chote beep/tone sounds generate karta hai (bina kisi extra audio file ke)
taaki 'Correct form' aur 'Incorrect form' par alag alag sound alert de sake.

Pure Python 'wave' module se WAV bytes banata hai aur base64 me encode
karke ek HTML <audio autoplay> tag return karta hai jise Streamlit me
st.markdown(..., unsafe_allow_html=True) se play kiya ja sakta hai.
"""

import base64
import io
import math
import struct
import wave


def _generate_tone(frequency=440.0, duration_ms=180, volume=0.5, sample_rate=22050):
    n_samples = int(sample_rate * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            t = i / sample_rate
            fade = min(1.0, (n_samples - i) / (sample_rate * 0.02))  # click hatane ke liye chota fade-out
            value = int(volume * fade * 32767.0 * math.sin(2 * math.pi * frequency * t))
            wf.writeframesraw(struct.pack("<h", value))
    return buf.getvalue()


def get_audio_html(kind="good"):
    """
    kind: 'good' -> high-pitch pleasant beep (sahi form / rep complete)
          'bad'  -> low-pitch buzzer (galat form)
    """
    if kind == "good":
        wav_bytes = _generate_tone(frequency=880.0, duration_ms=150, volume=0.4)
    else:
        wav_bytes = _generate_tone(frequency=220.0, duration_ms=260, volume=0.5)

    b64 = base64.b64encode(wav_bytes).decode()
    return (
        f'<audio autoplay="true" style="display:none">'
        f'<source src="data:audio/wav;base64,{b64}" type="audio/wav">'
        f"</audio>"
    )
