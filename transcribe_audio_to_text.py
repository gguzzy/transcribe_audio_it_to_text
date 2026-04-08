import queue
import sys
import tempfile
import threading
from pathlib import Path
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
from faster_whisper import WhisperModel
import pyttsx3


# Parametri di configurazione audio
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
MODEL_NAME = "large-v3"
COMPUTE_TYPE = "int8" # se gpu usare float16
LANGUAGE = "it"

"""
Scelta del dispositivo di input
"""
def choose_input_device():
    devices = sd.query_devices()
    input_devices = []
    print("\nDispositivi microfono disponibili:\n") # printo dispositivi disponibili
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            input_devices.append(idx)
            print(f"[{idx}] {dev['name']}")
    if not input_devices:
        raise RuntimeError("Nessun dispositivo di input audio trovato.")

    # Scelta da tastiera del dispositivo di input
    print("\nPremi INVIO per usare il default, oppure scrivi l'indice del microfono.")
    choice = input("Scelta: ").strip()
    if not choice:
        return None
    try:
        device_index = int(choice)
    except ValueError as e:
        raise ValueError("Indice dispositivo non valido.") from e
    if device_index not in input_devices:
        raise ValueError("Il dispositivo scelto non è un input valido.")
    return device_index

def record_until_enter(device=None):
    audio_queue = queue.Queue()
    frames = []
    stop_event = threading.Event()
    def callback(indata, frames_count, time_info, status):
        if status:
            print(f"[Audio warning] {status}", file=sys.stderr)
        audio_queue.put(indata.copy())
    def wait_for_enter():
        input("\nRegistrazione in corso... premi INVIO per fermare.\n")
        stop_event.set()
    print("Avvio registrazione...")
    waiter = threading.Thread(target=wait_for_enter, daemon=True)
    waiter.start()
    with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            device=device,
            callback=callback,
    ):
        while not stop_event.is_set():
            try:
                data = audio_queue.get(timeout=0.2)
                frames.append(data)
            except queue.Empty:
                pass
    print("Registrazione terminata.")
    if not frames:
        raise RuntimeError("Nessun audio registrato.")
    audio = np.concatenate(frames, axis=0)
    return audio

def save_temp_wav(audio_np: np.ndarray) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    wav_write(tmp_path, SAMPLE_RATE, audio_np)
    return tmp_path

def transcribe_audio(file_path: Path) -> str:
    print("\nCaricamento modello ASR...")
    model = WhisperModel(MODEL_NAME, compute_type=COMPUTE_TYPE)
    print("Trascrizione in corso...")
    segments, info = model.transcribe(
        str(file_path),
        language=LANGUAGE,
        vad_filter=True,
        beam_size=5,
    )
    text_parts = []
    for segment in segments:
        text_parts.append(segment.text.strip())
    transcript = " ".join(part for part in text_parts if part).strip()
    return transcript

def read_text_aloud(text: str):
    print("\nLettura del testo generato...")
    engine = pyttsx3.init()
    # Forzo selezione lingua italiana
    try:
        voices = engine.getProperty("voices")
        selected_voice = None
        for v in voices:
            voice_blob = f"{getattr(v, 'name', '')} {getattr(v, 'id', '')}".lower()
            if "ital" in voice_blob or "it_" in voice_blob or "italian" in voice_blob:
                selected_voice = v.id
                break
        if selected_voice:
            engine.setProperty("voice", selected_voice)
    except Exception:
        pass
    engine.setProperty("rate", 175)
    engine.say(text if text else "Non sono riuscito a trascrivere nulla.")
    engine.runAndWait()

def main():
    try:
        device = choose_input_device()
        audio = record_until_enter(device=device)
        wav_path = save_temp_wav(audio)
        print(f"\nAudio salvato temporaneamente in: {wav_path}")
        text = transcribe_audio(wav_path)
        print("\n\n\nTESTO TRASCRITTO\n\n")
        print(text if text else "[vuoto]")
        if text:
            answer = input("\nVuoi che lo legga ad alta voce? [s/N]: ").strip().lower()
            if answer == "s":
                read_text_aloud(text)
    except KeyboardInterrupt:
        print("\nInterruzione da tastiera o chiusura anticipata da utente.")
    except Exception as e:
        print(f"\nErrore: {e}")

if __name__ == "__main__":
    main()