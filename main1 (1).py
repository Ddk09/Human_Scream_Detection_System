
import sounddevice as sd
import numpy as np
import librosa
import joblib
import time
from datetime import datetime
from rich.console import Console
import socketio
from scipy.signal import butter, lfilter

MODEL_PATH = "scream_rf_model.pkl"
SAMPLE_RATE = 44100
DURATION = 4.0  
N_MFCC = 13

console = Console()

try:
    rf_model = joblib.load(MODEL_PATH)
except:
    raise Exception("Model file missing?? But it was here yesterday.")

sio = socketio.Client()
try:
    sio.connect("http://127.0.0.1:5000")
    print("CONNECTED TO BACKEND (nice!)")
except Exception as e:
    print("CONNECT ERROR:", e)
    exit()

screams_heard = 0
calm_events = 0


def high_pass_filter(audio_data, cutoff_val=1000, sr=SAMPLE_RATE, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff_val / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return lfilter(b, a, audio_data)


def beep_alert(conf):
    global screams_heard
    screams_heard += 1  

    sio.emit("scream_event", {
        "status": "scream",
        "confidence": float(conf)
    })

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"[bold red]SCREAM detected![/bold red] at {now}  (Conf: {conf:.2f})")


def safe_alert(conf):
    global calm_events
    calm_events += 1

    sio.emit("scream_event", {
        "status": "safe",
        "confidence": float(conf)
    })
    console.print(f"[green]Safe[/green] (Conf: {conf:.2f})")


def extract_features(audio_raw, sr, n_mfcc=N_MFCC):
    mfcc_values = librosa.feature.mfcc(y=audio_raw, sr=sr, n_mfcc=n_mfcc)
    mfcc_avg = np.mean(mfcc_values.T, axis=0)
    zcr_v = np.mean(librosa.feature.zero_crossing_rate(y=audio_raw))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=audio_raw, sr=sr))
    roll_v = np.mean(librosa.feature.spectral_rolloff(y=audio_raw, sr=sr))

    feat_vec = np.hstack([mfcc_avg, zcr_v, spec_cent, roll_v])

    return feat_vec


def callback(indata, frames, t_info, status):
    audio_chunk = indata[:, 0]  
    audio_chunk = high_pass_filter(audio_chunk)

    feats = extract_features(audio_chunk, SAMPLE_RATE).reshape(1, -1)

    THRESHOLD = 0.62 
    prob = rf_model.predict_proba(feats)[0][1]

    conf_percent = prob * 100

    if prob >= THRESHOLD:
        beep_alert(conf_percent)
    else:
        safe_alert(conf_percent)


console.print("[yellow]Starting scream detection... (wish me luck)[/yellow]")

with sd.InputStream(
    channels=1,
    samplerate=SAMPLE_RATE,
    blocksize=int(SAMPLE_RATE * DURATION),
    callback=callback
):
    console.print("[cyan]Listening... Press Ctrl+C to stop.[/cyan]")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nDetection stopped. (manual shutdown)")
        sio.disconnect()
