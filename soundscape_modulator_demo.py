# soundscape_modulator_demo.py
import streamlit as st
import numpy as np
import scipy.signal as sps
import soundfile as sf
import time
import io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# ------- CONFIG ----------
SR = 16000  # sample rate
BUFFER_SECONDS = 6  # rolling window length to display
BUFFER_SIZE = SR * BUFFER_SECONDS

st.set_page_config(layout="wide", page_title="Soundscape Modulator Demo")

# ------- UI controls ----------
st.title("Soundscape Modulator — Live Demo (Simulated)")
col1, col2, col3 = st.columns([3,2,2])

with col1:
    start = st.button("Start Simulation")
    stop  = st.button("Stop")
    snapshot = st.button("Snapshot / Export CSV")

with col2:
    mod_rate = st.slider("Modulation rate (Hz)", 0.1, 5.0, 0.5, step=0.1)
    mod_depth = st.slider("Modulation depth", 0.0, 1.0, 0.7, step=0.05)

with col3:
    stimulus_profile = st.selectbox("Stimulus profile", ["Sine+Noise", "Burst-Pulse", "Ambient Layers"])
    show_spectrogram = st.checkbox("Show Spectrogram", True)

# ------- State ----------
if 'running' not in st.session_state:
    st.session_state.running = False
if 'buffer' not in st.session_state:
    st.session_state.buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
if 'time_cursor' not in st.session_state:
    st.session_state.time_cursor = 0.0
if 'events' not in st.session_state:
    st.session_state.events = []

# handle start/stop
if start:
    st.session_state.running = True
    st.session_state.events.append((time.time(), "start"))
if stop:
    st.session_state.running = False
    st.session_state.events.append((time.time(), "stop"))

# ------- helper functions ----------
def synth_block(t0, duration, profile, sr=SR, mod_rate=0.5, mod_depth=0.7):
    t = np.arange(int(duration*sr)) / sr
    out = np.zeros_like(t)
    if profile == "Sine+Noise":
        carrier = 440.0  # A4
        # amplitude-modulated sine plus noise
        am = 1.0 + mod_depth * np.sin(2*np.pi*mod_rate*(t0 + t))
        out = am * 0.5 * np.sin(2*np.pi*carrier*(t0 + t))
        out += 0.15 * np.random.randn(len(t))
    elif profile == "Burst-Pulse":
        # create periodic bursts
        burst_env = (np.sin(2*np.pi*mod_rate*(t0 + t)) > 0.9).astype(float)
        out = burst_env * (np.sin(2*np.pi*660*(t0 + t))) * (0.6 + 0.4*np.random.rand())
        out += 0.05 * np.random.randn(len(t))
    else:
        # ambient layers: a couple of detuned sines + slow mod
        c1 = np.sin(2*np.pi*220*(t0 + t))
        c2 = np.sin(2*np.pi*222*(t0 + t)) * 0.6
        slow = 1.0 + 0.5 * np.sin(2*np.pi*(mod_rate/4)*(t0 + t))
        out = slow * (0.3*c1 + 0.2*c2) + 0.08*np.random.randn(len(t))
    # normalize
    out = out / (np.max(np.abs(out)) + 1e-9) * 0.9
    return out

def compute_features(block, sr=SR):
    # RMS
    rms = np.sqrt(np.mean(block**2))
    # zero-crossing rate
    zcr = ((block[:-1]*block[1:]) < 0).sum() / len(block)
    # spectral centroid
    f, t, Sxx = spectrogram(block, fs=sr, nperseg=512, noverlap=256)
    mag = Sxx + 1e-12
    centroid = (f[:,None]*mag).sum()/mag.sum()
    # band energy (low: <500Hz, mid: 500-3000, high >3000)
    total = mag.sum()
    low = mag[f < 500,:].sum()/total
    mid = mag[(f>=500)&(f<3000),:].sum()/total
    high = mag[f>=3000,:].sum()/total
    return {"rms": float(rms), "zcr": float(zcr), "centroid": float(centroid), "low": float(low), "mid": float(mid), "high": float(high)}

# ------- Layout placeholders ----------
wave_col, spec_col = st.columns([2,1])
wave_plot = wave_col.empty()
features_col = wave_col.container()
spec_plot = spec_col.empty()
timeline = st.container()

# ------- Main loop (simulate / update) ----------
# We'll push blocks of ~0.2s to the buffer per frame for a smooth demo.
BLOCK_SECONDS = 0.2
block_len = int(BLOCK_SECONDS * SR)

if st.session_state.running:
    # generate block
    t0 = st.session_state.time_cursor
    block = synth_block(t0, BLOCK_SECONDS, stimulus_profile, SR, mod_rate, mod_depth)
    st.session_state.time_cursor += BLOCK_SECONDS

    # shift buffer and append
    buf = st.session_state.buffer
    buf = np.roll(buf, -block_len)
    buf[-block_len:] = block
    st.session_state.buffer = buf

    # compute features on the whole buffer (or last 1s)
    features = compute_features(buf[-SR:])  # last 1 second features
    # append events occasionally to demo timeline
    if np.random.rand() < 0.02:
        st.session_state.events.append((time.time(), "detected_event"))

    # render waveform
    fig_w, axw = plt.subplots(figsize=(9,2))
    tvec = np.linspace(-BUFFER_SECONDS, 0, len(buf))
    axw.plot(tvec, buf, linewidth=0.6)
    axw.set_xlabel("Seconds (relative)")
    axw.set_ylabel("Amplitude")
    axw.set_ylim(-1.1, 1.1)
    axw.set_title("Rolling waveform (last {} s)".format(BUFFER_SECONDS))
    fig_w.tight_layout()
    wave_plot.pyplot(fig_w)
    plt.close(fig_w)

    # render spectrogram if requested
    if show_spectrogram:
        fig_s, axs = plt.subplots(1,1, figsize=(4,3))
        f, t_spec, Sxx = spectrogram(buf, fs=SR, nperseg=512, noverlap=400)
        axs.pcolormesh(t_spec - BUFFER_SECONDS, f, 10*np.log10(Sxx+1e-12), shading='gouraud')
        axs.set_ylim(0, 8000)
        axs.set_xlabel("Seconds (relative)")
        axs.set_ylabel("Frequency (Hz)")
        axs.set_title("Spectrogram (log power)")
        fig_s.tight_layout()
        spec_plot.pyplot(fig_s)
        plt.close(fig_s)

    # features display
    with features_col:
        f1, f2, f3 = st.columns(3)
        f1.metric("RMS", f"{features['rms']:.3f}")
        f2.metric("Spectral centroid (Hz)", f"{features['centroid']:.1f}")
        f3.metric("ZCR", f"{features['zcr']:.4f}")
        bar = st.progress(int(100 * features['mid']))  # just to show activity
        # small dataframe for band energies
        df = pd.DataFrame({
            "band": ["low <500Hz", "mid 500-3000Hz", "high >3000Hz"],
            "relative_energy": [features['low'], features['mid'], features['high']]
        })
        st.table(df)

    # timeline / events
    with timeline:
        st.subheader("Events timeline (latest first)")
        evs = sorted(st.session_state.events, key=lambda x: x[0], reverse=True)[:10]
        rows = []
        for ts, label in evs:
            rows.append({"time": time.strftime("%H:%M:%S", time.localtime(ts)), "event": label})
        st.table(pd.DataFrame(rows))

    # add to feature history for export
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({"t": time.time(), **features})

    # snapshot / export CSV button handling
    if snapshot:
        df_hist = pd.DataFrame(st.session_state.history)
        csv_buf = io.BytesIO()
        df_hist.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        st.download_button("Download features CSV", data=csv_buf, file_name="soundscape_features.csv", mime="text/csv")

    # small sleep to pace UI updates
    time.sleep(0.12)
else:
    st.info("Simulation stopped. Press Start Simulation to run the modulator demo.")
    # show static waveform from buffer
    buf = st.session_state.buffer
    fig_w, axw = plt.subplots(figsize=(9,2))
    tvec = np.linspace(-BUFFER_SECONDS, 0, len(buf))
    axw.plot(tvec, buf, linewidth=0.6)
    axw.set_xlabel("Seconds (relative)")
    axw.set_ylabel("Amplitude")
    axw.set_ylim(-1.1, 1.1)
    axw.set_title("Rolling waveform (last {} s)".format(BUFFER_SECONDS))
    fig_w.tight_layout()
    wave_plot.pyplot(fig_w)
    plt.close(fig_w)

# Footer: quick usage tips
st.markdown("---")
st.markdown("**Usage tips:** To demo real audio input replace the synth_block with a live microphone callback (e.g., using sounddevice) or read blocks from a pre-recorded WAV and push them to the buffer. Use `st.session_state.history` to save features and `st.download_button` to export CSV snapshots.")
