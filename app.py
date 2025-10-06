"""
Green O2 Exchange - Streamlit v2 (QR scanner + upload + form)
Features:
- QR scanner via webcam (streamlit-webrtc)
- Upload image with QR
- Read query params via st.query_params (support: ?tube=...&branch=...)
- After QR decoded -> show peminjaman form with fields:
  tube_id, branch, name, NIK, phone, address, purpose, est_duration, return_date
- Save to data/borrow_log.csv
"""

import os
from datetime import datetime
from urllib.parse import urlparse, parse_qs

import streamlit as st
import pandas as pd
import qrcode
import networkx as nx
import folium
from streamlit_folium import st_folium
from math import radians, sin, cos, asin, sqrt

# QR decode libs
from PIL import Image
import numpy as np

# For upload decoding
try:
    from pyzbar.pyzbar import decode as zbar_decode
except Exception as e:
    zbar_decode = None

# For webcam streamer
try:
    from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode, VideoTransformerBase
    import av
    import cv2
except Exception:
    webrtc_streamer = None
    VideoTransformerBase = object  # dummy


st.set_page_config(layout="wide", page_title="Green O2 Exchange - Scanner Demo")

# --- Paths ---
DATA_DIR = "data"
BRANCHES_CSV = os.path.join(DATA_DIR, "branches.csv")
BORROW_LOG = os.path.join(DATA_DIR, "borrow_log.csv")
QRCODE_DIR = os.path.join(DATA_DIR, "qrcodes")

os.makedirs(QRCODE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------
# Utilities
# -----------------
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

def ensure_borrow_log():
    if not os.path.exists(BORROW_LOG):
        df = pd.DataFrame(columns=[
            "timestamp", "tube_id", "branch", "action",
            "name", "nik", "phone", "address", "purpose",
            "est_duration", "return_date", "notes"
        ])
        df.to_csv(BORROW_LOG, index=False)

def log_borrow_row(row: dict):
    ensure_borrow_log()
    df = pd.read_csv(BORROW_LOG)
    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(BORROW_LOG, index=False)

def generate_qr_image(data: str, path: str):
    img = qrcode.make(data)
    img.save(path)

def decode_qr_from_pil(img_pil: Image.Image):
    """Return list of decoded texts (strings)."""
    if zbar_decode is None:
        return []
    decoded = zbar_decode(img_pil)
    texts = [d.data.decode("utf-8") for d in decoded]
    return texts

def parse_tube_branch_from_text(text: str):
    """
    Accepts:
      - query string like "?tube=TUBE-...&branch=Kuningan"
      - full url ".../?tube=...&branch=..."
      - plain 'TUBE-...' or 'tube:...,branch:...'
    Returns (tube, branch) or (None, None)
    """
    if not text:
        return None, None
    text = text.strip()
    # if looks like URL or query
    if "?" in text:
        try:
            qs = text.split("?", 1)[1]
            params = parse_qs(qs)
            tube = params.get("tube", [""])[0]
            branch = params.get("branch", [""])[0]
            if tube or branch:
                return tube, branch
        except Exception:
            pass
    # if contains & or =
    if "=" in text and "&" in text:
        try:
            params = parse_qs(text)
            tube = params.get("tube", [""])[0]
            branch = params.get("branch", [""])[0]
            return tube, branch
        except Exception:
            pass
    # if contains 'tube' or 'branch' words
    lowered = text.lower()
    if "tube" in lowered or "branch" in lowered:
        # try simple parsing: split by comma or semicolon
        parts = [p.strip() for p in text.replace(";", ",").split(",")]
        tube, branch = None, None
        for p in parts:
            if "tube" in p.lower() and ":" in p:
                tube = p.split(":",1)[1].strip()
            if "branch" in p.lower() and ":" in p:
                branch = p.split(":",1)[1].strip()
        return tube, branch
    # fallback: if text looks like TUBE-...
    if text.upper().startswith("TUBE-"):
        return text, ""
    return None, None

# -----------------
# Load branches (if exists)
# -----------------
if not os.path.exists(BRANCHES_CSV):
    st.warning("branches.csv tidak ditemukan di folder data/. Aplikasi tetap berjalan, tetapi daftar cabang kosong.")
    branches = pd.DataFrame(columns=["branch", "lat", "lon", "capacity"])
else:
    branches = pd.read_csv(BRANCHES_CSV)

# Ensure some QR codes exist for existing branches (demo)
for _, r in branches.iterrows():
    branch = r["branch"]
    for i in range(1, 4):
        tube_id = f"TUBE-{branch.replace(' ', '_')}-{i:03d}"
        qr_url = f"?tube={tube_id}&branch={branch}"
        qr_path = os.path.join(QRCODE_DIR, f"{tube_id}.png")
        if not os.path.exists(qr_path):
            generate_qr_image(qr_url, qr_path)

# -----------------
# State
# -----------------
if "decoded_tube" not in st.session_state:
    st.session_state.decoded_tube = ""
if "decoded_branch" not in st.session_state:
    st.session_state.decoded_branch = ""
if "scan_method" not in st.session_state:
    st.session_state.scan_method = ""

# -----------------
# Read query params (support ?tube=...&branch=...)
# -----------------
qparams = st.query_params
qp_tube = ""
qp_branch = ""
if qparams:
    # st.query_params might contain values as lists or strings
    try:
        qp_tube = qparams.get("tube") or qparams.get("tube", "")
        if isinstance(qp_tube, list):
            qp_tube = qp_tube[0] if qp_tube else ""
    except Exception:
        qp_tube = ""
    try:
        qp_branch = qparams.get("branch") or qparams.get("branch", "")
        if isinstance(qp_branch, list):
            qp_branch = qp_branch[0] if qp_branch else ""
    except Exception:
        qp_branch = ""

if qp_tube:
    st.session_state.decoded_tube = qp_tube
    st.session_state.decoded_branch = qp_branch
    st.session_state.scan_method = "query_param"

# -----------------
# UI Layout
# -----------------
st.title("Green O₂ Exchange — Scanner & Borrow Form")
col_map, col_actions = st.columns([2, 1])

with col_map:
    st.header("Peta Mitra (Jakarta)")
    m = folium.Map(location=[-6.200000, 106.816666], zoom_start=11)
    for _, r in branches.iterrows():
        try:
            folium.Marker([r["lat"], r["lon"]], popup=f"{r['branch']}\nCapacity: {r.get('capacity','')}" ).add_to(m)
        except Exception:
            pass
    st_data = st_folium(m, width=700, height=420)

with col_actions:
    st.header("Scanner")
    st.write("Pilih metode scan:")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Scan via Webcam"):
            st.session_state.scan_method = "webcam"
    with col2:
        if st.button("Upload QR Image"):
            st.session_state.scan_method = "upload"

    st.markdown("---")

    # Webcam streamer
    if st.session_state.scan_method == "webcam":
        if webrtc_streamer is None:
            st.error("Webcam scanner tidak tersedia. Pastikan `streamlit-webrtc` terinstal.")
        else:
            st.info("Memulai webcam. Tampilkan QR ke kamera untuk decode.")
            class QRVideoTransformer(VideoTransformerBase):
                def __init__(self):
                    self.last_decoded = ""
                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    # convert to PIL for pyzbar
                    try:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(img_rgb)
                        decoded = []
                        if zbar_decode:
                            decoded = zbar_decode(pil)
                        if decoded:
                            # take first
                            txt = decoded[0].data.decode("utf-8")
                            # set session_state via javascript callback (webrtc cannot set directly)
                            # But we can show text in overlay and also set last_decoded
                            self.last_decoded = txt
                    except Exception:
                        pass
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

            ctx = webrtc_streamer(
                key="qr-webrtc",
                mode=WebRtcMode.SENDRECV,
                video_transformer_factory=QRVideoTransformer,
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                media_stream_constraints={"video": True, "audio": False},
                async_transform=True,
                desired_playing_state=True,
            )

            # Try to read decoded text from the transformer state if available
            if ctx.video_transformer:
                # transformer holds last_decoded
                txt = getattr(ctx.video_transformer, "last_decoded", "")
                if txt:
                    tube, branch = parse_tube_branch_from_text(txt)
                    if tube:
                        st.session_state.decoded_tube = tube
                        st.session_state.decoded_branch = branch or ""
                        st.success(f"QR terdeteksi: {tube} / {branch}")
                        # stop the streamer automatically (optional)
                        # ctx.stop()  # careful: stop may not always be available

    # Upload flow
    if st.session_state.scan_method == "upload":
        uploaded = st.file_uploader("Upload gambar QR (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            try:
                pil = Image.open(uploaded).convert("RGB")
                texts = decode_qr_from_pil(pil)
                if texts:
                    txt = texts[0]
                    tube, branch = parse_tube_branch_from_text(txt)
                    if tube:
                        st.session_state.decoded_tube = tube
                        st.session_state.decoded_branch = branch or ""
                        st.success(f"QR terdeteksi: {tube} / {branch}")
                    else:
                        st.error(f"QR terbaca tapi tidak mengandung tube/branch: {txt}")
                else:
                    st.error("Tidak ada QR terbaca pada gambar (pastikan QR jelas).")
            except Exception as e:
                st.error(f"Gagal memproses gambar: {e}")

    st.markdown("---")
    # Manual quick input (in case QR not available)
    st.write("Atau masukkan manual (opsional):")
    manual_tube = st.text_input("Tube ID (manual)", value=st.session_state.decoded_tube or "")
    manual_branch = st.text_input("Branch (manual)", value=st.session_state.decoded_branch or "")
    if manual_tube and (manual_tube != st.session_state.decoded_tube):
        st.session_state.decoded_tube = manual_tube
    if manual_branch and (manual_branch != st.session_state.decoded_branch):
        st.session_state.decoded_branch = manual_branch

st.markdown("---")
# -----------------
# If we have decoded tube -> show form
# -----------------
if st.session_state.decoded_tube:
    st.subheader("Form Peminjaman")
    with st.form("borrow_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            tube_id = st.text_input("Tube ID", value=st.session_state.decoded_tube, disabled=True)
            branch_name = st.text_input("Branch", value=st.session_state.decoded_branch or "", disabled=True)
            name = st.text_input("Nama Peminjam")
            nik = st.text_input("NIK")
            phone = st.text_input("Nomor Telepon")
        with col_b:
            address = st.text_area("Alamat", height=100)
            purpose = st.text_input("Keperluan")
            est_duration = st.text_input("Estimasi waktu peminjaman (misal: 3 hari)")
            return_date = st.date_input("Perkiraan tanggal pengembalian")
        submitted = st.form_submit_button("Submit Peminjaman")
        if submitted:
            row = {
                "timestamp": datetime.utcnow().isoformat(),
                "tube_id": tube_id,
                "branch": branch_name,
                "action": "borrow",
                "name": name,
                "nik": nik,
                "phone": phone,
                "address": address,
                "purpose": purpose,
                "est_duration": est_duration,
                "return_date": return_date.isoformat() if (hasattr(return_date, "isoformat")) else str(return_date),
                "notes": f"scan_method={st.session_state.scan_method or 'manual'}"
            }
            try:
                log_borrow_row(row)
                st.success("Peminjaman disimpan ke borrow_log.csv")
                # clear decoded after submit if you want
                st.session_state.decoded_tube = ""
                st.session_state.decoded_branch = ""
                st.session_state.scan_method = ""
            except Exception as e:
                st.error(f"Gagal menyimpan: {e}")

# Show recent logs
st.markdown("---")
st.header("Log Peminjaman Terakhir")
ensure_borrow_log()
try:
    df_log = pd.read_csv(BORROW_LOG)
    st.dataframe(df_log.sort_values(by="timestamp", ascending=False).head(30))
except Exception:
    st.info("Tidak ada log untuk ditampilkan.")

st.markdown("---")
st.caption("Demo: untuk produksi, pindahkan penyimpanan ke database dan amankan endpoint QR.")
