import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import numpy as np
import mediapipe as mp
import os
import threading

# ================================================================
# 1. MONKEY-PATCH: Fix the "_polling_thread is None" crash
#    in older versions of streamlit-webrtc
# ================================================================
try:
    import streamlit_webrtc.shutdown as _shutdown

    _original_stop = _shutdown.SessionShutdownObserver.stop

    def _safe_stop(self):
        if getattr(self, "_polling_thread", None) is not None:
            _original_stop(self)

    _shutdown.SessionShutdownObserver.stop = _safe_stop
except Exception:
    pass


# ================================================================
# 2. ICE / TURN SERVER CONFIGURATION
#    Without a TURN server, WebRTC will NOT work on Streamlit Cloud.
#    Sign up for free TURN credentials at https://www.metered.ca/
#    then add them in the Streamlit Cloud secrets dashboard.
# ================================================================
def get_ice_servers():
    ice_servers = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]

    # --- Try loading TURN credentials from Streamlit secrets ---
    try:
        turn_urls = st.secrets["TURN_URLS"]        # e.g. "turn:a]relay1.expressturn.com:443"
        turn_user = st.secrets["TURN_USERNAME"]
        turn_cred = st.secrets["TURN_CREDENTIAL"]

        # Handle both single string and list
        if isinstance(turn_urls, str):
            turn_urls = [turn_urls]

        ice_servers.append(
            {
                "urls": turn_urls,
                "username": turn_user,
                "credential": turn_cred,
            }
        )
    except Exception:
        st.sidebar.warning(
            "⚠️ No TURN server configured.\n\n"
            "WebRTC **will not work** on Streamlit Cloud without one.\n\n"
            "Add `TURN_URLS`, `TURN_USERNAME`, `TURN_CREDENTIAL` "
            "to your app's **Secrets** in the Streamlit Cloud dashboard.\n\n"
            "Get free credentials → [metered.ca](https://www.metered.ca/)"
        )

    return ice_servers


RTC_CONFIGURATION = RTCConfiguration({"iceServers": get_ice_servers()})


# ================================================================
# 3. MEDIAPIPE FACE-MESH LANDMARK INDICES
# ================================================================
# Earlobe attachment points
LEFT_EARLOBE = 177      # Bottom of left ear / jawline
RIGHT_EARLOBE = 401     # Bottom of right ear / jawline

# Ear tragion points (for sizing reference)
LEFT_EAR_TRAGION = 234
RIGHT_EAR_TRAGION = 454

# Face width reference
FACE_LEFT = 234
FACE_RIGHT = 454


# ================================================================
# 4. EARRING VIDEO PROCESSOR
# ================================================================
class EarringProcessor(VideoProcessorBase):
    def __init__(self):
        self._earring_bgr = None          # Earring image (BGRA, 4 channels)
        self._lock = threading.Lock()
        self._scale = 0.22                 # Earring size relative to face width
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    # --- Thread-safe earring getter / setter ---
    @property
    def earring_image(self):
        with self._lock:
            return self._earring_bgr

    @earring_image.setter
    def earring_image(self, value):
        with self._lock:
            self._earring_bgr = value

    @property
    def scale(self):
        with self._lock:
            return self._scale

    @scale.setter
    def scale(self, value):
        with self._lock:
            self._scale = value

    # --- Alpha-composite overlay ---
    @staticmethod
    def overlay_transparent(background, overlay, x, y, ow, oh):
        """Place a BGRA overlay onto a BGR background at (x, y) with size (ow, oh)."""
        if overlay is None or ow <= 0 or oh <= 0:
            return background

        try:
            overlay_resized = cv2.resize(overlay, (ow, oh), interpolation=cv2.INTER_AREA)
        except Exception:
            return background

        bg_h, bg_w = background.shape[:2]

        # Clip to image boundaries
        y1, y2 = max(0, y), min(bg_h, y + oh)
        x1, x2 = max(0, x), min(bg_w, x + ow)
        oy1, oy2 = y1 - y, y1 - y + (y2 - y1)
        ox1, ox2 = x1 - x, x1 - x + (x2 - x1)

        if y2 <= y1 or x2 <= x1:
            return background

        if overlay_resized.shape[2] == 4:
            alpha = overlay_resized[oy1:oy2, ox1:ox2, 3].astype(np.float32) / 255.0
            a3 = np.stack([alpha] * 3, axis=-1)
            roi = background[y1:y2, x1:x2].astype(np.float32)
            fg = overlay_resized[oy1:oy2, ox1:ox2, :3].astype(np.float32)
            background[y1:y2, x1:x2] = (a3 * fg + (1.0 - a3) * roi).astype(np.uint8)
        else:
            background[y1:y2, x1:x2] = overlay_resized[oy1:oy2, ox1:ox2, :3]

        return background

    # --- Process each video frame ---
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        earring = self.earring_image

        if earring is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for fl in results.multi_face_landmarks:
                # --- Compute face width for proportional earring sizing ---
                lf = fl.landmark[FACE_LEFT]
                rf = fl.landmark[FACE_RIGHT]
                face_w = abs(rf.x - lf.x) * w

                # --- Earring dimensions ---
                ew = max(10, int(face_w * self.scale))
                aspect = earring.shape[0] / max(earring.shape[1], 1)
                eh = max(10, int(ew * aspect))

                # --- Left earring ---
                ll = fl.landmark[LEFT_EARLOBE]
                lx = int(ll.x * w) - ew // 2
                ly = int(ll.y * h)
                img = self.overlay_transparent(img, earring, lx, ly, ew, eh)

                # --- Right earring (mirror) ---
                earring_flip = cv2.flip(earring, 1)
                rl = fl.landmark[RIGHT_EARLOBE]
                rx = int(rl.x * w) - ew // 2
                ry = int(rl.y * h)
                img = self.overlay_transparent(img, earring_flip, rx, ry, ew, eh)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ================================================================
# 5. EARRING IMAGE HELPERS
# ================================================================
EARRING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "earrings")


def get_earring_catalog():
    """Return dict {display_name: filepath} of earring PNGs."""
    if not os.path.isdir(EARRING_DIR):
        os.makedirs(EARRING_DIR, exist_ok=True)

    catalog = {}
    for f in sorted(os.listdir(EARRING_DIR)):
        if f.lower().endswith((".png", ".webp", ".jpg", ".jpeg")):
            name = os.path.splitext(f)[0].replace("_", " ").replace("-", " ").title()
            catalog[name] = os.path.join(EARRING_DIR, f)
    return catalog


def create_sample_earrings():
    """Generate simple placeholder earrings if the folder is empty."""
    os.makedirs(EARRING_DIR, exist_ok=True)
    if any(f.endswith(".png") for f in os.listdir(EARRING_DIR)):
        return  # already have images

    samples = {
        "gold_stud":    (0, 215, 255),
        "silver_hoop":  (192, 192, 192),
        "ruby_drop":    (60, 20, 220),
        "emerald_gem":  (50, 180, 50),
        "sapphire_gem": (200, 100, 30),
    }

    for name, bgr in samples.items():
        canvas = np.zeros((100, 50, 4), dtype=np.uint8)
        # connector line
        cv2.line(canvas, (25, 0), (25, 25), (*bgr, 200), 2)
        # gem body
        cv2.circle(canvas, (25, 55), 20, (*bgr, 255), -1)
        # highlight
        cv2.circle(canvas, (18, 48), 6, (255, 255, 255, 160), -1)

        cv2.imwrite(os.path.join(EARRING_DIR, f"{name}.png"), canvas)


def load_earring_cv2(path: str):
    """Load an earring image as BGRA numpy array."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None and img.shape[2] == 3:
        # Add alpha channel if missing
        alpha = np.full((*img.shape[:2], 1), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=-1)
    return img


# ================================================================
# 6. STREAMLIT UI
# ================================================================
def main():
    st.set_page_config(
        page_title="✨ Virtual Earring Try-On",
        page_icon="💎",
        layout="wide",
    )

    st.title("✨ Virtual Earring Try-On")
    st.caption("See how earrings look on you — in real time!")

    # Make sure we have at least sample earrings
    create_sample_earrings()

    # ------ SIDEBAR ------
    st.sidebar.header("💎 Earring Selection")

    catalog = get_earring_catalog()

    # Selector
    earring_choice = st.sidebar.selectbox(
        "Choose from collection",
        options=["— None —"] + list(catalog.keys()),
    )

    # Upload
    uploaded_file = st.sidebar.file_uploader(
        "…or upload your own (PNG with transparency)",
        type=["png", "webp"],
    )

    # Scale slider
    earring_scale = st.sidebar.slider(
        "Earring size",
        min_value=0.08,
        max_value=0.50,
        value=0.22,
        step=0.02,
    )

    # Resolve the earring image
    earring_img = None

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        earring_img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        if earring_img is not None and earring_img.shape[2] == 3:
            alpha = np.full((*earring_img.shape[:2], 1), 255, dtype=np.uint8)
            earring_img = np.concatenate([earring_img, alpha], axis=-1)
        st.sidebar.image(uploaded_file, caption="Your earring", width=80)

    elif earring_choice != "— None —":
        earring_img = load_earring_cv2(catalog[earring_choice])
        st.sidebar.image(catalog[earring_choice], caption=earring_choice, width=80)

    # Instructions
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        ### 📋 How to use
        1. **Allow** camera access when prompted
        2. **Pick** an earring or upload your own PNG
        3. **Adjust** the size slider
        4. Face the camera — earrings appear automatically!

        ### 💡 Tips
        - Use **good lighting** for best detection
        - Face the camera **straight on**
        - PNG with **transparent background** works best
        """
    )

    # ------ MAIN AREA: WebRTC ------
    col1, col2 = st.columns([3, 1])

    with col1:
        ctx = webrtc_streamer(
            key="earring-tryon",
            video_processor_factory=EarringProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 24},
                },
                "audio": False,
            },
            async_processing=True,
        )

    with col2:
        st.markdown("#### 🎥 Status")
        if ctx.state.playing:
            st.success("Camera is active")
        else:
            st.info("Click **START** to begin")

    # Push selected earring into the running processor
    if ctx.video_processor is not None:
        ctx.video_processor.earring_image = earring_img
        ctx.video_processor.scale = earring_scale

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:gray;'>"
        "Built with Streamlit · MediaPipe · streamlit-webrtc"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
