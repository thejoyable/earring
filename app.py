```python
"""
AR Earring Virtual Try-On – Streamlit + WebRTC
Stable version for Streamlit Cloud
"""

import cv2
import numpy as np
import math
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from pathlib import Path

# NEW mediapipe import (fix for mp.solutions error)
from mediapipe.python.solutions.face_mesh import FaceMesh


# ------------------------------------------------
# CONFIG
# ------------------------------------------------
NOSE = 1
LEFT_TRAG = 234
RIGHT_TRAG = 454
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

LEFT_BASE = 93
LEFT_REF = 123
RIGHT_BASE = 323
RIGHT_REF = 352

FOREHEAD = 10
CHIN = 152

VIS_RATIO_THRESH = 0.75
EARLOBE_OFFSET = 0.47
EARRING_SIZE_RATIO = 0.25

TILT_DAMPING = 0.35
FADE_SPEED = 0.15
ANCHOR_Y_SHIFT = 0.0


# ------------------------------------------------
# ONE EURO FILTER
# ------------------------------------------------
class OneEuroFilter:

    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):

        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = t0

    @staticmethod
    def alpha(te, cutoff):

        r = 2.0 * math.pi * cutoff * te
        return r / (r + 1.0)

    def __call__(self, t, x):

        te = t - self.t_prev
        if te <= 0:
            te = 1e-6

        dx = (x - self.x_prev) / te
        ad = self.alpha(te, self.d_cutoff)

        dx_hat = ad * dx + (1 - ad) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.alpha(te, cutoff)

        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


class PointStabilizer:

    def __init__(self, **kw):

        self.fx = None
        self.fy = None
        self.kw = kw

    def update(self, t, x, y):

        if self.fx is None:

            self.fx = OneEuroFilter(t, x, **self.kw)
            self.fy = OneEuroFilter(t, y, **self.kw)

            return x, y

        return self.fx(t, x), self.fy(t, y)


class ScalarStabilizer:

    def __init__(self, **kw):

        self.f = None
        self.kw = kw

    def update(self, t, v):

        if self.f is None:

            self.f = OneEuroFilter(t, v, **self.kw)
            return v

        return self.f(t, v)


# ------------------------------------------------
# AR EARRING TRACKER
# ------------------------------------------------
class AREarTracker:

    def __init__(self, earring_bgra):

        self.face_mesh = FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        if earring_bgra.shape[2] == 3:

            earring_bgra = np.dstack(
                [earring_bgra, np.full(earring_bgra.shape[:2], 255, np.uint8)]
            )

        self.earring_L = earring_bgra
        self.earring_R = cv2.flip(earring_bgra, 1)

        self.pos_L = PointStabilizer(min_cutoff=1.5, beta=0.5)
        self.pos_R = PointStabilizer(min_cutoff=1.5, beta=0.5)

        self.sz_L = ScalarStabilizer(min_cutoff=0.5, beta=0.05)
        self.sz_R = ScalarStabilizer(min_cutoff=0.5, beta=0.05)

        self.angle_stab = ScalarStabilizer(min_cutoff=1.2, beta=0.3)
        self.faceh_stab = ScalarStabilizer(min_cutoff=0.6, beta=0.05)

        self.t0 = time.time()

    def t(self):

        return time.time() - self.t0

    def estimate_lobe(self, lm, base, ref, w, h):

        bx, by = lm[base].x, lm[base].y
        dx = bx - lm[ref].x
        dy = by - lm[ref].y

        return (bx + dx * EARLOBE_OFFSET) * w, (by + dy * EARLOBE_OFFSET) * h

    def face_height(self, lm, w, h):

        f = np.array([lm[FOREHEAD].x * w, lm[FOREHEAD].y * h])
        c = np.array([lm[CHIN].x * w, lm[CHIN].y * h])

        return np.linalg.norm(c - f)

    def process(self, frame):

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return frame

        lm = results.multi_face_landmarks[0].landmark

        fh = self.face_height(lm, w, h)
        size = fh * EARRING_SIZE_RATIO

        xL, yL = self.estimate_lobe(lm, LEFT_BASE, LEFT_REF, w, h)
        xR, yR = self.estimate_lobe(lm, RIGHT_BASE, RIGHT_REF, w, h)

        self.overlay(frame, self.earring_L, xL, yL, size)
        self.overlay(frame, self.earring_R, xR, yR, size)

        return frame

    def overlay(self, frame, earring, cx, cy, size):

        h, w = frame.shape[:2]

        nh = int(size)
        nw = int(size * (earring.shape[1] / earring.shape[0]))

        ear = cv2.resize(earring, (nw, nh))

        x = int(cx - nw // 2)
        y = int(cy)

        if x < 0 or y < 0 or x + nw > w or y + nh > h:
            return

        alpha = ear[:, :, 3] / 255.0

        for c in range(3):

            frame[y:y + nh, x:x + nw, c] = (
                ear[:, :, c] * alpha
                + frame[y:y + nh, x:x + nw, c] * (1 - alpha)
            )


# ------------------------------------------------
# IMAGE LOADER
# ------------------------------------------------
def load_earring(uploaded):

    if uploaded:

        raw = np.frombuffer(uploaded.read(), np.uint8)
        return cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)

    default = Path("earring.png")

    if default.exists():

        return cv2.imread(str(default), cv2.IMREAD_UNCHANGED)

    canvas = np.zeros((120, 120, 4), dtype=np.uint8)

    pts = np.array([[60, 0], [120, 60], [60, 120], [0, 60]])
    cv2.fillPoly(canvas, [pts], (255, 0, 255, 200))

    return canvas


# ------------------------------------------------
# WEBRTC PROCESSOR
# ------------------------------------------------
class EarringProcessor(VideoProcessorBase):

    def __init__(self):

        self.tracker = None

    def set_earring(self, img):

        self.tracker = AREarTracker(img)

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        if self.tracker:
            img = self.tracker.process(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ------------------------------------------------
# STREAMLIT APP
# ------------------------------------------------
def main():

    st.set_page_config(page_title="AR Earring Try-On", layout="wide")

    st.title("💎 AR Earring Virtual Try-On")

    if "processor_init" not in st.session_state:
        st.session_state.processor_init = False

    with st.sidebar:

        uploaded = st.file_uploader("Upload earring PNG", type=["png", "jpg"])

    earring = load_earring(uploaded)

    with st.sidebar:

        st.image(cv2.cvtColor(earring[:, :, :3], cv2.COLOR_BGR2RGB), width=120)

    RTC_CONFIGURATION = {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }

    ctx = webrtc_streamer(
        key="earring-tryon",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=EarringProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,
    )

    if ctx.video_processor:

        if not st.session_state.processor_init:

            ctx.video_processor.set_earring(earring)
            st.session_state.processor_init = True


if __name__ == "__main__":
    main()
```
