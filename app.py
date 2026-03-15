```python
"""
AR Earring Virtual Try-On – Streamlit + streamlit-webrtc
"""

import cv2
import numpy as np
import math
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from pathlib import Path

# FIXED MEDIAPIPE IMPORT
from mediapipe.python.solutions.face_mesh import FaceMesh


# CONFIG
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


# ONE EURO FILTER
class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = t0

    @staticmethod
    def _alpha(te, cutoff):
        r = 2.0 * math.pi * cutoff * te
        return r / (r + 1.0)

    def __call__(self, t, x):
        te = t - self.t_prev
        if te < 1e-9:
            te = 1e-6
        ad = self._alpha(te, self.d_cutoff)
        dx = (x - self.x_prev) / te
        dx_hat = ad * dx + (1 - ad) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(te, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class PointStabilizer:
    def __init__(self, **kw):
        self._fx = self._fy = None
        self._kw = kw

    def update(self, t, x, y):
        if self._fx is None:
            self._fx = OneEuroFilter(t, x, **self._kw)
            self._fy = OneEuroFilter(t, y, **self._kw)
            return x, y
        return self._fx(t, x), self._fy(t, y)


class ScalarStabilizer:
    def __init__(self, **kw):
        self._f = None
        self._kw = kw

    def update(self, t, v):
        if self._f is None:
            self._f = OneEuroFilter(t, v, **self._kw)
            return v
        return self._f(t, v)


# AR TRACKER
class AREarTracker:
    def __init__(self, earring_bgra):

        self.face_mesh = FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
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

        self.opacity_L = 0.0
        self.opacity_R = 0.0
        self.cache_L = None
        self.cache_R = None
        self.cache_angle = 0.0
        self.t0 = time.time()

    def _t(self):
        return time.time() - self.t0

    def process(self, frame):

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            return frame

        lm = results.multi_face_landmarks[0].landmark

        bx = lm[LEFT_BASE].x * w
        by = lm[LEFT_BASE].y * h
        rx = lm[LEFT_REF].x * w
        ry = lm[LEFT_REF].y * h

        lx = bx + (bx - rx) * EARLOBE_OFFSET
        ly = by + (by - ry) * EARLOBE_OFFSET

        bx = lm[RIGHT_BASE].x * w
        by = lm[RIGHT_BASE].y * h
        rx = lm[RIGHT_REF].x * w
        ry = lm[RIGHT_REF].y * h

        rx = bx + (bx - rx) * EARLOBE_OFFSET
        ry = by + (by - ry) * EARLOBE_OFFSET

        self._overlay(frame, self.earring_L, lx, ly)
        self._overlay(frame, self.earring_R, rx, ry)

        return frame

    def _overlay(self, frame, earring, cx, cy):

        h, w = frame.shape[:2]

        size = int(min(h, w) * 0.12)

        ear = cv2.resize(earring, (size, size))

        x = int(cx - size // 2)
        y = int(cy)

        if x < 0 or y < 0 or x + size > w or y + size > h:
            return

        alpha = ear[:, :, 3] / 255.0

        for c in range(3):
            frame[y:y + size, x:x + size, c] = (
                ear[:, :, c] * alpha
                + frame[y:y + size, x:x + size, c] * (1 - alpha)
            )


# LOAD EARRING
def load_earring_image(uploaded_file=None):

    if uploaded_file is not None:
        raw = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img

    default = Path("earring.png")
    if default.exists():
        return cv2.imread(str(default), cv2.IMREAD_UNCHANGED)

    canvas = np.zeros((120, 120, 4), dtype=np.uint8)
    pts = np.array([[60, 0], [120, 60], [60, 120], [0, 60]])
    cv2.fillPoly(canvas, [pts], (255, 0, 255, 220))
    return canvas


class EarringProcessor(VideoProcessorBase):

    def __init__(self):
        self.tracker = None

    def set_earring(self, earring_bgra):
        self.tracker = AREarTracker(earring_bgra)

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        if self.tracker:
            img = self.tracker.process(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():

    st.set_page_config(page_title="AR Earring Try-On", layout="wide")
    st.title("💎 AR Earring Virtual Try-On")

    with st.sidebar:
        uploaded = st.file_uploader(
            "Upload earring image",
            type=["png", "jpg", "jpeg", "webp"],
        )

    earring_img = load_earring_image(uploaded)

    with st.sidebar:
        st.image(cv2.cvtColor(earring_img[:, :, :3], cv2.COLOR_BGR2RGB), width=120)

    RTC_CONFIG = {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }

    ctx = webrtc_streamer(
        key="earring-tryon",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=EarringProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,
    )

    if ctx.video_processor:
        ctx.video_processor.set_earring(earring_img)


if __name__ == "__main__":
    main()
```
