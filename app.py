"""
AR Earring Virtual Try-On – Streamlit + streamlit-webrtc
"""

import cv2
import numpy as np
import mediapipe as mp
import math
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from pathlib import Path


# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
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


# -------------------------------------------------------
# ONE EURO FILTER
# -------------------------------------------------------
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


# -------------------------------------------------------
# AR EAR TRACKER
# -------------------------------------------------------
class AREarTracker:

    def __init__(self, earring_bgra):

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
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

        self.opacity_L = 0
        self.opacity_R = 0

        self.cache_L = None
        self.cache_R = None
        self.cache_angle = 0

        self.t0 = time.time()

    def t(self):
        return time.time() - self.t0

    def visibility(self, lm, w, h):

        nose = np.array([lm[NOSE].x * w, lm[NOSE].y * h])
        L = np.array([lm[LEFT_TRAG].x * w, lm[LEFT_TRAG].y * h])
        R = np.array([lm[RIGHT_TRAG].x * w, lm[RIGHT_TRAG].y * h])

        ld = np.linalg.norm(L - nose)
        rd = np.linalg.norm(R - nose)

        return ld / (rd + 1e-6) > VIS_RATIO_THRESH, rd / (ld + 1e-6) > VIS_RATIO_THRESH

    def head_roll(self, lm, w, h):

        le = np.array([lm[LEFT_EYE_OUTER].x * w, lm[LEFT_EYE_OUTER].y * h])
        re = np.array([lm[RIGHT_EYE_OUTER].x * w, lm[RIGHT_EYE_OUTER].y * h])

        return math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))

    def face_height(self, lm, w, h):

        f = np.array([lm[FOREHEAD].x * w, lm[FOREHEAD].y * h])
        c = np.array([lm[CHIN].x * w, lm[CHIN].y * h])

        return np.linalg.norm(c - f)

    def estimate_lobe(self, lm, base, ref, w, h):

        bx, by = lm[base].x, lm[base].y
        dx = bx - lm[ref].x
        dy = by - lm[ref].y

        return (bx + dx * EARLOBE_OFFSET) * w, (by + dy * EARLOBE_OFFSET) * h

    def overlay(self, frame, earring, cx, cy, size, angle, opacity):

        if size < 5 or opacity < 0.01:
            return

        fh, fw = frame.shape[:2]

        aspect = earring.shape[1] / earring.shape[0]

        nh = int(size)
        nw = int(nh * aspect)

        ear = cv2.resize(earring, (nw, nh))

        att_x = nw // 2
        att_y = int(nh * ANCHOR_Y_SHIFT)

        M = cv2.getRotationMatrix2D((att_x, att_y), -angle * TILT_DAMPING, 1)

        ear = cv2.warpAffine(
            ear,
            M,
            (nw, nh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        ox = int(cx - att_x)
        oy = int(cy - att_y)

        eh, ew = ear.shape[:2]

        if ox < 0 or oy < 0 or ox + ew > fw or oy + eh > fh:
            return

        alpha = ear[:, :, 3] / 255.0 * opacity

        for c in range(3):
            frame[oy:oy + eh, ox:ox + ew, c] = (
                ear[:, :, c] * alpha
                + frame[oy:oy + eh, ox:ox + ew, c] * (1 - alpha)
            )

    def process(self, frame):

        h, w = frame.shape[:2]
        t = self.t()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return frame

        lm = results.multi_face_landmarks[0].landmark

        show_L, show_R = self.visibility(lm, w, h)

        fh = self.faceh_stab.update(t, self.face_height(lm, w, h))
        angle = self.angle_stab.update(t, self.head_roll(lm, w, h))

        base_size = fh * EARRING_SIZE_RATIO

        if show_L:

            x, y = self.estimate_lobe(lm, LEFT_BASE, LEFT_REF, w, h)
            x, y = self.pos_L.update(t, x, y)

            size = self.sz_L.update(t, base_size)

            self.overlay(frame, self.earring_L, x, y, size, angle, 1)

        if show_R:

            x, y = self.estimate_lobe(lm, RIGHT_BASE, RIGHT_REF, w, h)
            x, y = self.pos_R.update(t, x, y)

            size = self.sz_R.update(t, base_size)

            self.overlay(frame, self.earring_R, x, y, size, angle, 1)

        return frame


# -------------------------------------------------------
# EARRING IMAGE LOADER
# -------------------------------------------------------
def load_earring(uploaded):

    if uploaded:

        raw = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)

        return img

    default = Path("earring.png")

    if default.exists():

        return cv2.imread(str(default), cv2.IMREAD_UNCHANGED)

    canvas = np.zeros((120, 120, 4), dtype=np.uint8)

    pts = np.array([[60, 0], [120, 60], [60, 120], [0, 60]])

    cv2.fillPoly(canvas, [pts], (255, 0, 255, 200))

    return canvas


# -------------------------------------------------------
# WEBRTC PROCESSOR
# -------------------------------------------------------
class EarringProcessor(VideoProcessorBase):

    def __init__(self):

        self.tracker = None
        self.img = None

    def set_earring(self, img):

        self.img = img
        self.tracker = AREarTracker(img)

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        if self.tracker:

            img = self.tracker.process(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -------------------------------------------------------
# STREAMLIT APP
# -------------------------------------------------------
def main():

    st.set_page_config(page_title="AR Earring Try-On", layout="wide")

    st.title("💎 AR Earring Virtual Try-On")

    if "earring_processor" not in st.session_state:
        st.session_state.earring_processor = False

    with st.sidebar:

        uploaded = st.file_uploader("Upload earring PNG", type=["png", "jpg", "jpeg"])

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
        async_processing=False
    )

    if ctx.video_processor:

        if not st.session_state.earring_processor:

            ctx.video_processor.set_earring(earring)

            st.session_state.earring_processor = True


if __name__ == "__main__":
    main()
