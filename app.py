"""
AR Earring Virtual Try-On — Streamlit + streamlit-webrtc
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

# ───────────────────── CONFIG ─────────────────────
NOSE, LEFT_TRAG, RIGHT_TRAG       = 1, 234, 454
LEFT_EYE_OUTER, RIGHT_EYE_OUTER   = 33, 263
LEFT_BASE, LEFT_REF               = 93, 123
RIGHT_BASE, RIGHT_REF             = 323, 352
FOREHEAD, CHIN                    = 10, 152

VIS_RATIO_THRESH   = 0.75
EARLOBE_OFFSET     = 0.47
EARRING_SIZE_RATIO = 0.25
TILT_DAMPING       = 0.35
FADE_SPEED         = 0.15
ANCHOR_Y_SHIFT     = 0.0


# ───────────────── ONE-EURO FILTER ────────────────
class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self.x_prev     = float(x0)
        self.dx_prev    = 0.0
        self.t_prev     = t0

    @staticmethod
    def _alpha(te, cutoff):
        r = 2.0 * math.pi * cutoff * te
        return r / (r + 1.0)

    def __call__(self, t, x):
        te = max(t - self.t_prev, 1e-6)
        ad     = self._alpha(te, self.d_cutoff)
        dx     = (x - self.x_prev) / te
        dx_hat = ad * dx + (1.0 - ad) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a      = self._alpha(te, cutoff)
        x_hat  = a * x + (1.0 - a) * self.x_prev
        self.x_prev  = x_hat
        self.dx_prev = dx_hat
        self.t_prev  = t
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


# ───────────────── AR EAR TRACKER ─────────────────
class AREarTracker:
    def __init__(self, earring_bgra: np.ndarray):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        if earring_bgra.shape[2] == 3:
            earring_bgra = np.dstack(
                [earring_bgra,
                 np.full(earring_bgra.shape[:2], 255, np.uint8)]
            )
        self.earring_L = earring_bgra
        self.earring_R = cv2.flip(earring_bgra, 1)

        self.pos_L = PointStabilizer(min_cutoff=1.5, beta=0.5)
        self.pos_R = PointStabilizer(min_cutoff=1.5, beta=0.5)
        self.sz_L  = ScalarStabilizer(min_cutoff=0.5, beta=0.05)
        self.sz_R  = ScalarStabilizer(min_cutoff=0.5, beta=0.05)
        self.angle_stab = ScalarStabilizer(min_cutoff=1.2, beta=0.3)
        self.faceh_stab = ScalarStabilizer(min_cutoff=0.6, beta=0.05)

        self.opacity_L = self.opacity_R = 0.0
        self.cache_L   = self.cache_R   = None
        self.cache_angle = 0.0
        self.t0 = time.time()

    def _t(self):
        return time.time() - self.t0

    # ── geometry helpers ──
    def _visibility(self, lm, w, h):
        nose = np.array([lm[NOSE].x * w, lm[NOSE].y * h])
        L = np.array([lm[LEFT_TRAG].x * w,  lm[LEFT_TRAG].y * h])
        R = np.array([lm[RIGHT_TRAG].x * w, lm[RIGHT_TRAG].y * h])
        ld, rd = np.linalg.norm(L - nose), np.linalg.norm(R - nose)
        return (ld / (rd + 1e-6) > VIS_RATIO_THRESH,
                rd / (ld + 1e-6) > VIS_RATIO_THRESH)

    def _head_roll_deg(self, lm, w, h):
        le = np.array([lm[LEFT_EYE_OUTER].x * w,  lm[LEFT_EYE_OUTER].y * h])
        re = np.array([lm[RIGHT_EYE_OUTER].x * w, lm[RIGHT_EYE_OUTER].y * h])
        return math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))

    def _face_height(self, lm, w, h):
        f = np.array([lm[FOREHEAD].x * w, lm[FOREHEAD].y * h])
        c = np.array([lm[CHIN].x * w,     lm[CHIN].y * h])
        return np.linalg.norm(c - f)

    def _yaw_factors(self, lm, w, h):
        nose = np.array([lm[NOSE].x * w, lm[NOSE].y * h])
        L = np.array([lm[LEFT_TRAG].x * w,  lm[LEFT_TRAG].y * h])
        R = np.array([lm[RIGHT_TRAG].x * w, lm[RIGHT_TRAG].y * h])
        ld, rd = np.linalg.norm(L - nose), np.linalg.norm(R - nose)
        s = ld + rd + 1e-6
        return (np.clip(ld / s * 2, 0.4, 1.0),
                np.clip(rd / s * 2, 0.4, 1.0))

    def _estimate_lobe(self, lm, base_id, ref_id, w, h):
        bx, by = lm[base_id].x, lm[base_id].y
        dx = bx - lm[ref_id].x
        dy = by - lm[ref_id].y
        return (bx + dx * EARLOBE_OFFSET) * w, (by + dy * EARLOBE_OFFSET) * h

    # ── overlay ──
    def _overlay(self, frame, earring_src, cx, cy, size, angle_deg, opacity):
        if size < 4 or opacity < 0.01:
            return
        fh, fw = frame.shape[:2]
        aspect = earring_src.shape[1] / earring_src.shape[0]
        nh = max(int(size), 2)
        nw = max(int(nh * aspect), 2)
        ear = cv2.resize(earring_src, (nw, nh), interpolation=cv2.INTER_AREA)

        att_x, att_y = nw // 2, int(nh * ANCHOR_Y_SHIFT)
        M = cv2.getRotationMatrix2D(
            (att_x, att_y), -angle_deg * TILT_DAMPING, 1.0
        )
        cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
        rw = int(nh * sin_a + nw * cos_a) + 2
        rh = int(nh * cos_a + nw * sin_a) + 2
        M[0, 2] += (rw - nw) / 2
        M[1, 2] += (rh - nh) / 2
        ear = cv2.warpAffine(ear, M, (rw, rh),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))

        new_att = M @ np.array([att_x, att_y, 1.0])
        ox, oy = int(cx - new_att[0]), int(cy - new_att[1])
        eh, ew = ear.shape[:2]
        x1, y1, x2, y2     = ox, oy, ox + ew, oy + eh
        sx1, sy1, sx2, sy2 = 0,  0,  ew,      eh
        if x1 < 0:  sx1 -= x1; x1 = 0
        if y1 < 0:  sy1 -= y1; y1 = 0
        if x2 > fw: sx2 -= (x2 - fw); x2 = fw
        if y2 > fh: sy2 -= (y2 - fh); y2 = fh
        if x1 >= x2 or y1 >= y2:
            return
        crop = ear[sy1:sy2, sx1:sx2]
        if crop.size == 0:
            return
        alpha = (crop[:, :, 3:4].astype(np.float32) / 255.0) * opacity
        fg  = crop[:, :, :3].astype(np.float32)
        roi = frame[y1:y2, x1:x2].astype(np.float32)
        frame[y1:y2, x1:x2] = (fg * alpha + roi * (1 - alpha)).astype(np.uint8)

    # ── main process ──
    def process(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        t = self._t()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            self.opacity_L = max(0.0, self.opacity_L - FADE_SPEED)
            self.opacity_R = max(0.0, self.opacity_R - FADE_SPEED)
            if self.cache_L and self.opacity_L > 0.01:
                self._overlay(frame, self.earring_L,
                              *self.cache_L, self.cache_angle, self.opacity_L)
            if self.cache_R and self.opacity_R > 0.01:
                self._overlay(frame, self.earring_R,
                              *self.cache_R, self.cache_angle, self.opacity_R)
            return frame

        lm = results.multi_face_landmarks[0].landmark
        show_L, show_R = self._visibility(lm, w, h)
        fh      = self.faceh_stab.update(t, self._face_height(lm, w, h))
        angle   = self.angle_stab.update(t, self._head_roll_deg(lm, w, h))
        lf, rf  = self._yaw_factors(lm, w, h)
        base_sz = fh * EARRING_SIZE_RATIO

        # LEFT
        if show_L:
            x, y = self._estimate_lobe(lm, LEFT_BASE, LEFT_REF, w, h)
            x, y = self.pos_L.update(t, x, y)
            sz   = self.sz_L.update(t, base_sz * lf)
            self.opacity_L = min(1.0, self.opacity_L + FADE_SPEED)
            self.cache_L = (x, y, sz)
        else:
            self.opacity_L = max(0.0, self.opacity_L - FADE_SPEED)
        if self.cache_L and self.opacity_L > 0.01:
            self._overlay(frame, self.earring_L,
                          *self.cache_L, angle, self.opacity_L)

        # RIGHT
        if show_R:
            x, y = self._estimate_lobe(lm, RIGHT_BASE, RIGHT_REF, w, h)
            x, y = self.pos_R.update(t, x, y)
            sz   = self.sz_R.update(t, base_sz * rf)
            self.opacity_R = min(1.0, self.opacity_R + FADE_SPEED)
            self.cache_R = (x, y, sz)
        else:
            self.opacity_R = max(0.0, self.opacity_R - FADE_SPEED)
        if self.cache_R and self.opacity_R > 0.01:
            self._overlay(frame, self.earring_R,
                          *self.cache_R, angle, self.opacity_R)

        self.cache_angle = angle
        return frame


# ───────────── LOAD EARRING IMAGE ─────────────────
def load_earring_image(uploaded_file=None) -> np.ndarray:
    if uploaded_file is not None:
        raw = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img

    default = Path("earring.png")
    if default.exists():
        img = cv2.imread(str(default), cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img

    # fallback placeholder
    sz = 120
    canvas = np.zeros((sz, sz, 4), dtype=np.uint8)
    pts = np.array([[sz//2, 0], [sz, sz//2], [sz//2, sz], [0, sz//2]])
    cv2.fillPoly(canvas, [pts], (255, 0, 255, 220))
    return canvas


# ──────────── WEBRTC VIDEO PROCESSOR ──────────────
class EarringProcessor(VideoProcessorBase):
    def __init__(self):
        self.tracker = None
        self._earring_img = None

    def set_earring(self, earring_bgra: np.ndarray):
        self._earring_img = earring_bgra
        self.tracker = AREarTracker(earring_bgra)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # selfie mirror
        if self.tracker is not None:
            img = self.tracker.process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ──────────────── STREAMLIT APP ───────────────────
def main():
    st.set_page_config(page_title="AR Earring Try-On", layout="wide")
    st.title("💎 AR Earring Virtual Try-On")

    with st.sidebar:
        st.header("Settings")
        uploaded = st.file_uploader(
            "Upload earring PNG (transparent bg)",
            type=["png", "webp", "jpg", "jpeg"],
        )

    earring_img = load_earring_image(uploaded)

    with st.sidebar:
        preview = earring_img[:, :, :3]
        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                 caption="Current earring", width=120)

    ctx = webrtc_streamer(
        key="earring-tryon",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        video_processor_factory=EarringProcessor,
        media_stream_constraints={
            "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
            "audio": False,
        },
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.set_earring(earring_img)


if __name__ == "__main__":
    main()
