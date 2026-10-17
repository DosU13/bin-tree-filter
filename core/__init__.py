"""Shared runtime for live.py, Align Perform and perform.py (no Qt imports here)."""

import os

# Must be set before cv2 loads: with hardware transforms on, MSMF takes ~13 s
# to open some webcams (measured on a Logitech C920) and runs them at ~17 fps.
# Entry points import core before cv2 so this takes effect.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
