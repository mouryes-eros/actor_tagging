from __future__ import annotations

import os
import sys
import json
import math
import time
import pickle
import argparse
import hashlib
from pathlib import Path

_CACHE = {}

def _lazy(key: str, loader):
    v = _CACHE.get(key)
    if v is None:
        v = loader()
        _CACHE[key] = v
    return v

def toml():
    def _load():
        try:
            import tomllib as t
            return t
        except Exception:
            import tomli as t
            return t
    return _lazy("toml", _load)

def np():
    return _lazy("np", lambda: __import__("numpy"))

def tqdm():
    def _load():
        from tqdm import tqdm as tq
        return tq
    return _lazy("tqdm_fn", _load)

def ray():
    return _lazy("ray", lambda: __import__("ray"))

def torch():
    return _lazy("torch", lambda: __import__("torch"))

def cv2():
    return _lazy("cv2", lambda: __import__("cv2"))

def YOLO():
    def _load():
        from ultralytics import YOLO as cls
        return cls
    return _lazy("YOLO_cls", _load)

def FaceAnalysis():
    def _load():
        from insightface.app import FaceAnalysis as cls
        return cls
    return _lazy("FaceAnalysis_cls", _load)

def PIL_Image():
    def _load():
        from PIL import Image as Image
        return Image
    return _lazy("PIL_Image", _load)

def PIL_ImageDraw():
    def _load():
        from PIL import ImageDraw as ImageDraw
        return ImageDraw
    return _lazy("PIL_ImageDraw", _load)

def PIL_ImageFont():
    def _load():
        from PIL import ImageFont as ImageFont
        return ImageFont
    return _lazy("PIL_ImageFont", _load)
