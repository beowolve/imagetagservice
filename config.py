import sys
from pathlib import Path


def _base_dir() -> Path:
    # In a Nuitka onefile EXE, __file__ points to the temp extraction dir.
    # sys.executable always points to the real EXE on disk.
    if "__compiled__" in dir():
        return Path(sys.executable).parent
    return Path(__file__).parent


_BASE = _base_dir()

MODEL_DIR = _BASE / "models"
MODEL_FILENAME = "ram_swin_large_14m.pth"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

IMAGE_SIZE = 384
CONFIDENCE_THRESHOLD = 0.95

RUNTIME_DIR = _BASE / "runtime"
PORT_FILE = RUNTIME_DIR / "port.txt"

DEFAULT_HOST = "127.0.0.1"

# Number of (image_hash, known_tags) results to keep in memory
CACHE_SIZE = 128
