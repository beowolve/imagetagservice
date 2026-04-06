from pathlib import Path

# Resolve paths relative to this file so the service works from any cwd
_BASE = Path(__file__).parent

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
