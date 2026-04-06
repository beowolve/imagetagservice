import hashlib
import socket
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from config import PORT_FILE, RUNTIME_DIR


def find_free_port() -> int:
    """Bind to port 0 to let the OS assign a free port, then return it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def write_port_file(port: int) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    PORT_FILE.write_text(str(port), encoding="utf-8")


def validate_image(data: bytes) -> Image.Image:
    """Open image bytes, verify it's a valid image, return RGB PIL image."""
    try:
        img = Image.open(BytesIO(data))
        img.verify()  # raises on corrupt files
        # Re-open after verify (verify() closes the internal stream)
        img = Image.open(BytesIO(data)).convert("RGB")
        return img
    except (UnidentifiedImageError, Exception) as exc:
        raise ValueError(f"Invalid or unreadable image: {exc}") from exc


def image_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()
