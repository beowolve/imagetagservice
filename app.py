"""
ImageTagService – entrypoint.

Server mode (no args):
    python app.py
    → starts FastAPI on a free port, writes port to runtime/port.txt

CLI mode (image path as first arg):
    python app.py image.jpg
    python app.py image.jpg --tags "cat,dog"
"""

from __future__ import annotations

import logging
import sys
import warnings

# Suppress FutureWarnings from transformers / timm (version mismatch noise)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Silence chatty third-party loggers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("timm").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    from model import load_model
    load_model()
    yield


app = FastAPI(
    title="ImageTagService",
    description="Local image tagging via the Recognize Anything Model (RAM).",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
def health() -> dict:
    return {"status": "ok"}


@app.post("/tag", summary="Tag a single image")
async def tag_endpoint(
    file: UploadFile = File(..., description="Image file to tag."),
    known_tags: str = Form(
        default="",
        description="Comma-separated list of known tags to match against.",
    ),
) -> JSONResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    from utils import image_hash, validate_image

    try:
        pil_image = validate_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    tags_list = [t.strip() for t in known_tags.split(",") if t.strip()]

    from model import tag_image_with_hash

    try:
        result = tag_image_with_hash(
            pil_image,
            image_hash(data),
            tags_list or None,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(content=result)


@app.post("/tag-batch", summary="Tag multiple images")
async def tag_batch_endpoint(
    files: List[UploadFile] = File(..., description="Image files to tag."),
    known_tags: str = Form(
        default="",
        description="Comma-separated list of known tags (applied to all images).",
    ),
) -> JSONResponse:
    if not files:
        return JSONResponse(content={"results": []})

    from model import tag_image_with_hash
    from utils import image_hash, validate_image

    tags_list = [t.strip() for t in known_tags.split(",") if t.strip()]
    results = []

    for upload in files:
        data = await upload.read()
        entry: dict = {"filename": upload.filename, "error": None}

        if not data:
            entry["error"] = "Empty file."
            results.append(entry)
            continue

        try:
            pil_image = validate_image(data)
            result = tag_image_with_hash(
                pil_image,
                image_hash(data),
                tags_list or None,
            )
            entry.update(result)
        except ValueError as exc:
            entry["error"] = str(exc)
        except RuntimeError as exc:
            entry["error"] = str(exc)

        results.append(entry)

    return JSONResponse(content={"results": results})


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _run_server() -> None:
    import uvicorn

    from config import DEFAULT_HOST
    from utils import find_free_port, write_port_file

    port = find_free_port()
    write_port_file(port)
    logger.info("Starting server on http://%s:%d", DEFAULT_HOST, port)
    logger.info("Port written to runtime/port.txt")

    uvicorn.run(
        app,
        host=DEFAULT_HOST,
        port=port,
        log_level="info",
    )


def _run_cli(args: list[str]) -> None:
    from model import load_model

    load_model()

    from cli import run_cli

    sys.exit(run_cli(args))


if __name__ == "__main__":
    # Detect CLI vs server mode based on whether arguments were supplied.
    # Arguments that start with "--" are uvicorn/internal flags; a positional
    # first argument is treated as an image path → CLI mode.
    cli_args = sys.argv[1:]

    if cli_args and not cli_args[0].startswith("--"):
        _run_cli(cli_args)
    else:
        # Server mode: load model before uvicorn so errors surface immediately.
        # (The @app.on_event("startup") also calls load_model, which is a no-op
        #  if the model is already loaded.)
        from model import load_model

        load_model()
        _run_server()
