# ImageTagService

Local image tagging service using the [Recognize Anything Model (RAM)](https://github.com/xinyu1205/recognize-anything), designed to be easily integrated into other applications. It exposes a simple HTTP API so any program can request image tags without bundling ML dependencies itself — just send an image, get tags back.

Runs as a FastAPI HTTP server or standalone CLI. No external services — fully local, CPU-first with optional CUDA.

> Developed and tested on Windows. Should work on Linux and macOS as well, but has not been tested on those platforms.

---

## Project structure

```
ImageTagService/
├── app.py           entrypoint — routes to CLI or server
├── model.py         RAM singleton, inference, tag matching
├── cli.py           argparse CLI with csv/json output
├── config.py        all paths & constants
├── utils.py         free port, port file, image validation, hash
├── requirements.txt
├── build.bat        Nuitka build script → dist/ImageTagService.exe
└── models/          place model file here (not in git)
```

---

## Quick start

### 1. Create and activate a virtual environment

> **Requires Python 3.11 or 3.12.** Newer versions (3.13+) are not yet supported by RAM's dependencies (`transformers`, `tokenizers`).

**Recommended — using [uv](https://github.com/astral-sh/uv):**
```bat
scoop install uv
uv venv --python 3.11
.venv\Scripts\activate
```

**Alternative — using the `py` launcher:**
```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bat
uv pip install -r requirements.txt
```

Or with plain pip:
```bat
pip install -r requirements.txt
```

> **Note:** RAM does not declare all its dependencies. The following packages are pinned explicitly in `requirements.txt` to cover the gaps: `transformers<4.36`, `timm`, `scipy`, `fairscale`.

### 3. Download the model (~5.6 GB)

```bat
mkdir models
curl -L -o models\ram_swin_large_14m.pth https://huggingface.co/xinyu1205/recognize_anything_model/resolve/main/ram_swin_large_14m.pth
```

### 4. CLI mode

```bat
python app.py image.jpg
python app.py image.jpg --tags "cat,dog,sky"
python app.py image.jpg --tags "cat,dog" --format json
python app.py image.jpg --all
```

### 5. Server mode

```bat
python app.py
```

Check the assigned port:
```bat
type runtime\port.txt
```

Then use the API:
```bat
curl http://127.0.0.1:<port>/health
curl -X POST -F "file=@image.jpg" -F "known_tags=cat,dog" http://127.0.0.1:<port>/tag
curl -X POST -F "files=@a.jpg" -F "files=@b.jpg" http://127.0.0.1:<port>/tag-batch
```

### 6. Build EXE

Make sure the venv is activated, then:

```bat
uv pip install nuitka ordered-set zstandard
build.bat
```

Copy the model alongside the executable:
```bat
xcopy /E /I models dist\models
```

Run:
```bat
dist\ImageTagService.exe image.jpg
```

---

## CLI reference

| Argument | Description |
|---|---|
| `image` | Path to the image file |
| `--tags "cat,dog"` | Comma-separated known tags to match against |
| `--all` | Print all RAM tags instead of matched/suggested |
| `--format csv\|json` | Output format (default: `csv`) |

---

## API

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /tag`

| Field | Type | Description |
|---|---|---|
| `file` | file | Image to tag |
| `known_tags` | string (form) | Comma-separated known tags (optional) |

### `POST /tag-batch`

| Field | Type | Description |
|---|---|---|
| `files` | file[] | Images to tag |
| `known_tags` | string (form) | Comma-separated known tags, applied to all images (optional) |

### Response shape

```json
{
  "matched_tags":   ["cat", "sky"],
  "suggested_tags": ["grass", "field"],
  "all_tags":       ["cat", "sky", "grass", "field"]
}
```

- `matched_tags` — known tags that were found in the image
- `suggested_tags` — RAM tags that did not match any known tag
- `all_tags` — every tag RAM returned above the confidence threshold

When no `known_tags` are provided, `matched_tags` is empty and all RAM tags appear in `suggested_tags`.

---

## Configuration

Edit `config.py` to change defaults:

| Constant | Default | Description |
|---|---|---|
| `MODEL_FILENAME` | `ram_swin_large_14m.pth` | Model file name inside `models/` |
| `IMAGE_SIZE` | `384` | Input resolution (px) |
| `CONFIDENCE_THRESHOLD` | `0.68` | Tag confidence cutoff (0–1). Higher = fewer but more certain tags |
| `CACHE_SIZE` | `128` | In-memory result cache entries |
| `DEFAULT_HOST` | `127.0.0.1` | Server bind address |

**Confidence threshold guidance:**

| Value | Effect |
|---|---|
| `0.68` | RAM default — many tags |
| `0.80`–`0.90` | Fewer, more reliable tags |
| `0.95`+ | Only very confident tags |

### Available RAM tags

RAM knows **4585 tags**. The full list is at:
```
https://github.com/xinyu1205/recognize-anything/blob/main/ram/data/ram_tag_list.txt
```

---

## License

[MIT](LICENSE) — Copyright (c) 2026 Andreas Ebner
