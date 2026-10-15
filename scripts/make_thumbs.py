"""Generate small thumbnails for every image under assets/images.

Mirrors the assets/images tree into assets/images/thumbs/ (same relative
paths), resizing to at most 480x480 px. Skips files that already have a
thumbnail, so it is cheap to re-run. Run by .github/workflows/
generate-thumbs.yml on every push that touches assets/images.
"""

import os

from PIL import Image

ROOT = "assets/images"
THUMBS = os.path.join(ROOT, "thumbs")
MAX_SIZE = (480, 480)
QUALITY = 72
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    generated = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        if dirpath == ROOT:
            # Never thumbnail the thumbnails.
            dirnames[:] = [d for d in dirnames if d != "thumbs"]
        for name in filenames:
            if os.path.splitext(name)[1].lower() not in EXTS:
                continue
            src = os.path.join(dirpath, name)
            rel = os.path.relpath(src, ROOT)
            dst = os.path.join(THUMBS, rel)
            if os.path.exists(dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            img = Image.open(src)
            img.thumbnail(MAX_SIZE)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img.save(dst, quality=QUALITY, optimize=True)
            generated += 1
    print(f"generated {generated} thumbnail(s)")


if __name__ == "__main__":
    main()
