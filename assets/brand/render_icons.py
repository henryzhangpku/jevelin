"""Render the PNG icons from the same geometry as docs/favicon.svg (4x supersampled)."""
from pathlib import Path
from PIL import Image, ImageDraw

INK, BLUE = (17, 20, 24, 255), (57, 135, 229, 255)
OUT = Path(__file__).resolve().parents[2] / "docs"


def mark(size):
    s = 4 * size
    k = s / 64.0
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=14 * k, fill=INK)
    d.line([(14 * k, 50 * k), (41 * k, 23 * k)], fill=BLUE, width=round(6.5 * k))
    for x, y in ((14, 50), (41, 23)):                     # round caps
        r = 3.25 * k
        d.ellipse([x * k - r, y * k - r, x * k + r, y * k + r], fill=BLUE)
    d.polygon([(53 * k, 11 * k), (49.6 * k, 28 * k), (36 * k, 14.4 * k)], fill=BLUE)
    return img.resize((size, size), Image.LANCZOS)


for name, size in (("favicon-32.png", 32), ("apple-touch-icon.png", 180), ("icon-512.png", 512)):
    mark(size).save(OUT / name)
    print("wrote", name)
