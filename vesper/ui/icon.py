"""A real icon, drawn in code, because the alternative was a stock one.

The tray used `IDI_APPLICATION`, the generic Windows program icon. In a tray of
fifteen things that is indistinguishable from everything else, and the whole
point of the icon is that a glance tells you whether Vesper is listening.

Drawn rather than shipped as an asset file for two reasons. Pillow is not
installed and would be a new dependency to draw one 16x16 mark, on a machine
that has had wheels blocked. And an icon generated from parameters can have a
state variant added by changing a colour, rather than by opening an editor.

The mark is the evening star, which is what Vesper means. A four point star on a
disc reads at 16 pixels where a wordmark or a waveform turns to mush, and the
colour carries the state: lit when listening, dimmed when paused.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# Drawn large and shrunk, which is the cheapest anti-aliasing there is and the
# only reason the star's points survive at 16 pixels.
SUPERSAMPLE = 8
SIZES = (16, 32, 48)

# Deliberate, not decorative. Listening is the resting state and should look
# calm rather than alarming; paused is drained of colour rather than red,
# because paused is a choice you made, not a fault.
LISTENING = ((28, 32, 56), (252, 219, 149))    # deep night, warm star
PAUSED = ((58, 60, 70), (166, 170, 182))       # slate, cold star
WORKING = ((28, 32, 56), (126, 220, 226))      # the same night, a cool star


def _disc(size: int) -> np.ndarray:
    """A filled circle with a soft edge, as a 0 to 1 coverage mask."""
    axis = (np.arange(size) + 0.5) / size * 2 - 1
    x, y = np.meshgrid(axis, axis)
    return (np.sqrt(x * x + y * y) <= 0.96).astype(np.float32)


def _star(size: int, fatness: float) -> np.ndarray:
    """A four point star: two tapered spikes crossed at the centre.

    Built from a distance field rather than polygons so the points stay sharp
    and the waist stays smooth when it is shrunk down.

    `fatness` exists because the elegant version disappears. At 128 pixels a
    needle-thin star looks right; shrunk to the 16 pixels you actually see in
    the tray, those points fall below a pixel and the whole mark turns into a
    grey smudge. Small sizes get a blunter star on purpose.
    """
    axis = (np.arange(size) + 0.5) / size * 2 - 1
    x, y = np.meshgrid(axis, axis)
    ax, ay = np.abs(x), np.abs(y)

    # A four pointed star is the set where |x|^p + |y|^p is small, for p below
    # one. The exponent decides how needle-like the points are.
    with np.errstate(invalid="ignore"):
        field = np.power(ax, fatness) + np.power(ay, fatness)
    return (field <= 0.95).astype(np.float32)


def _render(size: int, background, foreground) -> np.ndarray:
    """One RGBA image, drawn big and averaged down."""
    big = size * SUPERSAMPLE
    disc = _disc(big)
    # Blunter the smaller it gets, so the points survive the downsample.
    fatness = 0.68 if size <= 20 else (0.60 if size <= 40 else 0.52)
    star = _star(big, fatness) * disc  # the star never spills past the disc

    rgba = np.zeros((big, big, 4), dtype=np.float32)
    for channel in range(3):
        rgba[..., channel] = (
            background[channel] * (1 - star) + foreground[channel] * star
        )
    rgba[..., 3] = disc * 255.0

    # Average each block down to one pixel. Premultiplied, so edge pixels do
    # not pick up colour from the transparent surround.
    rgba[..., :3] *= rgba[..., 3:4] / 255.0
    small = rgba.reshape(size, SUPERSAMPLE, size, SUPERSAMPLE, 4).mean(axis=(1, 3))
    alpha = np.clip(small[..., 3:4], 1e-6, 255.0)
    small[..., :3] = np.clip(small[..., :3] / (alpha / 255.0), 0, 255)
    return np.clip(small, 0, 255).astype(np.uint8)


def _bmp_entry(image: np.ndarray) -> bytes:
    """One ICO image in the classic DIB form: header, BGRA rows, mask."""
    size = image.shape[0]
    header = struct.pack(
        "<IiiHHIIiiII",
        40,            # header size
        size,          # width
        size * 2,      # height, doubled: colour rows then the mask rows
        1,             # planes
        32,            # bits per pixel
        0, 0, 0, 0, 0, 0,
    )
    # ICO stores rows bottom up, and channels as BGRA.
    flipped = image[::-1]
    pixels = flipped[..., [2, 1, 0, 3]].tobytes()

    # The AND mask is unused for 32 bit icons but must still be present, padded
    # to four byte rows. Omitting it produces an icon Windows silently ignores.
    row_bytes = ((size + 31) // 32) * 4
    mask = b"\x00" * (row_bytes * size)
    return header + pixels + mask


def write_ico(path: Path, background, foreground, sizes=SIZES) -> Path:
    """Write a multi-resolution .ico. Returns the path."""
    images = [_render(size, background, foreground) for size in sizes]
    entries = [_bmp_entry(image) for image in images]

    offset = 6 + 16 * len(entries)
    directory = struct.pack("<HHH", 0, 1, len(entries))
    for image, blob in zip(images, entries):
        size = image.shape[0]
        directory += struct.pack(
            "<BBBBHHII",
            size if size < 256 else 0,
            size if size < 256 else 0,
            0, 0, 1, 32, len(blob), offset,
        )
        offset += len(blob)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(directory + b"".join(entries))
    return path


STATES = {"listening": LISTENING, "paused": PAUSED, "working": WORKING}


def ensure(directory: Path) -> dict[str, Path]:
    """Draw the icon set if it is not already there. Returns state to path.

    Regenerated only when missing, so startup does not pay for it every time,
    and deleting the folder is all it takes to pick up a change to the colours.
    """
    icons: dict[str, Path] = {}
    for state, (background, foreground) in STATES.items():
        target = Path(directory) / f"vesper-{state}.ico"
        try:
            if not target.exists() or target.stat().st_size == 0:
                write_ico(target, background, foreground)
            icons[state] = target
        except (OSError, ValueError):
            # No icon is survivable; refusing to start over one is not.
            continue
    return icons
