"""Generate fictional Egyptian-style plates for DEMONSTRATION ONLY.
50 different photos of one fictional ID, 100 photos of other fictional IDs.
This is not a real-world dataset and contains no genuine personal vehicle records.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter, features

ROOT = Path(__file__).resolve().parent
TARGET = ('س ب م', '١٢٣٤')
LETTERS = 'أبجدرسصطعفقلمهنوي'
DIGITS = '١٢٣٤٥٦٧٨٩'


def font(size: int, arabic: bool = False):
    paths = ([
        '/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf',
        '/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf',
        'C:/Windows/Fonts/arialbd.ttf',
        'C:/Windows/Fonts/arial.ttf',
        '/System/Library/Fonts/Supplemental/Arial.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ] if arabic else [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        'C:/Windows/Fonts/arialbd.ttf',
        '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
    ])
    for path in paths:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def centered(draw: ImageDraw.ImageDraw, xy, s, f, fill='black', rtl=False):
    opts = {'direction': 'rtl'} if rtl and features.check('raqm') else {}
    draw.text(xy, s, font=f, fill=fill, anchor='mm', **opts)


def create_plate(letters: str, digits: str, seed: int) -> Image.Image:
    r = random.Random(seed)
    # Enlarged render followed by photographic downsampling.
    w, h = 700, 340
    im = Image.new('RGB', (w, h), (236, 238, 240))
    draw = ImageDraw.Draw(im)
    draw.rounded_rectangle((12, 8, 688, 330), radius=18, fill=(r.randint(241, 255),)*3, outline=(34, 39, 45), width=8)
    # Typical Egyptian private-plate blue top band.
    draw.rectangle((22, 20, 678, 114), fill=(r.randint(100, 119), r.randint(179, 204), r.randint(216, 237)))
    centered(draw, (180, 66), 'EGYPT', font(51), fill=(17, 25, 35))
    centered(draw, (522, 61), 'مصر', font(64, True), fill=(17, 25, 35), rtl=True)
    draw.line((350, 122, 350, 318), fill=(43, 46, 53), width=5)
    centered(draw, (181, 215), digits, font(86, True), rtl=True)
    centered(draw, (525, 215), letters, font(82, True), rtl=True)
    # Subtle pin, light scratches, frame effects.
    for cx in (34, 665):
        for cy in (35, 304):
            draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=(90, 95, 100))
    if r.random() < .42:
        x = r.randint(30, 640)
        draw.line((x, 125, x + r.randint(12, 40), r.randint(260, 310)), fill=(190, 193, 196), width=r.randint(1, 2))
    # Modest rotation/shear; all four corners remain in view (plate-crop assumption).
    res = Image.Resampling.BICUBIC
    angle = r.uniform(-3.1, 3.1)
    im = im.rotate(angle, resample=res, expand=False, fillcolor=(r.randint(203, 224),)*3)
    # Simulated mild camera resolution, exposure and noise.
    downs = r.choice([(350, 170), (420, 204), (500, 243), (700, 340)])
    im = im.resize(downs, resample=Image.Resampling.LANCZOS)
    im = ImageEnhance.Contrast(im).enhance(r.uniform(.88, 1.11))
    im = ImageEnhance.Brightness(im).enhance(r.uniform(.84, 1.15))
    if r.random() < .4:
        im = im.filter(ImageFilter.GaussianBlur(radius=r.uniform(.15, .8)))
    a = np.asarray(im).astype(np.int16)
    noise = np.random.default_rng(seed).normal(0, r.uniform(1.1, 7.5), a.shape)
    a = np.uint8(np.clip(a + noise, 0, 255))
    return Image.fromarray(a)


def different_id(seed: int):
    r = random.Random(100000 + seed)
    while True:
        letters = ' '.join(r.sample(LETTERS, 3))
        digits = ''.join(r.choice(DIGITS) for _ in range(4))
        if (letters, digits) != TARGET:
            return letters, digits


def generate(force: bool = False):
    target = ROOT / 'data' / 'target'
    other = ROOT / 'data' / 'other'
    target.mkdir(parents=True, exist_ok=True)
    other.mkdir(parents=True, exist_ok=True)
    if not force and any(target.glob('*.jpg')) and any(other.glob('*.jpg')):
        print('Found existing data: leaving it unchanged. Add --force to replace DEMO images.')
        return
    if force:
        for folder in (target, other):
            for x in folder.glob('*.jpg'):
                x.unlink()
    for i in range(50):
        create_plate(*TARGET, 27000 + i).save(target / f'target_{i+1:03d}.jpg', quality=92)
    for i in range(100):
        create_plate(*different_id(i), 47000 + i).save(other / f'other_{i+1:03d}.jpg', quality=92)
    s = ROOT / 'samples'
    s.mkdir(exist_ok=True)
    create_plate(*TARGET, 99007).save(s / 'test_known.jpg', quality=94)
    create_plate(*different_id(919), 99222).save(s / 'test_unknown.jpg', quality=94)
    print(f'Generated: {len(list(target.glob("*.jpg")))} target, {len(list(other.glob("*.jpg")))} other.')
    print('IMPORTANT: these are fictional, synthetic DEMO photos. Real-world accuracy is NOT established.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--force', action='store_true', help='Replace demo images; NEVER use on real uploaded data!')
    generate(**vars(p.parse_args()))
