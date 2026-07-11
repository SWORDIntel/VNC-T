#!/usr/bin/env python3
"""Smart parallel VNC dataset augmentation using OpenCV + visual heuristics.

Uses cv2-based transformations inspired by screenshot_analyzer.py:
- Perspective warps (simulate different VNC viewing angles)
- Elastic distortions (deform without destroying text)
- JPEG compression artifacts (simulate low-bandwidth VNC)
- Scan-line / screen-capture artifacts
- Text-aware occlusions (overlay fake window/dialog rectangles)
- Color channel swaps (simulate different monitor calibrations)
- Grid-line augmentation for SCADA screens
- Region-specific blur (simulate motion / poor focus on part of screen)

Usage:
  python3 augment_dataset.py --data-dir data/training_dataset --target-per-class 900
"""
import argparse
import os
import random
import shutil
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def extract_visual_features(cv_img):
    """Extract features inspired by screenshot_analyzer._extract_visual_features."""
    h, w = cv_img.shape[:2]
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    brightness = float(gray.mean())
    edges = cv2.Canny(gray, 80, 180)
    edge_density = float(np.count_nonzero(edges)) / edges.size

    # Detect grid lines (SCADA indicator)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=max(50, min(w, h) // 12),
                            minLineLength=max(40, min(w, h) // 10),
                            maxLineGap=8)
    horizontal = vertical = 0
    if lines is not None:
        for line in lines[:500]:
            x1, y1, x2, y2 = line.reshape(-1)[:4]
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if dx > 3 * max(dy, 1):
                horizontal += 1
            elif dy > 3 * max(dx, 1):
                vertical += 1
    grid_score = min(1.0, (min(horizontal, vertical) + 0.25 * max(horizontal, vertical)) / 80.0)

    # Saturated color ratios (alarm indicators)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    hue = hsv[:, :, 0]
    valid = (sat > 110) & (val > 80)
    total = float(h * w)
    red_ratio = float((valid & ((hue < 10) | (hue > 170))).sum()) / total
    amber_ratio = float((valid & (hue >= 10) & (hue <= 35)).sum()) / total
    green_ratio = float((valid & (hue >= 36) & (hue <= 90)).sum()) / total

    return {
        "brightness": brightness,
        "dark_theme": brightness < 90.0,
        "edge_density": edge_density,
        "grid_score": grid_score,
        "red_ratio": red_ratio,
        "amber_ratio": amber_ratio,
        "green_ratio": green_ratio,
        "h": h, "w": w,
    }


def add_jpeg_artifacts(cv_img, quality=None):
    """Simulate low-bandwidth VNC JPEG compression."""
    if quality is None:
        quality = random.randint(35, 70)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf = cv2.imencode('.jpg', cv_img, encode_param)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def add_scan_lines(cv_img):
    """Add CRT/scan-line artifacts simulating older monitors."""
    h, w = cv_img.shape[:2]
    overlay = np.ones_like(cv_img, dtype=np.float32)
    for y in range(0, h, 2):
        overlay[y] = 0.85 + random.uniform(-0.05, 0.05)
    return np.clip(cv_img.astype(np.float32) * overlay, 0, 255).astype(np.uint8)


def add_perspective_warp(cv_img, intensity=None):
    """Apply mild perspective transform to simulate off-angle VNC capture."""
    h, w = cv_img.shape[:2]
    if intensity is None:
        intensity = random.uniform(0.002, 0.008)
    # Four source corners
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    # Perturbed destination corners
    margin_x = int(w * intensity)
    margin_y = int(h * intensity)
    dst = np.float32([
        [random.randint(-margin_x, margin_x), random.randint(-margin_y, margin_y)],
        [w + random.randint(-margin_x, margin_x), random.randint(-margin_y, margin_y)],
        [w + random.randint(-margin_x, margin_x), h + random.randint(-margin_y, margin_y)],
        [random.randint(-margin_x, margin_x), h + random.randint(-margin_y, margin_y)],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(cv_img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def add_elastic_distortion(cv_img, alpha=None, sigma=None):
    """Elastic deformation that preserves overall structure."""
    h, w = cv_img.shape[:2]
    if alpha is None:
        alpha = random.uniform(15, 40)
    if sigma is None:
        sigma = random.uniform(3, 6)
    dx = cv2.GaussianBlur((np.random.rand(h, w).astype(np.float32) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(h, w).astype(np.float32) * 2 - 1), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(cv_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def add_occlusion(cv_img, features):
    """Overlay a fake window/dialog rectangle to simulate UI popups."""
    h, w = cv_img.shape[:2]
    # Random rectangle size (5-25% of screen)
    rw = random.randint(int(w * 0.1), int(w * 0.35))
    rh = random.randint(int(h * 0.08), int(h * 0.25))
    rx = random.randint(0, max(1, w - rw))
    ry = random.randint(0, max(1, h - rh))

    # Pick a style: dark window, light dialog, or alarm banner
    style = random.choice(["dark_window", "light_dialog", "alarm_banner", "taskbar"])

    if style == "dark_window":
        color = (30, 30, 35)
        border = (60, 60, 70)
    elif style == "light_dialog":
        color = (240, 240, 240)
        border = (100, 100, 100)
    elif style == "alarm_banner":
        color = (0, 0, 180) if random.random() > 0.5 else (0, 140, 255)
        border = (255, 255, 255)
    else:  # taskbar
        rh = random.randint(25, 40)
        ry = h - rh
        rx = 0
        rw = w
        color = (40, 40, 45)
        border = (80, 80, 85)

    overlay = cv_img.copy()
    cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), color, -1)
    cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), border, 1)

    # Add a title bar for window styles
    if style in ("dark_window", "light_dialog"):
        bar_h = min(20, rh // 4)
        bar_color = (60, 60, 70) if style == "dark_window" else (200, 200, 210)
        cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + bar_h), bar_color, -1)

    # Blend with some transparency
    alpha = random.uniform(0.7, 0.95)
    return cv2.addWeighted(cv_img, 1 - alpha, overlay, alpha, 0)


def add_color_shift(cv_img, features):
    """Shift color channels to simulate different monitor calibrations."""
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Hue shift
    hue_shift = random.randint(-10, 10)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    # Saturation shift
    sat_shift = random.uniform(0.8, 1.2)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_shift, 0, 255)
    # Value shift
    val_shift = random.uniform(0.85, 1.15)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_shift, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def add_region_blur(cv_img):
    """Blur a random region to simulate motion or poor focus."""
    h, w = cv_img.shape[:2]
    rw = random.randint(int(w * 0.15), int(w * 0.4))
    rh = random.randint(int(h * 0.15), int(h * 0.4))
    rx = random.randint(0, max(1, w - rw))
    ry = random.randint(0, max(1, h - rh))
    ksize = random.choice([5, 7, 9, 11])
    blurred = cv2.GaussianBlur(cv_img, (ksize, ksize), 0)
    mask = np.zeros_like(cv_img)
    cv2.rectangle(mask, (rx, ry), (rx + rw, ry + rh), (255, 255, 255), -1)
    # Feather the mask edges
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    return np.where(mask > 0, blurred, cv_img).astype(np.uint8)


def add_vignette(cv_img):
    """Add subtle vignette to simulate older CRT monitors."""
    h, w = cv_img.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    vignette = 1.0 - 0.25 * (dist / max_dist) ** 2
    vignette = np.clip(vignette, 0.6, 1.0)
    result = cv_img.astype(np.float32) * vignette[:, :, np.newaxis]
    return np.clip(result, 0, 255).astype(np.uint8)


def augment_image_pil(img: Image.Image, variant: int) -> Image.Image:
    """Apply PIL-based augmentation variant (basic transforms).
    Always applies at least 2 transforms to avoid near-copies."""
    random.seed(variant * 999 + 42)
    out = img.copy()

    applied = 0
    if random.random() > 0.3:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
        applied += 1
    if random.random() > 0.7:
        out = out.transpose(Image.FLIP_TOP_BOTTOM)
        applied += 1
    if random.random() > 0.6:
        out = out.rotate(random.choice([-5, -3, 3, 5]), expand=False, fillcolor=(0, 0, 0))
        applied += 1
    # Always apply brightness/contrast to ensure minimum divergence
    out = ImageEnhance.Brightness(out).enhance(random.uniform(0.85, 1.15))
    applied += 1
    out = ImageEnhance.Contrast(out).enhance(random.uniform(0.88, 1.12))
    applied += 1
    if random.random() > 0.6:
        out = ImageEnhance.Color(out).enhance(random.uniform(0.85, 1.15))
        applied += 1

    return out


def augment_image_smart(img: Image.Image, variant: int) -> Image.Image:
    """Apply OpenCV-based smart augmentation using visual features.

    Uses feature extraction inspired by screenshot_analyzer.py to adapt
    augmentation type based on image content (SCADA grid, dark theme, etc).
    """
    cv_img = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
    random.seed(variant * 999 + hash(img.tobytes()[:1000]) % 100000 + 42)
    features = extract_visual_features(cv_img)

    # Build augmentation pipeline — pick 3-5 transforms per variant
    available = [
        "hflip", "vflip", "rotate", "brightness", "contrast",
        "jpeg_artifacts", "scan_lines", "perspective", "elastic",
        "occlusion", "color_shift", "region_blur", "vignette",
        "noise", "blur",
    ]

    # Weight transforms based on image features
    weights = {}
    for t in available:
        weights[t] = 1.0

    # SCADA/grid images: more perspective, elastic, color shift (preserve structure)
    if features["grid_score"] > 0.1:
        weights["perspective"] = 2.0
        weights["elastic"] = 1.5
        weights["color_shift"] = 1.5
        weights["occlusion"] = 0.5  # don't cover important SCADA data
        weights["scan_lines"] = 1.5

    # Dark theme: more brightness/contrast variation
    if features["dark_theme"]:
        weights["brightness"] = 2.0
        weights["contrast"] = 2.0
        weights["vignette"] = 1.5

    # High edge density: more JPEG artifacts, region blur
    if features["edge_density"] > 0.05:
        weights["jpeg_artifacts"] = 1.5
        weights["region_blur"] = 1.5

    # Alarm colors present: preserve color shift to vary alarm appearance
    if features["red_ratio"] > 0.01 or features["amber_ratio"] > 0.01:
        weights["color_shift"] = 2.0

    # Select 3-5 transforms — always at least 3 to avoid near-copies
    num_transforms = random.randint(3, 5)
    selected = []
    for _ in range(num_transforms):
        choices = [t for t in available if t not in selected]
        if not choices:
            break
        wts = [weights[t] for t in choices]
        pick = random.choices(choices, weights=wts, k=1)[0]
        selected.append(pick)

    # Always ensure at least one structural transform (not just color/brightness)
    structural = {"hflip", "vflip", "rotate", "perspective", "elastic", "occlusion", "region_blur", "scan_lines"}
    if not any(t in structural for t in selected):
        selected.append(random.choice(list(structural)))

    # Apply transforms
    for t in selected:
        if t == "hflip":
            cv_img = cv2.flip(cv_img, 1)
        elif t == "vflip":
            cv_img = cv2.flip(cv_img, 0)
        elif t == "rotate":
            angle = random.choice([-5, -3, 3, 5])
            h, w = cv_img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            cv_img = cv2.warpAffine(cv_img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        elif t == "brightness":
            factor = random.uniform(0.75, 1.25)
            cv_img = np.clip(cv_img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        elif t == "contrast":
            factor = random.uniform(0.8, 1.2)
            cv_img = np.clip((cv_img.astype(np.float32) - 128) * factor + 128, 0, 255).astype(np.uint8)
        elif t == "jpeg_artifacts":
            cv_img = add_jpeg_artifacts(cv_img)
        elif t == "scan_lines":
            cv_img = add_scan_lines(cv_img)
        elif t == "perspective":
            cv_img = add_perspective_warp(cv_img)
        elif t == "elastic":
            cv_img = add_elastic_distortion(cv_img)
        elif t == "occlusion":
            cv_img = add_occlusion(cv_img, features)
        elif t == "color_shift":
            cv_img = add_color_shift(cv_img, features)
        elif t == "region_blur":
            cv_img = add_region_blur(cv_img)
        elif t == "vignette":
            cv_img = add_vignette(cv_img)
        elif t == "noise":
            noise = np.random.normal(0, random.randint(5, 15), cv_img.shape).astype(np.float32)
            cv_img = np.clip(cv_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        elif t == "blur":
            ksize = random.choice([3, 5])
            cv_img = cv2.GaussianBlur(cv_img, (ksize, ksize), 0)

    # Convert back to PIL
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def process_one(args):
    src_path, out_path, variant = args
    try:
        img = Image.open(src_path).convert("RGB")
        # Use smart augmentation for most variants, PIL for a few (diversity)
        if variant % 5 == 0:
            aug = augment_image_pil(img, variant)
        else:
            aug = augment_image_smart(img, variant)
        # Verify minimum divergence — re-augment with stronger params if too similar
        orig_arr = np.array(img).astype(float)
        aug_arr = np.array(aug).astype(float)
        if orig_arr.shape == aug_arr.shape:
            diff = np.abs(orig_arr - aug_arr).mean()
            if diff < 3.0:
                # Too similar — apply additional noise + brightness shift
                aug_arr = aug_arr.astype(np.float32)
                noise = np.random.normal(0, 12, aug_arr.shape).astype(np.float32)
                aug_arr = np.clip(aug_arr * 1.1 + noise, 0, 255).astype(np.uint8)
                aug = Image.fromarray(aug_arr.astype(np.uint8))
        aug.save(out_path, "JPEG", quality=90)
        return 1
    except Exception as e:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Fast parallel VNC dataset augmentation")
    parser.add_argument("--data-dir", default="data/training_dataset", help="Source dataset directory")
    parser.add_argument("--output-dir", default="data/training_dataset_augmented", help="Output directory")
    parser.add_argument("--target-per-class", type=int, default=900, help="Target images per class")
    parser.add_argument("--max-augments-per-image", type=int, default=8, help="Max augmented variants per original")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    src_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # Build all tasks first
    tasks = []
    class_counts = {}

    for cls_dir in sorted(src_dir.iterdir()):
        if not cls_dir.is_dir():
            continue

        cls_name = cls_dir.name
        originals = sorted(cls_dir.glob("*.jpg"))
        count = len(originals)
        class_counts[cls_name] = count

        out_cls = out_dir / cls_name
        out_cls.mkdir(parents=True, exist_ok=True)

        # Copy originals
        for img_path in originals:
            shutil.copy2(img_path, out_cls / img_path.name)

        if count >= args.target_per_class:
            print(f"  {cls_name}: {count} (no augmentation needed)")
            continue

        needed = args.target_per_class - count
        augments_per = min(args.max_augments_per_image, (needed // count) + 1)
        cls_tasks = []

        for img_path in originals:
            for v in range(augments_per):
                if len(cls_tasks) >= needed:
                    break
                out_path = str(out_cls / f"aug_{v}_{img_path.stem}.jpg")
                cls_tasks.append((str(img_path), out_path, v))
            if len(cls_tasks) >= needed:
                break

        tasks.extend(cls_tasks[:needed])

    if not tasks:
        total = sum(class_counts.values())
        print(f"\nNo augmentation needed. Total: {total} images")
        return

    print(f"Generating {len(tasks)} augmented images using {args.workers} workers...")
    t0 = time.time()

    with Pool(args.workers) as pool:
        results = pool.map(process_one, tasks, chunksize=50)

    elapsed = time.time() - t0
    success = sum(results)

    # Print summary
    total = 0
    for cls_name in sorted(class_counts):
        orig = class_counts[cls_name]
        out_cls = out_dir / cls_name
        aug = len(list(out_cls.glob("aug_*.jpg")))
        total += orig + aug
        print(f"  {cls_name}: {orig} originals + {aug} augmented = {orig + aug}")

    print(f"\nTotal: {total} images ({success} augmented in {elapsed:.1f}s)")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
