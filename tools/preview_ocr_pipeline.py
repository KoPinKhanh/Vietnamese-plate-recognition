"""
Preview and verification tool for OCR dataset cropping and two-row plate rectification.
Generates:
1. outputs/ocr_dataset_preview/ with contact sheet (20 one_row, 20 two_rows, 20 xe may)
2. outputs/two_row_rectification_preview/ (50 images showing original | top | bottom | concat)
"""

import csv
import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def crop_plate_gt(img, bbox, pad_ratio=0.03):
    """Crops plate using ground-truth bbox with clamped padding."""
    ih, iw = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    pad_w = int(bw * pad_ratio)
    pad_h = int(bh * pad_ratio)

    cx1 = max(0, x1 - pad_w)
    cy1 = max(0, y1 - pad_h)
    cx2 = min(iw, x2 + pad_w)
    cy2 = min(ih, y2 + pad_h)

    crop = img[cy1:cy2, cx1:cx2]
    return crop


def find_horizontal_split(gray_crop):
    """
    Finds horizontal split valley between top row and bottom row
    using horizontal projection of pixel intensities in the middle 35%-60% zone.
    """
    h, w = gray_crop.shape
    min_y = int(h * 0.35)
    max_y = int(h * 0.60)

    if max_y <= min_y:
        return int(h * 0.48)

    # Use Sobel or edge gradient to find text gap (gap has minimal horizontal edges)
    grad_y = cv2.Sobel(gray_crop, cv2.CV_32F, 0, 1, ksize=3)
    edge_energy = np.sum(np.abs(grad_y), axis=1)

    # Search in middle zone
    search_zone = edge_energy[min_y:max_y]
    best_offset = int(np.argmin(search_zone))
    split_y = min_y + best_offset

    # Sanity check: must be reasonable split
    if split_y < int(h * 0.35) or split_y > int(h * 0.60):
        split_y = int(h * 0.48)

    return split_y


def rectify_two_row_plate(crop_bgr, target_h=48):
    """
    Splits two-row plate into top row and bottom row,
    then concatenates them horizontally: [top_row, spacer, bottom_row].
    Returns:
        rectified_bgr, top_row, bottom_row
    """
    h, w = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    split_y = find_horizontal_split(gray)

    top_crop = crop_bgr[:split_y, :]
    bot_crop = crop_bgr[split_y:, :]

    th, tw = top_crop.shape[:2]
    bh, bw = bot_crop.shape[:2]

    # Resize both to same height target_h
    scale_top = target_h / float(th) if th > 0 else 1.0
    new_tw = max(1, int(round(tw * scale_top)))
    top_resized = cv2.resize(top_crop, (new_tw, target_h), interpolation=cv2.INTER_LINEAR)

    scale_bot = target_h / float(bh) if bh > 0 else 1.0
    new_bw = max(1, int(round(bw * scale_bot)))
    bot_resized = cv2.resize(bot_crop, (new_bw, target_h), interpolation=cv2.INTER_LINEAR)

    # Small spacer between top and bottom (6px white)
    spacer = np.full((target_h, 6, 3), 255, dtype=np.uint8)
    rectified = np.hstack([top_resized, spacer, bot_resized])

    return rectified, top_crop, bot_crop


def letterbox_ocr(img_bgr, target_size=(160, 48), pad_color=(255, 255, 255)):
    """
    Resizes image maintaining aspect ratio and pads to target_size (width, height).
    Returns standardized (48, 160, 3) image.
    """
    tw, th = target_size
    h, w = img_bgr.shape[:2]

    scale = min(tw / float(w), th / float(h))
    new_w = max(1, min(tw, int(round(w * scale))))
    new_h = max(1, min(th, int(round(h * scale))))

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = tw - new_w
    pad_h = th - new_h

    # Right and bottom padding (or center padding)
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=pad_color
    )
    return padded


def build_ocr_preview(manifest_csv: str):
    preview_dir = Path("outputs/ocr_dataset_preview")
    rectify_dir = Path("outputs/two_row_rectification_preview")
    preview_dir.mkdir(parents=True, exist_ok=True)
    rectify_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    # Group by subset
    one_row_samples = [r for r in records if r["subset"] == "one_row"][:20]
    two_row_samples = [r for r in records if r["subset"] == "two_rows"][:20]
    moto_samples = [r for r in records if r["subset"] == "two_rows_label_xe_may"][:20]

    all_selected = (
        [("one_row", r) for r in one_row_samples] +
        [("two_rows", r) for r in two_row_samples] +
        [("motorcycle", r) for r in moto_samples]
    )

    contact_tiles = []

    for group_name, r in all_selected:
        img_p = r["image_path"]
        txt = r["plate_text"]
        bbox = [int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])]

        img = cv2.imread(img_p)
        if img is None:
            continue

        crop = crop_plate_gt(img, bbox, pad_ratio=0.03)
        if crop.size == 0:
            continue

        # Save individual crop
        cv2.imwrite(str(preview_dir / f"{group_name}_{r['plate_text']}.jpg"), crop)

        # Build preview tile with label banner
        ch, cw = crop.shape[:2]
        tile_w, tile_h = 240, 100
        scale = min(tile_w / float(cw), (tile_h - 25) / float(ch))
        nw, nh = int(cw * scale), int(ch * scale)
        r_crop = cv2.resize(crop, (nw, nh))

        tile = np.full((tile_h, tile_w, 3), 40, dtype=np.uint8)
        px = (tile_w - nw) // 2
        py = (tile_h - 25 - nh) // 2
        tile[py:py + nh, px:px + nw] = r_crop

        cv2.rectangle(tile, (0, tile_h - 25), (tile_w, tile_h), (20, 20, 20), -1)
        label_str = f"{group_name[:4]}: {txt}"
        cv2.putText(tile, label_str, (5, tile_h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        contact_tiles.append(tile)

    # Stitch contact sheet (5 cols x 12 rows = 60 tiles)
    cols = 5
    rows = (len(contact_tiles) + cols - 1) // cols
    sheet = np.zeros((rows * 100, cols * 240, 3), dtype=np.uint8)
    for idx, t in enumerate(contact_tiles):
        r = idx // cols
        c = idx % cols
        sheet[r * 100:(r + 1) * 100, c * 240:(c + 1) * 240] = t

    contact_path = preview_dir / "contact_sheet_preview.jpg"
    cv2.imwrite(str(contact_path), sheet)
    print(f"Saved OCR dataset contact sheet ({len(contact_tiles)} tiles) to: {contact_path}")

    # Generate 50 rectification previews
    two_row_all = [r for r in records if r["layout"] == "two_rows"][:50]
    print(f"\nGenerating 50 rectification previews for two-row plates...")

    for idx, r in enumerate(two_row_all, 1):
        img = cv2.imread(r["image_path"])
        if img is None:
            continue
        bbox = [int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])]
        crop = crop_plate_gt(img, bbox, pad_ratio=0.03)
        if crop.size == 0:
            continue

        rectified, top_row, bot_row = rectify_two_row_plate(crop, target_h=48)
        final_160x48 = letterbox_ocr(rectified, target_size=(160, 48))

        # Build comparison strip: [original | top | bot | rectified | final_160x48]
        ch = 70
        # Resize all to height ch for side-by-side display
        def resize_to_h(im, h):
            s = h / float(im.shape[0])
            return cv2.resize(im, (max(1, int(im.shape[1] * s)), h))

        v_orig = resize_to_h(crop, ch)
        v_top = resize_to_h(top_row, ch)
        v_bot = resize_to_h(bot_row, ch)
        v_rec = resize_to_h(rectified, ch)
        v_final = resize_to_h(final_160x48, ch)

        spacer = np.full((ch, 4, 3), 100, dtype=np.uint8)
        strip = np.hstack([v_orig, spacer, v_top, spacer, v_bot, spacer, v_rec, spacer, v_final])

        # Add top banner
        banner = np.zeros((24, strip.shape[1], 3), dtype=np.uint8)
        info_txt = f"#{idx:02d} [{r['subset']}] GT: {r['plate_text']} | Orig -> Top | Bot -> Concat -> Final 160x48"
        cv2.putText(banner, info_txt, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        combined = np.vstack([banner, strip])
        save_p = rectify_dir / f"rect_{idx:02d}_{r['plate_text']}.jpg"
        cv2.imwrite(str(save_p), combined)

    print(f"Saved 50 rectification previews to: {rectify_dir}")


if __name__ == "__main__":
    build_ocr_preview("datasets/VNLP_processed/train.csv")
