import os
import json
import argparse
from typing import Tuple
from PIL import Image

def _bbox_to_pixels(bbox, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """
    Convert MegaDetector bbox [x, y, w, h] in relative coords (0-1) to
    absolute pixel box (left, top, right, bottom), clamped to image bounds.
    """
    x, y, w, h = bbox
    left = max(0, int(round(x * img_w)))
    top = max(0, int(round(y * img_h)))
    right = min(img_w, int(round((x + w) * img_w)))
    bottom = min(img_h, int(round((y + h) * img_h)))
    # Ensure non-negative and at least within bounds
    left = max(0, min(left, img_w))
    top = max(0, min(top, img_h))
    right = max(0, min(right, img_w))
    bottom = max(0, min(bottom, img_h))
    return left, top, right, bottom

def crop_images_from_megadetector(
    json_path: str,
    input_root: str,
    output_root: str,
    confidence_threshold: float = 0.5,
    target_category: str = "1",
    min_width: int = 500,
    min_height: int = 500
):
    """
    Read a MegaDetector JSON file and crop detections from images.

    Filters:
    - Only detections with category == target_category (default: "1")
    - Only detections with conf >= confidence_threshold (default: 0.5)
    - Only detections whose pixel width >= min_width and height >= min_height (defaults: 500x500)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    os.makedirs(output_root, exist_ok=True)
    crop_count = 0
    skipped_small = 0
    skipped_lowconf = 0
    skipped_othercat = 0

    images = data.get("images", [])
    for img_entry in images:
        rel_path = img_entry.get("file") or img_entry.get("file_name") or img_entry.get("filename")
        if not rel_path:
            continue

        src_path = os.path.join(input_root, rel_path)
        if not os.path.exists(src_path):
            # Try a fallback if paths in JSON are basenames only
            basename = os.path.basename(rel_path)
            src_path_alt = os.path.join(input_root, basename)
            src_path = src_path if os.path.exists(src_path) else src_path_alt

        if not os.path.exists(src_path):
            # Can't find image, skip
            continue

        dets = img_entry.get("detections") or img_entry.get("objects") or []
        if not dets:
            continue

        # Prepare output dir mirroring relative path (without filename)
        rel_dir = os.path.dirname(rel_path)
        out_dir = os.path.join(output_root, rel_dir)
        os.makedirs(out_dir, exist_ok=True)

        try:
            with Image.open(src_path) as im:
                im_w, im_h = im.size

                idx = 0
                for det in dets:
                    cat = str(det.get("category", ""))
                    conf = float(det.get("conf", det.get("confidence", 0.0)))
                    if cat != str(target_category):
                        skipped_othercat += 1
                        continue
                    if conf < confidence_threshold:
                        skipped_lowconf += 1
                        continue

                    bbox = det.get("bbox") or det.get("box") or det.get("rect")
                    if not bbox or len(bbox) != 4:
                        continue

                    left, top, right, bottom = _bbox_to_pixels(bbox, im_w, im_h)
                    width_px = max(0, right - left)
                    height_px = max(0, bottom - top)

                    # Size filter
                    if width_px < min_width or height_px < min_height:
                        skipped_small += 1
                        continue

                    crop = im.crop((left, top, right, bottom))

                    base = os.path.splitext(os.path.basename(rel_path))[0]
                    ext = os.path.splitext(src_path)[1].lower() or ".jpg"
                    out_name = f"{base}_det{idx}_x{left}_y{top}_w{width_px}_h{height_px}{ext}"
                    out_path = os.path.join(out_dir, out_name)
                    # Ensure parent exists (should already due to os.makedirs above)
                    crop.save(out_path)
                    crop_count += 1
                    idx += 1
        except Exception as e:
            # If an image fails to open or process, continue gracefully
            print(f"Warning: failed processing {src_path}: {e}")

    print(f"Saved crops: {crop_count}")
    print(f"Skipped (too small): {skipped_small}")
    print(f"Skipped (low confidence): {skipped_lowconf}")
    print(f"Skipped (other category): {skipped_othercat}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop images using MegaDetector JSON output, with min size filtering.")
    parser.add_argument("--json_path", required=True, help="Path to MegaDetector output JSON file")
    parser.add_argument("--input_root", required=True, help="Root directory containing the source images")
    parser.add_argument("--output_root", required=True, help="Root directory to save cropped images")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Minimum confidence threshold for detections")
    parser.add_argument("--target_category", default="1", help="Target category to crop (default: '1')")
    parser.add_argument("--min_width", type=int, default=500, help="Minimum crop width in pixels (default: 500)")
    parser.add_argument("--min_height", type=int, default=500, help="Minimum crop height in pixels (default: 500)")

    args = parser.parse_args()

    crop_images_from_megadetector(
        json_path=args.json_path,
        input_root=args.input_root,
        output_root=args.output_root,
        confidence_threshold=args.confidence_threshold,
        target_category=args.target_category,
        min_width=args.min_width,
        min_height=args.min_height
    )
