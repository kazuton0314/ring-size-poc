from __future__ import annotations
import argparse, csv, glob, os
from typing import Dict, Optional
import cv2
import numpy as np

from src.measure_mediapipe import (
    TEN_YEN_DIAMETER_CM, MAX_WIDTH,
    ROI_SIZE, BAND_PX, CANNY_LO, CANNY_HI,
    load_and_resize, detect_coin_scale, detect_hand_landmarks, draw_landmarks,
    compute_pip_axes, crop_roi, make_strip_only_mask, edges_in_strip,
    first_edge, visualize_rays, safe_pt
)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def write_img(path: str, img: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)

def evaluate_one(image_path: str, out_dir: str) -> Dict[str, Optional[float]]:
    """1枚評価：デバッグ出力を out_dir に保存し、結果を辞書で返す"""
    res = {"image": os.path.basename(image_path), "status": "NG",
           "px_per_cm": None, "width_px": None, "width_cm": None, "note": ""}

    try:
        # 入力
        img = load_and_resize(image_path, MAX_WIDTH)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        write_img(os.path.join(out_dir, "00_input.jpg"), img)

        # 10円スケール
        px_per_cm, vis_circle = detect_coin_scale(gray, TEN_YEN_DIAMETER_CM)
        write_img(os.path.join(out_dir, "10_circle.jpg"), vis_circle)
        if px_per_cm is not None:
            res["px_per_cm"] = float(px_per_cm)

        # 手ランドマーク
        hand_lms = detect_hand_landmarks(img)
        if hand_lms is None:
            res["note"] = "hand landmarks not found"
            return res
        vis_lm = draw_landmarks(img, hand_lms)
        write_img(os.path.join(out_dir, "15_landmarks.jpg"), vis_lm)

        h, w = img.shape[:2]
        p_pip, v_unit, perp = compute_pip_axes(hand_lms, w, h)

        # ROI
        roi_gray, (x1, y1, x2, y2) = crop_roi(gray, p_pip, ROI_SIZE)
        write_img(os.path.join(out_dir, "30_roi_gray.png"), roi_gray)

        # 帯
        strip = make_strip_only_mask(roi_gray.shape, (x1, y1, x2, y2), p_pip, v_unit, BAND_PX)
        write_img(os.path.join(out_dir, "31_strip_only.png"), strip)

        # エッジ
        edges = edges_in_strip(roi_gray, strip)
        write_img(os.path.join(out_dir, "40_edges_in_strip.png"), edges)

        # 端探索
        center = p_pip.copy()
        max_len = int(ROI_SIZE * 0.9)
        right_edge = first_edge(center,  perp, x1, y1, edges, max_len)
        left_edge  = first_edge(center, -perp, x1, y1, edges, max_len)

        # 可視化（矢印）
        vis_rays = visualize_rays(edges, center, v_unit, perp, x1, y1, max_len)
        write_img(os.path.join(out_dir, "50_rays.png"), vis_rays)

        # 結果描画
        out = img.copy()
        cv2.circle(out, safe_pt(center), 3, (255, 0, 0), -1)
        if right_edge is not None and left_edge is not None:
            cv2.line(out, safe_pt(left_edge), safe_pt(right_edge), (0, 0, 255), 2)
            width_px = float(np.linalg.norm(right_edge - left_edge))
            res["width_px"] = width_px
            if px_per_cm:
                width_cm = width_px / px_per_cm
                res["width_cm"] = float(width_cm)
                label = f"PIP width: {width_cm:.2f} cm"
            else:
                label = f"PIP width: {width_px:.1f} px"
            res["status"] = "OK"
        else:
            label = "Edge not found"
            res["note"] = "edge not found on one or both sides"
        cv2.putText(out, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        write_img(os.path.join(out_dir, "60_result.jpg"), out)

        return res

    except Exception as e:
        res["note"] = f"{type(e).__name__}: {e}"
        return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", type=str, default="", help='Glob e.g. "images/sample*.jpg"')
    ap.add_argument("--range", nargs=2, type=int, metavar=("START", "END"),
                    help="Evaluate images/sample{START..END}.jpg")
    ap.add_argument("--out", type=str, default="output_batch", help="Output root")
    args = ap.parse_args()

    # 入力リスト
    paths = []
    if args.pattern:
        paths += glob.glob(args.pattern)
    if args.range:
        s, e = args.range
        paths += [f"images/sample{i}.jpg" for i in range(s, e+1)]
    paths = sorted({p for p in paths if os.path.exists(p)})
    if not paths:
        print("[ERROR] no inputs. Use --pattern or --range"); return

    # 出力先/CSV
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "summary.csv")
    fields = ["image","status","px_per_cm","width_px","width_cm","note"]
    rows = []

    print(f"[INFO] Evaluating {len(paths)} images...")
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        r = evaluate_one(p, out_dir)
        rows.append(r)
        msg = f" - {r['image']}: {r['status']}"
        if r.get("width_cm") is not None: msg += f" | {r['width_cm']:.2f} cm"
        if r.get("note"): msg += f" | note={r['note']}"
        print(msg)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"[DONE] summary -> {csv_path}")
    print(f"[DONE] per-image outputs -> {args.out}/sampleX/")
if __name__ == "__main__":
    main()
