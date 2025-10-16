# pip_width_refactored.py
from __future__ import annotations
import os
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

# ========= 設定（必要に応じて編集） =========
IMG_PATH = 'images/sample8.jpg'
TEN_YEN_DIAMETER_CM = 2.3
MAX_WIDTH = 600

# 探索関連
ROI_SIZE = 70                  # PIP周辺の切り出し半径(px)
BAND_PX  = 10                  # PIPを中心とした指ストリップ半幅(px)
CANNY_LO, CANNY_HI = 50, 150   # Canny閾値

DEBUG_DIR = 'output'
os.makedirs(DEBUG_DIR, exist_ok=True)


# ========= ユーティリティ =========
def safe_pt(p: np.ndarray | Tuple[float, float]) -> Tuple[int, int]:
    """(x, y) を四捨五入して int にする"""
    return int(round(float(p[0]))), int(round(float(p[1])))


def save(path: str, img: np.ndarray) -> None:
    """デバッグ出力ヘルパー（フォルダがなければ作る）"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


# ========= 画像IO・前処理 =========
def load_and_resize(path: str, max_width: int) -> np.ndarray:
    """画像を読み込み、max_width を超える場合のみ縮小"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    if w > max_width:
        s = max_width / w
        img = cv2.resize(img, None, fx=s, fy=s)
    return img


# ========= スケール（10円の円検出） =========
def detect_coin_scale(gray: np.ndarray, ten_yen_diam_cm: float) -> Tuple[Optional[float], np.ndarray]:
    """HoughCirclesで10円を検出し、px/cmを返す。描画済み画像も返す"""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=50, param1=100, param2=45,
        minRadius=20, maxRadius=150
    )
    px_per_cm: Optional[float] = None
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x_c, y_c, r = circles[0, 0]
        cv2.circle(vis, (x_c, y_c), r, (0, 255, 0), 2)
        px_per_cm = (2 * r) / ten_yen_diam_cm
        print(f'[Scale] px/cm = {px_per_cm:.2f}')
    else:
        print('[WARN] 10円未検出（cm換算はスキップ）')
    return px_per_cm, vis


# ========= MediaPipe Hands =========
def detect_hand_landmarks(img_bgr: np.ndarray) -> Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList]:
    """手のランドマークを1手だけ検出して返す"""
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        res = hands.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        return None
    return res.multi_hand_landmarks[0]


def draw_landmarks(img_bgr: np.ndarray, hand_lms) -> np.ndarray:
    """ランドマークを可視化"""
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    vis = img_bgr.copy()
    mp_drawing.draw_landmarks(vis, hand_lms, mp_hands.HAND_CONNECTIONS)
    return vis


# ========= 軸計算（PIPの横幅方向） =========
def compute_pip_axes(hand_lms, w: int, h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    薬指PIPの位置Pと、指の長手方向v_unit、横方向perpを返す
    v_unitは MCP→PIP と PIP→DIP の合成で安定化
    """
    mp_hands = mp.solutions.hands
    p_mcp = np.array([hand_lms.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * w,
                      hand_lms.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * h], dtype=np.float32)
    p_pip = np.array([hand_lms.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * w,
                      hand_lms.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * h], dtype=np.float32)
    p_dip = np.array([hand_lms.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * w,
                      hand_lms.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * h], dtype=np.float32)

    v1 = p_pip - p_mcp
    v2 = p_dip - p_pip
    v = v1 + v2
    if np.linalg.norm(v) < 1e-3:
        v = v1
    v_unit = v / (np.linalg.norm(v) + 1e-6)
    perp = np.array([-v_unit[1], v_unit[0]], dtype=np.float32)
    return p_pip, v_unit, perp

# ========= ROI/ストリップ/エッジ =========
def crop_roi(gray: np.ndarray, center: np.ndarray, radius: int):
    h, w = gray.shape[:2]
    x1 = int(max(center[0] - radius, 0))
    x2 = int(min(center[0] + radius, w - 1))
    y1 = int(max(center[1] - radius, 0))
    y2 = int(min(center[1] + radius, h - 1))
    roi_gray = gray[y1:y2, x1:x2]
    return roi_gray, (x1, y1, x2, y2)

# ===== 指ストリップ =====
def make_strip_only_mask(roi_shape, roi_xyxy, pip_xy, v_unit, band_px):
    H, W = roi_shape
    x1, y1, x2, y2 = roi_xyxy
    yy, xx = np.mgrid[0:H, 0:W]
    XX = xx + x1
    YY = yy + y1
    P = np.stack([XX, YY], axis=-1).astype(np.float32)
    proj_long = ((P - pip_xy) @ v_unit).astype(np.float32)  # 長手方向距離
    strip = (np.abs(proj_long) <= band_px).astype(np.uint8) * 255
    return strip

# ===== エッジ抽出 =====
def edges_in_strip(roi_gray: np.ndarray, strip_mask: np.ndarray):
    blur = cv2.GaussianBlur(roi_gray, (3,3), 0)
    edges = cv2.Canny(blur, CANNY_LO, CANNY_HI)
    edges = cv2.bitwise_and(edges, strip_mask)
    return edges

# ========= 端探索・可視化 =========
def first_edge(center_xy, dir_unit, x1, y1, edge_img, max_len):
    H, W = edge_img.shape
    for t in range(1, max_len + 1):
        p = center_xy + dir_unit * t
        x, y = int(round(p[0])) - x1, int(round(p[1])) - y1
        if x < 0 or x >= W or y < 0 or y >= H:
            break
        if edge_img[y, x] != 0:
            return np.array([x + x1, y + y1], dtype=np.float32)
    return None

# ===== 可視化（探索矢印）=====
def visualize_rays(edge_img, center_xy, v_unit, perp, x1, y1, max_len,
                   offsets=(-6, -3, 0, 3, 6)):
    vis = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)
    for off in offsets:
        s = center_xy + v_unit * off
        p0 = (int(round(s[0] - x1)), int(round(s[1] - y1)))
        p1 = (int(round(s[0] + perp[0] * max_len - x1)),
              int(round(s[1] + perp[1] * max_len - y1)))
        cv2.arrowedLine(vis, p0, p1, (200, 200, 200), 1, tipLength=0.08)
    return vis

# ========= メイン測定フロー =========
def main(img_path: str):
    # 入力
    img = load_and_resize(img_path, MAX_WIDTH)
    save(os.path.join(DEBUG_DIR, '00_input.jpg'), img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    px_per_cm, vis_circle = detect_coin_scale(gray, TEN_YEN_DIAMETER_CM)
    save(os.path.join(DEBUG_DIR, '10_circle.jpg'), vis_circle)

    hand_lms = detect_hand_landmarks(img)
    if hand_lms is None:
        print('[ERROR] 手が検出できません'); return
    vis_lm = draw_landmarks(img, hand_lms)
    save(os.path.join(DEBUG_DIR, '15_landmarks.jpg'), vis_lm)

    h, w = img.shape[:2]
    p_pip, v_unit, perp = compute_pip_axes(hand_lms, w, h)

    # ROI
    roi_gray, (x1, y1, x2, y2) = crop_roi(gray, p_pip, ROI_SIZE)
    save(os.path.join(DEBUG_DIR, '30_roi_gray.png'), roi_gray)

    # 帯
    strip = make_strip_only_mask(roi_gray.shape, (x1,y1,x2,y2), p_pip, v_unit, BAND_PX)
    save(os.path.join(DEBUG_DIR, '31_strip_only.png'), strip)

    # エッジ
    edges = edges_in_strip(roi_gray, strip)
    save(os.path.join(DEBUG_DIR, '40_edges_in_strip.png'), edges)

    # 端探索
    center = p_pip.copy()
    max_len = int(ROI_SIZE * 0.9)
    right_edge = first_edge(center,  perp, x1, y1, edges, max_len)
    left_edge  = first_edge(center, -perp, x1, y1, edges, max_len)

    # 可視化（矢印）
    vis_rays = visualize_rays(edges, center, v_unit, perp, x1, y1, max_len)
    save(os.path.join(DEBUG_DIR, '50_rays.png'), vis_rays)

    # 出力
    out = img.copy()
    cv2.circle(out, safe_pt(center), 3, (255, 0, 0), -1)
    label = "Edge not found"
    if right_edge is not None and left_edge is not None:
        cv2.line(out, safe_pt(left_edge), safe_pt(right_edge), (0, 0, 255), 2)
        width_px = float(np.linalg.norm(right_edge - left_edge))
        if px_per_cm:
            width_cm = width_px / px_per_cm
            label = f"PIP width: {width_cm:.2f} cm"
            print(f"[PIP] width_px={width_px:.1f} -> {width_cm:.2f} cm")
        else:
            label = f"PIP width: {width_px:.1f} px"
    cv2.putText(out, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    save(os.path.join(DEBUG_DIR, '60_result.jpg'), out)

    cv2.imshow('Result', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(IMG_PATH)
