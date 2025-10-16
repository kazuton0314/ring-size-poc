# Ring Size POC – 薬指PIP幅の自動推定（OpenCV + MediaPipe)

本リポジトリは、静止画像に写った左手薬指 + 10円玉から
薬指PIP関節の幅（px と cm換算）を自動測定する Proof of Concept (POC) です。

## 機能概要

| ステップ | 内容 | 使用ライブラリ |
|---|---|---|
| 1 | 10円玉を HoughCircles で検出 → px/cm スケール算出 | OpenCV |
| 2 | MediaPipe Hands で手の21ランドマーク検出 | MediaPipe |
| 3 | PIP関節周辺ROIを切り出し（指の横断線にフォーカス） | OpenCV / NumPy |
| 4 | 指の長軸を推定 → 直交方向(perp)を測定軸として生成 | NumPy |
| 5 | 細い帯(strip) 内のみ Canny エッジを適用 | OpenCV |
| 6 | PIP中心から左右に走査 → 最初のエッジ同士の距離を幅とする | NumPy |
| 7 | 結果を画像に可視化（単体）＋ CSV にまとめる（バッチ） | OpenCV / CSV |

## 実行例（単体での計測）

```
python src/measure_mediapipe.py
```

出力結果は `output/` に保存されます：

- `00_input.jpg` : 入力画像
- `10_circle.jpg` : 10円検出
- `15_landmarks.jpg` : 手のランドマーク可視化
- `30_roi_gray.png` : ROI切り出し
- `31_strip_only.png` : 指の横断帯
- `40_edges_in_strip.png` : エッジ抽出結果
- `50_rays.png` : スキャン方向確認
- `60_result.jpg` : 幅検出の最終可視化

## 複数画像の一括評価（sample1〜7）

```
# 範囲指定
python -m src.scripts.eval_batch --range 1 7 --out output_batch

# パターン指定
python -m src.scripts.eval_batch --pattern "images/sample*.jpg" --out output_batch
```

出力構成：

```
output_batch/
├─ sample1/
│   ├─ 00_input.jpg
│   ├─ ...
│   ├─ 60_result.jpg
├─ sample2/
│   └─ ...
└─ summary.csv  ← 成功/失敗・px/cm幅・備考の一覧
```

## 環境構築

前提：
- OS: Windows
- Python: 3.9〜3.12

セットアップ手順（例：Windows PowerShell）:

```
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## ディレクトリ構成

```
/
├─ src/
│  ├─ measure_mediapipe.py         # メイン処理（単体実行）
│  └─ scripts/
│     └─ eval_batch.py             # バッチ評価スクリプト
├─ images/
│  ├─ sample1.jpg ... sample8.jpg  # テスト用画像
├─ output/                         # 単体実行時のデバッグ出力
├─ output_batch/                   # バッチ評価結果
├─ requirements.txt
├─ README.md
└─ .gitignore
```

## 測定アルゴリズム（概要）

```
[画像読み込み] → [10円スケール取得] → [ランドマーク検出]
→ [PIP中心座標推定] → [ROI切り出し]
→ [指方向ベクトル算出] → [直交方向帯の生成]
→ [Cannyエッジ検出] → [左右方向へ探索]
→ [最初のエッジ同士の距離 = 幅(px)] → [スケール変換でcm]
```

## 現状の精度と制約

- 単色背景・明るい環境では成功率は高いが誤差が生じる(sample1~4)。特に撮影距離・角度の影響、誤差が大きく出る可能性がある
- 影や複雑な背景、10円玉が不鮮明の場合には誤検出が発生することがある(sample5~7)
- MediaPipeのランドマークはAI推定のためズレる可能性あり（デバッグ画像で可視化）
- 幅測定：PIP 近傍の横断帯 + Canny + 直交スキャンで安定。極端な屈曲・遮蔽・強い影ではエッジが途切れる可能性あり。
- 解像度：いまはリサイズ後の座標で処理。将来的に 測定は原解像度で行う構成に見直す余地あり（精度向上）。

## 改善提案（ロードマップ）
- 10円検出の精度向上：HoughCirclesからテンプレートマッチング／AI検出に切り替え
- 角度・距離補正：MediaPipeの3D座標で平面補正や撮影姿勢の影響を低減
- 手ランドマークの信頼性向上：左右識別、パラメータ調整
- Cannyエッジ改善：自動閾値化・多ライン平均で堅牢化
- 測定領域処理：ROIを原画像ベースに変更して高解像度保持
- 信頼度スコア出力：結果の安定性を数値で示す

## 注意事項

- 本リポジトリは実証実験目的（POC）です。
- 医療用途・精密測定用としての保証はありません。
- 画像データは匿名化・マスク処理を推奨します。

## License
© 2025 kazuton. All rights reserved.

This repository is provided for demonstration and evaluation purposes only.
Commercial use, redistribution, or modification without permission is prohibited.

Third-party libraries used in this project:
- OpenCV (BSD-3-Clause License)
- MediaPipe (Apache License 2.0)
- NumPy (BSD License)