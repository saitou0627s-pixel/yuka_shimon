r"""床指紋の先行研究（B-ORB + RANSAC方式）を論文記述から再実装したコード。

対象：藤田悟・藤田貴大・内田薫「床指紋：床の模様に基づく位置推定」
      情報処理学会論文誌 58(12), 2023-2033, 2017。
      論文の3.2、3.3.2、3.4.2、3.5、4.1に対応。

【使い方】
1. このファイルを yuka_shimon フォルダに置く（gazou の中ではない）。
2. gazou/touroku.jpg と gazou/satsuei_real.jpg を用意する。
   satsuei_real.jpg がない場合だけ、gazou/satsuei.jpg を使用する。
3. VS Codeのターミナルで実行する：
   .\.venv\Scripts\python.exe .\step3_senko_saigen.py
4. 結果は kekka/step3_senko_日時/ に保存される。既存の結果は上書きしない。

画像名を指定する例：
   .\.venv\Scripts\python.exe .\step3_senko_saigen.py --satsuei gazou/test.jpg
各撮影画像をすべてのDB画像と照合し、該当する床を選ぶ例：
   .\.venv\Scripts\python.exe -u .\step3_senko_saigen.py --all-db --db-dir gazou\db --satsuei-dir gazou\satsuei --scale 1.0 --fast-threshold 5
登録DBを事前作成する（初回だけ1回）：
   .\.venv\Scripts\python.exe -u .\step3_senko_saigen.py --build-db --db-dir gazou\db --db-cache gazou\db_cache --scale 1.0 --features 2500 --fast-threshold 5 --trials 20000
撮影画像1枚を4分割し、事前登録DBで地点を識別する：
   .\.venv\Scripts\python.exe -u .\step3_senko_saigen.py --identify --db-cache gazou\db_cache --satsuei gazou\satsuei\shoot_01.png --output kekka\identify_01 --trials 20000
登録画像の2x2タイルだけを4分割し、実際の撮影画像1枚を全DBタイルと照合する例：
   .\.venv\Scripts\python.exe -u .\step3_senko_saigen_tile.py --split-db-one --db-dir gazou\db --satsuei gazou\test.jpg --output kekka\tile_one_test_01 --scale 1.0 --features 2500 --fast-threshold 5 --trials 20000
複数の撮影画像を1枚ずつ全DBタイルと照合する例：
   .\.venv\Scripts\python.exe -u .\step3_senko_saigen_tile.py --split-all-db --db-dir gazou\db --satsuei-dir gazou\satsuei --output kekka\tile_batch_test_01 --scale 1.0 --features 2500 --fast-threshold 5 --trials 20000
登録画像の4分割だけを行い、切り出し結果を確認する例：
   .\.venv\Scripts\python.exe -u .\step3_senko_saigen_tile.py --split-only --db-dir gazou\db --output kekka\tile_split_check
コードだけの動作検証：
   .\.venv\Scripts\python.exe .\step3_senko_saigen_tile.py --self-test

【再現の範囲・限界】
・著者の原コードではなく、公開論文の記述からの再実装。
・平均値差分、多階調版B-ORB、相似三角形RANSAC、事後の妥当性評価を実装。
・ヒストグラム方式、AKAZE方式は今回の対象外。
・--all-dbでは1組用の照合処理を全DB画像に適用し、成功候補中の最高得点を識別結果とする。
・--build-dbで登録画像の4分割とB-ORB抽出を事前に行い、--identifyで再利用する。
・--identifyでは撮影写真1枚を4分割し、撮影タイル4枚×DBタイル40枚の160組を照合する。
・--split-only / --split-db-one / --split-all-dbでは、登録側だけを中央付近の縦横目地で4タイルへ分割する。
・実際の照合で使う撮影画像は分割せず、1枚を全登録タイルと照合する。
・--split-all-dbは実験用で、フォルダ内の複数の撮影画像をそれぞれ1枚のまま照合する。
・タイル分割は先行研究の処理ではなく、指導教員の指摘に基づく追加処理。
・入力画像の事前透視補正は行わない。
・後段のHomographyは照合結果の評価用。入力の事前補正とは別。
・論文に未記載の数値は下記Configのコメントに明記した。
・元論文の画像データと原コードがないため、同じ成功率・数値は保証しない。
・インライア率は幾何モデルに合った対応の割合。照合の正解率ではない。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import time

try:
    import cv2
    import numpy as np
except ImportError:
    print("OpenCVとNumPyが必要です。ターミナルで次を実行してください。")
    print(r".\.venv\Scripts\python.exe -m pip install opencv-python numpy")
    raise SystemExit(2)


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    # 論文4.1、4.2：縦横それぞれ1/4、B-ORB上限10000点。
    image_scale: float = 0.25
    nfeatures: int = 10000
    # 論文3.2：上下左右20画素まで。中心込みの41x41で実装。
    filter_radius: int = 20
    # 論文3.3.2：bundle数は4-10程度。具体的な配置は未記載。
    bundle_rows: int = 3
    bundle_cols: int = 3
    # 以下のORB細部は論文に数値がなく、OpenCV標準値で補完。
    scale_factor: float = 1.2
    nlevels: int = 8
    fast_threshold: int = 20
    edge_threshold: int = 31
    patch_size: int = 31
    harris_block_size: int = 7
    harris_k: float = 0.04
    # 論文3.4.2の80000回は説明中の試行回数例。本実装の上限に採用。
    ransac_trials: int = 80000
    # 論文に具体値なし。座標誤差は縮小後の撮影画像上の画素単位。
    geometry_tolerance_px: float = 5.0
    min_consensus: int = 20
    random_seed: int = 0
    # 論文3.5：近傍探索の半径は未記載。5画素で補完。
    validation_radius_px: float = 5.0
    # 論文3.5、4.1：Hamming <= 100の割合が6%を超えると成功。
    validation_hamming: int = 100
    validation_threshold: float = 0.06
    # 見やすさのための描画上限。処理・集計には全対応を使用する。
    draw_matches: int = 10000


def check_config(c: Config) -> None:
    if not 0 < c.image_scale <= 1:
        raise ValueError("画像縮小率は0より大きく1以下にしてください。")
    if c.nfeatures < 1 or c.nlevels < 1 or c.ransac_trials < 1:
        raise ValueError("特徴点上限・レイヤ数・試行上限は1以上が必要です。")
    if c.min_consensus < 4:
        raise ValueError("事後のHomography推定のため、最低対応数は4以上が必要です。")
    if c.geometry_tolerance_px <= 0 or c.validation_radius_px <= 0:
        raise ValueError("座標誤差と近傍半径は0より大きくしてください。")
    if c.scale_factor <= 1 or c.filter_radius < 0:
        raise ValueError("ピラミッド倍率は1より大きく、フィルタ半径は0以上にしてください。")
    if c.bundle_rows < 1 or c.bundle_cols < 1 or c.fast_threshold < 1:
        raise ValueError("bundleの行数・列数とFAST閾値は1以上が必要です。")


def read_image(path: Path) -> np.ndarray:
    """日本語を含むWindowsのパスにも対応する画像読み込み。"""
    if not path.is_file():
        raise FileNotFoundError(f"画像がありません：{path}")
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"画像として読み込めません：{path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"画像を保存できません：{path}")
    encoded.tofile(str(path))


TILE_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")


def detect_center_grout(gray: np.ndarray, direction: str) -> tuple[int, float]:
    """画像中央付近を横断する暗い目地の座標を1本検出する。

    目地は周囲のタイル面より暗く、画像の広い範囲を横断するという前提で、
    行・列ごとの平均輝度を局所平均との差で評価する。探索範囲は中央30%から
    70%までに限定し、外周の目地や画像端を選ばないようにする。
    戻り値は座標と、周囲に対する暗さ（大きいほど目地らしい）である。
    """
    if gray.ndim != 2:
        raise ValueError("目地検出にはグレースケール画像が必要です。")
    if direction == "vertical":
        profile = gray.astype(np.float32).mean(axis=0)
    elif direction == "horizontal":
        profile = gray.astype(np.float32).mean(axis=1)
    else:
        raise ValueError("directionはverticalまたはhorizontalを指定してください。")
    length = len(profile)
    if length < 80:
        raise ValueError("4分割するには画像が小さすぎます。")
    fine = cv2.GaussianBlur(profile.reshape(1, -1), (9, 1), 0).ravel()
    coarse = cv2.GaussianBlur(profile.reshape(1, -1), (61, 1), 0).ravel()
    darkness = coarse - fine
    start, stop = int(length * 0.30), int(length * 0.70)
    coordinate = start + int(np.argmax(darkness[start:stop]))
    confidence = float(darkness[coordinate])
    # 暗い線が確認できない場合は、安全に画像中央へ戻す。
    if confidence < 0.5:
        coordinate = length // 2
    return coordinate, confidence


def split_four_tiles(image: np.ndarray, margin: int = 4,
                     mode: str = "auto") -> tuple[list[np.ndarray], dict]:
    """2x2の床画像を左上・右上・左下・右下の4枚に分割する。

    marginは中央目地の両側から除外する画素数。mode="auto"は目地を
    自動検出し、mode="center"は画像中心を分割位置として用いる。
    """
    if margin < 0:
        raise ValueError("分割余白は0以上にしてください。")
    height, width = image.shape[:2]
    if mode == "auto":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        split_x, confidence_x = detect_center_grout(gray, "vertical")
        split_y, confidence_y = detect_center_grout(gray, "horizontal")
    elif mode == "center":
        split_x, split_y = width // 2, height // 2
        confidence_x = confidence_y = 0.0
    else:
        raise ValueError("分割方式はautoまたはcenterを指定してください。")
    x1, x2 = split_x - margin, split_x + margin + 1
    y1, y2 = split_y - margin, split_y + margin + 1
    if min(x1, width - x2, y1, height - y2) < 64:
        raise ValueError(
            f"分割位置が画像端に近すぎます：x={split_x}, y={split_y}, "
            f"画像={width}x{height}")
    boxes = ((0, 0, x1, y1), (x2, 0, width, y1),
             (0, y2, x1, height), (x2, y2, width, height))
    tiles = [image[top:bottom, left:right].copy()
             for left, top, right, bottom in boxes]
    return tiles, {
        "split_x": split_x, "split_y": split_y,
        "margin": margin, "mode": mode,
        "vertical_confidence": confidence_x,
        "horizontal_confidence": confidence_y,
        "source_width": width, "source_height": height,
        "boxes": [list(box) for box in boxes],
    }


def draw_split_preview(image: np.ndarray, split_info: dict) -> np.ndarray:
    """検出した縦横目地と、除外する幅を確認する画像を作る。"""
    preview = image.copy()
    x, y = split_info["split_x"], split_info["split_y"]
    margin = split_info["margin"]
    cv2.line(preview, (x, 0), (x, image.shape[0] - 1), (0, 0, 255), 2)
    cv2.line(preview, (0, y), (image.shape[1] - 1, y), (0, 0, 255), 2)
    if margin:
        cv2.line(preview, (x - margin, 0), (x - margin, image.shape[0] - 1),
                 (0, 255, 255), 1)
        cv2.line(preview, (x + margin, 0), (x + margin, image.shape[0] - 1),
                 (0, 255, 255), 1)
        cv2.line(preview, (0, y - margin), (image.shape[1] - 1, y - margin),
                 (0, 255, 255), 1)
        cv2.line(preview, (0, y + margin), (image.shape[1] - 1, y + margin),
                 (0, 255, 255), 1)
    return preview


def mean_difference_filter(gray: np.ndarray, radius: int = 20) -> np.ndarray:
    """論文3.2の多階調版。多階調化の式は未記載なので差分+128を採用。"""
    values = gray.astype(np.float32)
    local_mean = cv2.blur(values, (2 * radius + 1, 2 * radius + 1),
                          borderType=cv2.BORDER_REFLECT_101)
    return np.clip(values - local_mean + 128.0, 0, 255).astype(np.uint8)


def distribute_quotas(total: int, weights: np.ndarray) -> np.ndarray:
    """重みに比例した整数配分。端数は小数部分が大きい順に配る。"""
    raw = total * weights.astype(np.float64) / weights.sum()
    quotas = np.floor(raw).astype(np.int64)
    remaining = total - int(quotas.sum())
    order = np.argsort(-(raw - quotas), kind="stable")
    quotas[order[:remaining]] += 1
    return quotas


def level_quotas(c: Config) -> list[int]:
    """OpenCV ORBと同じ比率で、各ピラミッドレイヤに通常上限を配分。"""
    factor = 1.0 / c.scale_factor
    desired = c.nfeatures * (1 - factor) / (1 - factor ** c.nlevels)
    quotas = []
    for _ in range(c.nlevels - 1):
        quotas.append(int(math.floor(desired + 0.5)))
        desired *= factor
    # 非常に小さいnfeaturesでも合計が上限を超えないよう補完する。
    if sum(quotas) > c.nfeatures:
        return distribute_quotas(c.nfeatures, factor ** np.arange(c.nlevels)).tolist()
    return quotas + [c.nfeatures - sum(quotas)]


def keypoint_angle(layer: np.ndarray, x: int, y: int, radius: int) -> float:
    """ORBのIntensity Centroid方向。円形パッチの離散化は本実装の補完。"""
    patch = layer[y - radius:y + radius + 1, x - radius:x + radius + 1]
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    values = patch.astype(np.float32) * (xx * xx + yy * yy <= radius * radius)
    return math.degrees(math.atan2(float((values * yy).sum()),
                                  float((values * xx).sum()))) % 360.0


def balanced_orb(filtered: np.ndarray, c: Config):
    """論文3.3.2：2倍上限 -> 面積比例bundle選択 -> 全体Harris選択。

    bundle内の順位基準は具体的に未記載のため、ORBの予備選択に合わせ
    FAST応答値を使う。最後のHarrisはcv2.cornerHarrisで計算する。
    OpenCV ORB内部とビット単位で同一という保証はない。
    """
    height, width = filtered.shape
    normal_limits = level_quotas(c)
    pyramid, records, stats = [], [], []
    fast = cv2.FastFeatureDetector_create(c.fast_threshold, True,
                                         cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    for level, normal_limit in enumerate(normal_limits):
        scale = c.scale_factor ** level
        size = (max(1, round(width / scale)), max(1, round(height / scale)))
        layer = filtered if level == 0 else cv2.resize(
            pyramid[-1], size, interpolation=cv2.INTER_LINEAR)
        pyramid.append(layer)
        h, w = layer.shape
        x_edges = np.linspace(0, w, c.bundle_cols + 1, dtype=int)
        y_edges = np.linspace(0, h, c.bundle_rows + 1, dtype=int)
        areas = np.outer(np.diff(y_edges), np.diff(x_edges)).ravel()
        quotas = distribute_quotas(2 * normal_limit, areas)
        groups = [[] for _ in quotas]
        candidates = fast.detect(layer, None) if normal_limit else []
        candidate_count = 0
        for kp in candidates:
            x, y = kp.pt
            if not (c.edge_threshold <= x < w - c.edge_threshold
                    and c.edge_threshold <= y < h - c.edge_threshold):
                continue
            col = int(np.searchsorted(x_edges, x, side="right") - 1)
            row = int(np.searchsorted(y_edges, y, side="right") - 1)
            groups[row * c.bundle_cols + col].append(kp)
            candidate_count += 1
        selected = []
        bundle_counts = []
        for group, quota in zip(groups, quotas):
            group.sort(key=lambda kp: kp.response, reverse=True)
            selected.extend(group[:int(quota)])
            bundle_counts.append(min(len(group), int(quota)))
        if selected:
            harris = cv2.cornerHarris(layer, c.harris_block_size, 3, c.harris_k)
            for kp in selected:
                x, y = map(lambda v: int(round(v)), kp.pt)
                records.append((float(harris[y, x]), level, kp))
        stats.append({"level": level, "normal_limit": normal_limit,
                      "doubled_limit": 2 * normal_limit,
                      "fast_candidates": candidate_count,
                      "bundle_selected": len(selected),
                      "bundle_quotas": quotas.tolist(),
                      "bundle_selected_counts": bundle_counts,
                      "final_selected": 0, "descriptors": 0})

    # 全レイヤを束ねて、通常上限までHarris基準で絞り込む。
    records.sort(key=lambda r: r[0], reverse=True)
    selected_records = records[:c.nfeatures]
    by_level = [[] for _ in pyramid]
    for response, level, kp in selected_records:
        by_level[level].append((response, kp))
        stats[level]["final_selected"] += 1

    keypoints, descriptors = [], []
    orb = cv2.ORB_create(nfeatures=c.nfeatures, nlevels=1,
                         edgeThreshold=c.edge_threshold, patchSize=c.patch_size,
                         WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE)
    for level, level_records in enumerate(by_level):
        if not level_records:
            continue
        layer = pyramid[level]
        local_points = []
        for response, kp in level_records:
            x, y = kp.pt
            angle = keypoint_angle(layer, round(x), round(y), c.patch_size // 2)
            local_points.append(cv2.KeyPoint(x, y, float(c.patch_size),
                                             angle, response, 0, -1))
        computed, desc = orb.compute(layer, local_points)
        if desc is None:
            continue
        scale = c.scale_factor ** level
        for kp, descriptor in zip(computed, desc):
            keypoints.append(cv2.KeyPoint(kp.pt[0] * scale, kp.pt[1] * scale,
                                          kp.size * scale, kp.angle,
                                          kp.response, level, -1))
            descriptors.append(descriptor)
        stats[level]["descriptors"] = len(desc)
    desc = np.asarray(descriptors, dtype=np.uint8).reshape(-1, 32)
    return keypoints, desc, stats


def match_descriptors(desc1: np.ndarray, desc2: np.ndarray):
    """対応候補。論文で詳細未記載のためBFMatcher+相互一致で補完。

    ハミング距離による足切りやratio testは、この段階では追加しない。
    """
    if not len(desc1) or not len(desc2):
        return []
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(desc1, desc2)
    return sorted(matches, key=lambda m: (m.distance, m.queryIdx, m.trainIdx))


def paper_ransac(src: np.ndarray, dst: np.ndarray, c: Config):
    """論文3.4.2のRANSAC。OpenCV findHomographyのRANSACではない。

    1. 異なる2対応から回転・等方拡大縮小・移動の相似変換を作る。
    2. 別の第3対応が変換に合うかを確認する（相似三角形の検査）。
    3. 合った場合のみ全対応を調べ、最低対応数以上なら終了する。
    論文記述に沿って最初の成立モデルで停止。最大インライアを探索
    し続ける方式や、任意のせん断を含む一般Affine推定ではない。
    """
    count = len(src)
    empty_mask = np.zeros(count, dtype=bool)
    details = {"trials": 0, "full_checks": 0, "best_consensus": 0}
    if count < c.min_consensus:
        return None, empty_mask, details
    rng = np.random.default_rng(c.random_seed)
    tol_sq = c.geometry_tolerance_px ** 2
    # まとめて候補を作るが、検査順・停止条件は1回ずつと同じ。
    for start in range(0, c.ransac_trials, 2048):
        batch_size = min(2048, c.ransac_trials - start)
        triples = rng.integers(0, count, size=(batch_size, 3))
        repeated = ((triples[:, 0] == triples[:, 1])
                    | (triples[:, 0] == triples[:, 2])
                    | (triples[:, 1] == triples[:, 2]))
        while repeated.any():
            triples[repeated] = rng.integers(0, count, size=(int(repeated.sum()), 3))
            repeated = ((triples[:, 0] == triples[:, 1])
                        | (triples[:, 0] == triples[:, 2])
                        | (triples[:, 1] == triples[:, 2]))
        p0, p1, p2 = (src[triples[:, i]] for i in range(3))
        q0, q1, q2 = (dst[triples[:, i]] for i in range(3))
        u, v = p1 - p0, q1 - q0
        denominator = (u * u).sum(axis=1)
        safe_denominator = np.maximum(denominator, 1e-12)
        a = (u * v).sum(axis=1) / safe_denominator
        b = (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) / safe_denominator
        tx = q0[:, 0] - a * p0[:, 0] + b * p0[:, 1]
        ty = q0[:, 1] - b * p0[:, 0] - a * p0[:, 1]
        predicted = np.column_stack((a * p2[:, 0] - b * p2[:, 1] + tx,
                                     b * p2[:, 0] + a * p2[:, 1] + ty))
        # 同一点・一直線上の退化三角形だけは数値保護として除外。
        tri_src = u[:, 0] * (p2 - p0)[:, 1] - u[:, 1] * (p2 - p0)[:, 0]
        tri_dst = v[:, 0] * (q2 - q0)[:, 1] - v[:, 1] * (q2 - q0)[:, 0]
        passes = (denominator > 1e-12) & ((v * v).sum(axis=1) > 1e-12)
        passes &= (np.abs(tri_src) > 1e-9) & (np.abs(tri_dst) > 1e-9)
        passes &= ((predicted - q2) ** 2).sum(axis=1) <= tol_sq
        for index in np.flatnonzero(passes):
            matrix = np.array([[a[index], -b[index], tx[index]],
                               [b[index], a[index], ty[index]]], dtype=np.float64)
            transformed = src @ matrix[:, :2].T + matrix[:, 2]
            mask = ((transformed - dst) ** 2).sum(axis=1) <= tol_sq
            consensus = int(mask.sum())
            details["full_checks"] += 1
            details["best_consensus"] = max(details["best_consensus"], consensus)
            if consensus >= c.min_consensus:
                details["trials"] = start + int(index) + 1
                return matrix, mask, details
        details["trials"] = start + batch_size
    return None, empty_mask, details


def homography_from_consensus(src: np.ndarray, dst: np.ndarray):
    """論文3.5：精査後の対応からHomographyを推定（RANSACを使わない）。"""
    if len(src) < 4:
        return None
    if (np.linalg.matrix_rank(src - src.mean(axis=0)) < 2
            or np.linalg.matrix_rank(dst - dst.mean(axis=0)) < 2):
        return None
    try:
        matrix, _ = cv2.findHomography(src, dst, method=0)
    except cv2.error:
        return None
    if matrix is None or not np.isfinite(matrix).all():
        return None
    if np.linalg.matrix_rank(matrix) < 3:
        return None
    return matrix


BIT_COUNTS = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def validate_alignment(points1: np.ndarray, desc1: np.ndarray,
                       points2: np.ndarray, desc2: np.ndarray,
                       homography: np.ndarray | None,
                       query_shape: tuple[int, int], c: Config):
    """論文3.5：座標近傍のうちHamming距離が最小の特徴点を探す。

    インライアだけでなく、登録画像の全特徴点を再評価する。
    分母の厳密な扱いは論文に未記載のため「全登録特徴点数」に固定。
    画角外・近傍なしは不一致として数える。近傍がなかった点を分母から
    除外すると高い割合になり得るため、除外しない。
    """
    result = {"denominator": len(points1), "projected_inside": 0,
              "with_nearby_feature": 0, "hamming_le_100": 0,
              "fraction": 0.0, "success": False, "homography_available": False}
    if homography is None or not len(points1):
        return result
    result["homography_available"] = True
    projected_h = np.column_stack((points1, np.ones(len(points1)))) @ homography.T
    finite = np.isfinite(projected_h).all(axis=1) & (np.abs(projected_h[:, 2]) > 1e-12)
    projected = np.full((len(points1), 2), np.nan, dtype=np.float64)
    projected[finite] = projected_h[finite, :2] / projected_h[finite, 2, None]
    height, width = query_shape
    inside = finite & (projected[:, 0] >= 0) & (projected[:, 0] < width)
    inside &= (projected[:, 1] >= 0) & (projected[:, 1] < height)
    result["projected_inside"] = int(inside.sum())
    radius = c.validation_radius_px
    bins = {}
    for i, point in enumerate(points2):
        cell = (math.floor(point[0] / radius), math.floor(point[1] / radius))
        bins.setdefault(cell, []).append(i)
    good = 0
    for index in np.flatnonzero(inside):
        point = projected[index]
        cell_x, cell_y = math.floor(point[0] / radius), math.floor(point[1] / radius)
        candidates = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                candidates.extend(bins.get((cell_x + dx, cell_y + dy), []))
        if not candidates:
            continue
        ids = np.asarray(candidates, dtype=np.int64)
        ids = ids[((points2[ids] - point) ** 2).sum(axis=1) <= radius ** 2]
        if not len(ids):
            continue
        result["with_nearby_feature"] += 1
        distances = BIT_COUNTS[np.bitwise_xor(desc2[ids], desc1[index])].sum(axis=1)
        good += int(int(distances.min()) <= c.validation_hamming)
    result["hamming_le_100"] = good
    result["fraction"] = good / len(points1)
    result["success"] = result["fraction"] > c.validation_threshold
    return result


def points_from_keypoints(keypoints) -> np.ndarray:
    return np.asarray([kp.pt for kp in keypoints], dtype=np.float64).reshape(-1, 2)


def draw_features(filtered: np.ndarray, keypoints) -> np.ndarray:
    return cv2.drawKeypoints(filtered, keypoints, None, color=(0, 255, 0))


def write_visuals(output: Path, image1: np.ndarray, image2: np.ndarray,
                  filter1: np.ndarray, filter2: np.ndarray,
                  kp1, kp2, matches, inliers: np.ndarray,
                  homography: np.ndarray | None, c: Config) -> None:
    for filename, picture in [("01_touroku.jpg", image1), ("02_satsuei.jpg", image2),
                              ("03_touroku_filter.png", filter1),
                              ("04_satsuei_filter.png", filter2),
                              ("05_touroku_features.jpg", draw_features(filter1, kp1)),
                              ("06_satsuei_features.jpg", draw_features(filter2, kp2))]:
        save_image(output / filename, picture)
    subset = matches[:c.draw_matches]
    common = {"flags": cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS}
    before = cv2.drawMatches(image1, kp1, image2, kp2, subset, None,
                             matchColor=(0, 200, 255), **common)
    # 前後比較は同じ対応集合。後の図だけ別の対応集合に差し替えない。
    retained_subset = [m for m, keep in zip(subset, inliers[:len(subset)]) if keep]
    after = cv2.drawMatches(image1, kp1, image2, kp2, retained_subset, None,
                            matchColor=(0, 255, 0), **common)
    # 全インライアからの抜粋は別ファイルとして区別する。
    all_inliers = [m for m, keep in zip(matches, inliers) if keep]
    overview = cv2.drawMatches(image1, kp1, image2, kp2,
                               all_inliers[:c.draw_matches], None,
                               matchColor=(0, 255, 0), **common)
    save_image(output / "07_ransac_before.jpg", before)
    save_image(output / "08_ransac_after_same_points.jpg", after)
    save_image(output / "09_inlier_examples.jpg", overview)
    if homography is not None:
        h, w = image2.shape[:2]
        warped = cv2.warpPerspective(image1, homography, (w, h))
        mask = cv2.warpPerspective(np.full(image1.shape[:2], 255, np.uint8),
                                    homography, (w, h)) > 0
        overlap = image2.copy()
        blend = cv2.addWeighted(warped, 0.5, image2, 0.5, 0)
        overlap[mask] = blend[mask]
        save_image(output / "10_alignment_after_matching.jpg", overlap)


def run_pair(path1: Path, path2: Path, output: Path, c: Config,
             verbose: bool = True, save_visuals: bool = True) -> dict:
    check_config(c)
    # 新規ディレクトリだけに書き込む。既存出力の上書きを防ぐ。
    output.mkdir(parents=True, exist_ok=False)
    clock = time.perf_counter()
    image1, image2 = read_image(path1), read_image(path2)
    source_shapes = [list(image1.shape[:2]), list(image2.shape[:2])]
    for index, image in enumerate((image1, image2)):
        h, w = image.shape[:2]
        size = (max(1, round(w * c.image_scale)), max(1, round(h * c.image_scale)))
        resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        if index == 0:
            image1 = resized
        else:
            image2 = resized
    timings = {"load_resize_seconds": time.perf_counter() - clock}
    if verbose:
        print(f"登録画像：{path1}", flush=True)
        print(f"撮影画像：{path2}", flush=True)
        print("平均値差分フィルタとB-ORBを実行しています…", flush=True)
    clock = time.perf_counter()
    filter1 = mean_difference_filter(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), c.filter_radius)
    filter2 = mean_difference_filter(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), c.filter_radius)
    kp1, desc1, stats1 = balanced_orb(filter1, c)
    kp2, desc2, stats2 = balanced_orb(filter2, c)
    timings["filter_extract_seconds"] = time.perf_counter() - clock
    clock = time.perf_counter()
    matches = match_descriptors(desc1, desc2)
    points1, points2 = points_from_keypoints(kp1), points_from_keypoints(kp2)
    src = np.asarray([kp1[m.queryIdx].pt for m in matches], np.float64).reshape(-1, 2)
    dst = np.asarray([kp2[m.trainIdx].pt for m in matches], np.float64).reshape(-1, 2)
    timings["match_seconds"] = time.perf_counter() - clock
    if verbose:
        print(f"特徴点数：登録{len(kp1)} / 撮影{len(kp2)}", flush=True)
        print(f"対応候補{len(matches)}組を論文方式RANSACで精査しています…", flush=True)
    clock = time.perf_counter()
    similarity, inliers, ransac_info = paper_ransac(src, dst, c)
    timings["ransac_seconds"] = time.perf_counter() - clock
    clock = time.perf_counter()
    homography = homography_from_consensus(src[inliers], dst[inliers])
    validation = validate_alignment(points1, desc1, points2, desc2,
                                     homography, image2.shape[:2], c)
    timings["homography_validation_seconds"] = time.perf_counter() - clock
    inlier_count = int(inliers.sum())
    rate = inlier_count / len(matches) if matches else 0.0
    if len(matches) < c.min_consensus:
        reason = "対応候補が最低対応数に届いていません。"
    elif similarity is None:
        reason = "試行上限までに幾何モデルが成立しませんでした。同じ場所でないとは断定できません。"
    elif homography is None:
        reason = "精査後の対応から妥当なHomographyを推定できませんでした。"
    elif not validation["success"]:
        reason = "幾何モデルは成立しましたが、事後の特徴量一致が判定閾値を超えていません。"
    else:
        reason = "幾何モデルと事後の特徴量一致の基準を満たしました。"
    summary = [
        "床指紋：先行研究の論文記述からの再実装（原コードではありません）",
        f"登録画像のB-ORB特徴点数：{len(kp1)}",
        f"撮影画像のB-ORB特徴点数：{len(kp2)}",
        f"RANSAC前の対応点数：{len(matches)}",
        f"インライア数：{inlier_count}",
        f"インライア率：{rate * 100:.1f}%（照合の正解率ではありません）",
        f"RANSAC試行数：{ransac_info['trials']} / 上限{c.ransac_trials}",
        f"RANSAC全対応検査回数：{ransac_info['full_checks']}",
        f"Homography推定：{'成功' if homography is not None else '不可'}",
        f"妥当性評価の分母（全登録特徴点）：{validation['denominator']}",
        f"変換先が撮影画像内の特徴点：{validation['projected_inside']}",
        f"近傍に撮影側の特徴点がある数：{validation['with_nearby_feature']}",
        f"近傍Hamming距離100以下：{validation['hamming_le_100']}",
        f"妥当性評価の割合：{validation['fraction'] * 100:.2f}%",
        f"照合判定：{'成功' if validation['success'] else '不成立'}"
        f"（妥当性評価が{c.validation_threshold * 100:g}%を超える場合に成功）",
        f"判定の理由：{reason}",
        "※判定はアルゴリズムの推定。撮影場所の正解ラベルとの比較は別途必要です。",
        f"保存先：{output}",
    ]
    if save_visuals:
        summary.extend([
            f"07・08は同じ先頭{min(c.draw_matches, len(matches))}対応の前後比較。",
            f"09は全インライアから最大{c.draw_matches}対応を別途抜粋した図。",
        ])
    else:
        summary.append("図の保存：省略（全DB照合モード）")
    report = {"implementation": "paper-based reimplementation, not author source",
              "paper": "Fujita et al., IPSJ Journal 58(12), 2023-2033 (2017)",
              "scope": "B-ORB + similar-triangle RANSAC + post-match validation",
              "opencv_version": cv2.__version__, "numpy_version": np.__version__,
              "python_version": sys.version.split()[0], "config": asdict(c),
              "registered_image": str(path1), "query_image": str(path2),
              "registered_sha256": hashlib.sha256(path1.read_bytes()).hexdigest(),
              "query_sha256": hashlib.sha256(path2.read_bytes()).hexdigest(),
              "source_shapes_hw": source_shapes,
              "resized_shapes_hw": [list(image1.shape[:2]), list(image2.shape[:2])],
              "registered_features": len(kp1), "query_features": len(kp2),
              "candidate_matches": len(matches), "inliers": inlier_count,
              "inlier_fraction": rate, "ransac": ransac_info,
              "decision_reason": reason,
              "similarity_matrix": similarity.tolist() if similarity is not None else None,
              "homography": homography.tolist() if homography is not None else None,
              "validation": validation, "registered_layers": stats1,
              "query_layers": stats2, "timings": timings,
              "visuals_saved": save_visuals,
              "assumptions": [
                  "diff+128 with clipping; neighborhood 41x41, reflect101 borders",
                  "OpenCV ORB default pyramid/FAST settings; no adaptive threshold",
                  "3x3 bundles, per-bundle FAST score, global cornerHarris score",
                  "BFMatcher Hamming crossCheck; no ratio or distance cutoff",
                  "first accepted similarity consensus; 5px tolerance, 20 pairs",
                  "post-consensus homography without a second RANSAC",
                  "validation within 5px; denominator is all registered features",
              ]}
    clock = time.perf_counter()
    if save_visuals:
        write_visuals(output, image1, image2, filter1, filter2,
                      kp1, kp2, matches, inliers, homography, c)
    timings["visual_save_seconds"] = time.perf_counter() - clock
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    (output / "result.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    if verbose:
        print("\n" + "\n".join(summary), flush=True)
    return report


def prepare_cached_image(path: Path, c: Config) -> dict:
    """全DB照合用に、画像読込・フィルタ・B-ORBを1回だけ実行する。"""
    clock = time.perf_counter()
    source = read_image(path)
    source_shape = list(source.shape[:2])
    height, width = source.shape[:2]
    size = (max(1, round(width * c.image_scale)),
            max(1, round(height * c.image_scale)))
    image = cv2.resize(source, size, interpolation=cv2.INTER_AREA)
    filtered = mean_difference_filter(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), c.filter_radius)
    keypoints, descriptors, stats = balanced_orb(filtered, c)
    return {
        "path": path,
        "image": image,
        "filtered": filtered,
        "keypoints": keypoints,
        "descriptors": descriptors,
        "points": points_from_keypoints(keypoints),
        "stats": stats,
        "source_shape": source_shape,
        "seconds": time.perf_counter() - clock,
    }


def compare_cached_images(registered: dict, query: dict, c: Config) -> dict:
    """事前抽出済みの2画像をBFMatcher・RANSAC・事後評価で照合する。"""
    clock = time.perf_counter()
    matches = match_descriptors(
        registered["descriptors"], query["descriptors"])
    points1, points2 = registered["points"], query["points"]
    src = np.asarray([points1[m.queryIdx] for m in matches],
                     np.float64).reshape(-1, 2)
    dst = np.asarray([points2[m.trainIdx] for m in matches],
                     np.float64).reshape(-1, 2)
    match_seconds = time.perf_counter() - clock
    clock = time.perf_counter()
    similarity, inliers, ransac_info = paper_ransac(src, dst, c)
    ransac_seconds = time.perf_counter() - clock
    clock = time.perf_counter()
    homography = homography_from_consensus(src[inliers], dst[inliers])
    query_shape = (tuple(query["image_shape"])
                   if "image_shape" in query else query["image"].shape[:2])
    validation = validate_alignment(
        points1, registered["descriptors"],
        points2, query["descriptors"], homography,
        query_shape, c)
    validation_seconds = time.perf_counter() - clock
    inlier_count = int(inliers.sum())
    return {
        "candidate_matches": len(matches),
        "inliers": inlier_count,
        "inlier_fraction": inlier_count / len(matches) if matches else 0.0,
        "ransac": ransac_info,
        "similarity_matrix": (
            similarity.tolist() if similarity is not None else None),
        "homography": homography.tolist() if homography is not None else None,
        "homography_array": homography,
        "validation": validation,
        "matches": matches,
        "inlier_mask": inliers,
        "timings": {
            "match_seconds": match_seconds,
            "ransac_seconds": ransac_seconds,
            "homography_validation_seconds": validation_seconds,
        },
    }


def numbered_images(folder: Path, prefix: str) -> list[tuple[int, Path]]:
    """db_01 / shoot_01 のような番号付き画像を番号順に返す。"""
    if not folder.is_dir():
        raise FileNotFoundError(f"画像フォルダがありません：{folder}")
    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    found: list[tuple[int, Path]] = []
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        if not path.stem.lower().startswith(prefix.lower()):
            continue
        match = re.search(r"(\d+)$", path.stem)
        if match:
            found.append((int(match.group(1)), path.resolve()))
    found.sort(key=lambda item: (item[0], item[1].name.lower()))
    if not found:
        raise FileNotFoundError(
            f"{folder} に {prefix}01.png のような画像がありません。")
    numbers = [number for number, _ in found]
    if len(numbers) != len(set(numbers)):
        raise ValueError(f"{folder} 内で同じ番号の画像が重複しています。")
    return found


def split_numbered_dataset(source_dir: Path, destination_dir: Path,
                           source_prefix: str, tile_prefix: str,
                           margin: int, mode: str) -> list[dict]:
    """番号付き2x2画像を4分割し、平坦なタイル番号で保存する。

    db_01.pngはdb_011.png～db_014.pngとなる。末尾1～4は順に
    左上・右上・左下・右下を表す。元画像番号と分割情報はmanifest.csv
    に保存する。
    """
    sources = numbered_images(source_dir, source_prefix)
    destination_dir.mkdir(parents=True, exist_ok=False)
    preview_dir = destination_dir / "split_previews"
    preview_dir.mkdir()
    rows: list[dict] = []
    for source_number, source_path in sources:
        image = read_image(source_path)
        tiles, split_info = split_four_tiles(image, margin=margin, mode=mode)
        preview_path = preview_dir / f"{source_prefix}{source_number:02d}_split.jpg"
        save_image(preview_path, draw_split_preview(image, split_info))
        for tile_index, (tile_name, tile) in enumerate(zip(TILE_NAMES, tiles), start=1):
            flat_id = source_number * 10 + tile_index
            filename = f"{tile_prefix}{flat_id:03d}.png"
            save_image(destination_dir / filename, tile)
            left, top, right, bottom = split_info["boxes"][tile_index - 1]
            rows.append({
                "flat_id": flat_id,
                "source_id": source_number,
                "tile_index": tile_index,
                "tile_name": tile_name,
                "source_file": str(source_path),
                "tile_file": filename,
                "split_x": split_info["split_x"],
                "split_y": split_info["split_y"],
                "split_margin": margin,
                "vertical_confidence": f"{split_info['vertical_confidence']:.4f}",
                "horizontal_confidence": f"{split_info['horizontal_confidence']:.4f}",
                "crop_left": left, "crop_top": top,
                "crop_right": right, "crop_bottom": bottom,
                "tile_width": tile.shape[1], "tile_height": tile.shape[0],
            })
    write_csv(destination_dir / "manifest.csv", list(rows[0].keys()), rows)
    return rows


def prepare_db_tiles(db_dir: Path, output: Path, margin: int,
                     mode: str) -> tuple[Path, list[dict]]:
    """2x2の登録画像だけを4分割し、DBタイルとして保存する。"""
    # 入力不足で空の結果フォルダだけが残らないよう、作成前に検査する。
    numbered_images(db_dir, "db_")
    output.mkdir(parents=True, exist_ok=False)
    tile_root = output / "split_images"
    tile_root.mkdir()
    db_tiles = tile_root / "db_tiles"
    print("登録画像を4分割しています…", flush=True)
    db_rows = split_numbered_dataset(
        db_dir, db_tiles, "db_", "db_", margin, mode)
    summary = [
        "2x2登録画像のタイル分割結果",
        f"登録元画像：{len(db_rows) // 4}枚 → 登録タイル：{len(db_rows)}枚",
        "撮影画像：分割しない（実際に撮影した1画像をそのまま照合）",
        f"分割方式：{mode}",
        f"中央目地から除外した幅：左右・上下それぞれ{margin}画素",
        f"登録タイル保存先：{db_tiles.resolve()}",
        "split_previews内の赤線が検出位置、黄線が除外範囲です。",
    ]
    (output / "split_summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8")
    print("\n" + "\n".join(summary), flush=True)
    return db_tiles, db_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_cached_tile_database(db_dir: Path,
                             queries: list[tuple[int, Path]],
                             output: Path, c: Config,
                             save_pair_visuals: bool = False) -> dict:
    """撮影画像を全登録タイルと照合する。

    登録側は2x2画像から切り出したタイル。撮影側は分割せず、
    実際に撮影した1画像をそのまま使う。B-ORB特徴は各画像で
    1回だけ抽出し、BFMatcherとRANSACによる各組の照合に再利用する。
    """
    check_config(c)
    databases = numbered_images(db_dir, "db_")
    if not queries:
        raise FileNotFoundError("撮影画像がありません。")
    for _, query_path in queries:
        if not query_path.is_file():
            raise FileNotFoundError(f"撮影画像がありません：{query_path}")
    output.mkdir(parents=True, exist_ok=False)
    visual_root = output / "pair_visuals"
    if save_pair_visuals:
        visual_root.mkdir()
    prepared: dict[Path, dict] = {}
    all_images = [path for _, path in databases] + [path for _, path in queries]
    print(f"登録タイル{len(databases)}枚と撮影画像{len(queries)}枚から"
          "B-ORB特徴を抽出しています…", flush=True)
    extraction_started = time.perf_counter()
    for index, path in enumerate(all_images, start=1):
        print(f"  特徴抽出 [{index:02d}/{len(all_images):02d}] {path.name}", flush=True)
        prepared[path] = prepare_cached_image(path, c)
    extraction_seconds = time.perf_counter() - extraction_started

    total = len(databases) * len(queries)
    pair_rows: list[dict] = []
    score_lookup: dict[tuple[int, int], float] = {}
    started = time.perf_counter()
    for query_position, (query_number, query_path) in enumerate(queries):
        for db_position, (db_number, db_path) in enumerate(databases):
            count = query_position * len(databases) + db_position + 1
            pair_name = f"shoot_{query_number:03d}_vs_db_{db_number:03d}"
            if total <= 100 or count == 1 or count == total or count % 20 == 0:
                print(f"[{count:04d}/{total:04d}] {pair_name}", flush=True)
            pair_started = time.perf_counter()
            report = compare_cached_images(
                prepared[db_path], prepared[query_path], c)
            validation = report["validation"]
            score_lookup[(query_number, db_number)] = validation["fraction"]
            predicted_same = bool(validation["success"])
            pair_rows.append({
                "shoot_id": f"shoot_{query_number:03d}",
                "shoot_file": query_path.name,
                "db_id": f"db_{db_number:03d}",
                "predicted_same": int(predicted_same),
                "final_match_percent": f"{validation['fraction'] * 100:.4f}",
                "candidate_matches": report["candidate_matches"],
                "inliers": report["inliers"],
                "inlier_percent": f"{report['inlier_fraction'] * 100:.4f}",
                "homography_success": int(report["homography"] is not None),
                "ransac_trials": report["ransac"]["trials"],
                "seconds": f"{time.perf_counter() - pair_started:.3f}",
            })
            if save_pair_visuals:
                pair_output = visual_root / pair_name
                pair_output.mkdir()
                write_visuals(
                    pair_output,
                    prepared[db_path]["image"], prepared[query_path]["image"],
                    prepared[db_path]["filtered"], prepared[query_path]["filtered"],
                    prepared[db_path]["keypoints"], prepared[query_path]["keypoints"],
                    report["matches"], report["inlier_mask"],
                    report["homography_array"], c)
    matching_seconds = time.perf_counter() - started
    write_csv(output / "all_pair_results.csv", list(pair_rows[0].keys()), pair_rows)

    # 各撮影画像について、全登録タイルの最高得点を記録する。
    best_rows: list[dict] = []
    for query_number, query_path in queries:
        ranked = sorted(
            databases,
            key=lambda item: score_lookup[(query_number, item[0])], reverse=True)
        best_number, _ = ranked[0]
        second_number, _ = ranked[1] if len(ranked) > 1 else ranked[0]
        best_score = score_lookup[(query_number, best_number)]
        second_score = score_lookup[(query_number, second_number)]
        best_rows.append({
            "shoot_id": f"shoot_{query_number:03d}",
            "shoot_file": query_path.name,
            "predicted_db_tile": (
                f"db_{best_number:03d}"
                if best_score > c.validation_threshold else "no_match"),
            "best_score_percent": f"{best_score * 100:.4f}",
            "second_db_tile": f"db_{second_number:03d}",
            "second_score_percent": f"{second_score * 100:.4f}",
        })
    write_csv(output / "tile_identification_results.csv",
              list(best_rows[0].keys()), best_rows)

    with (output / "score_matrix_percent.csv").open(
            "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["shoot/db"] + [f"db_{number:03d}" for number, _ in databases])
        for query_number, _ in queries:
            writer.writerow(
                [f"shoot_{query_number:03d}"]
                + [f"{score_lookup[(query_number, db_number)] * 100:.4f}"
                   for db_number, _ in databases])

    summary = {
        "database_tiles": len(databases),
        "query_images": len(queries),
        "total_pairs": total,
        "feature_extraction_seconds": extraction_seconds,
        "matching_seconds": matching_seconds,
        "elapsed_seconds": extraction_seconds + matching_seconds,
        "feature_cache_used": True,
    }
    (output / "tile_pair_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "床指紋：撮影画像と全DBタイルの照合",
        f"登録タイル数：{len(databases)}",
        f"撮影画像数：{len(queries)}（撮影側は分割しない）",
        f"総照合数：{total}",
        f"特徴抽出時間：{extraction_seconds:.1f}秒（各画像1回のみ）",
        f"照合時間：{matching_seconds:.1f}秒",
        f"合計時間：{summary['elapsed_seconds']:.1f}秒",
        "※BFMatcherとRANSACは各照合で使用。",
    ]
    (output / "tile_pair_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    return summary


def run_all_database(db_dir: Path, query_dir: Path, output: Path, c: Config,
                     save_pair_visuals: bool = False) -> dict:
    """各撮影画像を全DB画像と照合し、該当する床を1件選んで保存する。"""
    databases = numbered_images(db_dir, "db_")
    queries = numbered_images(query_dir, "shoot_")
    output.mkdir(parents=True, exist_ok=False)
    pair_root = output / "pairs"
    pair_root.mkdir()
    total = len(databases) * len(queries)
    pair_rows: list[dict] = []
    reports: dict[tuple[int, int], dict] = {}
    started = time.perf_counter()

    for query_position, (query_number, query_path) in enumerate(queries):
        for db_position, (db_number, db_path) in enumerate(databases):
            count = query_position * len(databases) + db_position + 1
            pair_name = f"shoot_{query_number:02d}_vs_db_{db_number:02d}"
            print(f"[{count:03d}/{total:03d}] {pair_name}", flush=True)
            pair_started = time.perf_counter()
            report = run_pair(db_path, query_path, pair_root / pair_name, c,
                              verbose=False, save_visuals=save_pair_visuals)
            reports[(query_number, db_number)] = report
            validation = report["validation"]
            predicted_same = bool(validation["success"])
            ground_truth_same = query_number == db_number
            pair_rows.append({
                "shoot_id": f"shoot_{query_number:02d}",
                "db_id": f"db_{db_number:02d}",
                "ground_truth_same": int(ground_truth_same),
                "predicted_same": int(predicted_same),
                "correct": int(ground_truth_same == predicted_same),
                "final_match_percent": f"{validation['fraction'] * 100:.4f}",
                "candidate_matches": report["candidate_matches"],
                "inliers": report["inliers"],
                "inlier_percent": f"{report['inlier_fraction'] * 100:.4f}",
                "homography_success": int(report["homography"] is not None),
                "ransac_trials": report["ransac"]["trials"],
                "seconds": f"{time.perf_counter() - pair_started:.3f}",
                "result_folder": str((pair_root / pair_name).resolve()),
            })

    pair_fields = list(pair_rows[0].keys())
    write_csv(output / "all_pair_results.csv", pair_fields, pair_rows)

    # 撮影画像ごとに、判定に成功したDBのうち最終一致割合が最大のものを
    # 「この撮影画像の床」として選ぶ。全DBが不成立なら該当なしとする。
    best_rows: list[dict] = []
    for query_number, query_path in queries:
        ranked = sorted(
            databases,
            key=lambda item: (
                reports[(query_number, item[0])]["validation"]["fraction"],
                reports[(query_number, item[0])]["inliers"],
            ),
            reverse=True,
        )
        passed_databases = [
            item for item in ranked
            if reports[(query_number, item[0])]["validation"]["success"]
        ]
        best_number, best_path = (passed_databases[0] if passed_databases else ranked[0])
        second_number, _ = ranked[1] if len(ranked) > 1 else ranked[0]
        best_report = reports[(query_number, best_number)]
        second_report = reports[(query_number, second_number)]
        best_score = best_report["validation"]["fraction"] * 100
        second_score = second_report["validation"]["fraction"] * 100
        passed = bool(passed_databases)
        predicted = f"db_{best_number:02d}" if passed else "no_match"
        best_rows.append({
            "shoot_id": f"shoot_{query_number:02d}",
            "expected_db": f"db_{query_number:02d}",
            "predicted_db": predicted,
            "correct_identification": int(passed and best_number == query_number),
            "best_score_percent": f"{best_score:.4f}",
            "second_db": f"db_{second_number:02d}",
            "second_score_percent": f"{second_score:.4f}",
            "margin_percent_point": f"{best_score - second_score:.4f}",
        })
    write_csv(output / "identification_results.csv", list(best_rows[0].keys()), best_rows)

    identification_lines = [
        "床指紋：撮影画像ごとの識別結果",
        "（各撮影画像をすべてのDB画像と照合し、成功候補の最高得点を採用）",
        "",
    ]
    for row in best_rows:
        if row["predicted_db"] == "no_match":
            identification_lines.append(
                f"{row['shoot_id']} → 該当する床なし "
                f"（最高得点 {float(row['best_score_percent']):.2f}%）")
        else:
            identification_lines.append(
                f"{row['shoot_id']} → {row['predicted_db']} の床 "
                f"（一致割合 {float(row['best_score_percent']):.2f}%）")
    (output / "identification_results.txt").write_text(
        "\n".join(identification_lines) + "\n", encoding="utf-8")

    # 10×10の最終一致割合を、そのまま表計算で開ける行列にする。
    matrix_path = output / "score_matrix_percent.csv"
    with matrix_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["shoot/db"] + [f"db_{number:02d}" for number, _ in databases])
        for query_number, _ in queries:
            writer.writerow(
                [f"shoot_{query_number:02d}"]
                + [f"{reports[(query_number, db_number)]['validation']['fraction'] * 100:.4f}"
                   for db_number, _ in databases]
            )

    tp = sum(row["ground_truth_same"] == 1 and row["predicted_same"] == 1
             for row in pair_rows)
    fn = sum(row["ground_truth_same"] == 1 and row["predicted_same"] == 0
             for row in pair_rows)
    fp = sum(row["ground_truth_same"] == 0 and row["predicted_same"] == 1
             for row in pair_rows)
    tn = sum(row["ground_truth_same"] == 0 and row["predicted_same"] == 0
             for row in pair_rows)
    identification_correct = sum(row["correct_identification"] for row in best_rows)
    same_total, different_total = tp + fn, fp + tn
    elapsed = time.perf_counter() - started
    summary = {
        "database_images": len(databases),
        "query_images": len(queries),
        "total_pairs": total,
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "pairwise_accuracy": (tp + tn) / total if total else 0.0,
        "same_location_detection_rate": tp / same_total if same_total else 0.0,
        "different_location_false_match_rate": fp / different_total if different_total else 0.0,
        "identification_correct": identification_correct,
        "identification_accuracy": identification_correct / len(queries),
        "elapsed_seconds": elapsed,
        "config": asdict(c),
    }
    (output / "batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_lines = [
        "床指紋：全DB照合結果",
        "",
        *identification_lines[3:],
        "",
        f"DB画像数：{len(databases)}",
        f"撮影画像数：{len(queries)}",
        f"総照合数：{total}",
        f"同地点を成功判定：{tp}/{same_total}",
        f"同地点の見逃し：{fn}",
        f"異地点の誤照合：{fp}/{different_total}",
        f"異地点を正しく拒否：{tn}/{different_total}",
        f"全組合せの正解率：{summary['pairwise_accuracy'] * 100:.2f}%",
        f"撮影画像ごとのDB識別成功：{identification_correct}/{len(queries)}",
        f"DB識別正解率：{summary['identification_accuracy'] * 100:.2f}%",
        f"処理時間：{elapsed:.1f}秒",
        f"保存先：{output.resolve()}",
    ]
    (output / "batch_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(summary_lines), flush=True)
    return summary


def aggregate_tile_locations(pair_csv: Path, db_rows: list[dict],
                             output: Path, c: Config,
                             expected_by_query: dict[int, int] | None = None) -> dict:
    """全登録タイルの得点から、元の登録地点を1件選ぶ。

    1地点は4枚の登録タイルで構成される。1枚の撮影画像に対し、
    その4タイルのうち最も高い一致割合をその地点の得点とする。
    撮影画像自体は分割しない。
    """
    score: dict[tuple[int, int], float] = {}
    query_names: dict[int, str] = {}
    with pair_csv.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            query_id = int(row["shoot_id"].split("_")[-1])
            db_id = int(row["db_id"].split("_")[-1])
            score[(query_id, db_id)] = float(row["final_match_percent"]) / 100.0
            query_names[query_id] = row.get("shoot_file", f"shoot_{query_id:02d}")

    db_groups: dict[int, list[int]] = {}
    for row in db_rows:
        db_groups.setdefault(int(row["source_id"]), []).append(int(row["flat_id"]))
    for values in db_groups.values():
        values.sort()

    matrix_rows: list[list[str]] = []
    result_rows: list[dict] = []
    for query_id in sorted(query_names):
        candidates = []
        for db_location, db_tiles in sorted(db_groups.items()):
            ranked_tiles = sorted(
                ((score[(query_id, db_tile)], db_tile) for db_tile in db_tiles),
                reverse=True)
            best_tile_score, best_tile_id = ranked_tiles[0]
            candidates.append({
                "db_location": db_location,
                "score": best_tile_score,
                "best_tile_id": best_tile_id,
                "successful_tiles": sum(
                    value > c.validation_threshold for value, _ in ranked_tiles),
            })
        candidates.sort(
            key=lambda item: (item["score"], item["successful_tiles"]), reverse=True)
        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else candidates[0]
        passed = best["score"] > c.validation_threshold
        predicted = best["db_location"] if passed else None
        expected = (expected_by_query or {}).get(query_id)
        result_rows.append({
            "shoot_id": f"shoot_{query_id:03d}",
            "shoot_file": query_names[query_id],
            "expected_location": (
                f"db_{expected:02d}" if expected is not None else "not_specified"),
            "predicted_location": (
                f"db_{predicted:02d}" if predicted is not None else "no_match"),
            "matched_db_tile": (
                f"db_{best['best_tile_id']:03d}" if passed else "no_match"),
            "correct": (
                int(predicted == expected) if expected is not None else ""),
            "location_score_percent": f"{best['score'] * 100:.4f}",
            "passed_db_tiles_in_location": best["successful_tiles"],
            "second_location": f"db_{second['db_location']:02d}",
            "second_score_percent": f"{second['score'] * 100:.4f}",
            "margin_percent_point": f"{(best['score'] - second['score']) * 100:.4f}",
        })
        candidate_by_location = {item["db_location"]: item for item in candidates}
        matrix_rows.append(
            [query_names[query_id]]
            + [f"{candidate_by_location[db_location]['score'] * 100:.4f}"
               for db_location in sorted(db_groups)]
        )

    write_csv(output / "tile_location_results.csv",
              list(result_rows[0].keys()), result_rows)
    with (output / "tile_location_score_matrix_percent.csv").open(
            "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["shoot/db"] + [f"db_{number:02d}" for number in sorted(db_groups)])
        writer.writerows(matrix_rows)

    lines = [
        "床指紋：撮影画像ごとの地点識別結果",
        "（撮影画像は分割せず、全DBタイルの中から最高得点を採用）",
        "",
    ]
    for row in result_rows:
        lines.append(
            f"{row['shoot_file']} → {row['predicted_location']} "
            f"（対応タイル {row['matched_db_tile']}、"
            f"一致割合 {float(row['location_score_percent']):.2f}%）")
    evaluated = [row for row in result_rows if row["correct"] != ""]
    correct = sum(int(row["correct"]) for row in evaluated)
    accuracy = correct / len(evaluated) if evaluated else None
    if evaluated:
        lines.extend([
            "",
            f"地点識別成功：{correct}/{len(evaluated)}",
            f"地点識別正解率：{accuracy * 100:.2f}%",
        ])
    lines.append(
        "※地点得点は、その地点に属する4枚のDBタイルの最高一致割合。")
    (output / "tile_location_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    return {"query_images": len(result_rows), "evaluated": len(evaluated),
            "correct": correct, "accuracy": accuracy, "rows": result_rows}


def run_split_all_database(db_dir: Path, query_dir: Path, output: Path,
                           c: Config, margin: int, mode: str,
                           save_pair_visuals: bool = False) -> dict:
    """登録側だけを4分割し、複数の撮影画像を1枚ずつ照合する。"""
    started = time.perf_counter()
    queries = numbered_images(query_dir, "shoot_")
    db_tiles, db_rows = prepare_db_tiles(db_dir, output, margin, mode)
    matching_output = output / "tile_pair_matching"
    raw_summary = run_cached_tile_database(
        db_tiles, queries, matching_output, c,
        save_pair_visuals=save_pair_visuals)
    expected = {query_number: query_number for query_number, _ in queries}
    location = aggregate_tile_locations(
        matching_output / "all_pair_results.csv", db_rows, output, c, expected)
    final = {
        "registered_source_images": len(db_rows) // 4,
        "query_images": len(queries),
        "registered_tiles": len(db_rows),
        "tile_pair_comparisons": raw_summary["total_pairs"],
        "location_identification_correct": location["correct"],
        "location_identification_accuracy": location["accuracy"],
        "elapsed_seconds": time.perf_counter() - started,
        "split_mode": mode,
        "split_margin": margin,
        "config": asdict(c),
    }
    (output / "tile_run_summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    return final


def run_split_one_query(db_dir: Path, query_path: Path, output: Path,
                        c: Config, margin: int, mode: str,
                        save_pair_visuals: bool = False) -> dict:
    """登録側だけを4分割し、実際の撮影画像1枚で地点識別する。"""
    if not query_path.is_file():
        raise FileNotFoundError(f"撮影画像がありません：{query_path}")
    started = time.perf_counter()
    db_tiles, db_rows = prepare_db_tiles(db_dir, output, margin, mode)
    matching_output = output / "tile_pair_matching"
    raw_summary = run_cached_tile_database(
        db_tiles, [(1, query_path)], matching_output, c,
        save_pair_visuals=save_pair_visuals)
    location = aggregate_tile_locations(
        matching_output / "all_pair_results.csv", db_rows, output, c)
    final = {
        "registered_source_images": len(db_rows) // 4,
        "registered_tiles": len(db_rows),
        "query_images": 1,
        "tile_pair_comparisons": raw_summary["total_pairs"],
        "prediction": location["rows"][0],
        "elapsed_seconds": time.perf_counter() - started,
        "split_mode": mode,
        "split_margin": margin,
        "config": asdict(c),
    }
    (output / "tile_run_summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    return final


def save_feature_file(path: Path, prepared: dict) -> None:
    """DBタイルの照合に必要な特徴量を永続保存する。"""
    np.savez_compressed(
        path,
        points=np.asarray(prepared["points"], dtype=np.float64),
        descriptors=np.asarray(prepared["descriptors"], dtype=np.uint8).reshape(-1, 32),
        image_shape=np.asarray(prepared["image"].shape[:2], dtype=np.int32),
        source_shape=np.asarray(prepared["source_shape"], dtype=np.int32),
    )


def build_database_cache(db_dir: Path, cache_dir: Path, c: Config,
                         margin: int, mode: str) -> dict:
    """2x2登録画像の分割とB-ORB抽出を事前に1回だけ行う。"""
    check_config(c)
    sources = numbered_images(db_dir, "db_")
    cache_dir.mkdir(parents=True, exist_ok=False)
    db_tiles_dir = cache_dir / "db_tiles"
    features_dir = cache_dir / "features"
    print("登録画像を4分割しています…", flush=True)
    rows = split_numbered_dataset(
        db_dir, db_tiles_dir, "db_", "db_", margin, mode)
    features_dir.mkdir()

    tile_paths = dict(numbered_images(db_tiles_dir, "db_"))
    started = time.perf_counter()
    tile_records: list[dict] = []
    for index, row in enumerate(rows, start=1):
        flat_id = int(row["flat_id"])
        tile_path = tile_paths[flat_id]
        print(f"  DB特徴抽出 [{index:02d}/{len(rows):02d}] {tile_path.name}",
              flush=True)
        prepared = prepare_cached_image(tile_path, c)
        feature_name = f"db_{flat_id:03d}.npz"
        save_feature_file(features_dir / feature_name, prepared)
        tile_records.append({
            **row,
            "image_file": str(Path("db_tiles") / tile_path.name),
            "feature_file": str(Path("features") / feature_name),
            "feature_count": len(prepared["points"]),
            "tile_sha256": hashlib.sha256(tile_path.read_bytes()).hexdigest(),
        })
    elapsed = time.perf_counter() - started
    metadata = {
        "format": "floor-fingerprint-db-cache-v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_directory": str(db_dir.resolve()),
        "source_images": len(sources),
        "database_tiles": len(tile_records),
        "split_mode": mode,
        "split_margin": margin,
        "config": asdict(c),
        "feature_extraction_seconds": elapsed,
        "tiles": tile_records,
    }
    (cache_dir / "database.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "床指紋：登録DBの事前作成完了",
        f"登録元画像：{len(sources)}枚",
        f"登録タイル：{len(tile_records)}枚",
        f"特徴抽出時間：{elapsed:.1f}秒",
        f"保存先：{cache_dir.resolve()}",
        "※次回以降は --identify でこのDBを読み込みます。",
    ]
    (cache_dir / "database_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    return metadata


def load_database_cache(cache_dir: Path) -> tuple[list[tuple[int, dict]], dict, Config]:
    """事前作成したDB特徴量を読み込む。"""
    metadata_path = cache_dir / "database.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"登録DBがありません：{metadata_path}\n"
            "先に --build-db を実行してください。")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "floor-fingerprint-db-cache-v1":
        raise ValueError("登録DBの形式がこのコードと一致しません。")
    config_values = {
        key: value for key, value in metadata["config"].items()
        if key in Config.__dataclass_fields__
    }
    c = Config(**config_values)
    databases: list[tuple[int, dict]] = []
    for row in sorted(metadata["tiles"], key=lambda item: int(item["flat_id"])):
        feature_path = cache_dir / row["feature_file"]
        image_path = cache_dir / row["image_file"]
        if not feature_path.is_file():
            raise FileNotFoundError(f"DB特徴量がありません：{feature_path}")
        with np.load(feature_path, allow_pickle=False) as saved:
            points = saved["points"].copy()
            descriptors = saved["descriptors"].copy()
            image_shape = tuple(int(value) for value in saved["image_shape"])
            source_shape = [int(value) for value in saved["source_shape"]]
        databases.append((int(row["flat_id"]), {
            "path": image_path,
            "points": points,
            "descriptors": descriptors,
            "image_shape": image_shape,
            "source_shape": source_shape,
        }))
    if not databases:
        raise ValueError("登録DBにタイルがありません。")
    return databases, metadata, c


def prepare_one_query_photo(query_path: Path, output: Path, c: Config,
                            margin: int, mode: str) -> tuple[list[tuple[int, dict]], dict, float]:
    """実際の撮影写真1枚を4タイルに分割し、特徴を抽出する。"""
    image = read_image(query_path)
    tiles, split_info = split_four_tiles(image, margin=margin, mode=mode)
    query_dir = output / "query_tiles"
    query_dir.mkdir()
    save_image(output / "query_split_preview.jpg",
               draw_split_preview(image, split_info))
    started = time.perf_counter()
    prepared_tiles: list[tuple[int, dict]] = []
    rows: list[dict] = []
    for tile_index, (tile_name, tile) in enumerate(zip(TILE_NAMES, tiles), start=1):
        tile_path = query_dir / f"shoot_tile_{tile_index}.png"
        save_image(tile_path, tile)
        print(f"  撮影特徴抽出 [{tile_index}/4] {tile_path.name}", flush=True)
        prepared = prepare_cached_image(tile_path, c)
        prepared_tiles.append((tile_index, prepared))
        left, top, right, bottom = split_info["boxes"][tile_index - 1]
        rows.append({
            "tile_index": tile_index,
            "tile_name": tile_name,
            "tile_file": tile_path.name,
            "feature_count": len(prepared["points"]),
            "crop_left": left, "crop_top": top,
            "crop_right": right, "crop_bottom": bottom,
        })
    elapsed = time.perf_counter() - started
    write_csv(output / "query_tiles.csv", list(rows[0].keys()), rows)
    (output / "query_split.json").write_text(
        json.dumps(split_info, ensure_ascii=False, indent=2), encoding="utf-8")
    return prepared_tiles, split_info, elapsed


def identify_one_photo_from_cache(cache_dir: Path, query_path: Path,
                                  output: Path, runtime_c: Config,
                                  margin: int, mode: str,
                                  expected_db: int | None = None) -> dict:
    """撮影写真1枚を4分割し、事前登録DBから地点を1件選ぶ。"""
    if not query_path.is_file():
        raise FileNotFoundError(f"撮影画像がありません：{query_path}")
    databases, metadata, stored_c = load_database_cache(cache_dir)
    # 特徴抽出設定はDB作成時と同じ値を強制する。
    # RANSACの試行上限と乱数シードだけは実行時に変更できる。
    c = replace(stored_c, ransac_trials=runtime_c.ransac_trials,
                random_seed=runtime_c.random_seed)
    check_config(c)
    output.mkdir(parents=True, exist_ok=False)
    overall_started = time.perf_counter()
    print("撮影画像1枚を4タイルに分割しています…", flush=True)
    queries, split_info, query_feature_seconds = prepare_one_query_photo(
        query_path, output, c, margin, mode)

    total = len(queries) * len(databases)
    print(f"撮影タイル4枚 × DBタイル{len(databases)}枚 "
          f"= {total}組を照合します…", flush=True)
    scores: dict[tuple[int, int], float] = {}
    pair_rows: list[dict] = []
    matching_started = time.perf_counter()
    for query_position, (query_id, query) in enumerate(queries):
        for db_position, (db_id, database) in enumerate(databases):
            count = query_position * len(databases) + db_position + 1
            if total <= 40 or count == 1 or count == total or count % 20 == 0:
                print(f"  照合 [{count:03d}/{total:03d}] "
                      f"shoot_tile_{query_id} vs db_{db_id:03d}", flush=True)
            pair_started = time.perf_counter()
            report = compare_cached_images(database, query, c)
            fraction = float(report["validation"]["fraction"])
            scores[(query_id, db_id)] = fraction
            pair_rows.append({
                "shoot_tile": f"shoot_tile_{query_id}",
                "db_tile": f"db_{db_id:03d}",
                "predicted_same": int(report["validation"]["success"]),
                "final_match_percent": f"{fraction * 100:.4f}",
                "candidate_matches": report["candidate_matches"],
                "inliers": report["inliers"],
                "inlier_percent": f"{report['inlier_fraction'] * 100:.4f}",
                "homography_success": int(report["homography"] is not None),
                "ransac_trials": report["ransac"]["trials"],
                "seconds": f"{time.perf_counter() - pair_started:.3f}",
            })
    matching_seconds = time.perf_counter() - matching_started
    write_csv(output / "all_pair_results.csv", list(pair_rows[0].keys()), pair_rows)

    db_groups: dict[int, list[int]] = {}
    for row in metadata["tiles"]:
        db_groups.setdefault(int(row["source_id"]), []).append(int(row["flat_id"]))
    for tile_ids in db_groups.values():
        tile_ids.sort()

    candidates: list[dict] = []
    query_ids = [query_id for query_id, _ in queries]
    for location, db_tile_ids in sorted(db_groups.items()):
        tile_choices = []
        for query_id in query_ids:
            best_score, best_db_tile = max(
                (scores[(query_id, db_id)], db_id) for db_id in db_tile_ids)
            tile_choices.append({
                "query_tile": query_id,
                "db_tile": best_db_tile,
                "score": best_score,
            })
        location_score = float(np.mean([item["score"] for item in tile_choices]))
        candidates.append({
            "location": location,
            "score": location_score,
            "successful_query_tiles": sum(
                item["score"] > c.validation_threshold for item in tile_choices),
            "tile_choices": tile_choices,
        })
    candidates.sort(
        key=lambda item: (item["score"], item["successful_query_tiles"]),
        reverse=True)
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else candidates[0]
    passed = best["score"] > c.validation_threshold
    predicted = best["location"] if passed else None
    correct = (predicted == expected_db) if expected_db is not None else None

    with (output / "location_score_matrix_percent.csv").open(
            "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        by_location = {item["location"]: item for item in candidates}
        writer.writerow(
            ["shoot"] + [f"db_{location:02d}" for location in sorted(by_location)])
        writer.writerow(
            [query_path.name]
            + [f"{by_location[location]['score'] * 100:.4f}"
               for location in sorted(by_location)])

    total_seconds = time.perf_counter() - overall_started
    result = {
        "query_image": str(query_path.resolve()),
        "database_cache": str(cache_dir.resolve()),
        "database_source_images": int(metadata["source_images"]),
        "database_tiles": len(databases),
        "query_photos": 1,
        "query_tiles": len(queries),
        "total_comparisons": total,
        "predicted_location": (
            f"db_{predicted:02d}" if predicted is not None else "no_match"),
        "expected_location": (
            f"db_{expected_db:02d}" if expected_db is not None else None),
        "correct": correct,
        "location_score": best["score"],
        "second_location": f"db_{second['location']:02d}",
        "second_score": second["score"],
        "margin": best["score"] - second["score"],
        "successful_query_tiles": best["successful_query_tiles"],
        "best_tile_matches": best["tile_choices"],
        "query_split": split_info,
        "query_feature_seconds": query_feature_seconds,
        "matching_seconds": matching_seconds,
        "elapsed_seconds": total_seconds,
        "config": asdict(c),
    }
    (output / "identification_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "床指紋：事前登録DBを使った1撮影画像の地点識別",
        f"撮影画像：{query_path.resolve()}",
        "撮影枚数：1枚（4タイルに分割）",
        f"登録DB：{metadata['source_images']}地点 / {len(databases)}タイル",
        f"総照合数：{total}",
        f"推定地点：{result['predicted_location']}",
        f"地点得点：{best['score'] * 100:.2f}%",
        f"2位：{result['second_location']} ({second['score'] * 100:.2f}%)",
        f"得点差：{result['margin'] * 100:.2f}ポイント",
        f"判定基準を超えた撮影タイル："
        f"{best['successful_query_tiles']}/4",
    ]
    if expected_db is not None:
        lines.append(
            f"正解地点：db_{expected_db:02d} / "
            f"識別：{'正解' if correct else '不正解'}")
    lines.extend([
        f"撮影側の分割・特徴抽出：{query_feature_seconds:.1f}秒",
        f"照合時間：{matching_seconds:.1f}秒",
        f"合計時間：{total_seconds:.1f}秒",
        "※DB側の分割・特徴抽出は事前登録済みのため実行時間に含まれません。",
        "※透視変換は使用していません。BFMatcherとRANSACは各照合で使用します。",
        f"保存先：{output.resolve()}",
    ])
    (output / "identification_result.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    return result


def self_test() -> None:
    """合成データで処理の実装を検証。論文の性能再現を証明するものではない。"""
    print("合成データで自己テストしています…", flush=True)
    constant = np.full((70, 90), 83, dtype=np.uint8)
    assert np.all(mean_difference_filter(constant) == 128)
    assert int(distribute_quotas(123, np.array([10, 20, 30])).sum()) == 123
    split_test = np.full((400, 420, 3), 180, dtype=np.uint8)
    split_test[:, 207:213] = 30
    split_test[194:200, :] = 30
    tiles, split_info = split_four_tiles(split_test, margin=3, mode="auto")
    assert abs(split_info["split_x"] - 210) <= 2
    assert abs(split_info["split_y"] - 197) <= 2
    assert len(tiles) == 4 and all(tile.size for tile in tiles)
    c = Config()
    assert sum(level_quotas(c)) == c.nfeatures
    rng = np.random.default_rng(123)
    src = rng.uniform(0, 600, (700, 2))
    angle = math.radians(23)
    expected = 1.12 * np.array([[math.cos(angle), -math.sin(angle)],
                                [math.sin(angle), math.cos(angle)]])
    dst = rng.uniform(0, 600, (700, 2))
    dst[:35] = src[:35] @ expected.T + [12, -7]
    dst[:35] += rng.normal(0, 0.05, (35, 2))
    test_c = replace(c, geometry_tolerance_px=0.5)
    matrix, mask, _ = paper_ransac(src, dst, test_c)
    assert matrix is not None and mask[:35].sum() == 35
    assert mask[35:].sum() <= 1
    assert np.allclose(matrix[:, :2], expected, atol=0.01)
    # Hammingでは座標が最も近い点でなく、近傍内で記述子が最も近い点を選ぶ。
    p1 = np.array([[10., 10.], [20., 20.], [500., 500.]])
    p2 = np.array([[10., 10.], [11., 10.], [20., 20.]])
    d1 = np.zeros((3, 32), dtype=np.uint8)
    d2 = np.vstack((np.full(32, 255, np.uint8), np.zeros(32, np.uint8),
                     np.full(32, 255, np.uint8)))
    v = validate_alignment(p1, d1, p2, d2, np.eye(3), (100, 100), c)
    assert v["hamming_le_100"] == 1 and v["denominator"] == 3
    assert abs(v["fraction"] - 1 / 3) < 1e-12
    # 論文4.1は「6%を超える」。6%ちょうどは成功扱いにしない。
    yy, xx = np.mgrid[0:10, 0:10]
    boundary_points = np.column_stack((xx.ravel(), yy.ravel())).astype(float) * 20 + 10
    source_desc = np.zeros((100, 32), np.uint8)
    target_desc = np.full((100, 32), 255, np.uint8)
    target_desc[:6] = 0
    boundary = validate_alignment(boundary_points, source_desc, boundary_points,
                                   target_desc, np.eye(3), (220, 220), c)
    assert boundary["fraction"] == 0.06 and not boundary["success"]
    target_desc[6] = 0
    boundary = validate_alignment(boundary_points, source_desc, boundary_points,
                                   target_desc, np.eye(3), (220, 220), c)
    assert boundary["fraction"] == 0.07 and boundary["success"]
    assert homography_from_consensus(np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]]),
                                      np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]])) is None
    # 別ファイルとして読み込み、全パイプラインと実際の画像・JSON保存を検証。
    gray = rng.integers(30, 220, (640, 720), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.7)
    cv2.line(gray, (15, 320), (710, 320), 30, 3)
    cv2.line(gray, (360, 15), (360, 630), 30, 3)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    with tempfile.TemporaryDirectory(prefix="floor_fingerprint_test_") as directory:
        root = Path(directory)
        registered, query = root / "登録.jpg", root / "撮影.jpg"
        save_image(registered, image)
        save_image(query, cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
        smoke_c = replace(c, nfeatures=600, image_scale=1.0, min_consensus=10)
        positive = run_pair(registered, query, root / "positive", smoke_c, verbose=False)
        assert positive["validation"]["success"], positive["validation"]
        assert positive["inliers"] >= 10
        assert (root / "positive" / "10_alignment_after_matching.jpg").is_file()
        saved = json.loads((root / "positive" / "result.json").read_text(encoding="utf-8"))
        assert saved["inliers"] == positive["inliers"]
        # 別の模様では成功判定しないことを確認。
        independent = cv2.GaussianBlur(rng.integers(30, 220, (640, 720),
                                                   dtype=np.uint8), (3, 3), 0.7)
        save_image(query, independent)
        negative = run_pair(registered, query, root / "negative", smoke_c, verbose=False)
        assert not negative["validation"]["success"], negative["validation"]
        # 特徴点がない画像でもゼロ除算・描画エラーなし。
        save_image(registered, constant)
        save_image(query, constant)
        blank = run_pair(registered, query, root / "blank", smoke_c, verbose=False)
        assert blank["candidate_matches"] == 0 and not blank["validation"]["success"]
        assert (root / "blank" / "08_ransac_after_same_points.jpg").is_file()
    print("自己テスト成功：フィルタ、配分、95%外れ値RANSAC、近傍Hamming、")
    print("               同一床模様の回転、別模様、特徴点なし、日本語パス、結果保存。")
    print("※合成データの動作検証であり、実際の床や論文の成功率の再現ではありません。")


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (BASE_DIR / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="床指紋の先行研究：B-ORB + 論文方式RANSAC")
    parser.add_argument("--touroku", default="gazou/touroku.jpg", help="登録画像のパス")
    parser.add_argument("--satsuei", default=None, help="撮影画像のパス")
    parser.add_argument("--all-db", action="store_true",
                        help="各撮影画像を全DB画像と照合して、該当する床を選ぶ")
    parser.add_argument("--build-db", action="store_true",
                        help="登録画像の4分割とB-ORB特徴抽出を事前に1回行う")
    parser.add_argument("--identify", action="store_true",
                        help="撮影画像1枚を4分割し、事前登録DBで地点識別する")
    parser.add_argument("--split-only", action="store_true",
                        help="2x2の登録画像だけを目地で4分割して保存する")
    parser.add_argument("--split-db-one", action="store_true",
                        help="登録画像だけを4分割し、--satsueiの1画像を全DBタイルと照合する")
    parser.add_argument("--split-all-db", action="store_true",
                        help="登録画像だけを4分割し、撮影フォルダの各1画像を全DBタイルと照合する")
    parser.add_argument("--db-dir", default="gazou/db",
                        help="全DB照合または4分割で使うDB画像フォルダ")
    parser.add_argument("--db-cache", default="gazou/db_cache",
                        help="--build-dbの保存先または--identifyで読み込む登録DB")
    parser.add_argument("--satsuei-dir", default="gazou/satsuei",
                        help="全DB照合または4分割で使う撮影画像フォルダ")
    parser.add_argument("--split-mode", choices=("auto", "center"), default="auto",
                        help="auto=中央付近の目地を検出、center=画像中心で分割")
    parser.add_argument("--split-margin", type=int, default=4,
                        help="中央目地の両側から除外する画素数（元画像上）")
    parser.add_argument("--save-pair-visuals", action="store_true",
                        help="--all-dbの全組合せで照合画像も保存する（容量大）")
    parser.add_argument("--output", default=None, help="新規の結果フォルダ（既存は不可）")
    parser.add_argument("--features", type=int, default=Config.nfeatures)
    parser.add_argument("--scale", type=float, default=Config.image_scale)
    parser.add_argument("--fast-threshold", type=int, default=Config.fast_threshold)
    parser.add_argument("--trials", type=int, default=Config.ransac_trials)
    parser.add_argument("--seed", type=int, default=Config.random_seed)
    parser.add_argument("--expected-db", type=int, default=None,
                        help="--identifyの実験用正解地点番号（実運用では省略）")
    parser.add_argument("--self-test", action="store_true", help="合成データで実装を検証")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    c = replace(Config(), nfeatures=args.features, image_scale=args.scale,
                fast_threshold=args.fast_threshold, ransac_trials=args.trials,
                random_seed=args.seed)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    selected_modes = sum((args.all_db, args.build_db, args.identify,
                          args.split_only, args.split_db_one,
                          args.split_all_db))
    if selected_modes > 1:
        raise ValueError("実行モードは1つだけ指定してください。")
    if args.split_margin < 0:
        raise ValueError("--split-marginは0以上にしてください。")
    if args.build_db:
        build_database_cache(
            resolve_path(args.db_dir), resolve_path(args.db_cache), c,
            args.split_margin, args.split_mode)
        return
    if args.identify:
        if args.satsuei is None:
            raise ValueError("--identifyでは --satsuei に撮影画像1枚を指定してください。")
        output = (resolve_path(args.output) if args.output else
                  BASE_DIR / "kekka" / f"step3_identify_{stamp}")
        identify_one_photo_from_cache(
            resolve_path(args.db_cache), resolve_path(args.satsuei),
            output, c, args.split_margin, args.split_mode,
            expected_db=args.expected_db)
        return
    if args.split_only or args.split_db_one or args.split_all_db:
        db_dir = resolve_path(args.db_dir)
        if args.split_db_one:
            prefix = "step3_db_tiles_one"
        elif args.split_all_db:
            prefix = "step3_db_tiles_all"
        else:
            prefix = "step3_db_tile_split"
        output = (resolve_path(args.output) if args.output else
                  BASE_DIR / "kekka" / f"{prefix}_{stamp}")
        if args.split_only:
            prepare_db_tiles(
                db_dir, output, args.split_margin, args.split_mode)
        elif args.split_db_one:
            if args.satsuei is None:
                raise ValueError("--split-db-oneでは --satsuei に撮影画像1枚を指定してください。")
            run_split_one_query(
                db_dir, resolve_path(args.satsuei), output, c,
                args.split_margin, args.split_mode,
                save_pair_visuals=args.save_pair_visuals)
        else:
            query_dir = resolve_path(args.satsuei_dir)
            run_split_all_database(
                db_dir, query_dir, output, c, args.split_margin, args.split_mode,
                save_pair_visuals=args.save_pair_visuals)
        return
    if args.all_db:
        db_dir = resolve_path(args.db_dir)
        query_dir = resolve_path(args.satsuei_dir)
        output = (resolve_path(args.output) if args.output else
                  BASE_DIR / "kekka" / f"step3_all_db_{stamp}")
        run_all_database(db_dir, query_dir, output, c,
                         save_pair_visuals=args.save_pair_visuals)
        return
    registered = resolve_path(args.touroku)
    if args.satsuei is not None:
        query = resolve_path(args.satsuei)
    else:
        query = resolve_path("gazou/satsuei_real.jpg")
        if not query.is_file():
            query = resolve_path("gazou/satsuei.jpg")
    # 入力不足では結果フォルダを作らず、先に分かりやすく伝える。
    for path in (registered, query):
        if not path.is_file():
            raise FileNotFoundError(f"画像がありません：{path}\n"
                                    "画像名を確認するか、--touroku / --satsueiで指定してください。")
    output = resolve_path(args.output) if args.output else BASE_DIR / "kekka" / f"step3_senko_{stamp}"
    run_pair(registered, query, output, c)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, cv2.error) as exc:
        print(f"\n実行できませんでした：{exc}", file=sys.stderr)
        raise SystemExit(1)
