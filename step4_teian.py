r"""床指紋の提案手法：目地検出・透視変換・座標近傍照合。

先行研究再現の step3_senko_saigen.py と同じフォルダに置いて使う。
step3 の平均値差分フィルタと B-ORB 実装を再利用し、次を変更する。

  先行研究再現：BFMatcher -> RANSAC -> Homographyによる事後評価
  提案手法　　：目地検出 -> 透視変換 -> タイル分割
                  -> 同一座標近傍だけでHamming距離を比較

注意：本コードの目地検出・座標近傍照合・判定閾値は提案手法であり、
藤田らの先行研究に記載された処理ではない。判定閾値は10地点以上の
同地点・異地点データから決める必要があるため、初期値は暫定値である。

実行例：
  .\.venv\Scripts\python.exe .\step4_teian.py ^
    --touroku gazou\reg_01.jpg --satsuei gazou\query_01.jpg

自動目地検出が難しい画像では、4隅を左上、右上、右下、左下の順で指定：
  --touroku-points "120,100;980,120;960,1010;100,990"
  --satsuei-points "90,130;1000,90;1030,980;110,1020"

合成画像による動作確認：
  .\.venv\Scripts\python.exe .\step4_teian.py --self-test
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time

try:
    import cv2
    import numpy as np
except ImportError:
    print("OpenCVとNumPyが必要です。")
    raise SystemExit(2)

try:
    from step3_senko_saigen import (
        Config as BaseConfig,
        BIT_COUNTS,
        balanced_orb,
        mean_difference_filter,
        points_from_keypoints,
        read_image,
        save_image,
    )
except ImportError as exc:
    print("step3_senko_saigen.pyをstep4_teian.pyと同じフォルダに置いてください。")
    raise SystemExit(2) from exc


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    # 透視変換後の2x2全体の一辺。4タイルは各512pxになる。
    warp_size: int = 1024
    # 目地と外周を特徴抽出から除くため、各タイルの内側を切り出す割合。
    tile_margin_ratio: float = 0.06
    # 目地検出用。大きな画像だけ縮小し、検出した座標は元画像へ戻す。
    detection_max_side: int = 1400
    canny_low: int = 35
    canny_high: int = 110
    hough_threshold: int = 70
    hough_min_length_ratio: float = 0.32
    hough_max_gap_ratio: float = 0.04
    line_merge_ratio: float = 0.025
    # 透視変換後、同じ場所とみなす座標の半径。
    spatial_radius_px: float = 18.0
    # ORB記述子は256 bit。先行研究の妥当性評価と同じ上限を初期値にする。
    hamming_threshold: int = 100
    # 提案手法の判定閾値は未確定。10地点の実験後に決定する暫定値。
    decision_threshold: float = 0.06
    # 1タイルあたりのB-ORB上限。4枚合計で最大10000点。
    features_per_tile: int = 2500
    fast_threshold: int = 20
    draw_matches: int = 300


def check_config(c: Config) -> None:
    if c.warp_size < 256 or c.warp_size % 2:
        raise ValueError("warp-sizeは256以上の偶数にしてください。")
    if not 0 <= c.tile_margin_ratio < 0.25:
        raise ValueError("tile-marginは0以上0.25未満にしてください。")
    if c.spatial_radius_px <= 0 or not 0 <= c.hamming_threshold <= 256:
        raise ValueError("座標半径とHamming閾値を確認してください。")
    if not 0 <= c.decision_threshold <= 1:
        raise ValueError("判定閾値は0から1の範囲にしてください。")


def parse_points(text: str | None) -> np.ndarray | None:
    if text is None:
        return None
    try:
        points = [[float(value) for value in pair.split(",")]
                  for pair in text.split(";")]
    except ValueError as exc:
        raise ValueError("4隅は x,y;x,y;x,y;x,y の形式で指定してください。") from exc
    array = np.asarray(points, dtype=np.float32)
    if array.shape != (4, 2):
        raise ValueError("4隅を左上、右上、右下、左下の順に4点指定してください。")
    return array


def order_quad(points: np.ndarray) -> np.ndarray:
    """4点を左上、右上、右下、左下の順に並べる。"""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    cyclic = points[np.argsort(angles)]
    start = int(np.argmin(cyclic[:, 0] + cyclic[:, 1]))
    cyclic = np.roll(cyclic, -start, axis=0)
    # 画像座標でTLから時計回りなら2点目がTR、反時計回りならBL。
    if cyclic[1, 0] < cyclic[-1, 0]:
        cyclic = cyclic[[0, 3, 2, 1]]
    return cyclic.astype(np.float32)


def line_intersection(line1: np.ndarray, line2: np.ndarray) -> np.ndarray | None:
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    determinant = a1 * b2 - a2 * b1
    if abs(determinant) < 1e-8:
        return None
    return np.array([(b1 * c2 - b2 * c1) / determinant,
                     (c1 * a2 - c2 * a1) / determinant], dtype=np.float64)


def segment_to_line(segment: np.ndarray) -> np.ndarray | None:
    x1, y1, x2, y2 = map(float, segment)
    a, b = y1 - y2, x2 - x1
    norm = math.hypot(a, b)
    if norm < 1e-8:
        return None
    a, b = a / norm, b / norm
    return np.array([a, b, -(a * x1 + b * y1)], dtype=np.float64)


def _cluster_orientations(angles: np.ndarray) -> np.ndarray:
    """180度周期の線分角を2方向へ分ける。"""
    samples = np.column_stack((np.cos(2 * angles), np.sin(2 * angles))).astype(np.float32)
    if len(samples) < 6:
        raise ValueError("目地候補の直線が不足しています。")
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    _, labels, centers = cv2.kmeans(samples, 2, None, criteria, 10,
                                     cv2.KMEANS_PP_CENTERS)
    directions = 0.5 * np.arctan2(centers[:, 1], centers[:, 0])
    difference = abs(math.sin(float(directions[0] - directions[1])))
    if difference < math.sin(math.radians(55)):
        raise ValueError("ほぼ直交する2方向の目地線を検出できませんでした。")
    return labels.ravel()


def _merge_lines(records: list[tuple[np.ndarray, float, float]],
                 merge_distance: float) -> list[dict]:
    """画像中心での符号付き位置が近い線を同じ目地として束ねる。"""
    records.sort(key=lambda item: item[1])
    groups: list[list[tuple[np.ndarray, float, float]]] = []
    for record in records:
        if not groups or abs(record[1] - np.average(
                [r[1] for r in groups[-1]], weights=[r[2] for r in groups[-1]])) > merge_distance:
            groups.append([record])
        else:
            groups[-1].append(record)
    merged = []
    for group in groups:
        weights = np.asarray([r[2] for r in group], dtype=np.float64)
        reference = group[0][0].copy()
        aligned = []
        for line, _, _ in group:
            aligned.append(line if np.dot(line[:2], reference[:2]) >= 0 else -line)
        line = np.average(np.asarray(aligned), axis=0, weights=weights)
        line /= max(np.linalg.norm(line[:2]), 1e-12)
        position = float(np.average([r[1] for r in group], weights=weights))
        merged.append({"line": line, "position": position,
                       "score": float(weights.sum()), "segments": len(group)})
    return merged


def _select_three_grid_lines(lines: list[dict], extent: float) -> list[np.ndarray]:
    """中央目地と、その両側の外周目地を選ぶ。"""
    if len(lines) < 3:
        raise ValueError("2x2タイルに必要な3本の目地を検出できませんでした。")
    minimum_spacing = 0.12 * extent
    central_candidates = sorted(lines, key=lambda x: abs(x["position"]))[:min(4, len(lines))]
    best = None
    for center in central_candidates:
        negative = [x for x in lines
                    if x["position"] < center["position"] - minimum_spacing]
        positive = [x for x in lines
                    if x["position"] > center["position"] + minimum_spacing]
        for left in negative:
            for right in positive:
                spacing1 = center["position"] - left["position"]
                spacing2 = right["position"] - center["position"]
                ratio = min(spacing1, spacing2) / max(spacing1, spacing2)
                if ratio < 0.45:
                    continue
                # 長い線を優先しつつ、中央線が画像中央付近にある組を選ぶ。
                span = right["position"] - left["position"]
                score = (left["score"] + center["score"] + right["score"])
                # 2x2全体の外周を選ぶため、等間隔性に加えて広い範囲を
                # 覆う3本を優先する。これがないとタイル内部の長い傷を
                # 外周目地と誤認しやすい。
                score *= (0.5 + 0.5 * ratio) * (span / max(extent, 1.0)) ** 2
                score /= 1.0 + abs(center["position"]) / max(extent, 1.0)
                if best is None or score > best[0]:
                    best = (score, [left["line"], center["line"], right["line"]])
    if best is None:
        raise ValueError("等間隔に並ぶ3本の目地線を選べませんでした。")
    return best[1]


def detect_grid_quad(image: np.ndarray, c: Config):
    """Hough線分から2x2タイル外周の4点を推定する。"""
    original_h, original_w = image.shape[:2]
    scale = min(1.0, c.detection_max_side / max(original_h, original_w))
    work = cv2.resize(image, (round(original_w * scale), round(original_h * scale)),
                      interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, c.canny_low, c.canny_high, L2gradient=True)
    extent = float(min(work.shape[:2]))
    segments = cv2.HoughLinesP(
        edges, 1, np.pi / 360, c.hough_threshold,
        minLineLength=max(30, round(c.hough_min_length_ratio * extent)),
        maxLineGap=max(5, round(c.hough_max_gap_ratio * extent)))
    if segments is None or len(segments) < 6:
        raise ValueError("目地候補の線分を十分に検出できませんでした。")
    # OpenCV 4系は(N,1,4)、5系は環境により(N,4)で返るため両対応。
    segments = np.asarray(segments).reshape(-1, 4)
    vectors = segments[:, 2:4] - segments[:, 0:2]
    lengths = np.linalg.norm(vectors, axis=1)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    labels = _cluster_orientations(angles)
    center = np.array([work.shape[1] / 2, work.shape[0] / 2], dtype=np.float64)
    families = []
    all_merged = []
    for family_id in (0, 1):
        records = []
        family_indices = np.flatnonzero(labels == family_id)
        if len(family_indices) < 3:
            raise ValueError("一方向の目地候補が不足しています。")
        normal_reference = None
        for index in family_indices:
            line = segment_to_line(segments[index])
            if line is None:
                continue
            if normal_reference is None:
                normal_reference = line[:2].copy()
            if np.dot(line[:2], normal_reference) < 0:
                line = -line
            position = float(line[0] * center[0] + line[1] * center[1] + line[2])
            records.append((line, position, float(lengths[index])))
        merged = _merge_lines(records, c.line_merge_ratio * extent)
        selected = _select_three_grid_lines(merged, extent)
        families.append(selected)
        all_merged.append(merged)

    outer1 = [families[0][0], families[0][2]]
    outer2 = [families[1][0], families[1][2]]
    corners = []
    for line1 in outer1:
        for line2 in outer2:
            point = line_intersection(line1, line2)
            if point is None:
                raise ValueError("目地線の交点を計算できませんでした。")
            corners.append(point / scale)
    quad = order_quad(np.asarray(corners))
    margin = 0.18 * max(original_h, original_w)
    if ((quad[:, 0] < -margin).any() or (quad[:, 0] > original_w + margin).any()
            or (quad[:, 1] < -margin).any() or (quad[:, 1] > original_h + margin).any()):
        raise ValueError("検出した2x2タイル領域が画像外へ大きく外れています。")

    overlay = work.copy()
    colors = ((255, 80, 30), (30, 220, 255))
    for family_id, selected in enumerate(families):
        for line in selected:
            a, b, cc = line
            # 十分長い2点を作って描画する。
            if abs(b) > abs(a):
                p1 = (0, round(-cc / b))
                p2 = (work.shape[1] - 1,
                      round(-(a * (work.shape[1] - 1) + cc) / b))
            else:
                p1 = (round(-cc / a), 0)
                p2 = (round(-(b * (work.shape[0] - 1) + cc) / a),
                      work.shape[0] - 1)
            cv2.line(overlay, p1, p2, colors[family_id], 3, cv2.LINE_AA)
    scaled_quad = np.round(quad * scale).astype(np.int32)
    cv2.polylines(overlay, [scaled_quad], True, (0, 255, 0), 4, cv2.LINE_AA)
    return quad, overlay, edges, {
        "detected_segments": int(len(segments)),
        "merged_lines_per_direction": [len(group) for group in all_merged],
        "detection_scale": scale,
    }


def warp_grid(image: np.ndarray, quad: np.ndarray, size: int):
    destination = np.array([[0, 0], [size - 1, 0],
                            [size - 1, size - 1], [0, size - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(order_quad(quad), destination)
    warped = cv2.warpPerspective(image, matrix, (size, size),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
    return warped, matrix


def rotate_mosaic(image: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(image, quarter_turns).copy()


def tile_bounds(size: int, margin_ratio: float):
    half = size // 2
    margin = round(half * margin_ratio)
    return [
        (margin, margin, half - margin, half - margin),
        (half + margin, margin, size - margin, half - margin),
        (margin, half + margin, half - margin, size - margin),
        (half + margin, half + margin, size - margin, size - margin),
    ]


def extract_tile_features(mosaic: np.ndarray, c: Config):
    """4タイルを別々に処理し、座標だけ2x2全体へ戻す。"""
    base = replace(BaseConfig(), image_scale=1.0, nfeatures=c.features_per_tile,
                   fast_threshold=c.fast_threshold)
    all_keypoints, all_descriptors, tile_stats = [], [], []
    filtered_mosaic = np.full(mosaic.shape[:2], 128, np.uint8)
    for tile_id, (x0, y0, x1, y1) in enumerate(
            tile_bounds(c.warp_size, c.tile_margin_ratio)):
        tile = mosaic[y0:y1, x0:x1]
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        filtered = mean_difference_filter(gray, base.filter_radius)
        filtered_mosaic[y0:y1, x0:x1] = filtered
        keypoints, descriptors, layers = balanced_orb(filtered, base)
        for keypoint in keypoints:
            all_keypoints.append(cv2.KeyPoint(
                keypoint.pt[0] + x0, keypoint.pt[1] + y0,
                keypoint.size, keypoint.angle, keypoint.response,
                keypoint.octave, tile_id))
        if len(descriptors):
            all_descriptors.extend(descriptors)
        tile_stats.append({"tile": tile_id, "bounds": [x0, y0, x1, y1],
                           "features": len(keypoints), "layers": layers})
    descriptors = np.asarray(all_descriptors, np.uint8).reshape(-1, 32)
    return all_keypoints, descriptors, filtered_mosaic, tile_stats


def spatial_match(keypoints1, desc1: np.ndarray, keypoints2, desc2: np.ndarray,
                  c: Config):
    """同じタイルかつ座標半径内だけで記述子を比較し、1対1対応を作る。"""
    points1, points2 = points_from_keypoints(keypoints1), points_from_keypoints(keypoints2)
    result = {"source_features": len(points1), "target_features": len(points2),
              "with_spatial_candidate": 0, "accepted": 0,
              "fraction_of_source": 0.0, "fraction_of_candidate": 0.0,
              "mean_hamming": None, "median_hamming": None}
    if not len(points1) or not len(points2):
        return [], result
    radius = c.spatial_radius_px
    bins: dict[tuple[int, int, int], list[int]] = {}
    for index, (point, kp) in enumerate(zip(points2, keypoints2)):
        cell = (kp.class_id, math.floor(point[0] / radius), math.floor(point[1] / radius))
        bins.setdefault(cell, []).append(index)
    proposals = []
    for query_index, (point, kp) in enumerate(zip(points1, keypoints1)):
        cell_x, cell_y = math.floor(point[0] / radius), math.floor(point[1] / radius)
        candidates = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                candidates.extend(bins.get((kp.class_id, cell_x + dx, cell_y + dy), []))
        if not candidates:
            continue
        ids = np.asarray(candidates, dtype=np.int64)
        ids = ids[((points2[ids] - point) ** 2).sum(axis=1) <= radius ** 2]
        if not len(ids):
            continue
        result["with_spatial_candidate"] += 1
        distances = BIT_COUNTS[np.bitwise_xor(desc2[ids], desc1[query_index])].sum(axis=1)
        best_position = int(np.argmin(distances))
        distance = int(distances[best_position])
        if distance <= c.hamming_threshold:
            proposals.append((distance, query_index, int(ids[best_position])))
    # 同じ撮影側特徴点を複数回使わない。Hamming距離が小さい対応を優先。
    proposals.sort()
    used_query, used_train, matches = set(), set(), []
    for distance, query_index, train_index in proposals:
        if query_index in used_query or train_index in used_train:
            continue
        used_query.add(query_index)
        used_train.add(train_index)
        matches.append(cv2.DMatch(query_index, train_index, float(distance)))
    result["accepted"] = len(matches)
    result["fraction_of_source"] = len(matches) / len(points1)
    result["fraction_of_candidate"] = (
        len(matches) / result["with_spatial_candidate"]
        if result["with_spatial_candidate"] else 0.0)
    if matches:
        distances = np.asarray([m.distance for m in matches])
        result["mean_hamming"] = float(distances.mean())
        result["median_hamming"] = float(np.median(distances))
    return matches, result


def draw_grid_features(filtered: np.ndarray, keypoints):
    return cv2.drawKeypoints(filtered, keypoints, None, color=(0, 255, 0))


def make_manual_overlay(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    cv2.polylines(overlay, [np.round(order_quad(quad)).astype(np.int32)],
                    True, (0, 255, 0), max(2, round(max(image.shape[:2]) / 500)))
    return overlay


def analyze_orientation(registered: np.ndarray, query: np.ndarray,
                        registered_features, c: Config, turns: int):
    rotated = rotate_mosaic(query, turns)
    kp1, desc1, _, _ = registered_features
    kp2, desc2, filtered2, stats2 = extract_tile_features(rotated, c)
    matches, metrics = spatial_match(kp1, desc1, kp2, desc2, c)
    metrics["quarter_turns_ccw"] = turns
    return {"mosaic": rotated, "keypoints": kp2, "descriptors": desc2,
            "filtered": filtered2, "tile_stats": stats2,
            "matches": matches, "metrics": metrics}


def run_pair(path1: Path, path2: Path, output: Path, c: Config,
             quad1: np.ndarray | None = None, quad2: np.ndarray | None = None,
             verbose: bool = True) -> dict:
    check_config(c)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    image1, image2 = read_image(path1), read_image(path2)
    timings = {}

    clock = time.perf_counter()
    if quad1 is None:
        quad1, overlay1, edges1, detection1 = detect_grid_quad(image1, c)
        detection1["mode"] = "automatic"
    else:
        quad1 = order_quad(quad1)
        overlay1, edges1 = make_manual_overlay(image1, quad1), None
        detection1 = {"mode": "manual"}
    if quad2 is None:
        quad2, overlay2, edges2, detection2 = detect_grid_quad(image2, c)
        detection2["mode"] = "automatic"
    else:
        quad2 = order_quad(quad2)
        overlay2, edges2 = make_manual_overlay(image2, quad2), None
        detection2 = {"mode": "manual"}
    timings["grout_detection_seconds"] = time.perf_counter() - clock

    clock = time.perf_counter()
    warped1, perspective1 = warp_grid(image1, quad1, c.warp_size)
    warped2, perspective2 = warp_grid(image2, quad2, c.warp_size)
    timings["perspective_seconds"] = time.perf_counter() - clock

    clock = time.perf_counter()
    registered_features = extract_tile_features(warped1, c)
    kp1, desc1, filtered1, stats1 = registered_features
    orientations = [analyze_orientation(warped1, warped2, registered_features, c, turns)
                    for turns in range(4)]
    best = max(orientations,
               key=lambda item: (item["metrics"]["fraction_of_source"],
                                 item["metrics"]["accepted"]))
    timings["feature_and_spatial_match_seconds"] = time.perf_counter() - clock
    metrics = best["metrics"]
    success = metrics["fraction_of_source"] > c.decision_threshold
    metrics["success"] = success
    metrics["decision_threshold"] = c.decision_threshold
    metrics["threshold_status"] = "provisional; calibrate with same/different-location data"

    save_image(output / "01_touroku_original.jpg", image1)
    save_image(output / "02_satsuei_original.jpg", image2)
    save_image(output / "03_touroku_grout_detection.jpg", overlay1)
    save_image(output / "04_satsuei_grout_detection.jpg", overlay2)
    if edges1 is not None:
        save_image(output / "05_touroku_edges.png", edges1)
    if edges2 is not None:
        save_image(output / "06_satsuei_edges.png", edges2)
    save_image(output / "07_touroku_perspective.jpg", warped1)
    save_image(output / "08_satsuei_perspective_best_rotation.jpg", best["mosaic"])
    save_image(output / "09_touroku_filter.png", filtered1)
    save_image(output / "10_satsuei_filter.png", best["filtered"])
    save_image(output / "11_touroku_features.jpg", draw_grid_features(filtered1, kp1))
    save_image(output / "12_satsuei_features.jpg",
               draw_grid_features(best["filtered"], best["keypoints"]))
    drawn = cv2.drawMatches(
        warped1, kp1, best["mosaic"], best["keypoints"],
        best["matches"][:c.draw_matches], None, matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    save_image(output / "13_spatial_matches.jpg", drawn)
    timings["total_seconds"] = time.perf_counter() - started

    if metrics["source_features"] == 0:
        reason = "登録画像から特徴点を抽出できませんでした。"
    elif metrics["with_spatial_candidate"] == 0:
        reason = "透視変換後の同一座標近傍に撮影側特徴点がありませんでした。"
    elif success:
        reason = "座標近傍かつHamming距離閾値以下の対応割合が暫定閾値を超えました。"
    else:
        reason = "座標近傍の特徴量一致割合が暫定閾値を超えませんでした。"

    summary = [
        "床指紋：提案手法（目地検出＋透視変換＋座標近傍照合）",
        f"登録画像：{path1}",
        f"撮影画像：{path2}",
        f"目地指定：登録{detection1['mode']} / 撮影{detection2['mode']}",
        f"採用した撮影画像の回転：反時計回り{90 * metrics['quarter_turns_ccw']}度",
        f"登録側B-ORB特徴点数：{len(kp1)}",
        f"撮影側B-ORB特徴点数：{len(best['keypoints'])}",
        f"座標近傍に候補がある登録特徴点：{metrics['with_spatial_candidate']}",
        f"Hamming距離{c.hamming_threshold}以下の1対1対応：{metrics['accepted']}",
        f"全登録特徴点に対する一致割合：{metrics['fraction_of_source'] * 100:.2f}%",
        f"近傍候補あり特徴点に対する一致割合：{metrics['fraction_of_candidate'] * 100:.2f}%",
        f"判定：{'成功' if success else '不成立'}（暫定閾値{c.decision_threshold * 100:g}%）",
        f"判定理由：{reason}",
        "※提案手法の判定閾値は未確定。10地点の同地点・異地点データで決定する。",
        "※BFMatcherとRANSACは使用していない。座標近傍内だけでHamming距離を比較。",
        f"処理時間：{timings['total_seconds']:.3f}秒",
        f"保存先：{output}",
    ]
    report = {
        "implementation": "proposed method prototype",
        "scope": "grout detection + perspective normalization + tile-wise B-ORB + spatial matching",
        "opencv_version": cv2.__version__, "numpy_version": np.__version__,
        "python_version": sys.version.split()[0], "config": asdict(c),
        "registered_image": str(path1), "query_image": str(path2),
        "registered_sha256": hashlib.sha256(path1.read_bytes()).hexdigest(),
        "query_sha256": hashlib.sha256(path2.read_bytes()).hexdigest(),
        "registered_quad": quad1.tolist(), "query_quad": quad2.tolist(),
        "registered_perspective": perspective1.tolist(),
        "query_perspective": perspective2.tolist(),
        "registered_detection": detection1, "query_detection": detection2,
        "registered_tile_stats": stats1, "query_tile_stats": best["tile_stats"],
        "orientation_trials": [item["metrics"] for item in orientations],
        "best_metrics": metrics, "decision_reason": reason, "timings": timings,
        "assumptions": [
            "input contains one complete 2x2 tile block and its three grout lines per direction",
            "automatic grout detection uses Canny + probabilistic Hough segments",
            "each detected 2x2 block is warped to a square and split into four tiles",
            "tile borders are excluded before mean-difference filtering and B-ORB",
            "four 90-degree orientations are tested; the highest spatial-match score is used",
            "BFMatcher and RANSAC are not used",
            "the decision threshold is provisional and must be calibrated experimentally",
        ],
    }
    (output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "result.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    if verbose:
        print("\n" + "\n".join(summary), flush=True)
    return report


def make_synthetic_grid(seed: int = 1):
    rng = np.random.default_rng(seed)
    size, tile = 900, 360
    image = np.full((size, size, 3), 180, np.uint8)
    starts = (90, 450)
    for row, y0 in enumerate(starts):
        for col, x0 in enumerate(starts):
            texture = rng.normal(0, 12, (tile, tile)).astype(np.float32)
            texture = cv2.GaussianBlur(texture, (3, 3), 0.35)
            base = np.clip(175 + texture + 3 * (row * 2 + col), 0, 255).astype(np.uint8)
            image[y0:y0 + tile, x0:x0 + tile] = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            # 微細な汚れ・傷を模した点と短線。FAST/ORBの動作確認用。
            for _ in range(90):
                x = int(rng.integers(x0 + 25, x0 + tile - 25))
                y = int(rng.integers(y0 + 25, y0 + tile - 25))
                value = int(rng.integers(90, 245))
                if rng.random() < 0.7:
                    cv2.circle(image, (x, y), int(rng.integers(1, 4)),
                               (value, value, value), -1)
                else:
                    dx, dy = int(rng.integers(-10, 11)), int(rng.integers(-10, 11))
                    cv2.line(image, (x, y), (x + dx, y + dy),
                             (value, value, value), 1)
    for position in (88, 448, 808):
        cv2.line(image, (position, 88), (position, 812), (55, 55, 55), 7)
        cv2.line(image, (88, position), (812, position), (55, 55, 55), 7)
    quad = np.array([[88, 88], [808, 88], [808, 808], [88, 808]], np.float32)
    return image, quad


def self_test() -> None:
    print("提案手法を合成画像で自己テストしています…", flush=True)
    c = replace(Config(), warp_size=600, features_per_tile=500,
                fast_threshold=5, spatial_radius_px=18.0,
                decision_threshold=0.03)
    image, quad = make_synthetic_grid()
    destination = np.array([[70, 110], [830, 60], [850, 820], [100, 850]], np.float32)
    transform = cv2.getPerspectiveTransform(quad, destination)
    photographed = cv2.warpPerspective(image, transform, (920, 920),
                                        borderMode=cv2.BORDER_REFLECT_101)
    photographed = cv2.rotate(photographed, cv2.ROTATE_90_CLOCKWISE)
    transformed_quad = cv2.perspectiveTransform(quad[None], transform)[0]
    transformed_quad = np.column_stack((920 - 1 - transformed_quad[:, 1],
                                        transformed_quad[:, 0])).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="floor_proposal_test_") as directory:
        root = Path(directory)
        registered, query = root / "登録.jpg", root / "撮影.jpg"
        save_image(registered, image)
        save_image(query, photographed)
        positive = run_pair(registered, query, root / "positive", c,
                            quad, transformed_quad, verbose=False)
        assert positive["best_metrics"]["accepted"] > 0, positive["best_metrics"]
        assert positive["best_metrics"]["success"], positive["best_metrics"]
        assert (root / "positive" / "13_spatial_matches.jpg").is_file()
        automatic = run_pair(registered, query, root / "automatic", c,
                             verbose=False)
        assert automatic["registered_detection"]["mode"] == "automatic"
        assert automatic["query_detection"]["mode"] == "automatic"
        assert automatic["best_metrics"]["accepted"] > 0, automatic["best_metrics"]
        # 同じ4隅でも別模様なら一致割合が下がることを確認する。
        different, _ = make_synthetic_grid(seed=999)
        save_image(query, different)
        negative = run_pair(registered, query, root / "negative", c, quad, quad, verbose=False)
        assert (negative["best_metrics"]["fraction_of_source"]
                < positive["best_metrics"]["fraction_of_source"])
    print("自己テスト成功：透視変換、4タイル分割、回転探索、座標近傍照合、結果保存。")
    print("※合成画像の動作確認であり、実際の床に対する精度の保証ではありません。")


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (BASE_DIR / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="床指紋の提案手法：目地検出＋透視変換＋座標近傍照合")
    parser.add_argument("--touroku", default="gazou/touroku.jpg", help="登録画像のパス")
    parser.add_argument("--satsuei", default="gazou/satsuei_real.jpg", help="撮影画像のパス")
    parser.add_argument("--output", default=None, help="新規の結果フォルダ（既存は不可）")
    parser.add_argument("--touroku-points", default=None,
                        help="登録画像の4隅：x,y;x,y;x,y;x,y")
    parser.add_argument("--satsuei-points", default=None,
                        help="撮影画像の4隅：x,y;x,y;x,y;x,y")
    parser.add_argument("--warp-size", type=int, default=Config.warp_size)
    parser.add_argument("--features-per-tile", type=int, default=Config.features_per_tile)
    parser.add_argument("--fast-threshold", type=int, default=Config.fast_threshold)
    parser.add_argument("--spatial-radius", type=float, default=Config.spatial_radius_px)
    parser.add_argument("--hamming", type=int, default=Config.hamming_threshold)
    parser.add_argument("--threshold", type=float, default=Config.decision_threshold,
                        help="暫定判定閾値。実験後に校正する")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    c = replace(Config(), warp_size=args.warp_size,
                features_per_tile=args.features_per_tile,
                fast_threshold=args.fast_threshold,
                spatial_radius_px=args.spatial_radius,
                hamming_threshold=args.hamming,
                decision_threshold=args.threshold)
    registered, query = resolve_path(args.touroku), resolve_path(args.satsuei)
    for path in (registered, query):
        if not path.is_file():
            raise FileNotFoundError(f"画像がありません：{path}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (resolve_path(args.output) if args.output else
              BASE_DIR / "kekka" / f"step4_teian_{stamp}")
    run_pair(registered, query, output, c,
             parse_points(args.touroku_points), parse_points(args.satsuei_points))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, cv2.error) as exc:
        print(f"\n実行できませんでした：{exc}", file=sys.stderr)
        raise SystemExit(1)
