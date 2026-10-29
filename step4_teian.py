r"""床指紋の提案手法：1タイル透視補正 + 座標近傍照合 + RANSAC。

step3_senko_saigen.py と同じフォルダに保存してください。
既存の先行研究コードは変更しません。OpenCV、NumPyだけを追加で使います。

登録：既存 gazou/db_cache の「分割タイル画像」を読み、同じ正方形へ
正規化して特徴を再抽出し、提案手法専用 gazou/teian_db_cache に保存。
旧DB特徴量は画像サイズ・外周除外条件が異なるため、そのまま流用しません。
撮影：2x2全体は不要。完全に写ったタイルを個別に正方形へ透視補正。
各タイルを全DBタイルと比較し、0/90/180/270度を確認します。
対応候補は座標近傍のHamming距離から作り、step3のRANSACで精査します。

--satsuei-points は「1タイル」の4隅。x,y;x,y;x,y;x,y。
--quad は同じ指定を複数回指定できます。座標は元の撮影画像上の画素です。
未指定なら目地候補線の交点からタイル候補を自動検出します。
自動検出は実験実装であり、壁・影・傷の誤検出もあります。
query_detection.jpg で、選んだ領域がタイル1枚か必ず確認してください。
--detect-only は検出結果と切り出し画像だけを保存します。

初期検証の地点得点は、候補タイルとその地点の登録タイルの最高得点。
平均による4タイル集約はしません。部分的に写った場所も検証するためです。
6%の判定基準は暫定値です。検出候補数が増えると誤照合も増え得るため、
同地点・異地点データで検証してから決めてください。
このコードは先行研究の再現ではなく、新しい提案手法の実験用コードです。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import tempfile
import time

try:
    import cv2
    import numpy as np
    from step3_senko_saigen import (
        Config as BaseConfig, BIT_COUNTS, balanced_orb,
        mean_difference_filter, points_from_keypoints, read_image, save_image,
        numbered_images, split_four_tiles, paper_ransac,
        homography_from_consensus, validate_alignment,
    )
except ImportError as exc:
    print("OpenCV・NumPyをインストールし、step3_senko_saigen.pyを同じフォルダへ置いてください。")
    print(exc)
    raise SystemExit(2)

BASE_DIR = Path(__file__).resolve().parent
CACHE_FORMAT = "floor-fingerprint-proposal-single-tile-v1"


@dataclass(frozen=True)
class Config:
    tile_size: int = 512
    features_per_tile: int = 2500
    fast_threshold: int = 5
    margin_ratio: float = 0.04
    spatial_radius: float = 18.0
    hamming: int = 100
    threshold: float = 0.06
    trials: int = 20000
    seed: int = 0
    detection_max_side: int = 1400
    max_tiles: int = 12
    # 透視補正済み画像で、後段の変換が大きく崩れるモデルを除く補完設定。
    max_residual_corner_shift_ratio: float = 0.20


def resolve_path(value: str) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (BASE_DIR / p).resolve()


def check_config(c: Config) -> None:
    if c.tile_size < 128 or c.features_per_tile < 20 or c.fast_threshold < 1:
        raise ValueError("tile-sizeは128以上、features-per-tileは20以上、FASTは1以上が必要です。")
    if not 0 <= c.margin_ratio < 0.2:
        raise ValueError("tile-marginは0以上0.2未満にしてください。")
    if c.spatial_radius <= 0 or not 0 <= c.hamming <= 256:
        raise ValueError("spatial-radiusは正の数、hammingは0～256にしてください。")
    if not 0 <= c.threshold <= 1 or c.trials < 1 or c.max_tiles < 1:
        raise ValueError("thresholdは0～1、trialsとmax-tilesは1以上が必要です。")


def base_config(c: Config) -> BaseConfig:
    return replace(BaseConfig(), image_scale=1.0, nfeatures=c.features_per_tile,
                   fast_threshold=c.fast_threshold, ransac_trials=c.trials,
                   random_seed=c.seed, validation_hamming=c.hamming,
                   validation_threshold=c.threshold)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def order_quad(points: np.ndarray) -> np.ndarray:
    """凸な四隅を時計回りへ並べる。開始角の違いは4回転照合で吸収。"""
    p = np.asarray(points, np.float32)
    if p.shape != (4, 2) or not np.isfinite(p).all():
        raise ValueError("四隅は有限の座標4点で指定してください。")
    center = p.mean(axis=0)
    p = p[np.argsort(np.arctan2(p[:, 1] - center[1], p[:, 0] - center[0]))]
    p = np.roll(p, -int(np.argmin(p.sum(axis=1))), axis=0)
    if not cv2.isContourConvex(p.reshape(-1, 1, 2)) or abs(cv2.contourArea(p)) < 100:
        raise ValueError("四隅が凸な四角形になっていません。四隅の座標を確認してください。")
    return p


def parse_quad(text: str) -> np.ndarray:
    try:
        return order_quad(np.asarray([[float(v) for v in pair.split(",")]
                                      for pair in text.split(";")], np.float32))
    except (ValueError, TypeError) as exc:
        raise ValueError("四隅は x,y;x,y;x,y;x,y の形式で指定してください。") from exc


def require_inside(quad: np.ndarray, shape) -> None:
    h, w = shape[:2]
    if (quad[:, 0].min() < 0 or quad[:, 0].max() > w - 1
            or quad[:, 1].min() < 0 or quad[:, 1].max() > h - 1):
        raise ValueError(f"四隅が画像外です。元画像サイズは幅{w}×高さ{h}です。")


def normalize_tile(image: np.ndarray, c: Config) -> np.ndarray:
    square = cv2.resize(image, (c.tile_size, c.tile_size), interpolation=cv2.INTER_AREA)
    margin = round(c.tile_size * c.margin_ratio)
    if margin:
        square = cv2.resize(square[margin:-margin, margin:-margin],
                            (c.tile_size, c.tile_size), interpolation=cv2.INTER_LINEAR)
    return square


def rectify_tile(image: np.ndarray, quad: np.ndarray, c: Config) -> np.ndarray:
    quad = order_quad(quad)
    require_inside(quad, image.shape)
    s = c.tile_size - 1
    destination = np.float32([[0, 0], [s, 0], [s, s], [0, s]])
    matrix = cv2.getPerspectiveTransform(quad, destination)
    square = cv2.warpPerspective(image, matrix, (c.tile_size, c.tile_size),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return normalize_tile(square, c)


def extract_features(image: np.ndarray, c: Config) -> dict:
    filtered = mean_difference_filter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                      BaseConfig.filter_radius)
    keypoints, descriptors, _ = balanced_orb(filtered, base_config(c))
    return {"image": image, "filtered": filtered, "keypoints": keypoints,
            "points": points_from_keypoints(keypoints), "descriptors": descriptors}


def build_database(args, c: Config) -> None:
    """先行DBのタイル画像を利用。先行DBの特徴量・設定は変更しない。"""
    check_config(c)
    source = resolve_path(args.db_cache)
    metadata_path = source / "database.json"
    sources = []
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for row in metadata["tiles"]:
            path = source / Path(row["image_file"].replace("\\", "/"))
            sources.append((int(row["source_id"]), int(row["tile_index"]), path, None))
    else:
        # 元2x2画像からの登録も可能。真上から撮った登録画像を前提とする。
        for location, path in numbered_images(resolve_path(args.db_dir), "db_"):
            tiles, _ = split_four_tiles(read_image(path), margin=4, mode="auto")
            sources.extend((location, i + 1, path, tile) for i, tile in enumerate(tiles))
    if not sources:
        raise ValueError("登録タイルがありません。")
    for _, _, path, tile in sources:
        if tile is None and not path.is_file():
            raise FileNotFoundError(f"登録タイル画像がありません：{path}")
    keys = [(location, index) for location, index, _, _ in sources]
    if len(set(keys)) != len(keys):
        raise ValueError("登録タイルの地点番号・タイル番号が重複しています。")
    output = resolve_path(args.proposal_cache)
    output.mkdir(parents=True, exist_ok=False)
    (output / "tiles").mkdir()
    (output / "features").mkdir()
    started = time.perf_counter()
    rows = []
    for number, (location, index, path, tile) in enumerate(sorted(sources, key=lambda v: v[:2]), 1):
        name = f"db_{location:02d}_tile_{index}"
        print(f"登録特徴抽出 [{number}/{len(sources)}] 地点{location}・タイル{index}", flush=True)
        normalized = normalize_tile(read_image(path) if tile is None else tile, c)
        prepared = extract_features(normalized, c)
        save_image(output / "tiles" / f"{name}.png", normalized)
        np.savez_compressed(output / "features" / f"{name}.npz",
                            points=prepared["points"], descriptors=prepared["descriptors"])
        rows.append({"location": location, "tile_index": index, "name": name,
                     "source_image": str(path), "feature_count": len(prepared["points"]),
                     "image_file": f"tiles/{name}.png", "feature_file": f"features/{name}.npz"})
    elapsed = time.perf_counter() - started
    write_json(output / "database.json", {"format": CACHE_FORMAT, "config": asdict(c),
               "tiles": rows, "elapsed_seconds": elapsed})
    write_csv(output / "tile_names.csv", rows)
    print(f"\n提案手法DB作成完了：{len(set(k[0] for k in keys))}地点 / {len(rows)}タイル")
    print(f"登録処理時間：{elapsed:.1f}秒\n保存先：{output}", flush=True)


def load_database(path: Path, runtime: Config) -> tuple[list[dict], Config]:
    metadata_path = path / "database.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"提案手法DBがありません：{metadata_path}\n先に --build-db を実行してください。")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != CACHE_FORMAT:
        raise ValueError("先行研究用db_cacheではなく、提案手法用teian_db_cacheを指定してください。")
    stored = Config(**metadata["config"])
    c = replace(stored, spatial_radius=runtime.spatial_radius, hamming=runtime.hamming,
                threshold=runtime.threshold, trials=runtime.trials, seed=runtime.seed,
                max_tiles=runtime.max_tiles)
    check_config(c)
    rows = []
    for row in metadata["tiles"]:
        with np.load(path / row["feature_file"], allow_pickle=False) as data:
            points = data["points"].astype(np.float64)
            desc = data["descriptors"].astype(np.uint8)
        if points.ndim != 2 or points.shape[1] != 2 or desc.shape != (len(points), 32):
            raise ValueError(f"登録特徴量の形式が不正です：{row['name']}")
        rows.append({**row, "points": points, "descriptors": desc,
                     "image_path": path / row["image_file"]})
    if not rows:
        raise ValueError("提案手法DBにタイルがありません。")
    return rows, c


def line_intersection(first, second):
    point = np.cross(first, second)
    return None if abs(point[2]) < 1e-8 else point[:2] / point[2]


def detect_tile_quads(image: np.ndarray, c: Config):
    """目地候補2方向の隣接線が囲むセルを列挙。等間隔・2x2を要求しない。"""
    h, w = image.shape[:2]
    scale = min(1.0, c.detection_max_side / max(h, w))
    work = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 20, 60, L2gradient=True)
    extent = min(work.shape[:2])
    segments = cv2.HoughLinesP(edges, 1, np.pi / 360, 55,
                              minLineLength=max(40, round(extent * .13)),
                              maxLineGap=max(3, round(extent * .004)))
    diagnostics = {"mode": "automatic", "scale": scale, "segments": 0,
                   "warning": "壁・傷を含むタイル候補。検出画像を目視確認すること。"}
    if segments is None:
        return [], edges, diagnostics
    seg = np.asarray(segments).reshape(-1, 4).astype(float)
    vectors = seg[:, 2:] - seg[:, :2]
    lengths = np.linalg.norm(vectors, axis=1)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    samples = np.float32(np.column_stack((np.cos(2 * angles), np.sin(2 * angles))))
    if len(seg) < 4:
        return [], edges, diagnostics
    cv2.setRNGSeed(0)
    _, labels, centers = cv2.kmeans(samples, 2, None,
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 60, .0001),
        8, cv2.KMEANS_PP_CENTERS)
    direction = .5 * np.arctan2(centers[:, 1], centers[:, 0])
    if abs(math.sin(float(direction[0] - direction[1]))) < .4:
        return [], edges, {**diagnostics, "reason": "2方向の線を分離できません。"}
    center = np.array([work.shape[1] / 2, work.shape[0] / 2])
    families = []
    for family in (0, 1):
        normal = np.array([-math.sin(direction[family]), math.cos(direction[family])])
        records = []
        for index in np.flatnonzero(labels.ravel() == family):
            x1, y1, x2, y2 = seg[index]
            line = np.array([y1 - y2, x2 - x1, x1 * y2 - x2 * y1])
            line /= max(np.linalg.norm(line[:2]), 1e-8)
            if np.dot(line[:2], normal) < 0:
                line = -line
            if np.dot(line[:2], normal) < math.cos(math.radians(28)):
                continue
            records.append((float(np.dot(line[:2], center) + line[2]), line, lengths[index]))
        records.sort(key=lambda v: v[0])
        groups = []
        for record in records:
            if not groups or abs(record[0] - np.average(
                    [r[0] for r in groups[-1]], weights=[r[2] for r in groups[-1]])) > extent * .012:
                groups.append([record])
            else:
                groups[-1].append(record)
        merged = []
        for group in groups:
            line = np.average([r[1] for r in group], axis=0, weights=[r[2] for r in group])
            line /= np.linalg.norm(line[:2])
            merged.append(line)
        families.append(merged)
    # 境界が実際のエッジに沿っているか検査。画像外の辺を補わない。
    distances = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    candidates = []
    for a, b in zip(families[0], families[0][1:]):
        for d, e in zip(families[1], families[1][1:]):
            corners = [line_intersection(x, y) for x, y in ((a, d), (b, d), (b, e), (a, e))]
            if any(p is None for p in corners):
                continue
            try:
                quad = order_quad(np.float32(corners))
                require_inside(quad, work.shape)
            except ValueError:
                continue
            sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
            area = abs(cv2.contourArea(quad))
            if sides.min() < extent * .075 or sides.max() / sides.min() > 6:
                continue
            if not .005 * work.shape[0] * work.shape[1] <= area <= .55 * work.shape[0] * work.shape[1]:
                continue
            support = []
            for p, q in zip(quad, np.roll(quad, -1, axis=0)):
                t = np.linspace(.08, .92, 60)
                xy = np.rint(p[None] * (1 - t[:, None]) + q[None] * t[:, None]).astype(int)
                support.append(float(np.mean(distances[xy[:, 1], xy[:, 0]] < 5)))
            if min(support) < .45:
                continue
            candidates.append((float(np.mean(support)), area, quad / scale))
    candidates.sort(key=lambda v: (v[0], v[1]), reverse=True)
    accepted = []
    for support, area, quad in candidates:
        duplicate = False
        for other in accepted:
            overlap, _ = cv2.intersectConvexConvex(quad.astype(np.float32), other)
            if overlap / max(abs(cv2.contourArea(quad)) + abs(cv2.contourArea(other)) - overlap, 1) > .8:
                duplicate = True
                break
        if not duplicate:
            accepted.append(quad.astype(np.float32))
        if len(accepted) >= c.max_tiles:
            break
    accepted.sort(key=lambda q: (float(q.mean(axis=0)[1]), float(q.mean(axis=0)[0])))
    diagnostics.update(segments=len(seg), lines_per_direction=[len(f) for f in families],
                       candidate_count=len(accepted), max_tiles=c.max_tiles)
    return accepted, edges, diagnostics


def prepare_query(image: np.ndarray, output: Path, c: Config, texts: list[str]):
    if texts:
        quads = [parse_quad(text) for text in texts]
        for quad in quads:
            require_inside(quad, image.shape)
        edges, detection = None, {"mode": "manual", "candidate_count": len(quads)}
    else:
        quads, edges, detection = detect_tile_quads(image, c)
    output.mkdir(parents=True, exist_ok=False)
    (output / "query_tiles").mkdir()
    overlay = image.copy()
    for index, quad in enumerate(quads, 1):
        cv2.polylines(overlay, [np.rint(quad).astype(np.int32)], True, (0, 220, 0), 4)
        center = np.rint(quad.mean(axis=0)).astype(int)
        cv2.putText(overlay, f"tile {index}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX,
                    max(.6, max(image.shape[:2]) / 1800), (0, 0, 255), 3)
        save_image(output / "query_tiles" / f"shoot_tile_{index}.png", rectify_tile(image, quad, c))
    save_image(output / "query_detection.jpg", overlay)
    if edges is not None:
        save_image(output / "query_edges.png", edges)
    write_json(output / "query_quads.json", {"source_shape_hw": list(image.shape[:2]),
               "quads": [q.tolist() for q in quads], "detection": detection})
    return quads


def rotate_features(prepared: dict, turns: int, size: int) -> dict:
    """ORB記述子を再利用し、正方形上の座標を90度単位で回転する。"""
    p = prepared["points"].copy()
    for _ in range(turns):
        p = np.column_stack((p[:, 1], size - 1 - p[:, 0]))
    return {**prepared, "points": p, "image": np.rot90(prepared["image"], turns).copy()}


def spatial_matches(registered: dict, query: dict, c: Config):
    """座標近傍で最もHamming距離の小さい候補を選び、重複を除いて1対1に。"""
    radius = c.spatial_radius
    bins = {}
    for index, point in enumerate(query["points"]):
        cell = tuple(np.floor(point / radius).astype(int))
        bins.setdefault(cell, []).append(index)
    proposals = []
    for index, point in enumerate(registered["points"]):
        x, y = np.floor(point / radius).astype(int)
        ids = [i for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               for i in bins.get((x + dx, y + dy), [])]
        if not ids:
            continue
        ids = np.asarray(ids, np.int64)
        ids = ids[np.square(query["points"][ids] - point).sum(axis=1) <= radius ** 2]
        if not len(ids):
            continue
        distances = BIT_COUNTS[np.bitwise_xor(query["descriptors"][ids],
                                               registered["descriptors"][index])].sum(axis=1)
        best = int(np.argmin(distances))
        if distances[best] <= c.hamming:
            proposals.append((int(distances[best]), index, int(ids[best])))
    used, matches = set(), []
    for distance, source, target in sorted(proposals):
        if target not in used:
            used.add(target)
            matches.append(cv2.DMatch(source, target, float(distance)))
    return matches


def compare_tiles(registered: dict, query: dict, c: Config) -> dict:
    matches = spatial_matches(registered, query, c)
    src = np.asarray([registered["points"][m.queryIdx] for m in matches], np.float64).reshape(-1, 2)
    dst = np.asarray([query["points"][m.trainIdx] for m in matches], np.float64).reshape(-1, 2)
    _, mask, info = paper_ransac(src, dst, base_config(c))
    homography = homography_from_consensus(src[mask], dst[mask])
    sane = homography is not None
    if sane:
        s = c.tile_size - 1
        corners = np.float32([[0, 0], [s, 0], [s, s], [0, s]])
        projected = cv2.perspectiveTransform(corners[None], homography)[0]
        sane = bool(np.isfinite(projected).all()
                    and np.linalg.norm(projected - corners, axis=1).max()
                    <= c.max_residual_corner_shift_ratio * c.tile_size)
    validation = validate_alignment(registered["points"], registered["descriptors"],
        query["points"], query["descriptors"], homography if sane else None,
        (c.tile_size, c.tile_size), base_config(c))
    return {"score": validation["fraction"], "success": bool(validation["success"]),
            "candidate_matches": len(matches), "inliers": int(mask.sum()),
            "inlier_fraction": int(mask.sum()) / len(matches) if matches else 0.0,
            "ransac_trials": info["trials"], "ransac_full_checks": info["full_checks"],
            "homography_sane": sane, "validation": validation,
            "matches": matches, "mask": mask, "homography": homography}


def draw_pair(output: Path, registered: dict, query: dict, report: dict) -> None:
    source = read_image(registered["image_path"])
    first = [cv2.KeyPoint(float(p[0]), float(p[1]), 7.) for p in registered["points"]]
    second = [cv2.KeyPoint(float(p[0]), float(p[1]), 7.) for p in query["points"]]
    pairs = report["matches"][:300]
    retained = [m for m, keep in zip(pairs, report["mask"][:len(pairs)]) if keep]
    flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    save_image(output / "best_db_tile.png", source)
    save_image(output / "best_query_tile.png", query["image"])
    save_image(output / "matches_before_ransac.jpg", cv2.drawMatches(
        source, first, query["image"], second, pairs, None,
        matchColor=(0, 200, 255), flags=flags))
    save_image(output / "matches_after_ransac.jpg", cv2.drawMatches(
        source, first, query["image"], second, retained, None,
        matchColor=(0, 255, 0), flags=flags))


def identify(args, runtime: Config) -> dict:
    photo = resolve_path(args.satsuei)
    if not photo.is_file():
        raise FileNotFoundError(f"撮影画像がありません：{photo}")
    started = time.perf_counter()
    db, c = load_database(resolve_path(args.proposal_cache), runtime)
    loaded_seconds = time.perf_counter() - started
    output = resolve_path(args.output) if args.output else BASE_DIR / "kekka" / (
        "teian_identify_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    clock = time.perf_counter()
    texts = ([args.satsuei_points] if args.satsuei_points else []) + args.quad
    image = read_image(photo)
    quads = prepare_query(image, output, c, texts)
    detection_seconds = time.perf_counter() - clock
    if not quads:
        result = {"prediction": "no_tile_detected", "query_image": str(photo),
                  "reason": "目地から完全なタイル候補を検出できませんでした。照合は実行していません。"}
        write_json(output / "result.json", result)
        (output / "result.txt").write_text(result["reason"] + "\n", encoding="utf-8")
        print(result["reason"])
        print(f"確認画像：{output / 'query_detection.jpg'}\n--satsuei-pointsで1タイルの四隅を指定できます。")
        return result
    clock = time.perf_counter()
    queries = []
    for index in range(1, len(quads) + 1):
        print(f"撮影特徴抽出 [{index}/{len(quads)}] shoot_tile_{index}", flush=True)
        prepared = extract_features(read_image(output / "query_tiles" / f"shoot_tile_{index}.png"), c)
        queries.append([rotate_features(prepared, turns, c.tile_size) for turns in range(4)])
    extraction_seconds = time.perf_counter() - clock
    clock = time.perf_counter()
    rows, best_by_location = [], {}
    best_overall = None
    total = len(queries) * len(db)
    print(f"撮影候補{len(queries)}タイル × 登録{len(db)}タイル = {total}組（各組で4回転を確認）", flush=True)
    for index, rotations in enumerate(queries, 1):
        for db_index, registered in enumerate(db, 1):
            selected = None
            for turns, query in enumerate(rotations):
                report = compare_tiles(registered, query, c)
                if selected is None or (report["score"], report["inliers"]) > (selected[0]["score"], selected[0]["inliers"]):
                    selected = (report, turns, query)
            report, turns, query = selected
            row = {"query_tile": index, "db_location": registered["location"],
                   "db_tile": registered["tile_index"], "db_name": registered["name"],
                   "rotation_ccw": turns * 90, "score_percent": report["score"] * 100,
                   "success": int(report["success"]), "candidate_matches": report["candidate_matches"],
                   "inliers": report["inliers"], "inlier_percent": report["inlier_fraction"] * 100,
                   "ransac_trials": report["ransac_trials"],
                   "homography_sane": int(report["homography_sane"])}
            rows.append(row)
            location = registered["location"]
            rank = (report["score"], report["inliers"])
            if location not in best_by_location or rank > best_by_location[location]["rank"]:
                best_by_location[location] = {"location": location, "rank": rank, "row": row}
            if best_overall is None or rank > best_overall["rank"]:
                best_overall = {"rank": rank, "registered": registered, "query": query,
                                "report": report, "row": row}
            count = (index - 1) * len(db) + db_index
            if count == 1 or count == total or count % 20 == 0:
                print(f"照合 [{count}/{total}]", flush=True)
    matching_seconds = time.perf_counter() - clock
    ranked = sorted(best_by_location.values(), key=lambda v: (-v["rank"][0], -v["rank"][1], v["location"]))
    first = ranked[0]
    score = first["rank"][0]
    predicted = first["location"] if score > c.threshold else None
    second = ranked[1] if len(ranked) > 1 else None
    write_csv(output / "all_tile_results.csv", rows)
    location_rows = [{"location": v["location"], "score_percent": v["rank"][0] * 100,
                      "query_tile": v["row"]["query_tile"], "db_tile": v["row"]["db_tile"]} for v in ranked]
    write_csv(output / "location_scores.csv", location_rows)
    draw_pair(output, best_overall["registered"], best_overall["query"], best_overall["report"])
    elapsed = time.perf_counter() - started
    result = {"method": "single-tile perspective normalization + spatial Hamming + paper RANSAC + validation",
              "query_image": str(photo), "query_shape_hw": list(image.shape[:2]),
              "query_tiles": len(quads), "database_tiles": len(db), "tile_pair_comparisons": total,
              "orientation_comparisons": total * 4, "prediction": f"db_{predicted:02d}" if predicted is not None else "no_match",
              "location_score_percent": score * 100, "best_pair": first["row"],
              "location_scores": location_rows, "config": asdict(c),
              "expected_db": args.expected_db,
              "correct_identification": (predicted == args.expected_db) if args.expected_db is not None else None,
              "timings": {"db_load_seconds": loaded_seconds, "detection_rectify_seconds": detection_seconds,
                          "query_feature_seconds": extraction_seconds, "matching_seconds": matching_seconds,
                          "elapsed_seconds": elapsed},
              "notes": ["location score is maximum pair score; provisional decision rule",
                        "automatic quadrilaterals are candidates, not guaranteed tile detections",
                        "RANSAC is used for every orientation; BFMatcher is not used",
                        "no evaluation labels are used to choose the result"]}
    write_json(output / "result.json", result)
    lines = ["床指紋：提案手法の地点識別（1タイル単位）", f"撮影画像：{photo}",
             f"撮影タイル候補：{len(quads)}枚 / 登録DB：{len(ranked)}地点・{len(db)}タイル",
             f"総タイル照合数：{total}組 / 回転を含む照合数：{total * 4}",
             f"推定地点：{result['prediction']}", f"地点得点：{score * 100:.2f}%（地点内の最高タイル得点）"]
    if predicted is not None:
        lines.append(f"採用対応：撮影タイル{first['row']['query_tile']} → 地点{predicted}・タイル{first['row']['db_tile']}")
    if second is not None:
        lines.append(f"2位：db_{second['location']:02d} ({second['rank'][0] * 100:.2f}%)")
    if args.expected_db is not None:
        lines.append(f"正解地点：db_{args.expected_db:02d} / 識別：{'正解' if predicted == args.expected_db else '不正解'}")
    lines.extend([f"目地検出・補正：{detection_seconds:.1f}秒", f"撮影特徴抽出：{extraction_seconds:.1f}秒",
                  f"照合時間：{matching_seconds:.1f}秒 / 合計：{elapsed:.1f}秒",
                  "※透視変換・座標近傍照合・RANSACを使用。BFMatcherは使用していません。",
                  f"※判定基準{c.threshold * 100:g}%と最高得点による地点判定は暫定です。",
                  f"保存先：{output}"])
    (output / "result.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    return result


def self_test() -> None:
    """同一タイルの台形撮影・回転・別模様・自動検出の動作を検証。"""
    c = Config(tile_size=256, features_per_tile=450, fast_threshold=5)
    rng = np.random.default_rng(9)
    texture = cv2.GaussianBlur(rng.integers(30, 220, (256, 256), np.uint8), (3, 3), .6)
    tile = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
    p = np.float32([[0, 0], [255, 0], [255, 255], [0, 255]])
    q = np.float32([[90, 50], [430, 85], [390, 430], [60, 380]])
    photo = cv2.warpPerspective(tile, cv2.getPerspectiveTransform(p, q), (500, 500))
    registered = extract_features(normalize_tile(tile, c), c)
    query = extract_features(rectify_tile(photo, q, c), c)
    positive = compare_tiles(registered, query, c)
    assert positive["success"] and positive["inliers"] >= BaseConfig.min_consensus, positive
    different = cv2.cvtColor(cv2.GaussianBlur(
        rng.integers(30, 220, (256, 256), np.uint8), (3, 3), .6), cv2.COLOR_GRAY2BGR)
    negative = compare_tiles(registered, extract_features(normalize_tile(different, c), c), c)
    assert not negative["success"], negative
    rotated = extract_features(np.rot90(normalize_tile(tile, c)).copy(), c)
    scores = [compare_tiles(registered, rotate_features(rotated, n, c.tile_size), c)["score"] for n in range(4)]
    assert max(scores) > c.threshold
    empty = {"points": np.empty((0, 2)), "descriptors": np.empty((0, 32), np.uint8)}
    assert compare_tiles(empty, empty, c)["score"] == 0
    blank = np.full((600, 600, 3), 180, np.uint8)
    assert not detect_tile_quads(blank, c)[0]
    for x in (80, 300, 520):
        cv2.line(blank, (x, 20), (x, 580), (30, 30, 30), 3)
        cv2.line(blank, (20, x), (580, x), (30, 30, 30), 3)
    detected, _, _ = detect_tile_quads(blank, c)
    assert len(detected) == 4, len(detected)
    try:
        require_inside(np.float32([[-1, 0], [40, 0], [40, 40], [0, 40]]), tile.shape)
    except ValueError:
        pass
    else:
        raise AssertionError("画像外の四隅を受け付けました。")
    # 旧DBキャッシュの画像参照から提案DBを作成し、保存・読込・全DB識別も検証。
    with tempfile.TemporaryDirectory(prefix="teian_single_tile_test_") as directory:
        root = Path(directory)
        old = root / "先行DB"
        (old / "db_tiles").mkdir(parents=True)
        save_image(old / "db_tiles" / "db_011.png", tile)
        save_image(old / "db_tiles" / "db_021.png", different)
        write_json(old / "database.json", {"tiles": [
            {"source_id": 1, "tile_index": 1, "image_file": "db_tiles\\db_011.png"},
            {"source_id": 2, "tile_index": 1, "image_file": "db_tiles/db_021.png"}]})
        cache = root / "提案DB"
        build_args = argparse.Namespace(db_cache=str(old), db_dir=str(root / "unused"),
                                        proposal_cache=str(cache))
        build_database(build_args, c)
        loaded, stored = load_database(cache, c)
        assert len(loaded) == 2 and loaded[0]["location"] == 1
        assert np.array_equal(loaded[0]["descriptors"], registered["descriptors"])
        shot = root / "撮影.png"
        save_image(shot, photo)
        point_string = ";".join(f"{x},{y}" for x, y in q)
        runtime = argparse.Namespace(satsuei=str(shot), proposal_cache=str(cache),
            output=str(root / "結果"), satsuei_points=point_string, quad=[], expected_db=1)
        result = identify(runtime, c)
        assert result["prediction"] == "db_01", result
        assert result["orientation_comparisons"] == 8
        assert (root / "結果" / "matches_after_ransac.jpg").is_file()
        # タイルなしと照合不成立を区別し、検出診断を保存する。
        save_image(shot, np.full((500, 500, 3), 180, np.uint8))
        runtime.output = str(root / "目地なし")
        runtime.satsuei_points = None
        missing = identify(runtime, c)
        assert missing["prediction"] == "no_tile_detected"
        assert (root / "目地なし" / "query_detection.jpg").is_file()
    print("自己テスト成功：台形補正、RANSAC、回転、別模様、画像外、DB保存・識別・検出診断。")
    print("※合成データによる動作確認です。実際の床での識別性能は別途検証してください。")


def main() -> None:
    parser = argparse.ArgumentParser(description="床指紋：1タイル透視補正＋座標近傍照合＋RANSAC")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-db", action="store_true", help="提案手法専用DBを事前作成")
    mode.add_argument("--identify", action="store_true", help="撮影写真1枚を全登録タイルと照合")
    mode.add_argument("--detect-only", action="store_true", help="タイル候補検出・補正画像のみ保存")
    mode.add_argument("--self-test", action="store_true", help="合成データで動作確認")
    parser.add_argument("--db-cache", default="gazou/db_cache", help="先行研究DBの分割済み画像を読み込む")
    parser.add_argument("--db-dir", default="gazou/db", help="先行DBがない場合の真上2x2登録画像")
    parser.add_argument("--proposal-cache", default="gazou/teian_db_cache", help="提案手法用DBの保存先・読込先")
    parser.add_argument("--satsuei", help="撮影画像1枚のパス")
    parser.add_argument("--satsuei-points", help="1タイルの四隅 x,y;x,y;x,y;x,y。元画像座標")
    parser.add_argument("--quad", action="append", default=[], help="追加の1タイル四隅。複数回指定可能")
    parser.add_argument("--output", help="新規の結果フォルダ。省略すると日時で作成")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--features-per-tile", type=int, default=2500)
    parser.add_argument("--fast-threshold", type=int, default=5)
    parser.add_argument("--tile-margin", type=float, default=.04)
    parser.add_argument("--spatial-radius", type=float, default=18)
    parser.add_argument("--hamming", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=.06, help="暫定判定基準（割合）")
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tiles", type=int, default=12, help="自動検出の候補上限")
    parser.add_argument("--expected-db", type=int, help="実験用の正解地点。推定には使用しない")
    args = parser.parse_args()
    c = Config(tile_size=args.tile_size, features_per_tile=args.features_per_tile,
               fast_threshold=args.fast_threshold, margin_ratio=args.tile_margin,
               spatial_radius=args.spatial_radius, hamming=args.hamming,
               threshold=args.threshold, trials=args.trials, seed=args.seed, max_tiles=args.max_tiles)
    check_config(c)
    if args.self_test:
        self_test()
    elif args.build_db:
        build_database(args, c)
    elif not args.satsuei:
        raise ValueError("--satsueiに撮影画像1枚を指定してください。")
    elif args.identify:
        identify(args, c)
    else:
        output = resolve_path(args.output) if args.output else BASE_DIR / "kekka" / (
            "teian_detection_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        texts = ([args.satsuei_points] if args.satsuei_points else []) + args.quad
        quads = prepare_query(read_image(resolve_path(args.satsuei)), output, c, texts)
        print(f"タイル候補：{len(quads)}枚\n確認画像：{output / 'query_detection.jpg'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, cv2.error) as exc:
        print(f"\n実行できませんでした：{exc}", file=sys.stderr)
        raise SystemExit(1)
