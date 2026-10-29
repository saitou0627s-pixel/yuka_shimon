r"""床指紋：透視補正・目地領域・特徴点使用数の段階比較。

step3_senko_saigen.py / step4_teian.py と同じ場所へ、この名前で保存する。
既存の先行研究コードと提案手法コードは変更しない。

最初に登録DBを作成：
  python step4_teian_grout.py --build-db --db-dir gazou/db
目地特徴点の使用率を比較：
  python step4_teian_grout.py --sweep --satsuei gazou/satsuei/diagonal_angle_1.jpg
    --satsuei-points "880,652;1205,879;857,1145;554,861"
    --points-size "1536,2048" --expected-db 1 --expected-tile 1

【提案の中心】
目地の四隅から透視補正し、四辺の周囲に幅を持つ目地領域を作る。
画像の画素を消したり外周を切り捨てたりせず、B-ORBで一度だけ抽出した
固定候補点を目地上／タイル面に分け、目地上の点だけ100/75/50/25/0%残す。
タイル面の候補点は変えず、削った点を別の点で補充しない。
登録側と撮影側の両方へ同じ使用率を適用する。
座標近傍Hamming照合・step3のRANSAC・事後の位置合わせ評価を全条件で使う。
BFMatcherは使用しない。四隅の手動指定は補正・抑制処理の検証用。

【目地領域の定義】
検出または手動指定したタイル四辺を目地の中心線とみなし、その両側を
一定幅の帯にする。初期の半幅は正規化後12px（全幅24px）。
これは目地の画素を厳密に分割したものではなく、幾何的な目地領域の仮定。
mask画像を必ず確認し、幅は評価前に決め、使用率比較中は固定する。
特徴点の中心が帯に入るかで分類する。記述子全体の重なりでは分類しない。

【外周除外への対策】
1タイル512pxの周囲に実画像から96px分の文脈を残した704px画像で抽出する。
これにより目地をORBの画像端除外の内側に置く。元画像の外を埋めた画素は
有効領域マスクで識別し、記述子や平均値フィルタが埋め草に触れる点を除く。
元2x2登録画像に写っていない外側目地付近の特徴は復元できない。
旧db_cacheは中央目地を除いているため使わず、元gazou/dbから専用DBを作る。

【評価】
従来と同じ「選択後の全登録特徴点」を分母とする一致割合で地点を選ぶ。
分母が変わる影響を確認するため、削減前の固定候補点数を分母とする値も保存。
6%の判定基準は暫定。割合の最大値だけで最良条件を決めず、地点識別正解率、
異地点の成功判定数、実使用点数、処理時間を別データでも調べる。
時間は共有の特徴抽出と各条件の照合を分ける。キャッシュを読むため、
この比較は主に「照合段階」の時間比較であり抽出高速化を示すものではない。
正解ラベルは採点にだけ使い、タイルや地点の選択には使わない。
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
    from step3_senko_saigen import (
        Config as BaseConfig, balanced_orb, mean_difference_filter,
        read_image, save_image, numbered_images, detect_center_grout,
    )
    from step4_teian import (
        Config as TileConfig, order_quad, parse_quad, require_inside,
        detect_tile_quads, compare_tiles, rotate_features, draw_pair,
    )
except ImportError as exc:
    print("OpenCV・NumPyと、同じフォルダのstep3_senko_saigen.py・最新版step4_teian.pyが必要です。")
    print(exc)
    raise SystemExit(2)

BASE_DIR = Path(__file__).resolve().parent
FORMAT = "floor-grout-fixed-feature-pool-v1"


@dataclass(frozen=True)
class Config:
    tile_size: int = 512
    context: int = 96
    grout_half_width: int = 12
    features_per_tile: int = 2500
    fast_threshold: int = 5
    spatial_radius: float = 18.0
    hamming: int = 100
    threshold: float = .06
    trials: int = 20000
    seed: int = 0
    max_tiles: int = 12
    keep_method: str = "response"

    @property
    def canvas_size(self) -> int:
        return self.tile_size + 2 * self.context


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (BASE_DIR / path).resolve()


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def check_config(c: Config) -> None:
    if c.tile_size < 128 or c.context < 64 or c.features_per_tile < 20:
        raise ValueError("tile-sizeは128以上、contextは64以上、features-per-tileは20以上にしてください。")
    if not 1 <= c.grout_half_width < c.tile_size // 4:
        raise ValueError("grout-half-widthは1以上、タイルの一辺の1/4未満にしてください。")
    if c.fast_threshold < 1 or c.trials < 1 or c.max_tiles < 1:
        raise ValueError("FAST・RANSAC回数・タイル上限は1以上にしてください。")
    if c.spatial_radius <= 0 or not 0 <= c.hamming <= 256 or not 0 <= c.threshold <= 1:
        raise ValueError("座標半径、Hamming距離、判定基準の設定を確認してください。")
    if c.keep_method not in ("response", "random"):
        raise ValueError("使用点の選び方はresponseまたはrandomです。")


def matching_config(c: Config) -> TileConfig:
    return TileConfig(tile_size=c.canvas_size, features_per_tile=c.features_per_tile,
                      fast_threshold=c.fast_threshold, margin_ratio=0,
                      spatial_radius=c.spatial_radius, hamming=c.hamming,
                      threshold=c.threshold, trials=c.trials, seed=c.seed,
                      max_tiles=c.max_tiles)


def grout_mask(c: Config) -> np.ndarray:
    """正規化したタイル四辺の両側を、固定幅の帯として表す。"""
    yy, xx = np.indices((c.canvas_size, c.canvas_size))
    lo, hi = c.context, c.context + c.tile_size - 1
    distance = np.minimum.reduce([abs(xx - lo), abs(xx - hi), abs(yy - lo), abs(yy - hi)])
    return np.uint8(distance <= c.grout_half_width) * 255


def rectify_with_context(image: np.ndarray, quad: np.ndarray, c: Config):
    quad = order_quad(quad)
    require_inside(quad, image.shape)
    lo, hi = c.context, c.context + c.tile_size - 1
    target = np.float32([[lo, lo], [hi, lo], [hi, hi], [lo, hi]])
    matrix = cv2.getPerspectiveTransform(quad, target)
    size = (c.canvas_size, c.canvas_size)
    warped = cv2.warpPerspective(image, matrix, size, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.warpPerspective(np.full(image.shape[:2], 255, np.uint8), matrix, size,
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return warped, valid, matrix


def extract_pool(image: np.ndarray, quad: np.ndarray, c: Config) -> dict:
    warped, valid, matrix = rectify_with_context(image, quad, c)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    filtered = mean_difference_filter(gray, BaseConfig.filter_radius)
    # 文脈部分で候補が消費されるため、面積に応じて抽出上限を増やし、
    # 実タイル内の有効点だけを固定候補集合へ入れる。
    cap = math.ceil(c.features_per_tile * (c.canvas_size / c.tile_size) ** 2)
    base = replace(BaseConfig(), image_scale=1.0, nfeatures=cap,
                   fast_threshold=c.fast_threshold)
    keypoints, descriptors, _ = balanced_orb(filtered, base)
    distance = cv2.distanceTransform((valid > 0).astype(np.uint8), cv2.DIST_L2, 3)
    lo, hi = c.context, c.context + c.tile_size - 1
    ids = []
    rejected_context, rejected_invalid = 0, 0
    for index, kp in enumerate(keypoints):
        x, y = kp.pt
        if not (lo <= x <= hi and lo <= y <= hi):
            rejected_context += 1
            continue
        # 記述子の回転パッチと平均値差分フィルタが画像外補填を見ないように。
        support = .75 * kp.size + base.filter_radius + 2
        if distance[round(y), round(x)] <= support:
            rejected_invalid += 1
            continue
        ids.append(index)
    ids.sort(key=lambda i: (-keypoints[i].response, keypoints[i].pt[1], keypoints[i].pt[0], i))
    ids = np.asarray(ids[:c.features_per_tile], np.int64)
    selected = [keypoints[i] for i in ids]
    points = np.asarray([kp.pt for kp in selected], np.float64).reshape(-1, 2)
    responses = np.asarray([kp.response for kp in selected], np.float64)
    sizes = np.asarray([kp.size for kp in selected], np.float64)
    band = grout_mask(c)
    xy = np.rint(points).astype(int)
    on_grout = (band[xy[:, 1], xy[:, 0]] > 0) if len(xy) else np.zeros(0, bool)
    desc = descriptors[ids].reshape(-1, 32)
    return {"image": warped, "filtered": filtered, "valid": valid, "grout_mask": band,
            "matrix": matrix, "quad": order_quad(quad), "points": points,
            "descriptors": desc, "responses": responses, "sizes": sizes,
            "on_grout": on_grout, "pool_count": len(points),
            "rejected_context": rejected_context, "rejected_invalid": rejected_invalid}


def selection_indices(pool: dict, keep_percent: float, c: Config) -> np.ndarray:
    """目地外は全点固定。目地内だけ同じ順位の先頭を採用し、入れ子にする。"""
    grout = np.flatnonzero(pool["on_grout"])
    surface = np.flatnonzero(~pool["on_grout"])
    if c.keep_method == "random":
        order = np.random.default_rng(c.seed).permutation(grout)
    else:
        order = grout[np.argsort(-pool["responses"][grout], kind="stable")]
    count = math.floor(len(grout) * keep_percent / 100 + .5)
    return np.sort(np.concatenate((surface, order[:count]))).astype(np.int64)


def choose_features(pool: dict, keep_percent: float, c: Config) -> dict:
    ids = selection_indices(pool, keep_percent, c)
    n_grout = int(pool["on_grout"][ids].sum())
    return {**pool, "points": pool["points"][ids], "descriptors": pool["descriptors"][ids],
            "selected_indices": ids, "selected_grout": n_grout,
            "selected_surface": len(ids) - n_grout, "selected_total": len(ids)}


def draw_selection(pool: dict, keep_percent: float, c: Config) -> np.ndarray:
    selected = np.zeros(len(pool["points"]), bool)
    selected[selection_indices(pool, keep_percent, c)] = True
    image = pool["image"].copy()
    for i, point in enumerate(pool["points"]):
        color = (90, 90, 90) if not selected[i] else ((0, 150, 255) if pool["on_grout"][i] else (0, 235, 0))
        cv2.circle(image, tuple(np.rint(point).astype(int)), 2 if selected[i] else 1,
                   color, -1, cv2.LINE_AA)
    lo, hi = c.context, c.context + c.tile_size - 1
    cv2.rectangle(image, (lo, lo), (hi, hi), (255, 255, 0), 1)
    cv2.rectangle(image, (0, 0), (c.canvas_size - 1, 66), (25, 25, 25), -1)
    n_grout = int(np.sum(selected & pool["on_grout"]))
    surface = int(np.sum(~pool["on_grout"]))
    for y, text in [(25, f"Grout kept: {keep_percent:g}%  ({n_grout}/{int(pool['on_grout'].sum())})"),
                    (53, f"Surface kept: {surface}  |  Orange=grout  Green=surface  Gray=removed")]:
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def save_pool_visuals(folder: Path, pool: dict, c: Config) -> None:
    folder.mkdir(parents=True, exist_ok=False)
    save_image(folder / "01_rectified_with_context.png", pool["image"])
    save_image(folder / "02_valid_mask.png", pool["valid"])
    mask = cv2.bitwise_and(pool["grout_mask"], pool["valid"])
    save_image(folder / "03_grout_mask.png", mask)
    overlay = pool["image"].copy()
    tint = np.full_like(overlay, (0, 0, 255))
    blend = cv2.addWeighted(overlay, .65, tint, .35, 0)
    overlay[mask > 0] = blend[mask > 0]
    lo, hi = c.context, c.context + c.tile_size - 1
    cv2.rectangle(overlay, (lo, lo), (hi, hi), (255, 255, 0), 1)
    save_image(folder / "04_grout_overlay.jpg", overlay)
    save_image(folder / "05_all_candidate_features.jpg", draw_selection(pool, 100, c))
    save_image(folder / "06_filtered.png", pool["filtered"])


def pool_stats(pool: dict) -> dict:
    return {"pool_total": len(pool["points"]), "pool_grout": int(pool["on_grout"].sum()),
            "pool_surface": int((~pool["on_grout"]).sum()),
            "rejected_outside_tile": pool.get("rejected_context", 0),
            "rejected_near_invalid_pixels": pool.get("rejected_invalid", 0)}


def registration_quads(image: np.ndarray, mode: str):
    h, w = image.shape[:2]
    if mode == "center":
        x, y = w // 2, h // 2
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x, _ = detect_center_grout(gray, "vertical")
        y, _ = detect_center_grout(gray, "horizontal")
    quads = [np.float32(p) for p in [
        [[0, 0], [x, 0], [x, y], [0, y]],
        [[x, 0], [w - 1, 0], [w - 1, y], [x, y]],
        [[0, y], [x, y], [x, h - 1], [0, h - 1]],
        [[x, y], [w - 1, y], [w - 1, h - 1], [x, h - 1]]]]
    return quads, {"split_x": x, "split_y": y, "mode": mode}


def build_database(args, c: Config) -> dict:
    images = numbered_images(resolve_path(args.db_dir), "db_")
    output = resolve_path(args.grout_cache)
    output.mkdir(parents=True, exist_ok=False)
    (output / "tiles").mkdir()
    (output / "features").mkdir()
    (output / "registration_previews").mkdir()
    rows, clock = [], time.perf_counter()
    for location, path in images:
        image = read_image(path)
        quads, split = registration_quads(image, args.split_mode)
        overlay = image.copy()
        for tile_index, quad in enumerate(quads, 1):
            name = f"db_{location:02d}_tile_{tile_index}"
            print(f"登録候補抽出 [{len(rows) + 1}/{len(images) * 4}] {name}", flush=True)
            pool = extract_pool(image, quad, c)
            feature_name = f"features/{name}.npz"
            np.savez_compressed(output / feature_name, points=pool["points"],
                descriptors=pool["descriptors"], responses=pool["responses"], sizes=pool["sizes"],
                on_grout=pool["on_grout"], quad=quad, matrix=pool["matrix"])
            folder = output / "tiles" / name
            save_pool_visuals(folder, pool, c)
            rows.append({"location": location, "tile_index": tile_index, "name": name,
                         "feature_file": feature_name,
                         "image_file": f"tiles/{name}/01_rectified_with_context.png",
                         "source_image": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         **pool_stats(pool)})
            cv2.polylines(overlay, [np.rint(quad).astype(np.int32)], True, (0, 220, 0), 2)
            xy = tuple(np.rint(quad.mean(axis=0)).astype(int))
            cv2.putText(overlay, f"{location}-{tile_index}", xy, cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2)
        save_image(output / "registration_previews" / f"db_{location:02d}.jpg", overlay)
        write_json(output / "registration_previews" / f"db_{location:02d}.json", split)
    data = {"format": FORMAT, "created_at": datetime.now().isoformat(), "config": asdict(c),
            "opencv_version": cv2.__version__, "numpy_version": np.__version__,
            "python_version": sys.version.split()[0],
            "tiles": rows, "elapsed_seconds": time.perf_counter() - clock,
            "notes": "pool extracted once with real neighboring image context; no feature refill"}
    write_json(output / "database.json", data)
    write_csv(output / "feature_pool_counts.csv", rows)
    print(f"\n目地比較用DB作成完了：{len(images)}地点・{len(rows)}タイル")
    print(f"目地上候補点：合計{sum(row['pool_grout'] for row in rows)}点")
    print(f"処理時間：{data['elapsed_seconds']:.1f}秒\n保存先：{output}")
    return data


def load_database(path: Path, runtime: Config):
    metadata_path = path / "database.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"目地比較用DBがありません：{metadata_path}\n先に --build-db を実行してください。")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != FORMAT:
        raise ValueError("旧DBは使えません。元のgazou/dbから目地を残したDBを作成してください。")
    c = replace(Config(**metadata["config"]), spatial_radius=runtime.spatial_radius,
                hamming=runtime.hamming, threshold=runtime.threshold, trials=runtime.trials,
                seed=runtime.seed, max_tiles=runtime.max_tiles, keep_method=runtime.keep_method)
    pools = []
    for row in metadata["tiles"]:
        with np.load(path / row["feature_file"], allow_pickle=False) as data:
            values = {key: data[key].copy() for key in data.files}
        n = len(values["points"])
        if (values["points"].shape != (n, 2) or values["descriptors"].shape != (n, 32)
                or values["on_grout"].shape != (n,) or values["responses"].shape != (n,)):
            raise ValueError(f"特徴量キャッシュが不正です：{row['name']}")
        pools.append({**row, **values, "image_path": path / row["image_file"],
                      "pool_count": n, "on_grout": values["on_grout"].astype(bool)})
    if not pools:
        raise ValueError("登録タイルがありません。")
    return pools, c


def parse_points_size(text: str | None):
    if text is None:
        return None
    fields = re.split(r"[,xX]", text)
    if len(fields) != 2:
        raise ValueError("points-sizeは幅,高さ（例：1536,2048）で指定してください。")
    size = np.asarray([float(v) for v in fields])
    if not np.isfinite(size).all() or (size <= 1).any():
        raise ValueError("points-sizeの幅と高さは1より大きくしてください。")
    return size


def query_quads(image: np.ndarray, args, c: Config):
    texts = ([args.satsuei_points] if args.satsuei_points else []) + args.quad
    if texts:
        quads = [parse_quad(text) for text in texts]
        reference = parse_points_size(args.points_size)
        if reference is not None:
            scale = np.array([image.shape[1], image.shape[0]]) / reference
            quads = [np.float32(q * scale) for q in quads]
        for quad in quads:
            require_inside(quad, image.shape)
        return quads, None, {"mode": "manual", "reference_size_wh": reference.tolist() if reference is not None else None}
    quads, edges, details = detect_tile_quads(image, matching_config(c))
    return quads, edges, details


def prepare_queries(args, c: Config, output: Path):
    image = read_image(resolve_path(args.satsuei))
    quads, edges, details = query_quads(image, args, c)
    output.mkdir(parents=True, exist_ok=False)
    overlay = image.copy()
    for index, quad in enumerate(quads, 1):
        cv2.polylines(overlay, [np.rint(quad).astype(np.int32)], True, (0, 220, 0), 4)
        cv2.putText(overlay, f"tile {index}", tuple(np.rint(quad.mean(axis=0)).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(.6, max(image.shape[:2]) / 1800), (0, 0, 255), 3)
    save_image(output / "query_detection.jpg", overlay)
    if edges is not None:
        save_image(output / "query_edges.png", edges)
    write_json(output / "query_quads.json", {"source_shape_hw": list(image.shape[:2]),
               "quads": [q.tolist() for q in quads], "detection": details})
    if not quads:
        reason = "タイル候補を検出できませんでした。照合は実行していません。四隅指定または検出改善が必要です。"
        (output / "result.txt").write_text(reason + "\n", encoding="utf-8")
        raise ValueError(reason + f"\n診断画像：{output / 'query_detection.jpg'}")
    pools = []
    for index, quad in enumerate(quads, 1):
        print(f"撮影候補抽出 [{index}/{len(quads)}]", flush=True)
        pool = extract_pool(image, quad, c)
        save_pool_visuals(output / f"query_{index:02d}", pool, c)
        pool["query_index"] = index
        pools.append(pool)
    write_csv(output / "query_pool_counts.csv", [{"query_index": i + 1, **pool_stats(p)} for i, p in enumerate(pools)])
    return pools, details


def public_row(query_index, registered, report, turns, selected, query_selected) -> dict:
    good = report["validation"]["hamming_le_100"]
    pool_count = registered["pool_count"]
    return {"query_tile": query_index, "db_location": registered["location"],
            "db_tile": registered["tile_index"], "db_name": registered["name"],
            "rotation_ccw": turns * 90, "score_percent": report["score"] * 100,
            "fixed_pool_score_percent": good / pool_count * 100 if pool_count else 0.,
            "success": int(report["success"]), "validation_good": good,
            "selected_denominator": report["validation"]["denominator"], "fixed_denominator": pool_count,
            "registered_used_grout": selected["selected_grout"],
            "registered_used_surface": selected["selected_surface"],
            "query_used_grout": query_selected["selected_grout"],
            "query_used_surface": query_selected["selected_surface"],
            "candidate_matches": report["candidate_matches"], "inliers": report["inliers"],
            "inlier_percent": report["inlier_fraction"] * 100,
            "ransac_trials": report["ransac_trials"], "homography_sane": int(report["homography_sane"])}


def run_condition(db_pools, query_pools, keep: float, c: Config, output: Path, args):
    output.mkdir(parents=True, exist_ok=False)
    clock = time.perf_counter()
    selected_db = [choose_features(p, keep, c) for p in db_pools]
    selected_queries = [choose_features(p, keep, c) for p in query_pools]
    rotations = [[rotate_features(p, turn, c.canvas_size) for turn in range(4)] for p in selected_queries]
    selection_seconds = time.perf_counter() - clock
    clock = time.perf_counter()
    rows, best, by_location = [], None, {}
    for qi, query_rotations in enumerate(rotations, 1):
        for di, registered in enumerate(selected_db):
            selected = None
            for turns, query in enumerate(query_rotations):
                report = compare_tiles(registered, query, matching_config(c))
                rank = (report["score"], report["inliers"])
                if selected is None or rank > selected["rank"]:
                    selected = {"rank": rank, "report": report, "query": query, "turns": turns}
            row = public_row(qi, db_pools[di], selected["report"], selected["turns"],
                             registered, selected_queries[qi - 1])
            rows.append(row)
            location = registered["location"]
            if location not in by_location or selected["rank"] > by_location[location]["rank"]:
                by_location[location] = {"rank": selected["rank"], "row": row}
            if best is None or selected["rank"] > best["rank"]:
                best = {**selected, "registered": registered, "db_pool": db_pools[di], "row": row}
            count = (qi - 1) * len(selected_db) + di + 1
            if count == 1 or count % 20 == 0 or count == len(selected_db) * len(rotations):
                print(f"  目地使用率{keep:g}%：照合 [{count}/{len(selected_db) * len(rotations)}]", flush=True)
    matching_seconds = time.perf_counter() - clock
    ranked = sorted(by_location.items(), key=lambda v: (-v[1]["rank"][0], -v[1]["rank"][1], v[0]))
    top_location, top = ranked[0]
    predicted = top_location if top["rank"][0] > c.threshold else None
    second_score = ranked[1][1]["rank"][0] * 100 if len(ranked) > 1 else None
    # 正解はここから採点だけに使用。推定・切り出し・点選択には使用しない。
    other_false = (sum(row["success"] for row in rows if row["db_location"] != args.expected_db)
                   if args.expected_db is not None else None)
    truth_rows = [row for row in rows if row["db_location"] == args.expected_db]
    correct_score = max((row["score_percent"] for row in truth_rows), default=None)
    feature_rows = []
    for original, selected in zip(db_pools, selected_db):
        feature_rows.append({"side": "db", "name": original["name"], **pool_stats(original),
                             "used_total": selected["selected_total"], "used_grout": selected["selected_grout"],
                             "used_surface": selected["selected_surface"]})
    for qi, (original, selected) in enumerate(zip(query_pools, selected_queries), 1):
        feature_rows.append({"side": "query", "name": f"query_{qi:02d}", **pool_stats(original),
                             "used_total": selected["selected_total"], "used_grout": selected["selected_grout"],
                             "used_surface": selected["selected_surface"]})
    summary = {"grout_keep_percent": keep,
               "predicted_location": f"db_{predicted:02d}" if predicted is not None else "no_match",
               "best_candidate_location": f"db_{top_location:02d}",
               "best_db_tile": top["row"]["db_tile"], "best_query_tile": top["row"]["query_tile"],
               "score_percent": top["row"]["score_percent"],
               "fixed_pool_score_percent": top["row"]["fixed_pool_score_percent"],
               "second_score_percent": second_score,
               "expected_db": args.expected_db, "expected_tile": args.expected_tile,
               "correct_identification": int(predicted == args.expected_db) if args.expected_db is not None else None,
               "correct_tile_identification": (int(predicted == args.expected_db and top["row"]["db_tile"] == args.expected_tile)
                                               if args.expected_db is not None and args.expected_tile is not None else None),
               "correct_location_best_score_percent": correct_score,
               "false_matches_other_location_tile_pairs": other_false,
               "db_features_total": sum(p["selected_total"] for p in selected_db),
               "db_grout_features_total": sum(p["selected_grout"] for p in selected_db),
               "query_features_total": sum(p["selected_total"] for p in selected_queries),
               "query_grout_features_total": sum(p["selected_grout"] for p in selected_queries),
               "query_surface_features_total": sum(p["selected_surface"] for p in selected_queries),
               "tile_pair_comparisons": len(rows), "orientation_comparisons": len(rows) * 4,
               "selection_seconds": selection_seconds, "matching_seconds": matching_seconds}
    write_csv(output / "all_tile_results.csv", rows)
    write_csv(output / "feature_counts.csv", feature_rows)
    write_csv(output / "location_scores.csv", [{"location": loc, **data["row"]} for loc, data in ranked])
    draw_pair(output, best["registered"], best["query"], best["report"])
    for qi, pool in enumerate(query_pools, 1):
        save_image(output / f"query_{qi:02d}_features.jpg", draw_selection(pool, keep, c))
    db_image_pool = {**best["db_pool"], "image": read_image(best["db_pool"]["image_path"])}
    save_image(output / "best_db_features.jpg", draw_selection(db_image_pool, keep, c))
    write_json(output / "result.json", {"summary": summary, "best_pair": top["row"], "config": asdict(c)})
    lines = [f"目地上の特徴点使用率：{keep:g}%", f"推定地点：{summary['predicted_location']}",
             f"選択後の登録特徴点を分母とする一致割合：{summary['score_percent']:.2f}%",
             f"削減前の固定候補点を分母とする一致割合：{summary['fixed_pool_score_percent']:.2f}%",
             f"撮影側使用点数：目地{summary['query_grout_features_total']} / タイル面{summary['query_surface_features_total']}",
             f"照合時間：{matching_seconds:.2f}秒（共有の特徴抽出・画像保存を含まない）"]
    if args.expected_db is not None:
        lines.append(f"正解地点：db_{args.expected_db:02d} / 識別：{'正解' if summary['correct_identification'] else '不正解'}")
        lines.append(f"異地点タイルとの成功判定数：{other_false}")
    lines.extend(["透視補正・座標近傍Hamming照合・RANSAC・事後評価を全条件で使用。",
                  "同じ候補点から目地上の点だけ減らし、タイル面の点は固定。削減分の補充なし。",
                  f"判定基準{c.threshold * 100:g}%は暫定。一致割合は識別正解率ではありません。"])
    (output / "result.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    return summary


def parse_rates(value: str) -> list[float]:
    try:
        rates = [float(v) for v in value.split(",")]
    except ValueError as exc:
        raise ValueError("keep-ratesは100,75,50,25,0のように指定してください。") from exc
    if not rates or any(not math.isfinite(r) or not 0 <= r <= 100 for r in rates):
        raise ValueError("目地特徴点の使用率は0～100%で指定してください。")
    if len(set(rates)) != len(rates):
        raise ValueError("keep-ratesに同じ割合が重複しています。")
    return sorted(rates, reverse=True)


def contact_sheet(output: Path, rates: list[float], query_pools, c: Config):
    for qi, pool in enumerate(query_pools, 1):
        width = 350
        pictures = [cv2.resize(draw_selection(pool, rate, c), (width, width), interpolation=cv2.INTER_AREA)
                    for rate in rates]
        save_image(output / f"feature_comparison_query_{qi:02d}.jpg", np.concatenate(pictures, axis=1))


def japanese_summary(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        correct = row["correct_identification"]
        result.append({
            "目地上の特徴点使用率（％）": row["grout_keep_percent"],
            "推定地点": row["predicted_location"],
            "地点識別の正誤": "未採点" if correct is None else ("正解" if correct else "不正解"),
            "採用タイル番号": row["best_db_tile"],
            "選択後特徴点を分母とする一致割合（％）": row["score_percent"],
            "固定候補点を分母とする一致割合（％）": row["fixed_pool_score_percent"],
            "正解地点の最高一致割合（％）": row["correct_location_best_score_percent"],
            "異地点タイルとの成功判定数": row["false_matches_other_location_tile_pairs"],
            "撮影側で使用した目地上の点数": row["query_grout_features_total"],
            "撮影側で使用したタイル面の点数": row["query_surface_features_total"],
            "登録側で使用した目地上の点数（合計）": row["db_grout_features_total"],
            "照合時間（秒）": row["matching_seconds"],
        })
    return result


def run_sweep(args, runtime: Config) -> list[dict]:
    rates = parse_rates(args.keep_rates)
    started = time.perf_counter()
    db_pools, c = load_database(resolve_path(args.grout_cache), runtime)
    db_load_seconds = time.perf_counter() - started
    clock = time.perf_counter()
    output = resolve_path(args.output) if args.output else BASE_DIR / "kekka" / (
        "step4_grout_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    query_pools, detection = prepare_queries(args, c, output)
    query_preprocess_seconds = time.perf_counter() - clock
    warnings = []
    if not sum(int(p["on_grout"].sum()) for p in query_pools):
        warnings.append("撮影側の目地上候補点が0です。使用率を変えても撮影点は変わりません。目地マスクを確認してください。")
    if not sum(int(p["on_grout"].sum()) for p in db_pools):
        warnings.append("登録側の目地上候補点が0です。目地を含む元画像とマスクを確認してください。")
    for warning in warnings:
        print("注意：" + warning, flush=True)
    results = []
    for rate in rates:
        suffix = f"{rate:06.2f}".rstrip("0").rstrip(".").replace(".", "p")
        result = run_condition(db_pools, query_pools, rate, c, output / f"keep_{suffix}", args)
        results.append(result)
        write_csv(output / "sweep_summary.csv", results)
        write_csv(output / "comparison_summary.csv", japanese_summary(results))
    contact_sheet(output, rates, query_pools, c)
    elapsed = time.perf_counter() - started
    write_json(output / "experiment.json", {"config": asdict(c), "rates": rates,
        "source_image": str(resolve_path(args.satsuei)), "detection": detection,
        "source_sha256": hashlib.sha256(resolve_path(args.satsuei).read_bytes()).hexdigest(),
        "opencv_version": cv2.__version__, "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "db_load_seconds_shared": db_load_seconds,
        "query_preprocess_seconds_shared": query_preprocess_seconds,
        "elapsed_seconds_total": elapsed, "conditions": results, "warnings": warnings,
        "selection_definition": "grout center within fixed-width band; nested subset, surface fixed, no refill",
        "important": "manual corners are a controlled experiment; automatic grout detection is not validated"})
    lines = ["床指紋：目地上特徴点の使用率比較", f"撮影画像：{args.satsuei}",
             f"四隅指定：{detection['mode']}", f"目地領域：四辺から両側{c.grout_half_width}px",
             f"共有のDB読込：{db_load_seconds:.2f}秒 / 撮影補正・候補抽出・診断保存：{query_preprocess_seconds:.2f}秒",
             "使用率 / 推定地点 / 選択点分母の一致割合 / 固定点分母の一致割合 / 撮影目地点数 / 照合秒"]
    for row in results:
        lines.append(f"{row['grout_keep_percent']:g}% / {row['predicted_location']} / {row['score_percent']:.2f}% / "
                     f"{row['fixed_pool_score_percent']:.2f}% / {row['query_grout_features_total']}点 / {row['matching_seconds']:.2f}秒")
    lines.extend(warnings)
    lines.extend(["図の色：緑=使用するタイル面特徴点、橙=使用する目地特徴点、灰=除いた目地特徴点。",
                  "赤い帯=幾何的に作った目地領域。水色の枠=1タイルの境界。",
                  "主な比較は地点識別の正誤・異地点の成功判定数・使用点数・時間。最良割合は別データでも評価する。",
                  f"全体処理時間：{elapsed:.2f}秒", f"保存先：{output}"])
    (output / "sweep_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    return results


def self_test() -> None:
    c = Config(tile_size=256, context=80, grout_half_width=14, features_per_tile=600)
    # 削減以外の点が変わらないこと、入れ子関係と端数処理を直接確認。
    pool = {"points": np.column_stack((np.arange(12), np.arange(12))).astype(float),
            "descriptors": np.zeros((12, 32), np.uint8), "responses": np.arange(12)[::-1],
            "on_grout": np.array([True] * 8 + [False] * 4)}
    previous = set(range(12))
    for rate, expected in [(100, 8), (75, 6), (50, 4), (25, 2), (0, 0)]:
        chosen = choose_features(pool, rate, c)
        assert chosen["selected_grout"] == expected and chosen["selected_surface"] == 4
        assert set(chosen["selected_indices"]) <= previous
        previous = set(chosen["selected_indices"])
    assert parse_rates("0,100,50") == [100, 50, 0]
    rng = np.random.default_rng(42)
    gray = cv2.GaussianBlur(rng.integers(35, 220, (800, 800), np.uint8), (3, 3), .6)
    for axis in (150, 400, 650):
        cv2.line(gray, (axis, 0), (axis, 799), 30, 5)
        cv2.line(gray, (0, axis), (799, axis), 30, 5)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    quad = np.float32([[150, 150], [400, 150], [400, 400], [150, 400]])
    full = extract_pool(image, quad, c)
    assert full["on_grout"].any(), "文脈付き抽出でも目地上候補点がありません。"
    assert (~full["on_grout"]).any()
    lo, hi = c.context, c.context + c.tile_size - 1
    assert np.all((full["points"] >= lo) & (full["points"] <= hi))
    a = choose_features(full, 100, c)
    rep = compare_tiles(a, a, matching_config(c))
    assert rep["success"] and rep["inliers"] >= BaseConfig.min_consensus
    none = choose_features(full, 0, c)
    assert none["selected_grout"] == 0 and none["selected_surface"] == a["selected_surface"]
    # 四隅が画像端でも補填画素を参照する特徴点を使わない。
    edge = extract_pool(image, np.float32([[0, 0], [400, 0], [400, 400], [0, 400]]), c)
    assert edge["rejected_invalid"] > 0
    with tempfile.TemporaryDirectory(prefix="floor_grout_test_") as directory:
        root = Path(directory)
        db_dir = root / "登録"
        db_dir.mkdir()
        save_image(db_dir / "db_01.png", image)
        cache = root / "目地DB"
        args = argparse.Namespace(db_dir=str(db_dir), grout_cache=str(cache), split_mode="center")
        build_database(args, c)
        loaded, stored = load_database(cache, c)
        assert len(loaded) == 4 and stored.context == c.context
        assert sum(int(p["on_grout"].sum()) for p in loaded) > 0
        # 登録した左上タイルを指定。ラベル違いが推定を変えないことも確認。
        source_quad = np.float32([[0, 0], [400, 0], [400, 400], [0, 400]])
        qpool = extract_pool(image, source_quad, c)
        qpool["query_index"] = 1
        args.expected_db, args.expected_tile = 1, 1
        first = run_condition(loaded, [qpool], 100, c, root / "使用100", args)
        assert first["predicted_location"] == "db_01" and first["correct_tile_identification"] == 1
        args.expected_db = 999
        second = run_condition(loaded, [qpool], 100, c, root / "ラベルだけ変更", args)
        assert first["predicted_location"] == second["predicted_location"]
        assert first["score_percent"] == second["score_percent"] and second["correct_identification"] == 0
        assert (root / "使用100" / "matches_after_ransac.jpg").is_file()
    print("自己テスト成功：目地候補の保持、段階削減、タイル面固定、RANSAC、DB保存、正解ラベルの独立性。")
    print("※実画像での精度評価や目地自動検出の保証ではありません。")


def main() -> None:
    parser = argparse.ArgumentParser(description="透視補正＋目地領域＋特徴点使用率比較＋RANSAC")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-db", action="store_true")
    mode.add_argument("--sweep", action="store_true", help="目地使用率を段階比較")
    mode.add_argument("--detect-only", action="store_true", help="補正・目地マスク・候補点だけ保存")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--db-dir", default="gazou/db", help="目地を除いていない元2x2登録画像")
    parser.add_argument("--grout-cache", default="gazou/teian_grout_cache")
    parser.add_argument("--split-mode", choices=("auto", "center"), default="auto")
    parser.add_argument("--satsuei", help="撮影画像1枚")
    parser.add_argument("--satsuei-points", help="1タイルの四隅 x,y;x,y;x,y;x,y")
    parser.add_argument("--quad", action="append", default=[], help="追加のタイル四隅。複数回指定可能")
    parser.add_argument("--points-size", help="指定座標を読み取った画像の幅,高さ。自動で実画像サイズへ換算")
    parser.add_argument("--keep-rates", default="100,75,50,25,0", help="目地上の点を残す割合（%%）")
    parser.add_argument("--keep-method", choices=("response", "random"), default="response")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--context", type=int, default=96, help="目地を画像端から離す実画像の周辺領域")
    parser.add_argument("--grout-half-width", type=int, default=12, help="正規化後の目地帯の半幅")
    parser.add_argument("--features-per-tile", type=int, default=2500)
    parser.add_argument("--fast-threshold", type=int, default=5)
    parser.add_argument("--spatial-radius", type=float, default=18)
    parser.add_argument("--hamming", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=.06)
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tiles", type=int, default=12)
    parser.add_argument("--expected-db", type=int, help="採点専用の正解地点。推定には使用しない")
    parser.add_argument("--expected-tile", type=int, choices=(1, 2, 3, 4), help="採点専用の正解タイル番号")
    parser.add_argument("--output", help="新規の結果フォルダ。省略時は日時付き")
    args = parser.parse_args()
    c = Config(tile_size=args.tile_size, context=args.context, grout_half_width=args.grout_half_width,
               features_per_tile=args.features_per_tile, fast_threshold=args.fast_threshold,
               spatial_radius=args.spatial_radius, hamming=args.hamming, threshold=args.threshold,
               trials=args.trials, seed=args.seed, max_tiles=args.max_tiles, keep_method=args.keep_method)
    check_config(c)
    if args.self_test:
        self_test()
    elif args.build_db:
        build_database(args, c)
    else:
        if not args.satsuei:
            raise ValueError("--satsueiに撮影画像を指定してください。")
        if args.expected_tile is not None and args.expected_db is None:
            raise ValueError("expected-tileを使う場合はexpected-dbも指定してください。")
        if args.sweep:
            run_sweep(args, c)
        else:
            output = resolve_path(args.output) if args.output else BASE_DIR / "kekka" / (
                "step4_grout_check_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            pools, _ = prepare_queries(args, c, output)
            for i, pool in enumerate(pools, 1):
                print(f"撮影タイル{i}：{pool_stats(pool)}")
            print(f"保存先：{output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, cv2.error) as exc:
        print(f"\n実行できませんでした：{exc}", file=sys.stderr)
        raise SystemExit(1)
