r"""目地線・交点から1タイルを自動検出する実験用コード。

既存の step4_teian_grout.py は変更しない。
--run-sweep を付けると、自動検出した四隅を同ファイルへ渡して実行する。
したがって既存の透視補正、目地特徴点の使用率比較、RANSACは維持される。
検出単独ではDBや照合結果を参照しない（正解地点による候補選択をしない）。

処理：縮小→暗い帯の強調→Hough/LSD線分→暗さ・連続性評価→消失点2系統
→線の交点→四辺で目地を確認→候補の重複除去→幾何・画像情報で順位付け。
--clahe は縮小後の検出用グレー画像だけに局所コントラスト補正を行う。
既定はオフ。線分・四辺の評価には同じ補正画像を使い、照合へ渡す元画像は変えない。
DB作成側も利用する extract_lines 自体には補正を追加しない。
消失点が無限遠になる、ほぼ平行な線も同次座標で扱う。
これは床の意味的セグメンテーションではない。影・模様も誤検出し得る。
目地が周囲より暗い床を対象とする。設定値は論文由来ではない実験値。

使用例（VS CodeのPowerShell、yuka_shimon内）：
 .\.venv\Scripts\python.exe -u .\step4_auto_grout.py --satsuei "gazou\satsuei\diagonal_angle_1.jpg"
 .\.venv\Scripts\python.exe -u .\step4_auto_grout.py --satsuei "gazou\satsuei\diagonal_angle_1.jpg" --run-sweep --expected-db 1
 .\.venv\Scripts\python.exe .\step4_auto_grout.py --self-test
 .\.venv\Scripts\python.exe -u .\step4_auto_grout.py --satsuei "gazou\satsuei\night_5_1.jpg" --clahe

依存：opencv-python, numpy。検出単独ではstep3/step4のimportは不要。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import combinations
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    max_side: int = 1280
    min_length: float = 0.10  # 縮小画像の短辺に対する比
    dark_difference: float = 3.0  # 周囲との差（輝度0～255）
    line_support: float = 0.42
    side_support: float = 0.40
    angle_tolerance: float = 6.0
    min_area: float = 0.008
    max_area: float = 0.65
    max_candidates: int = 6
    seed: int = 0
    clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_grid: int = 8  # CLAHEの画像分割数（床のタイル数ではない）


def path_of(value: str) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (BASE / p).resolve()


def read_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"画像がありません：{path}")
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"画像を読み込めません：{path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"画像の保存に失敗：{path}")
    encoded.tofile(str(path))


def sample(gray: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return cv2.remap(gray, xy[:, 0].astype(np.float32).reshape(1, -1),
                     xy[:, 1].astype(np.float32).reshape(1, -1),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101).ravel()


def band_support(gray: np.ndarray, p: np.ndarray, q: np.ndarray,
                 difference: float, search: bool = False) -> dict:
    """中心帯が左右両側より暗い割合を計測。片側だけ暗い影境界を抑える。

    1本の線につき帯幅・中心ずれは一定とし、点ごとの都合よい選択はしない。
    """
    vector = q - p
    length = float(np.linalg.norm(vector))
    normal = np.array([-vector[1], vector[0]]) / max(length, 1e-9)
    n = max(24, min(240, round(length)))
    t = np.linspace(0.06, 0.94, n)
    xy = p[None] + t[:, None] * vector
    best = None
    for offset in ((-3., -1.5, 0., 1.5, 3.) if search else (0.,)):
        centered = xy + offset * normal
        center = (sample(gray, centered) + sample(gray, centered + normal)
                  + sample(gray, centered - normal)) / 3
        for width in (3., 5., 8., 12.):
            left = sample(gray, centered + width * normal)
            right = sample(gray, centered - width * normal)
            contrast = np.minimum(left, right) - center
            hits = contrast >= difference
            support = float(hits.mean())
            # 長さ方向を6区画に分け、局所的な汚れだけで高得点にしない。
            continuity = float(np.mean([part.mean() >= 0.25
                                       for part in np.array_split(hits, 6)]))
            strength = float(np.mean(np.clip(contrast, 0, 30)))
            score = support * 0.7 + continuity * 0.2 + min(strength / 20, 1) * 0.1
            record = dict(score=score, support=support, continuity=continuity,
                          strength=strength, offset=offset, width=width,
                          p=p + offset * normal, q=q + offset * normal)
            if best is None or score > best["score"]:
                best = record
    return best


def line_of(p, q):
    line = np.cross(np.r_[p, 1.], np.r_[q, 1.])
    return line / max(np.linalg.norm(line[:2]), 1e-12)


def extract_lines(gray: np.ndarray, c: Config):
    """複数幅のblack-hatで暗い帯を強調し、Hough線分を得る。"""
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0.6)
    byte = np.uint8(np.clip(smooth, 0, 255))
    blackhat = np.maximum.reduce([
        cv2.morphologyEx(byte, cv2.MORPH_BLACKHAT,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        for k in (9, 17, 27)])
    mask = np.uint8(blackhat >= max(4, c.dark_difference)) * 255
    extent = min(gray.shape)
    minimum = max(30, round(extent * c.min_length))
    # black-hatの二値画像にはタイル面の細かな模様も多い。
    # HoughはCannyエッジに適用し、LSDの長い線分も併用する。
    edges = cv2.Canny(byte, 25, 70, L2gradient=True)
    raw = cv2.HoughLinesP(edges, 1, np.pi / 720, max(25, minimum // 2),
                          minLineLength=minimum, maxLineGap=max(4, extent // 80))
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(byte)[0]
    parts = []
    if raw is not None:
        parts.append(raw.reshape(-1, 4))
    if lsd is not None:
        parts.append(lsd.reshape(-1, 4))
    raw = np.concatenate(parts) if parts else None
    records = []
    raw_count = 0 if raw is None else len(raw)
    if raw is not None:
        segments = raw.reshape(-1, 4).astype(float)
        lengths = np.linalg.norm(segments[:, :2] - segments[:, 2:], axis=1)
        indices = np.flatnonzero(lengths >= minimum)
        for index in indices[np.argsort(-lengths[indices])][:500]:
            p, q = segments[index, :2], segments[index, 2:]
            r = band_support(smooth, p, q, c.dark_difference, search=True)
            if r["support"] < c.line_support or r["continuity"] < 0.66:
                continue
            r["line"] = line_of(r["p"], r["q"])
            r["length"] = float(lengths[index])
            records.append(r)
    records.sort(key=lambda r: r["score"] * math.sqrt(r["length"]), reverse=True)
    # 同一目地に沿う線分を統合。離れた共線線分は端点の範囲を拡張する。
    merged = []
    for r in records:
        duplicate = None
        for other in merged:
            aligned = abs(float(np.dot(r["line"][:2], other["line"][:2])))
            mid = (r["p"] + r["q"]) / 2
            distance = abs(float(np.dot(other["line"], np.r_[mid, 1.])))
            if aligned > math.cos(math.radians(2.5)) and distance < 5:
                duplicate = other
                break
        if duplicate is None:
            merged.append(r)
        else:
            # 順位上位の中心線は維持し、実際の線分範囲だけ統合する。
            direction = duplicate["q"] - duplicate["p"]
            direction /= np.linalg.norm(direction)
            origin = duplicate["p"].copy()
            positions = (np.array([duplicate["p"], duplicate["q"], r["p"], r["q"]])
                         - origin) @ direction
            duplicate["p"] = origin + positions.min() * direction
            duplicate["q"] = origin + positions.max() * direction
            duplicate["length"] = float(positions.max() - positions.min())
    return merged[:100], blackhat, mask, raw_count


def vp_errors(vp, mids, directions):
    rays = vp[:2][None] - mids * vp[2]
    norms = np.linalg.norm(rays, axis=1)
    return np.abs(rays[:, 0] * directions[:, 1] - rays[:, 1] * directions[:, 0]) / np.maximum(norms, 1e-12)


def vp_families(records: list[dict], shape, c: Config):
    """線分の向きだけでなく消失点への整合性を用いる2モデルRANSAC。

    有限の消失点と無限遠（平行線）の両方を同次座標で処理。
    ここでのRANSACは線分類用。既存の特徴点照合RANSACとは別。
    """
    if len(records) < 6:
        return [], []
    center = np.array([shape[1] / 2, shape[0] / 2])
    scale = float(max(shape))
    ends = np.array([[r["p"], r["q"]] for r in records])
    normalized = (ends - center) / scale
    lines = np.array([line_of(p, q) for p, q in normalized])
    mids = normalized.mean(axis=1)
    directions = normalized[:, 1] - normalized[:, 0]
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    weights = np.array([r["score"] * math.sqrt(r["length"]) for r in records])
    active = np.arange(len(records))
    families, vps = [], []
    rng = np.random.default_rng(c.seed)
    threshold = math.sin(math.radians(c.angle_tolerance))
    for _ in range(2):
        if len(active) < 3:
            break
        pairs = list(combinations(active.tolist(), 2))
        if len(pairs) > 2500:
            pairs = [pairs[i] for i in rng.choice(len(pairs), 2500, replace=False)]
        best = None
        for a, b in pairs:
            vp = np.cross(lines[a], lines[b])
            if np.linalg.norm(vp) < 1e-10:
                continue
            vp /= np.linalg.norm(vp)
            errors = vp_errors(vp, mids[active], directions[active])
            ids = active[errors < threshold]
            if len(ids) < 3:
                continue
            merit = float((weights[active] * np.maximum(0, 1 - errors / threshold)).sum())
            if best is None or merit > best[0]:
                best = merit, ids, vp
        if best is None:
            break
        _, ids, vp = best
        _, _, vt = np.linalg.svd(lines[ids] * np.sqrt(weights[ids, None]), full_matrices=True)
        refined = vt[-1]
        refined /= np.linalg.norm(refined)
        if np.sum(vp_errors(refined, mids[ids], directions[ids]) < threshold) >= len(ids) * 0.8:
            vp = refined
        ids = active[vp_errors(vp, mids[active], directions[active]) < threshold]
        if len(ids) < 3:
            break
        families.append(ids.tolist())
        # 正規化座標→縮小画像座標。無限遠もそのまま保存可能。
        vp_work = np.r_[vp[:2] * scale + center * vp[2], vp[2]]
        vps.append((vp_work / np.linalg.norm(vp_work)).tolist())
        active = active[~np.isin(active, ids)]
    return families, vps


def intersection(a, b):
    point = np.cross(a, b)
    if abs(point[2]) < 1e-9:
        return None
    return point[:2] / point[2]


def order_quad(points):
    p = np.asarray(points, np.float32)
    center = p.mean(axis=0)
    p = p[np.argsort(np.arctan2(p[:, 1] - center[1], p[:, 0] - center[0]))]
    return np.roll(p, -int(np.argmin(p.sum(axis=1))), axis=0)


def enumerate_quads(gray, records, families, c):
    if len(families) != 2:
        return []
    groups = [sorted(f, key=lambda i: records[i]["score"] * math.sqrt(records[i]["length"]),
                     reverse=True)[:16] for f in families]
    extent = min(gray.shape)
    area_total = gray.shape[0] * gray.shape[1]
    candidates = []
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0.6)
    for pair_a in combinations(groups[0], 2):
        for pair_b in combinations(groups[1], 2):
            a, b, d, e = [records[i]["line"] for i in (*pair_a, *pair_b)]
            corners = [intersection(x, y) for x, y in ((a, d), (b, d), (b, e), (a, e))]
            if any(p is None for p in corners):
                continue
            quad = order_quad(corners)
            if not np.isfinite(quad).all() or not cv2.isContourConvex(quad):
                continue
            if (quad.min() < 2 or quad[:, 0].max() >= gray.shape[1] - 2
                    or quad[:, 1].max() >= gray.shape[0] - 2):
                continue
            area = abs(float(cv2.contourArea(quad)))
            if not c.min_area * area_total <= area <= c.max_area * area_total:
                continue
            vectors = np.roll(quad, -1, axis=0) - quad
            lengths = np.linalg.norm(vectors, axis=1)
            if lengths.min() < extent * 0.06 or lengths.max() / lengths.min() > 7:
                continue
            # ほぼ一直線の四角形を除外。投影後の縦横比だけでは真偽は判別できない。
            angle_sines = np.abs(vectors[:, 0] * np.roll(vectors, 1, axis=0)[:, 1]
                                 - vectors[:, 1] * np.roll(vectors, 1, axis=0)[:, 0])
            angle_sines /= lengths * np.roll(lengths, 1)
            if angle_sines.min() < 0.18:
                continue
            # 候補内部を別の目地線が横断していたら複数タイルとみなし除外。
            used = set((*pair_a, *pair_b))
            crossed = False
            for i in set(groups[0] + groups[1]) - used:
                signed = np.c_[quad, np.ones(4)] @ records[i]["line"]
                if signed.min() < -8 and signed.max() > 8:
                    # 無限に延長した壁の線などでは除外しない。
                    # 四角形内を横断する区間にも暗い目地が存在するか調べる。
                    crossings = []
                    for j in range(4):
                        k = (j + 1) % 4
                        if signed[j] * signed[k] < 0:
                            fraction = signed[j] / (signed[j] - signed[k])
                            crossings.append(quad[j] + fraction * (quad[k] - quad[j]))
                    if len(crossings) == 2:
                        internal = band_support(smooth, crossings[0], crossings[1],
                                                c.dark_difference)
                        if internal["support"] >= c.line_support and internal["continuity"] >= 0.66:
                            crossed = True
                            break
            if crossed:
                continue
            sides = [band_support(smooth, p, q, c.dark_difference)
                     for p, q in zip(quad, np.roll(quad, -1, axis=0))]
            supports = [r["support"] for r in sides]
            if min(supports) < c.side_support:
                continue
            if min(r["continuity"] for r in sides) < 0.66:
                continue
            score = (0.55 * min(supports) + 0.30 * np.mean(supports)
                     + 0.15 * np.mean([min(r["strength"] / 20, 1) for r in sides]))
            candidates.append(dict(quad=quad, score=float(score), area=area,
                                   side_support=supports,
                                   side_width=[r["width"] for r in sides]))
    candidates.sort(key=lambda r: r["score"], reverse=True)
    accepted = []
    for r in candidates:
        duplicate = False
        for other in accepted:
            overlap, _ = cv2.intersectConvexConvex(r["quad"], other["quad"])
            union = r["area"] + other["area"] - overlap
            if overlap / max(union, 1) > 0.65:
                duplicate = True
                break
        if not duplicate:
            accepted.append(r)
        if len(accepted) >= c.max_candidates:
            break
    return accepted


def prepare_detection_gray(work: np.ndarray, c: Config):
    """検出用だけを補正。入力画像を変更せず、既定は従来と同じグレー化。"""
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    seconds = 0.0
    if c.clahe:
        if (not math.isfinite(c.clahe_clip_limit) or c.clahe_clip_limit <= 0
                or not 1 <= c.clahe_grid <= 64):
            raise ValueError("CLAHE設定が不正です。clip-limitは正の有限値、gridは1～64です。")
        started = time.perf_counter()
        operator = cv2.createCLAHE(clipLimit=c.clahe_clip_limit,
                                   tileGridSize=(c.clahe_grid, c.clahe_grid))
        gray = operator.apply(gray)
        seconds = time.perf_counter() - started
    details = dict(mode="clahe" if c.clahe else "none",
                   clip_limit=c.clahe_clip_limit if c.clahe else None,
                   grid_size=[c.clahe_grid, c.clahe_grid] if c.clahe else None,
                   scope="detection_only", matching_uses_original_image=True)
    return gray, details, seconds


def detect(image, c):
    started = time.perf_counter()
    factor = min(1., c.max_side / max(image.shape[:2]))
    size = (round(image.shape[1] * factor), round(image.shape[0] * factor))
    work = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    gray, preprocessing, clahe_seconds = prepare_detection_gray(work, c)
    prepared = time.perf_counter()
    records, blackhat, mask, raw_count = extract_lines(gray, c)
    extracted = time.perf_counter()
    families, vps = vp_families(records, gray.shape, c)
    classified = time.perf_counter()
    candidates = enumerate_quads(gray, records, families, c)
    # 実際のresize寸法に合わせてx,yを別々に復元する。
    xy_scale = np.array([work.shape[1] / image.shape[1], work.shape[0] / image.shape[0]])
    for r in candidates:
        r["quad_original"] = r["quad"] / xy_scale
    finished = time.perf_counter()
    timings = dict(resize_gray_and_preprocessing=prepared - started,
                   clahe=clahe_seconds,
                   line_extraction=extracted - prepared,
                   vanishing_point_classification=classified - extracted,
                   candidate_evaluation=finished - classified,
                   detection_total=finished - started)
    # claheはresize_gray_and_preprocessingの内数。合計時に二重加算しない。
    return dict(work=work, records=records, families=families, vps=vps,
                candidates=candidates, blackhat=blackhat, mask=mask,
                raw_segments=raw_count, xy_scale=xy_scale, detection_gray=gray,
                preprocessing=preprocessing, timings_seconds=timings)


def save_results(image, result, output, c):
    output.mkdir(parents=True, exist_ok=False)
    save_image(output / "00_detection_gray.png", result["detection_gray"])
    save_image(output / "01_grout_emphasis.png", result["blackhat"])
    save_image(output / "02_dark_band_mask.png", result["mask"])
    lines_picture = result["work"].copy()
    colors = ((0, 180, 255), (255, 180, 0))
    labels = {i: f for f, ids in enumerate(result["families"]) for i in ids}
    for i, r in enumerate(result["records"]):
        color = colors[labels[i]] if i in labels else (130, 130, 130)
        cv2.line(lines_picture, tuple(np.rint(r["p"]).astype(int)),
                 tuple(np.rint(r["q"]).astype(int)), color, 2)
    save_image(output / "03_line_families.jpg", lines_picture)
    overlay = result["work"].copy()
    rows = []
    for number, r in enumerate(result["candidates"], 1):
        quad = r["quad"]
        color = (0, 255, 0) if number == 1 else (0, 180, 255)
        cv2.polylines(overlay, [np.rint(quad).astype(np.int32)], True, color, 3)
        center = tuple(np.rint(quad.mean(axis=0)).astype(int))
        cv2.putText(overlay, f"#{number} score={r['score']:.2f}", center,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        for corner in quad:
            cv2.circle(overlay, tuple(np.rint(corner).astype(int)), 5, (0, 0, 255), -1)
        # 診断画像は目地を含めるため、正方形512pxの外に64pxの余白を残す。
        destination = np.float32([[64, 64], [575, 64], [575, 575], [64, 575]])
        transform = cv2.getPerspectiveTransform(r["quad_original"].astype(np.float32), destination)
        warped = cv2.warpPerspective(image, transform, (640, 640), borderValue=(0, 0, 0))
        save_image(output / f"tile_{number:02d}_rectified_preview.png", warped)
        # 黒い画像外領域と実写部分を区別する診断用マスク。
        valid = cv2.warpPerspective(np.full(image.shape[:2], 255, np.uint8),
                                    transform, (640, 640), flags=cv2.INTER_NEAREST)
        save_image(output / f"tile_{number:02d}_valid_mask.png", valid)
        points = r["quad_original"]
        text = ";".join(f"{x:.2f},{y:.2f}" for x, y in points)
        rows.append(dict(candidate=number, detection_score=r["score"],
                         quad_original_xy=points.tolist(), points_argument=text,
                         side_support=r["side_support"],
                         estimated_band_offset_px=r["side_width"]))
    save_image(output / "04_detected_tiles.jpg", overlay)
    report = dict(config=asdict(c), source_shape_hw=list(image.shape[:2]),
                  work_shape_hw=list(result["work"].shape[:2]),
                  raw_segments=result["raw_segments"], accepted_lines=len(result["records"]),
                  family_counts=[len(x) for x in result["families"]],
                  vanishing_points_homogeneous_work=result["vps"], candidates=rows,
                  preprocessing=result["preprocessing"],
                  timings_seconds=result["timings_seconds"],
                  selected_candidate=1 if rows else None,
                  warning="検出スコアは照合の一致割合・確率ではありません。目視確認が必要です。")
    (output / "detection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    correction = (f"CLAHEあり（clip-limit={c.clahe_clip_limit:g}、画像分割={c.clahe_grid}×{c.clahe_grid}）"
                  if c.clahe else "CLAHEなし（従来方式）")
    lines = ["目地交点の自動検出（実験実装）", f"検出用前処理：{correction}",
             "補正対象は検出用画像のみ。切り出し・照合には元画像を使用します。",
             f"CLAHE処理時間：{result['timings_seconds']['clahe'] * 1000:.2f}ミリ秒",
             f"検出計算時間：{result['timings_seconds']['detection_total']:.3f}秒（画像読込・診断保存を除く）",
             f"タイル候補数：{len(rows)}",
             "候補があれば04_detected_tiles.jpgの緑枠が採用候補、赤丸が四隅です。",
             "tile_*_rectified_preview.png は検出確認用。既存照合の補正画像とは別です。",
             "診断用画像の黒い領域は画像外。valid_maskで確認してください。",
             "未検出時に画像中心や任意の四角形へフォールバックはしません。"]
    if rows:
        lines += [f"検出スコア：{rows[0]['detection_score']:.3f}（一致率ではない）",
                  f"元画像座標の四隅：{rows[0]['points_argument']}"]
    (output / "detection_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"保存先：{output}", flush=True)
    return rows


def run_sweep(args, row, image_path, shape, output):
    script = path_of(args.proposal_script)
    if not script.is_file():
        raise FileNotFoundError(f"既存の照合コードがありません：{script}")
    # 引数リストで実行。shellを使わず、日本語・空白・セミコロンを安全に渡す。
    command = [sys.executable, "-u", str(script), "--sweep", "--satsuei", str(image_path),
               "--satsuei-points", row["points_argument"],
               "--points-size", f"{shape[1]},{shape[0]}", "--output", str(output / "matching")]
    # 正解地点・タイルは評価表示専用。検出関数には一切渡さない。
    if args.expected_db is not None:
        command += ["--expected-db", str(args.expected_db)]
    if args.expected_tile is not None:
        command += ["--expected-tile", str(args.expected_tile)]
    (output / "sweep_command.json").write_text(json.dumps(command, ensure_ascii=False, indent=2), encoding="utf-8")
    print("自動検出した第1候補を、既存の目地使用率比較へ渡します。", flush=True)
    completed = subprocess.run(command, cwd=str(BASE), check=False)
    if completed.returncode:
        raise RuntimeError("自動検出結果は保存済みですが、既存の照合処理でエラーになりました。上の表示を確認してください。")


def self_test():
    c = Config(max_side=1000)
    rng = np.random.default_rng(12)
    base = np.clip(rng.normal(180, 1, (800, 800)), 0, 255).astype(np.uint8)
    for position in (100, 300, 500, 700):
        cv2.line(base, (position, 0), (position, 799), 35, 6)
        cv2.line(base, (0, position), (799, position), 35, 6)
    image = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    destination = np.float32([[120, 70], [720, 170], [850, 770], [40, 690]])
    matrix = cv2.getPerspectiveTransform(np.float32([[0, 0], [799, 0], [799, 799], [0, 799]]), destination)
    tilted = cv2.warpPerspective(image, matrix, (900, 850), borderValue=(180, 180, 180))
    for label, photo, transform in (("parallel", image, np.eye(3)), ("perspective", tilted, matrix)):
        result = detect(photo, c)
        assert result["candidates"], f"合成{label}画像を検出できません。"
        truth = []
        for x in (100, 300, 500):
            for y in (100, 300, 500):
                quad = np.float32([[x, y], [x + 200, y], [x + 200, y + 200], [x, y + 200]])
                truth.append(cv2.perspectiveTransform(quad[None], transform)[0])
        predicted = result["candidates"][0]["quad_original"].astype(np.float32)
        overlaps = []
        for quad in truth:
            overlap, _ = cv2.intersectConvexConvex(predicted, quad)
            overlaps.append(overlap / max(cv2.contourArea(predicted) + cv2.contourArea(quad) - overlap, 1))
        assert max(overlaps) > 0.85, f"合成{label}の四隅誤差が大きい：{max(overlaps)}"
        print(f"自己テスト {label}: 第1候補IoU={max(overlaps):.3f}")
    assert not detect(np.full((500, 500, 3), 180, np.uint8), c)["candidates"]
    enabled = Config(max_side=1000, clahe=True)
    preserved = image.copy()
    plain, plain_details, plain_seconds = prepare_detection_gray(image, c)
    assert np.array_equal(plain, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    assert plain_details["mode"] == "none" and plain_seconds == 0
    corrected, details, elapsed = prepare_detection_gray(image, enabled)
    assert corrected.shape == plain.shape and corrected.dtype == np.uint8
    assert details["mode"] == "clahe" and elapsed >= 0
    assert np.array_equal(image, preserved), "検出前処理が元画像を変更しました。"
    assert detect(image, enabled)["candidates"], "CLAHEありの合成画像を検出できません。"
    assert not detect(np.full((500, 500, 3), 180, np.uint8), enabled)["candidates"]
    print("自己テスト CLAHE: オフの同一性・入力保持・オンの検出・無地の未検出を確認")
    print("自己テスト成功。合成画像の検証であり、実際の床での成功を保証しません。")


def main():
    parser = argparse.ArgumentParser(description="暗い目地線・消失点・交点によるタイル自動検出")
    parser.add_argument("--satsuei", help="撮影画像1枚")
    parser.add_argument("--output", help="新規の結果フォルダ。省略時は日時付き")
    parser.add_argument("--run-sweep", action="store_true", help="第1候補を既存step4_teian_grout.pyの--sweepに渡す")
    parser.add_argument("--proposal-script", default="step4_teian_grout.py")
    parser.add_argument("--expected-db", type=int, help="評価表示専用。検出・候補選択には使わない")
    parser.add_argument("--expected-tile", type=int, help="評価表示専用")
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--min-length", type=float, default=0.10)
    parser.add_argument("--dark-difference", type=float, default=3.0)
    parser.add_argument("--line-support", type=float, default=0.42)
    parser.add_argument("--side-support", type=float, default=0.40)
    parser.add_argument("--angle-tolerance", type=float, default=6.0)
    parser.add_argument("--min-area", type=float, default=0.008)
    parser.add_argument("--max-area", type=float, default=0.65)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clahe", action="store_true", help="検出用の縮小グレー画像だけをCLAHE補正（既定はオフ）")
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0, help="CLAHEのコントラスト制限。正の有限値（暫定値2.0）")
    parser.add_argument("--clahe-grid", type=int, default=8, help="CLAHEの各軸の画像分割数1～64。床タイル数ではない（既定8）")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.satsuei:
        parser.error("--satsueiを指定してください。")
    c = Config(**{k: getattr(args, k) for k in Config.__dataclass_fields__})
    if (c.max_side < 200 or not 0 < c.min_length < 1 or c.dark_difference <= 0
            or not 0 < c.line_support <= 1 or not 0 < c.side_support <= 1
            or not 0 < c.angle_tolerance < 30 or not 0 < c.min_area < c.max_area < 1
            or c.max_candidates < 1 or c.seed < 0):
        raise ValueError("検出設定の範囲が不正です。")
    if (not math.isfinite(c.clahe_clip_limit) or c.clahe_clip_limit <= 0
            or not 1 <= c.clahe_grid <= 64):
        raise ValueError("CLAHE設定が不正です。clip-limitは正の有限値、gridは1～64です。")
    image_path = path_of(args.satsuei)
    image = read_image(image_path)
    output = path_of(args.output) if args.output else BASE / "kekka" / ("auto_grout_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if output.exists():
        raise FileExistsError(f"結果フォルダが既にあります：{output}")
    started = time.perf_counter()
    result = detect(image, c)
    rows = save_results(image, result, output, c)
    print(f"検出・診断保存時間：{time.perf_counter() - started:.2f}秒", flush=True)
    if not rows:
        print("タイルを検出できませんでした。診断画像を確認してください。照合は実行しません。", file=sys.stderr)
        raise SystemExit(2)
    if args.run_sweep:
        run_sweep(args, rows[0], image_path, image.shape, output)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        print(f"実行できませんでした：{exc}", file=sys.stderr)
        raise SystemExit(1)
