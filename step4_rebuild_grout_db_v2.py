"""登録2×2タイルの縦3本・横3本の実目地からDBを作成する。

外周を画像端で代用しない。目地が写っていない、線が曖昧、交点が画像外なら
検出保留。対象はおおむね正面撮影した2×2登録画像（縦横から25度以内）。
暗い帯の判定は撮影側と同じ考え方を使うが、画像外の反射補間は証拠に使わない。
--check-db の確認画像を目視した後に --build-db を実行する。
既存スクリプト、既存DB、照合処理、判定基準は変更しない。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import step4_auto_grout as detector
import step4_teian_grout as grout

BASE = Path(__file__).resolve().parent
METHOD = "registration-visible-three-by-three-grout-grid-v2"
SETTINGS = dict(max_side=640, dark_difference=3., support=.42, continuity=.66,
                visible_fraction=.55, max_axis_angle=25., max_family_angle=12.,
                max_spacing_ratio=1.8, ambiguity_score_gap=.015)


class GridError(ValueError):
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def path_of(value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (BASE / p).resolve()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_hashes():
    names = ("step4_auto_grout", "step4_teian_grout", "step3_senko_saigen", "step4_teian")
    result = {name: digest(sys.modules[name].__file__) for name in names}
    result["builder"] = digest(__file__)
    return result


def dark_band(gray, p, q):
    """中心・両側すべてが画像内の位置だけで暗帯を評価（反射パディング禁止）。"""
    vector = np.asarray(q) - p
    length = float(np.linalg.norm(vector))
    if length < 20:
        return None
    normal = np.array([-vector[1], vector[0]]) / length
    t = np.linspace(.02, .98, 240)
    xy = np.asarray(p)[None] + t[:, None] * vector
    h, w = gray.shape
    best = None
    for width in (3., 5., 8., 12.):
        points = [xy, xy + normal, xy - normal, xy + width * normal, xy - width * normal]
        valid = np.logical_and.reduce([(a[:, 0] >= 0) & (a[:, 0] <= w - 1) &
            (a[:, 1] >= 0) & (a[:, 1] <= h - 1) for a in points])
        coverage = float(valid.mean())
        if coverage < SETTINGS["visible_fraction"]:
            continue
        # sample() の反射モードは使われないよう、評価対象を先に画像内へ限定。
        values = [detector.sample(gray, a[valid]) for a in points]
        contrast = np.minimum(values[3], values[4]) - (values[0] + values[1] + values[2]) / 3
        hits = contrast >= SETTINGS["dark_difference"]
        support = float(hits.mean())
        good = np.zeros(len(valid), bool)
        good[valid] = hits
        continuity_bins = []
        for ids in np.array_split(np.arange(len(valid)), 6):
            selected = ids[valid[ids]]
            if len(selected) >= 8:
                continuity_bins.append(float(good[selected].mean()) >= .25)
        if len(continuity_bins) < 4:
            continue
        continuity = float(np.mean(continuity_bins))
        strength = float(np.mean(np.clip(contrast, 0, 30)))
        score = .7 * support + .2 * continuity + .1 * min(strength / 20, 1)
        item = dict(score=score, support=support, continuity=continuity,
                    strength=strength, width=width, visible_fraction=coverage)
        if best is None or item["score"] > best["score"]:
            best = item
    return best


def raw_lines(gray):
    """Canny＋確率的ハフ変換とLSDから、目地候補の線分を得る。"""
    byte = np.uint8(np.clip(gray, 0, 255))
    edges = cv2.Canny(byte, 25, 70, L2gradient=True)
    minimum = max(24, round(min(gray.shape) * .06))
    hough = cv2.HoughLinesP(edges, 1, np.pi / 720, max(20, minimum // 2),
                           minLineLength=minimum, maxLineGap=8)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(byte)[0]
    # 撮影側の暗帯確認済み線分を先に使う。単純な長さ順だけだと、
    # 面模様や影から得た線分に予算を使い切り、細い目地を落としやすい。
    records, _, _, _ = detector.extract_lines(np.uint8(np.clip(gray, 0, 255)),
                                             detector.Config(max_side=SETTINGS["max_side"]))
    preferred = [(np.asarray(row["p"]), np.asarray(row["q"]), row["length"]) for row in records]
    arrays = [a.reshape(-1, 4) for a in (hough, lsd) if a is not None]
    if not arrays:
        return preferred
    segments = np.concatenate(arrays).astype(float)
    lengths = np.linalg.norm(segments[:, 2:] - segments[:, :2], axis=1)
    return preferred + [(segments[i, :2], segments[i, 2:], float(lengths[i]))
            for i in np.argsort(-lengths) if lengths[i] >= minimum]


def region_of(position, extent):
    fraction = position / (extent - 1)
    if 0 <= fraction <= .25:
        return 0
    if .30 <= fraction <= .70:
        return 1
    if .75 <= fraction <= 1:
        return 2
    return None


def line_candidates(gray, segments, axis):
    h, w = gray.shape
    vertical = axis == "vertical"
    component = 0 if vertical else 1
    extent = w if vertical else h
    raw_groups = [[], [], []]
    angle_limit = math.tan(math.radians(SETTINGS["max_axis_angle"]))
    for first, last, length in segments:
        a, b, c = detector.line_of(first, last)
        divisor = a if vertical else b
        other = b if vertical else a
        if abs(divisor) < 1e-9 or abs(other / divisor) > angle_limit:
            continue
        if vertical:
            p = np.array([-c / a, 0.])
            q = np.array([-(b * (h - 1) + c) / a, h - 1.])
        else:
            p = np.array([0., -c / b])
            q = np.array([w - 1., -(a * (w - 1) + c) / b])
        position = float((p[component] + q[component]) / 2)
        region = region_of(position, extent)
        if region is None:
            continue
        # 共線の短い線分を重複評価しない。画像外の端点は切り詰めない。
        duplicate = any(max(np.linalg.norm(p - item[0]), np.linalg.norm(q - item[1])) < 4
                        for item in raw_groups[region])
        if not duplicate and len(raw_groups[region]) < 20:
            raw_groups[region].append((p, q, length))
    result = [[], [], []]
    for group in raw_groups:
        for p, q, length in group:
            best = None
            # 線分から推定した傾きの小さな誤差を、実画像の暗帯に合わせて補正。
            for left, right in product((-6., -3., 0., 3., 6.), repeat=2):
                rp, rq = p.copy(), q.copy()
                rp[component] += left
                rq[component] += right
                evidence = dark_band(gray, rp, rq)
                if evidence is not None and (best is None or evidence["score"] > best["score"]):
                    best = {**evidence, "p": rp.tolist(), "q": rq.tolist()}
            if best is None or best["support"] < SETTINGS["support"] or best["continuity"] < SETTINGS["continuity"]:
                continue
            bp, bq = np.array(best["p"]), np.array(best["q"])
            best["position"] = float((bp[component] + bq[component]) / 2)
            best["line"] = detector.line_of(bp, bq).tolist()
            best["axis_angle"] = math.degrees(math.atan2(bq[component] - bp[component],
                                                (h - 1) if vertical else (w - 1)))
            region = region_of(best["position"], extent)
            if region is not None:
                result[region].append(best)
    for i, group in enumerate(result):
        group.sort(key=lambda item: item["score"], reverse=True)
        distinct = []
        for item in group:
            if not any(max(np.linalg.norm(np.array(item["p"]) - other["p"]),
                           np.linalg.norm(np.array(item["q"]) - other["q"])) < 6 for other in distinct):
                distinct.append(item)
        result[i] = distinct[:5]
    return result


def triples(groups):
    if any(not group for group in groups):
        return []
    options = []
    for rows in product(*groups):
        positions = [row["position"] for row in rows]
        gaps = np.diff(positions)
        angles = [row["axis_angle"] for row in rows]
        if min(gaps) <= 0 or max(gaps) / min(gaps) > SETTINGS["max_spacing_ratio"]:
            continue
        if max(angles) - min(angles) > SETTINGS["max_family_angle"]:
            continue
        options.append(rows)
    return options


def grid_quads(vertical, horizontal, image_shape):
    h, w = image_shape[:2]
    points = np.empty((3, 3, 2), np.float32)
    for row in range(3):
        for column in range(3):
            xy = np.cross(vertical[column]["line"], horizontal[row]["line"])
            if abs(xy[2]) < 1e-9:
                raise ValueError("目地線が交差しません。")
            xy = xy[:2] / xy[2]
            # 画像内へclampして四隅を捏造しない。
            if not (0 <= xy[0] <= w - 1 and 0 <= xy[1] <= h - 1):
                raise ValueError("目地の交点が画像外です。")
            points[row, column] = xy
    quads = [np.float32([points[r, c], points[r, c + 1],
                         points[r + 1, c + 1], points[r + 1, c]])
             for r in range(2) for c in range(2)]
    for quad in quads:
        if not cv2.isContourConvex(quad) or cv2.contourArea(quad) < .04 * w * h:
            raise ValueError("タイルの形または面積が不正です。")
    return quads, points


def split_quads(image):
    h, w = image.shape[:2]
    if min(h, w) < 160:
        raise GridError("短辺160px以上の元の登録画像が必要です。")
    scale = min(1., SETTINGS["max_side"] / max(h, w))
    size = (round(w * scale), round(h * scale))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA) if scale < 1 else image
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), .6)
    segments = raw_lines(gray)
    vg = line_candidates(gray, segments, "vertical")
    hg = line_candidates(gray, segments, "horizontal")
    diagnostics = {"method": METHOD, "settings": SETTINGS, "detection_size": list(size),
                   "vertical_candidates": vg, "horizontal_candidates": hg}
    missing = [name for name, group in zip(("左外周", "中央縦", "右外周", "上外周", "中央横", "下外周"), vg + hg) if not group]
    if missing:
        raise GridError("目地を確認できません：" + "・".join(missing) + "。画像端では代用しません。", diagnostics)
    candidates = []
    for vertical, horizontal in product(triples(vg), triples(hg)):
        try:
            quads, points = grid_quads(vertical, horizontal, small.shape)
        except ValueError:
            continue
        evidence = list(vertical) + list(horizontal)
        score = float(np.mean([row["score"] for row in evidence]))
        candidates.append((score, vertical, horizontal, quads, points))
    if not candidates:
        raise GridError("目地線はありますが、画像内に四隅がそろう2×2タイルを確認できません。", diagnostics)
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0]
    for other in candidates[1:]:
        distance = np.linalg.norm(best[4] - other[4], axis=2).max()
        if distance > .025 * min(gray.shape):
            if best[0] - other[0] < SETTINGS["ambiguity_score_gap"]:
                raise GridError("異なる目地配置の得点が僅差です。自動登録を保留します。", diagnostics)
            break
    factor = np.array([(w - 1) / (small.shape[1] - 1), (h - 1) / (small.shape[0] - 1)])
    quads = [np.float32(q * factor) for q in best[3]]
    diagnostics.update(score=best[0], selected_vertical=list(best[1]), selected_horizontal=list(best[2]),
                       intersections_original=(best[4] * factor).tolist(), outer_boundary="detected grout only")
    return quads, diagnostics


def draw_preview(image, location, quads=None, diagnostics=None):
    output = image.copy()
    thickness = max(2, round(max(image.shape[:2]) / 700))
    if quads is not None:
        for index, quad in enumerate(quads, 1):
            cv2.polylines(output, [np.rint(quad).astype(np.int32)], True, (0, 220, 0), thickness)
            cv2.putText(output, f"{location}-{index}", tuple(np.rint(quad.mean(axis=0)).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(.6, max(image.shape[:2]) / 1300), (0, 0, 255), thickness)
        for xy in np.array(diagnostics["intersections_original"]).reshape(-1, 2):
            cv2.circle(output, tuple(np.rint(xy).astype(int)), thickness * 2, (0, 0, 255), -1)
    elif diagnostics:
        sw, sh = diagnostics["detection_size"]
        h, w = image.shape[:2]
        factor = np.array([(w - 1) / (sw - 1), (h - 1) / (sh - 1)])
        for group in diagnostics["vertical_candidates"] + diagnostics["horizontal_candidates"]:
            if group:
                p, q = [tuple(np.rint(np.array(group[0][key]) * factor).astype(int)) for key in ("p", "q")]
                cv2.line(output, p, q, (0, 160, 255), thickness)
        cv2.putText(output, "HOLD: incomplete or ambiguous grout grid", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
    return output


def check_database(args):
    images = grout.numbered_images(path_of(args.db_dir), "db_")
    if not images or len({n for n, _ in images}) != len(images):
        raise ValueError("db_01.png等の元画像を番号の重複なく指定してください。")
    output = path_of(args.output)
    output.mkdir(parents=True, exist_ok=False)
    previews = output / "registration_previews"
    previews.mkdir()
    rows = []
    for location, source in images:
        print(f"外周を含む目地検出：db_{location:02d}", flush=True)
        image = grout.read_image(source)
        row = dict(location=location, source_image=str(source), source_sha256=digest(source))
        try:
            quads, info = split_quads(image)
            row.update(status="detected", quads=[q.tolist() for q in quads], split=info)
            preview = draw_preview(image, location, quads, info)
            print("  縦3本・横3本と9交点を確認。目視確認が必要です。", flush=True)
        except GridError as exc:
            row.update(status="failed", reason=str(exc), split=exc.diagnostics)
            preview = draw_preview(image, location, diagnostics=exc.diagnostics)
            print("  保留：" + str(exc), flush=True)
        grout.save_image(previews / f"db_{location:02d}.jpg", preview)
        grout.write_json(previews / f"db_{location:02d}.json", row)
        rows.append(row)
    plan = dict(method=METHOD, created_at=datetime.now().isoformat(),
                dependencies_sha256=dependency_hashes(), images=rows)
    grout.write_json(output / "split_plan.json", plan)
    failed = sum(row["status"] != "detected" for row in rows)
    print(f"検出 {len(rows) - failed}/{len(rows)}枚。DBはまだ作成していません。\n確認画像：{previews}")
    if failed:
        raise ValueError("検出保留があります。画像端による代用はしません。この状態ではDB作成不可です。")


def build_database(args):
    plan = json.loads(path_of(args.checked_plan).read_text(encoding="utf-8"))
    if plan.get("method") != METHOD or plan.get("dependencies_sha256") != dependency_hashes():
        raise ValueError("V2の確認結果が必要です。コード変更後は確認からやり直してください。")
    rows = plan.get("images", [])
    if not rows or any(row.get("status") != "detected" for row in rows):
        raise ValueError("保留の画像があります。全画像の確認が完了するまでDBを作成しません。")
    if len({row["location"] for row in rows}) != len(rows):
        raise ValueError("DB番号が重複しています。")
    for row in rows:
        if digest(row["source_image"]) != row["source_sha256"]:
            raise ValueError("確認後に元画像が変わりました：" + row["source_image"])
        image = grout.read_image(Path(row["source_image"]))
        quads = row.get("quads", [])
        if len(quads) != 4:
            raise ValueError("4タイルの四隅が必要です。")
        for quad in quads:
            q = np.float32(quad)
            if q.shape != (4, 2) or not np.isfinite(q).all() or not cv2.isContourConvex(q):
                raise ValueError("確認ファイルの四角形が不正です。")
            grout.require_inside(q, image.shape)
    output = path_of(args.grout_cache)
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("tiles", "features", "registration_previews"):
        (output / folder).mkdir()
    c = grout.Config()
    saved = []
    started = time.perf_counter()
    for row in rows:
        image = grout.read_image(Path(row["source_image"]))
        quads = [np.float32(q) for q in row["quads"]]
        for index, quad in enumerate(quads, 1):
            name = f"db_{row['location']:02d}_tile_{index}"
            print(f"登録候補抽出 [{len(saved) + 1}/{len(rows) * 4}] {name}", flush=True)
            pool = grout.extract_pool(image, quad, c)
            feature_file = f"features/{name}.npz"
            np.savez_compressed(output / feature_file, **{key: pool[key] for key in
                ("points", "descriptors", "responses", "sizes", "on_grout", "quad", "matrix")})
            grout.save_pool_visuals(output / "tiles" / name, pool, c)
            saved.append(dict(location=row["location"], tile_index=index, name=name,
                feature_file=feature_file, image_file=f"tiles/{name}/01_rectified_with_context.png",
                source_image=row["source_image"], source_sha256=row["source_sha256"], **grout.pool_stats(pool)))
        grout.save_image(output / "registration_previews" / f"db_{row['location']:02d}.jpg",
                         draw_preview(image, row["location"], quads, row["split"]))
        grout.write_json(output / "registration_previews" / f"db_{row['location']:02d}.json", row)
    metadata = dict(format=grout.FORMAT, registration_method=METHOD,
        created_at=datetime.now().isoformat(), config=asdict(c), opencv_version=cv2.__version__,
        numpy_version=np.__version__, python_version=sys.version.split()[0], tiles=saved,
        elapsed_seconds=time.perf_counter() - started,
        notes="Six visible grout lines; nine intersections; no image-edge fallback. Matching unchanged.")
    grout.write_json(output / "database.json", metadata)
    grout.write_json(output / "split_plan.json", plan)
    grout.write_csv(output / "feature_pool_counts.csv", saved)
    print(f"DB作成完了：{len(rows)}地点・{len(saved)}タイル\n保存先：{output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-db", action="store_true")
    mode.add_argument("--build-db", action="store_true")
    parser.add_argument("--db-dir", default="gazou/db_retake10_02")
    parser.add_argument("--output", default="kekka/db_grid_v2_check_01")
    parser.add_argument("--checked-plan")
    parser.add_argument("--grout-cache", default="gazou/teian_grout_cache_grid_v2_01")
    args = parser.parse_args()
    if args.check_db:
        check_database(args)
    else:
        if not args.checked_plan:
            parser.error("--checked-planに目視確認済みのV2 split_plan.jsonを指定してください。")
        build_database(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, cv2.error) as exc:
        print("実行できませんでした：" + str(exc), file=sys.stderr)
        raise SystemExit(1)
