"""登録用2×2画像の中央目地を再検出し、確認後に別のDBを作る。

既存ファイルは変更しない。step4_auto_grout の extract_lines / band_support
を再利用し、step4_teian_grout の特徴抽出・保存形式を維持する。
対象は、おおむね正面から撮影した2×2タイルの登録画像。
外周は元画像の端を使用するため、外周が目地に沿うことは別途目視確認が必要。
中央目地の検出失敗・曖昧さを画像中央へのフォールバックで隠さない。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
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
METHOD = "registration-central-dark-band-lines-v1"


def path_of(value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (BASE / p).resolve()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependencies():
    result = {name: digest(Path(module.__file__)) for name, module in (
        ("rebuild", sys.modules[__name__]), ("detector", detector), ("grout", grout))}
    for name in ("step3_senko_saigen", "step4_teian"):
        result[name] = digest(Path(sys.modules[name].__file__))
    return result


def central_line_candidates(gray, records, axis, config):
    """中心30～70%を通る、ほぼ縦／横の線を画像全体に延長して再評価。"""
    h, w = gray.shape
    candidates = []
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), .6)
    for record in records:
        a, b, c = record["line"]
        if axis == "vertical":
            if abs(a) < 1e-9 or abs(b / a) > math.tan(math.radians(25)):
                continue
            p = np.array([-c / a, 0.])
            q = np.array([-(b * (h - 1) + c) / a, h - 1.])
            position = (p[0] + q[0]) / 2
            extent = w
            inside = 0 <= min(p[0], q[0]) and max(p[0], q[0]) <= w - 1
        else:
            if abs(b) < 1e-9 or abs(a / b) > math.tan(math.radians(25)):
                continue
            p = np.array([0., -c / b])
            q = np.array([w - 1., -(a * (w - 1) + c) / b])
            position = (p[1] + q[1]) / 2
            extent = h
            inside = 0 <= min(p[1], q[1]) and max(p[1], q[1]) <= h - 1
        if not inside or not extent * .30 <= position <= extent * .70:
            continue
        # 撮影用と同じ「中心帯が両側より暗い割合」を再利用。
        # 短い線分の傾き誤差を全幅へ延長すると数pxずれるため、
        # 各端点を±6pxだけ動かし、全長で同じ暗帯条件を最も満たす線へ調整。
        tested = []
        component = 0 if axis == "vertical" else 1
        limit = w - 1 if component == 0 else h - 1
        for first in (-6., -3., 0., 3., 6.):
            for last in (-6., -3., 0., 3., 6.):
                rp, rq = p.copy(), q.copy()
                rp[component] += first
                rq[component] += last
                if not (0 <= rp[component] <= limit and 0 <= rq[component] <= limit):
                    continue
                candidate = detector.band_support(smooth, rp, rq, config.dark_difference, search=False)
                tested.append(candidate)
        if not tested:
            continue
        support = max(tested, key=lambda item: item["score"])
        p, q = support["p"], support["q"]
        position = float((p[component] + q[component]) / 2)
        if support["support"] < config.line_support or support["continuity"] < .66:
            continue
        item = {key: float(support[key]) for key in (
            "score", "support", "continuity", "strength", "width")}
        item.update(p=p.tolist(), q=q.tolist(), position=float(position))
        candidates.append(item)
    candidates.sort(key=lambda row: row["score"], reverse=True)
    # 同じ目地の近接候補をまとめる。別の線との僅差は曖昧として停止。
    distinct = []
    for item in candidates:
        if not any(max(np.linalg.norm(np.array(item["p"]) - other["p"]),
                       np.linalg.norm(np.array(item["q"]) - other["q"])) < 6
                   for other in distinct):
            distinct.append(item)
    return distinct


def select_line(candidates, axis, extent):
    if not candidates:
        raise ValueError(f"{axis}：両側より暗く連続する中央目地を確認できません。")
    best = candidates[0]
    for other in candidates[1:]:
        if abs(best["position"] - other["position"]) > extent * .03:
            if best["score"] - other["score"] < .05:
                raise ValueError(f"{axis}：離れた2本の目地候補が僅差です。自動分割を保留します。")
            break
    return best


def split_quads(image):
    h, w = image.shape[:2]
    if min(h, w) < 160:
        raise ValueError("画像が小さすぎます。短辺160px以上の元画像を指定してください。")
    # 登録写真の細かな面模様を縮小で抑える。照合画像は縮小しない。
    config = detector.Config(max_side=640)
    scale = min(1., config.max_side / max(h, w))
    small = cv2.resize(image, (round(w * scale), round(h * scale)),
                       interpolation=cv2.INTER_AREA) if scale < 1 else image
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sh, sw = gray.shape
    records, _, _, _ = detector.extract_lines(gray, config)
    vs = central_line_candidates(gray, records, "vertical", config)
    hs = central_line_candidates(gray, records, "horizontal", config)
    vertical = select_line(vs, "vertical", sw)
    horizontal = select_line(hs, "horizontal", sh)
    # 実際の縮小後寸法に合わせて端点を元画像座標へ戻す。
    factor = np.array([(w - 1) / (sw - 1), (h - 1) / (sh - 1)])
    vt, vb = [np.array(vertical[key]) * factor for key in ("p", "q")]
    hl, hr = [np.array(horizontal[key]) * factor for key in ("p", "q")]
    cross = np.cross(detector.line_of(vt, vb), detector.line_of(hl, hr))
    if abs(cross[2]) < 1e-9:
        raise ValueError("中央の縦目地と横目地が交差しません。")
    center = cross[:2] / cross[2]
    if not (.25 * w < center[0] < .75 * w and .25 * h < center[1] < .75 * h):
        raise ValueError("中央交点が2×2画像の中央範囲から外れています。")
    quads = [np.float32(points) for points in (
        [(0, 0), vt, center, hl], [vt, (w - 1, 0), hr, center],
        [hl, center, vb, (0, h - 1)], [center, hr, (w - 1, h - 1), vb])]
    for quad in quads:
        if not cv2.isContourConvex(quad) or cv2.contourArea(quad) < .06 * w * h:
            raise ValueError("分割後の四角形が不正です。")
    return quads, {"method": METHOD, "center": center.tolist(),
        "vertical": vertical, "horizontal": horizontal,
        "vertical_endpoints_original": [vt.tolist(), vb.tolist()],
        "horizontal_endpoints_original": [hl.tolist(), hr.tolist()],
        "detector_config": asdict(config), "vertical_candidates": vs[:5],
        "horizontal_candidates": hs[:5],
        "outer_boundary": "image edges; not automatically verified as grout"}


def overlay(image, location, quads):
    result = image.copy()
    thickness = max(2, round(max(image.shape[:2]) / 700))
    font_scale = max(.6, max(image.shape[:2]) / 1300)
    for index, quad in enumerate(quads, 1):
        cv2.polylines(result, [np.rint(quad).astype(np.int32)], True,
                      (0, 220, 0), thickness)
        xy = tuple(np.rint(quad.mean(axis=0)).astype(int))
        cv2.putText(result, f"{location}-{index}", xy, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 255), thickness)
    return result


def check_database(args):
    images = grout.numbered_images(path_of(args.db_dir), "db_")
    if not images:
        raise ValueError("登録画像がありません。db_01.png / db_10.jpgなどの元画像を指定してください。")
    locations = [location for location, _ in images]
    if len(locations) != len(set(locations)):
        raise ValueError("同じDB番号の画像が複数あります。重複を解消してください。")
    output = path_of(args.output)
    output.mkdir(parents=True, exist_ok=False)
    preview_dir = output / "registration_previews"
    preview_dir.mkdir()
    rows = []
    for location, path in images:
        print(f"中央目地の確認：db_{location:02d}", flush=True)
        row = {"location": location, "source_image": str(path),
               "source_sha256": digest(path)}
        image = grout.read_image(path)
        try:
            quads, info = split_quads(image)
            row.update(status="detected", quads=[q.tolist() for q in quads], split=info)
            picture = overlay(image, location, quads)
            print(f"  検出：中央交点 {info['center']}", flush=True)
        except ValueError as exc:
            row.update(status="failed", reason=str(exc))
            picture = image.copy()
            print(f"  保留：{exc}", flush=True)
        grout.save_image(preview_dir / f"db_{location:02d}.jpg", picture)
        grout.write_json(preview_dir / f"db_{location:02d}.json", row)
        rows.append(row)
    plan = {"method": METHOD, "created_at": datetime.now().isoformat(),
            "dependencies_sha256": dependencies(), "images": rows,
            "notice": "Preview central joints AND outer image borders before build. No matching run."}
    grout.write_json(output / "split_plan.json", plan)
    failed = sum(row["status"] != "detected" for row in rows)
    print(f"確認画像：{preview_dir}\n検出 {len(rows) - failed}/{len(rows)}枚", flush=True)
    print("DBはまだ作成していません。中央線と外周が実際の目地に合うか確認してください。")
    if failed:
        raise ValueError("検出保留の画像があります。この確認結果ではDBを作成できません。")


def build_checked_database(args):
    plan_path = path_of(args.checked_plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("method") != METHOD or plan.get("dependencies_sha256") != dependencies():
        raise ValueError("確認後にコードが変わったか、確認ファイルの形式が違います。確認からやり直してください。")
    rows = plan.get("images", [])
    if not rows or any(row.get("status") != "detected" for row in rows):
        raise ValueError("すべての登録画像の検出が完了した確認ファイルを指定してください。")
    if len({row["location"] for row in rows}) != len(rows):
        raise ValueError("確認ファイル内でDB番号が重複しています。")
    # 保存先を作る前に、全入力の不変性と四角形を確認。
    for row in rows:
        source = Path(row["source_image"])
        if digest(source) != row["source_sha256"]:
            raise ValueError(f"確認後に元画像が変わりました：{source}")
        image = grout.read_image(source)
        quads = row.get("quads", [])
        if len(quads) != 4:
            raise ValueError("確認ファイルの四角形は4つ必要です。")
        for q in quads:
            q = np.asarray(q, np.float32)
            if q.shape != (4, 2) or not np.isfinite(q).all() or not cv2.isContourConvex(q):
                raise ValueError("確認ファイルの四角形が不正です。")
            grout.require_inside(q, image.shape)
    output = path_of(args.grout_cache)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("tiles", "features", "registration_previews"):
        (output / name).mkdir()
    config = grout.Config()
    result_rows = []
    started = time.perf_counter()
    for row in rows:
        image = grout.read_image(Path(row["source_image"]))
        quads = [np.asarray(q, np.float32) for q in row["quads"]]
        location = row["location"]
        for index, quad in enumerate(quads, 1):
            name = f"db_{location:02d}_tile_{index}"
            print(f"登録候補抽出 [{len(result_rows) + 1}/{len(rows) * 4}] {name}", flush=True)
            pool = grout.extract_pool(image, quad, config)
            feature_file = f"features/{name}.npz"
            np.savez_compressed(output / feature_file,
                **{key: pool[key] for key in ("points", "descriptors", "responses",
                    "sizes", "on_grout", "quad", "matrix")})
            grout.save_pool_visuals(output / "tiles" / name, pool, config)
            result_rows.append({"location": location, "tile_index": index, "name": name,
                "feature_file": feature_file,
                "image_file": f"tiles/{name}/01_rectified_with_context.png",
                "source_image": row["source_image"], "source_sha256": row["source_sha256"],
                **grout.pool_stats(pool)})
        grout.save_image(output / "registration_previews" / f"db_{location:02d}.jpg",
                         overlay(image, location, quads))
        grout.write_json(output / "registration_previews" / f"db_{location:02d}.json", row)
    metadata = {"format": grout.FORMAT, "registration_method": METHOD,
        "created_at": datetime.now().isoformat(), "config": asdict(config),
        "opencv_version": cv2.__version__, "numpy_version": np.__version__,
        "python_version": sys.version.split()[0], "tiles": result_rows,
        "elapsed_seconds": time.perf_counter() - started,
        "notes": "Central dark-band lines corrected. Outer borders still assumed. Matching unchanged."}
    grout.write_json(output / "database.json", metadata)
    grout.write_json(output / "split_plan.json", plan)
    grout.write_csv(output / "feature_pool_counts.csv", result_rows)
    print(f"DB作成完了：{len(rows)}地点・{len(result_rows)}タイル\n保存先：{output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-db", action="store_true")
    mode.add_argument("--build-db", action="store_true")
    parser.add_argument("--db-dir", default="gazou/db_retake10_02")
    parser.add_argument("--output", default="kekka/db_split_darkband_check_01")
    parser.add_argument("--checked-plan")
    parser.add_argument("--grout-cache", default="gazou/teian_grout_cache_darkband_01")
    args = parser.parse_args()
    if args.check_db:
        check_database(args)
    else:
        if not args.checked_plan:
            parser.error("--checked-plan に目視確認済みの split_plan.json を指定してください。")
        build_checked_database(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, cv2.error) as exc:
        print(f"実行できませんでした：{exc}", file=sys.stderr)
        raise SystemExit(1)
