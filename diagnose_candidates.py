r"""タイル候補がどの段階で残る・消えるかを調べる診断用コード（検出・照合は変更しない）。

step4_auto_grout.py の検出処理（縮小→線分→消失点2系統→四角形の列挙→四辺の目地判定
→重複除去→上位max_candidates件）を同じ設定で再実行し、通常は保存されない情報を出す。

  * 各方向で四角形作りに使う上位16本の線分と、それ以外の線分（番号付き）
  * 四角形の除外理由ごとの件数（凸でない、面積、辺長比、角度、内部横断、四辺の目地らしさ）
  * 件数上限（既定6件）で打ち切られる前の全候補（番号・スコア・面積・四辺の支持率）
  * 全候補の一覧画像（1候補1枚）

既定設定では、診断内の候補列挙結果が step4_auto_grout.enumerate_quads と一致するかを
毎回確認し、一致しない場合は警告する（診断の再実装が本体とずれていないことの確認）。

正解のタイル・地点は一切使わない。どの候補が正しい完全タイルかは、出力画像を目視で判断する。

使用例（Windows PowerShell、yuka_shimon内）：
  & $env:USERPROFILE\.venvs\yuka_shimon\Scripts\python.exe -u diagnose_candidates.py --satsuei "gazou\satsuei\satsuei4_4.jpg" --min-length 0.05
  # 線分の上限16本を広げた場合に、正しい候補が現れるかを見る
  & $env:USERPROFILE\.venvs\yuka_shimon\Scripts\python.exe -u diagnose_candidates.py --satsuei "gazou\satsuei\satsuei4_4.jpg" --min-length 0.05 --group-limit 32
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, replace
from datetime import datetime
from itertools import combinations
import json
import math
import sys
import time

import cv2
import numpy as np

import step4_auto_grout as ag

DEFAULT_GROUP_LIMIT = 16  # step4_auto_grout.enumerate_quads 内の固定値


def line_rank_key(records):
    return lambda i: records[i]["score"] * math.sqrt(records[i]["length"])


def enumerate_quads_traced(gray, records, families, c, group_limit=DEFAULT_GROUP_LIMIT):
    """enumerate_quads と同じ判定を同じ順で行い、除外理由を記録する。

    戻り値：(重複除去・件数上限前の通過候補, 除外理由の集計, 各方向の使用線分番号)
    """
    reasons = Counter()
    if len(families) != 2:
        reasons["消失点の方向が2つ見つからない"] += 1
        return [], reasons, [[], []]
    groups = [sorted(f, key=line_rank_key(records), reverse=True)[:group_limit] for f in families]
    extent = min(gray.shape)
    area_total = gray.shape[0] * gray.shape[1]
    candidates = []
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0.6)
    for pair_a in combinations(groups[0], 2):
        for pair_b in combinations(groups[1], 2):
            a, b, d, e = [records[i]["line"] for i in (*pair_a, *pair_b)]
            corners = [ag.intersection(x, y) for x, y in ((a, d), (b, d), (b, e), (a, e))]
            if any(p is None for p in corners):
                reasons["交点なし（平行）"] += 1
                continue
            quad = ag.order_quad(corners)
            if not np.isfinite(quad).all() or not cv2.isContourConvex(quad):
                reasons["凸四角形でない"] += 1
                continue
            if (quad.min() < 2 or quad[:, 0].max() >= gray.shape[1] - 2
                    or quad[:, 1].max() >= gray.shape[0] - 2):
                reasons["四隅が画像外"] += 1
                continue
            area = abs(float(cv2.contourArea(quad)))
            if not c.min_area * area_total <= area <= c.max_area * area_total:
                reasons["面積が範囲外"] += 1
                continue
            vectors = np.roll(quad, -1, axis=0) - quad
            lengths = np.linalg.norm(vectors, axis=1)
            if lengths.min() < extent * 0.06 or lengths.max() / lengths.min() > 7:
                reasons["辺が短すぎる・辺長比が大きい"] += 1
                continue
            angle_sines = np.abs(vectors[:, 0] * np.roll(vectors, 1, axis=0)[:, 1]
                                 - vectors[:, 1] * np.roll(vectors, 1, axis=0)[:, 0])
            angle_sines /= lengths * np.roll(lengths, 1)
            if angle_sines.min() < 0.18:
                reasons["角が潰れている"] += 1
                continue
            used = set((*pair_a, *pair_b))
            crossed = False
            for i in set(groups[0] + groups[1]) - used:
                signed = np.c_[quad, np.ones(4)] @ records[i]["line"]
                if signed.min() < -8 and signed.max() > 8:
                    crossings = []
                    for j in range(4):
                        k = (j + 1) % 4
                        if signed[j] * signed[k] < 0:
                            fraction = signed[j] / (signed[j] - signed[k])
                            crossings.append(quad[j] + fraction * (quad[k] - quad[j]))
                    if len(crossings) == 2:
                        internal = ag.band_support(smooth, crossings[0], crossings[1],
                                                   c.dark_difference)
                        if internal["support"] >= c.line_support and internal["continuity"] >= 0.66:
                            crossed = True
                            break
            if crossed:
                reasons["内部を別の目地が横断（複数タイル）"] += 1
                continue
            sides = [ag.band_support(smooth, p, q, c.dark_difference)
                     for p, q in zip(quad, np.roll(quad, -1, axis=0))]
            supports = [r["support"] for r in sides]
            if min(supports) < c.side_support:
                reasons["四辺の目地支持率が不足"] += 1
                continue
            if min(r["continuity"] for r in sides) < 0.66:
                reasons["四辺の目地の連続性が不足"] += 1
                continue
            reasons["通過"] += 1
            score = (0.55 * min(supports) + 0.30 * np.mean(supports)
                     + 0.15 * np.mean([min(r["strength"] / 20, 1) for r in sides]))
            candidates.append(dict(quad=quad, score=float(score), area=area,
                                   side_support=supports,
                                   side_continuity=[r["continuity"] for r in sides],
                                   side_strength=[r["strength"] for r in sides],
                                   side_width=[r["width"] for r in sides],
                                   lines=[*pair_a, *pair_b]))
    candidates.sort(key=lambda r: r["score"], reverse=True)
    return candidates, reasons, groups


def deduplicate(candidates, limit):
    """enumerate_quads と同じ重複除去。件数上限で止めずに全件の状態を付ける。"""
    accepted = []
    for r in candidates:
        duplicate_of = None
        for number, other in accepted:
            overlap, _ = cv2.intersectConvexConvex(r["quad"], other["quad"])
            union = r["area"] + other["area"] - overlap
            if overlap / max(union, 1) > 0.65:
                duplicate_of = number
                break
        if duplicate_of is None:
            accepted.append((len(accepted) + 1, r))
            r["dedup_rank"] = len(accepted)
            r["status"] = "採用候補" if len(accepted) <= limit else f"上位{limit}件の外で打ち切り"
        else:
            r["dedup_rank"] = None
            r["status"] = f"重複（#{duplicate_of}と重なり）"
    return [r for _, r in accepted]


def same_quads(mine, reference):
    if len(mine) != len(reference):
        return False
    return all(np.allclose(a["quad"], b["quad"]) and abs(a["score"] - b["score"]) < 1e-9
               for a, b in zip(mine, reference))


def draw_lines(work, records, families, groups):
    picture = work.copy()
    colors = ((0, 180, 255), (255, 180, 0))
    family_of = {i: f for f, ids in enumerate(families) for i in ids}
    used = {i for g in groups for i in g}
    for i, r in enumerate(records):
        if i in family_of:
            color = colors[family_of[i]]
            thickness = 3 if i in used else 1
        else:
            color, thickness = (130, 130, 130), 1
        p = tuple(np.rint(r["p"]).astype(int))
        q = tuple(np.rint(r["q"]).astype(int))
        cv2.line(picture, p, q, color, thickness)
        middle = tuple(np.rint((r["p"] + r["q"]) / 2).astype(int))
        cv2.putText(picture, str(i), middle, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    return picture


def contact_sheet(work, rows, columns=4, width=360):
    scale = width / work.shape[1]
    height = round(work.shape[0] * scale)
    tiles = []
    for r in rows:
        tile = cv2.resize(work, (width, height), interpolation=cv2.INTER_AREA)
        quad = np.rint(r["quad"] * scale).astype(np.int32)
        color = (0, 255, 0) if r["status"] == "採用候補" else (0, 140, 255)
        cv2.polylines(tile, [quad], True, color, 2)
        label = f"#{r['dedup_rank']} s={r['score']:.3f} a={r['area_fraction']:.3f}"
        cv2.rectangle(tile, (0, 0), (width, 22), (255, 255, 255), -1)
        cv2.putText(tile, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return None
    blank = np.full_like(tiles[0], 255)
    while len(tiles) % columns:
        tiles.append(blank)
    rows_of = [np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
    return np.vstack(rows_of)


def main():
    parser = argparse.ArgumentParser(description="タイル候補の消え方の診断（検出・照合は変更しない）")
    parser.add_argument("--satsuei", required=True)
    parser.add_argument("--output")
    parser.add_argument("--group-limit", type=int, default=DEFAULT_GROUP_LIMIT,
                        help="各方向で四角形作りに使う線分の本数（本体は16固定）")
    parser.add_argument("--max-sheet", type=int, default=40, help="一覧画像に載せる候補数の上限")
    for name, field in ag.Config.__dataclass_fields__.items():
        if name in ("line_selection", "candidate_ranking"):
            continue
        flag = "--" + name.replace("_", "-")
        if field.type in ("bool", bool):
            parser.add_argument(flag, action="store_true")
        else:
            caster = {"int": int, "float": float}.get(str(field.type), type(field.default))
            parser.add_argument(flag, type=caster, default=field.default)
    parser.add_argument("--line-selection", choices=("balanced", "legacy", "diverse"), default="balanced")
    args = parser.parse_args()
    if args.group_limit < 2:
        parser.error("--group-limitは2以上です。")

    c = replace(ag.Config(), **{k: getattr(args, k) for k in ag.Config.__dataclass_fields__
                                if k != "candidate_ranking"})
    image_path = ag.path_of(args.satsuei)
    image = ag.read_image(image_path)
    output = (ag.path_of(args.output) if args.output else
              ag.BASE / "kekka" / ("diagnose_" + image_path.stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    output.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    factor = min(1., c.max_side / max(image.shape[:2]))
    size = (round(image.shape[1] * factor), round(image.shape[0] * factor))
    work = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    gray, _, _ = ag.prepare_detection_gray(work, c)
    line_info = {}
    records, _, _, raw_count = ag.extract_lines(gray, c, selection=c.line_selection, diagnostics=line_info)
    families, vps = ag.vp_families(records, gray.shape, c)

    passed, reasons, groups = enumerate_quads_traced(gray, records, families, c, args.group_limit)
    unique = deduplicate(passed, c.max_candidates)
    area_total = gray.shape[0] * gray.shape[1]
    top_score = unique[0]["score"] if unique else None
    for r in passed:
        r["area_fraction"] = r["area"] / area_total
        r["within_score_margin"] = top_score is not None and r["score"] + 1e-12 >= top_score - c.score_margin
        r["mean_y_fraction"] = float(np.mean(r["quad"][:, 1])) / gray.shape[0]

    consistent = None
    if args.group_limit == DEFAULT_GROUP_LIMIT:
        reference = ag.enumerate_quads(gray, records, families, c)
        consistent = same_quads(unique[:c.max_candidates], reference)
        if not consistent:
            print("警告：診断の候補が step4_auto_grout.enumerate_quads と一致しません。"
                  "本体が変更された可能性があります。結果を信用しないでください。", file=sys.stderr)

    xy_scale = np.array([work.shape[1] / image.shape[1], work.shape[0] / image.shape[0]])
    ag.save_image(output / "10_lines_numbered.jpg", draw_lines(work, records, families, groups))
    sheet = contact_sheet(work, unique[:args.max_sheet])
    if sheet is not None:
        ag.save_image(output / "11_candidates_sheet.jpg", sheet)

    with (output / "candidates_all.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["dedup_rank", "status", "score", "within_score_margin", "area_fraction",
                  "mean_y_fraction", "side_support", "side_continuity", "side_strength",
                  "lines", "quad_original_xy"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in passed:
            writer.writerow(dict(
                dedup_rank=r["dedup_rank"], status=r["status"], score=round(r["score"], 4),
                within_score_margin=r["within_score_margin"],
                area_fraction=round(r["area_fraction"], 4), mean_y_fraction=round(r["mean_y_fraction"], 3),
                side_support=[round(v, 3) for v in r["side_support"]],
                side_continuity=[round(v, 3) for v in r["side_continuity"]],
                side_strength=[round(v, 1) for v in r["side_strength"]],
                lines=r["lines"],
                quad_original_xy=";".join(f"{x:.1f},{y:.1f}" for x, y in r["quad"] / xy_scale)))

    family_sizes = [len(f) for f in families]
    lines = [
        "タイル候補の診断（検出・照合は変更していません）",
        f"撮影画像：{image_path.name}",
        f"設定：min-length={c.min_length:g}、line-selection={c.line_selection}、"
        f"CLAHE={'あり' if c.clahe else 'なし'}、四角形に使う線分=各方向{args.group_limit}本"
        + ("（本体と同じ）" if args.group_limit == DEFAULT_GROUP_LIMIT else "（本体は16本。診断用に変更）"),
        f"線分：検出{raw_count}本 → 目地条件を通過・統合後{len(records)}本",
        f"消失点の方向：{len(families)}系統、各系統の線分数 {family_sizes}"
        f"（四角形に使ったのは {[len(g) for g in groups]} 本）",
        "",
        "四角形の判定結果（件数）：",
        *[f"  {name}：{count}" for name, count in reasons.most_common()],
        "",
        f"重複除去後の候補：{len(unique)}件（採用候補は上位{c.max_candidates}件まで）",
        f"最高スコア：{top_score:.3f}" if top_score is not None else "最高スコア：候補なし",
        f"score-margin {c.score_margin:g} 以内の候補：{sum(r['within_score_margin'] for r in unique)}件",
        "",
        "見てほしいファイル：",
        "  11_candidates_sheet.jpg：重複除去後の全候補（緑=本体で採用される上位、橙=打ち切られる候補）",
        "  10_lines_numbered.jpg：線分（太線=四角形作りに使った線分、灰色=方向に分類されなかった線分）",
        "  candidates_all.csv：全候補のスコア・面積・四辺の支持率と、残った／消えた理由",
        "正しい完全タイルが一覧画像にあれば、その番号（#）を教えてください。無ければ「無い」と教えてください。",
    ]
    if consistent is not None:
        lines.append("本体との一致確認：" + ("一致" if consistent else "不一致（要確認）"))
    lines.append(f"診断時間：{time.perf_counter() - started:.2f}秒")
    (output / "diagnosis_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "diagnosis.json").write_text(json.dumps(dict(
        source_image=str(image_path), config=asdict(c), group_limit=args.group_limit,
        line_selection=line_info, family_sizes=family_sizes, groups=groups,
        vanishing_points=vps, reasons=dict(reasons), consistent_with_main=consistent,
        candidate_count_after_dedup=len(unique)), ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"保存先：{output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        print(f"実行できませんでした：{exc}", file=sys.stderr)
        raise SystemExit(1)
