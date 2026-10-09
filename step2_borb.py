import cv2
import numpy as np
import math
import os


# ============================================================
# 設定
# ============================================================

# 先行研究の基本実験：
# B-ORBの特徴点抽出数の上限は10000点
NFEATURES = 10000

# 先行研究の基本実験：
# 元画像を縦横それぞれ1/4に縮小
IMAGE_SCALE = 0.25

# 論文では具体値の記載なし
# OpenCV ORBの標準値に合わせる
SCALE_FACTOR = 1.2
NLEVELS = 8

# 論文：
# bundle数は4～10程度
# 今回は3×3 = 9 bundle
BUNDLE_ROWS = 3
BUNDLE_COLS = 3

# FASTの初期閾値
# 論文には具体値の記載なし
FAST_THRESHOLD = 20

# ORB標準値に近い設定
EDGE_THRESHOLD = 31
PATCH_SIZE = 31

# Harris応答値計算用
HARRIS_BLOCK_SIZE = 7
HARRIS_K = 0.04


# ============================================================
# 平均値差分フィルタ
# ============================================================

def mean_difference_filter(gray, radius=20):
    """
    平均値差分フィルタ

    論文：
    上下左右20ドット離れた点までの
    約1600点の正方形領域から平均輝度を求め、
    対象画素の輝度との差を数値化する。

    B-ORBでは多階調版を使用する。
    """

    ksize = radius * 2 + 1

    # 負の差分を扱えるようfloat32へ変換
    gray_float = gray.astype(np.float32)

    # 周囲の平均輝度
    local_mean = cv2.blur(
        gray_float,
        (ksize, ksize),
        borderType=cv2.BORDER_REFLECT
    )

    # 対象画素 - 周辺平均
    diff = gray_float - local_mean

    # 論文では多階調化の具体的変換式は記載されていない
    # ここでは差分0を128として8bit画像化する
    filtered = diff + 128.0

    filtered = np.clip(
        filtered,
        0,
        255
    ).astype(np.uint8)

    return filtered


def print_filter_check(name, gray, filtered):
    """B-ORBへ渡す画像がフィルタ後画像であることを数値でも確認する。"""
    detail = filtered.astype(np.float32) - 128.0
    p1, p99 = np.percentile(detail, [1, 99])
    changed_rate = float(np.mean(gray != filtered) * 100.0)
    print(
        f"{name}のB-ORB入力：平均値差分フィルタ後 / "
        f"元画像から変化した画素={changed_rate:.1f}% / "
        f"差分範囲={detail.min():.1f}～{detail.max():.1f} / "
        f"差分標準偏差={detail.std():.2f} / "
        f"1～99%範囲={p1:.1f}～{p99:.1f}"
    )


def make_visible_preview(filtered):
    """差分を見やすく伸ばした確認用画像。B-ORBの計算には使用しない。"""
    detail = filtered.astype(np.float32) - 128.0
    scale = max(float(np.percentile(np.abs(detail), 99)), 1.0)
    preview = 128.0 + detail * (127.0 / scale)
    return np.clip(preview, 0, 255).astype(np.uint8)


# ============================================================
# 各画像ピラミッドレイヤへの特徴点数の配分
# ============================================================

def calculate_level_limits(
    nfeatures,
    scale_factor,
    nlevels
):
    """
    ORBと同様の考え方で、
    総特徴点数を各レイヤへ配分する。
    """

    factor = 1.0 / scale_factor

    desired = (
        nfeatures
        * (1.0 - factor)
        / (1.0 - factor ** nlevels)
    )

    limits = []

    total = 0

    for _ in range(nlevels - 1):

        value = int(
            math.floor(
                desired + 0.5
            )
        )

        limits.append(value)

        total += value

        desired *= factor

    # 最終レイヤで合計を調整
    limits.append(
        max(
            nfeatures - total,
            0
        )
    )

    return limits


# ============================================================
# 画像ピラミッド作成
# ============================================================

def build_pyramid(
    gray,
    scale_factor,
    nlevels
):
    """
    解像度の異なる画像を作成する。
    """

    height, width = gray.shape[:2]

    pyramid = []
    scales = []

    for level in range(nlevels):

        scale = scale_factor ** level

        new_width = max(
            1,
            int(
                round(
                    width / scale
                )
            )
        )

        new_height = max(
            1,
            int(
                round(
                    height / scale
                )
            )
        )

        if level == 0:

            layer = gray.copy()

        else:

            layer = cv2.resize(
                gray,
                (new_width, new_height),
                interpolation=cv2.INTER_LINEAR
            )

        pyramid.append(layer)

        scales.append(scale)

    return pyramid, scales


# ============================================================
# bundleごとの特徴点上限
# ============================================================

def calculate_bundle_quotas(
    total_limit,
    rows,
    cols,
    width,
    height
):
    """
    各bundleの面積に比例して
    特徴点上限を配分する。
    """

    x_edges = np.linspace(
        0,
        width,
        cols + 1,
        dtype=int
    )

    y_edges = np.linspace(
        0,
        height,
        rows + 1,
        dtype=int
    )

    areas = []

    for row in range(rows):

        for col in range(cols):

            region_width = (
                x_edges[col + 1]
                - x_edges[col]
            )

            region_height = (
                y_edges[row + 1]
                - y_edges[row]
            )

            area = (
                region_width
                * region_height
            )

            areas.append(area)

    total_area = sum(areas)

    # 面積比例で上限数を計算
    raw_quotas = [
        total_limit
        * area
        / total_area
        for area in areas
    ]

    quotas = [
        int(
            math.floor(value)
        )
        for value in raw_quotas
    ]

    remaining = (
        total_limit
        - sum(quotas)
    )

    # 切り捨てで余った分を
    # 小数部分の大きいbundleから追加
    order = sorted(
        range(
            len(raw_quotas)
        ),
        key=lambda i:
        raw_quotas[i]
        - quotas[i],
        reverse=True
    )

    for index in order[:remaining]:

        quotas[index] += 1

    return quotas


# ============================================================
# FAST特徴点候補抽出
# ============================================================

def detect_fast_candidates(
    image,
    desired_count,
    edge_threshold
):
    """
    FASTで特徴点候補を取得する。

    閾値20から開始し、
    特徴点候補が不足した場合のみ
    10 → 5と下げる。

    ※この閾値変更は論文に具体記載がないため
      実装上の補完。
    """

    thresholds = [
        FAST_THRESHOLD,
        10,
        5
    ]

    final_keypoints = []

    used_threshold = FAST_THRESHOLD

    for threshold in thresholds:

        fast = (
            cv2.FastFeatureDetector_create(
                threshold=threshold,
                nonmaxSuppression=True,
                type=(
                    cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
                )
            )
        )

        keypoints = fast.detect(
            image,
            None
        )

        height, width = (
            image.shape[:2]
        )

        valid = []

        for kp in keypoints:

            x, y = kp.pt

            # ORB記述子を作成できない
            # 画像端付近を除外
            if (
                edge_threshold
                <= x
                < width - edge_threshold
                and
                edge_threshold
                <= y
                < height - edge_threshold
            ):

                valid.append(kp)

        final_keypoints = valid

        used_threshold = threshold

        # 必要な候補数が確保できれば終了
        if (
            len(final_keypoints)
            >= desired_count
        ):

            break

    return (
        final_keypoints,
        used_threshold
    )


# ============================================================
# Harris応答値
# ============================================================

def calculate_harris_map(image):
    """
    Harrisコーナー応答値を計算する。
    """

    image_float = (
        image.astype(
            np.float32
        )
    )

    harris = cv2.cornerHarris(
        image_float,
        blockSize=HARRIS_BLOCK_SIZE,
        ksize=3,
        k=HARRIS_K,
        borderType=cv2.BORDER_REFLECT101
    )

    return harris


# ============================================================
# ORBの方向計算
# ============================================================

def calculate_orientation(
    image,
    x,
    y,
    radius=15
):
    """
    Intensity Centroid法に近い方法で
    特徴点の方向を求める。
    """

    cx = int(round(x))
    cy = int(round(y))

    patch = image[
        cy - radius:
        cy + radius + 1,

        cx - radius:
        cx + radius + 1
    ].astype(
        np.float32
    )

    expected_size = (
        radius * 2 + 1
    )

    if (
        patch.shape[0]
        != expected_size
        or
        patch.shape[1]
        != expected_size
    ):

        return 0.0

    yy, xx = np.mgrid[
        -radius:
        radius + 1,

        -radius:
        radius + 1
    ]

    circle = (
        xx * xx
        + yy * yy
        <= radius * radius
    )

    values = (
        patch * circle
    )

    m10 = np.sum(
        xx * values
    )

    m01 = np.sum(
        yy * values
    )

    angle = (
        math.degrees(
            math.atan2(
                float(m01),
                float(m10)
            )
        )
    )

    if angle < 0:

        angle += 360.0

    return angle


# ============================================================
# B-ORB
# ============================================================

def balanced_orb_detect_and_compute(
    gray,
    nfeatures=10000,
    scale_factor=1.2,
    nlevels=8,
    bundle_rows=3,
    bundle_cols=3
):
    """
    論文記載のB-ORBを可能な範囲で再実装。

    処理：

    1. 画像ピラミッドを作成

    2. 各レイヤの特徴点上限を
       通常ORBの2倍に緩和

    3. FASTで候補特徴点を抽出

    4. 各レイヤをbundleへ分割

    5. bundle面積に比例して
       特徴点上限を配分

    6. bundleごとに上位特徴点を選択

    7. 全レイヤの特徴点をまとめる

    8. Harris応答値で全体をソート

    9. 本来の特徴点上限まで残す

    10. ORB記述子を計算
    """

    # --------------------------------------------------------
    # 画像ピラミッド
    # --------------------------------------------------------

    pyramid, scales = (
        build_pyramid(
            gray,
            scale_factor,
            nlevels
        )
    )

    # 通常ORBの各レイヤの上限
    normal_limits = (
        calculate_level_limits(
            nfeatures,
            scale_factor,
            nlevels
        )
    )

    all_records = []

    statistics = {}

    # ========================================================
    # 各レイヤ
    # ========================================================

    for level in range(
        nlevels
    ):

        layer = pyramid[level]

        scale = scales[level]

        height, width = (
            layer.shape[:2]
        )

        normal_limit = (
            normal_limits[level]
        )

        # ----------------------------------------------------
        # 論文手順(1)
        # 各レイヤの上限を2倍
        # ----------------------------------------------------

        relaxed_limit = (
            normal_limit * 2
        )

        # ----------------------------------------------------
        # FAST特徴点候補
        # ----------------------------------------------------

        (
            candidates,
            used_threshold
        ) = detect_fast_candidates(
            layer,
            relaxed_limit,
            EDGE_THRESHOLD
        )

        # ----------------------------------------------------
        # bundleごとの特徴点上限
        # ----------------------------------------------------

        bundle_quotas = (
            calculate_bundle_quotas(
                relaxed_limit,
                bundle_rows,
                bundle_cols,
                width,
                height
            )
        )

        bundle_count = (
            bundle_rows
            * bundle_cols
        )

        bundles = [
            []
            for _ in range(
                bundle_count
            )
        ]

        # ----------------------------------------------------
        # 論文手順(2)
        # 特徴点をbundleへ振り分ける
        # ----------------------------------------------------

        for kp in candidates:

            x, y = kp.pt

            col = min(
                int(
                    x
                    * bundle_cols
                    / width
                ),
                bundle_cols - 1
            )

            row = min(
                int(
                    y
                    * bundle_rows
                    / height
                ),
                bundle_rows - 1
            )

            bundle_index = (
                row
                * bundle_cols
                + col
            )

            bundles[
                bundle_index
            ].append(kp)

        # ----------------------------------------------------
        # bundleごとに特徴点を選択
        # ----------------------------------------------------

        selected_this_level = []

        bundle_before_counts = []

        bundle_after_counts = []

        for bundle_index in range(
            bundle_count
        ):

            bundle_points = (
                bundles[
                    bundle_index
                ]
            )

            bundle_before_counts.append(
                len(bundle_points)
            )

            # FAST応答値が高い順
            bundle_points.sort(
                key=lambda kp:
                kp.response,
                reverse=True
            )

            quota = (
                bundle_quotas[
                    bundle_index
                ]
            )

            selected = (
                bundle_points[
                    :quota
                ]
            )

            selected_this_level.extend(
                selected
            )

            bundle_after_counts.append(
                len(selected)
            )

        # ----------------------------------------------------
        # Harris応答値を計算
        # ----------------------------------------------------

        harris_map = (
            calculate_harris_map(
                layer
            )
        )

        for kp in (
            selected_this_level
        ):

            x = int(
                round(
                    kp.pt[0]
                )
            )

            y = int(
                round(
                    kp.pt[1]
                )
            )

            if (
                0 <= x < width
                and
                0 <= y < height
            ):

                harris_response = float(
                    harris_map[
                        y,
                        x
                    ]
                )

                all_records.append(
                    {
                        "level":
                            level,

                        "scale":
                            scale,

                        "keypoint":
                            kp,

                        "harris":
                            harris_response
                    }
                )

        statistics[level] = {

            "normal_limit":
                normal_limit,

            "relaxed_limit":
                relaxed_limit,

            "FAST_candidates":
                len(candidates),

            "FAST_threshold":
                used_threshold,

            "bundle_before":
                bundle_before_counts,

            "bundle_after":
                bundle_after_counts,

            "after_bundle":
                len(
                    selected_this_level
                ),

            "final":
                0
        }

    # ========================================================
    # 論文手順(3)
    # 全レイヤをまとめてHarris基準で最終選択
    # ========================================================

    all_records.sort(
        key=lambda record:
        record["harris"],
        reverse=True
    )

    # 元の特徴点上限まで残す
    final_records = (
        all_records[
            :nfeatures
        ]
    )

    for record in (
        final_records
    ):

        statistics[
            record["level"]
        ]["final"] += 1

    # ========================================================
    # ORB記述子作成
    # ========================================================

    final_keypoints = []

    final_descriptors = []

    for level in range(
        nlevels
    ):

        level_records = [
            record
            for record
            in final_records
            if record["level"]
            == level
        ]

        if (
            len(level_records)
            == 0
        ):

            continue

        layer = (
            pyramid[level]
        )

        scale = (
            scales[level]
        )

        local_keypoints = []

        # ----------------------------------------------------
        # 特徴点の向きを計算
        # ----------------------------------------------------

        for record in (
            level_records
        ):

            kp = (
                record[
                    "keypoint"
                ]
            )

            angle = (
                calculate_orientation(
                    layer,
                    kp.pt[0],
                    kp.pt[1],
                    radius=(
                        PATCH_SIZE // 2
                    )
                )
            )

            new_kp = cv2.KeyPoint(
                float(
                    kp.pt[0]
                ),
                float(
                    kp.pt[1]
                ),
                float(
                    PATCH_SIZE
                ),
                float(
                    angle
                ),
                float(
                    record[
                        "harris"
                    ]
                ),
                0,
                -1
            )

            local_keypoints.append(
                new_kp
            )

        # ----------------------------------------------------
        # ORB記述子
        # ----------------------------------------------------

        descriptor_orb = (
            cv2.ORB_create(
                nfeatures=max(
                    len(
                        local_keypoints
                    ),
                    1
                ),
                scaleFactor=(
                    scale_factor
                ),
                nlevels=1,
                edgeThreshold=(
                    EDGE_THRESHOLD
                ),
                patchSize=(
                    PATCH_SIZE
                ),
                scoreType=(
                    cv2.ORB_HARRIS_SCORE
                ),
                fastThreshold=(
                    FAST_THRESHOLD
                )
            )
        )

        (
            computed_keypoints,
            descriptors
        ) = descriptor_orb.compute(
            layer,
            local_keypoints
        )

        if (
            descriptors is None
        ):

            continue

        # ----------------------------------------------------
        # 元画像座標へ戻す
        # ----------------------------------------------------

        for (
            kp,
            descriptor
        ) in zip(
            computed_keypoints,
            descriptors
        ):

            original_kp = (
                cv2.KeyPoint(
                    float(
                        kp.pt[0]
                        * scale
                    ),
                    float(
                        kp.pt[1]
                        * scale
                    ),
                    float(
                        kp.size
                        * scale
                    ),
                    float(
                        kp.angle
                    ),
                    float(
                        kp.response
                    ),
                    int(
                        level
                    ),
                    int(
                        kp.class_id
                    )
                )
            )

            final_keypoints.append(
                original_kp
            )

            final_descriptors.append(
                descriptor
            )

    if (
        len(
            final_descriptors
        )
        == 0
    ):

        return (
            [],
            None,
            statistics
        )

    final_descriptors = (
        np.asarray(
            final_descriptors,
            dtype=np.uint8
        )
    )

    return (
        final_keypoints,
        final_descriptors,
        statistics
    )


# ============================================================
# B-ORB特徴点のみを描画
# ============================================================

def draw_borb_only(
    image,
    keypoints
):
    """
    B-ORB特徴点のみを緑色で表示する。
    bundle境界線は表示しない。
    """

    output = image.copy()

    height, width = (
        output.shape[:2]
    )

    for kp in keypoints:

        x = int(
            round(
                kp.pt[0]
            )
        )

        y = int(
            round(
                kp.pt[1]
            )
        )

        if (
            0 <= x < width
            and
            0 <= y < height
        ):

            cv2.circle(
                output,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )

    return output


def print_borb_balance_check(name, statistics):
    """bundle分割で候補が実際に絞られたかを表示する。"""
    candidates = sum(data["FAST_candidates"] for data in statistics.values())
    after_bundle = sum(data["after_bundle"] for data in statistics.values())
    final = sum(data["final"] for data in statistics.values())
    removed = candidates - after_bundle
    print(
        f"{name}のB-ORB絞り込み：FAST候補={candidates} / "
        f"bundle後={after_bundle} / bundleで除外={removed} / 最終={final}"
    )
    if removed == 0:
        print(
            "  注意：bundleの上限に達していないため、"
            "今回の画像ではB-ORBの領域別制限が発動していません。"
        )


# ============================================================
# B-ORB特徴点＋bundleを描画
# ============================================================

def draw_borb_grid(
    image,
    keypoints,
    rows,
    cols
):
    """
    同じB-ORB特徴点に対して
    bundle境界とbundleごとの特徴点数を表示する。
    """

    output = image.copy()

    height, width = (
        output.shape[:2]
    )

    counts = np.zeros(
        (
            rows,
            cols
        ),
        dtype=int
    )

    # --------------------------------------------------------
    # 特徴点
    # --------------------------------------------------------

    for kp in keypoints:

        x = int(
            round(
                kp.pt[0]
            )
        )

        y = int(
            round(
                kp.pt[1]
            )
        )

        if not (
            0 <= x < width
            and
            0 <= y < height
        ):

            continue

        # 特徴点を緑色で表示
        cv2.circle(
            output,
            (x, y),
            2,
            (0, 255, 0),
            -1
        )

        col = min(
            int(
                x
                * cols
                / width
            ),
            cols - 1
        )

        row = min(
            int(
                y
                * rows
                / height
            ),
            rows - 1
        )

        counts[
            row,
            col
        ] += 1

    # --------------------------------------------------------
    # bundle境界
    # --------------------------------------------------------

    for col in range(
        1,
        cols
    ):

        x = int(
            width
            * col
            / cols
        )

        cv2.line(
            output,
            (x, 0),
            (x, height),
            (0, 0, 255),
            2
        )

    for row in range(
        1,
        rows
    ):

        y = int(
            height
            * row
            / rows
        )

        cv2.line(
            output,
            (0, y),
            (width, y),
            (0, 0, 255),
            2
        )

    # --------------------------------------------------------
    # bundleごとの特徴点数
    # --------------------------------------------------------

    for row in range(
        rows
    ):

        for col in range(
            cols
        ):

            x = int(
                width
                * col
                / cols
            ) + 10

            y = int(
                height
                * row
                / rows
            ) + 30

            text = str(
                counts[
                    row,
                    col
                ]
            )

            # 白い縁
            cv2.putText(
                output,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                4,
                cv2.LINE_AA
            )

            # 青文字
            cv2.putText(
                output,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )

    return (
        output,
        counts
    )


# ============================================================
# 結果保存フォルダ
# ============================================================

os.makedirs(
    "kekka",
    exist_ok=True
)


# ============================================================
# 画像読み込み
# ============================================================

touroku = cv2.imread(
    "gazou/touroku.jpg"
)

satsuei = cv2.imread(
    "gazou/satsuei_real.jpg"
)

if (
    touroku is None
    or
    satsuei is None
):

    raise FileNotFoundError(
        "画像を読み込めませんでした"
    )


# ============================================================
# 先行研究と同じく縦横1/4に縮小
# ============================================================

if (
    IMAGE_SCALE
    != 1.0
):

    touroku = cv2.resize(
        touroku,
        None,
        fx=IMAGE_SCALE,
        fy=IMAGE_SCALE,
        interpolation=cv2.INTER_AREA
    )

    satsuei = cv2.resize(
        satsuei,
        None,
        fx=IMAGE_SCALE,
        fy=IMAGE_SCALE,
        interpolation=cv2.INTER_AREA
    )


# 縮小画像
cv2.imwrite(
    "kekka/10_touroku_resized.jpg",
    touroku
)

cv2.imwrite(
    "kekka/11_satsuei_resized.jpg",
    satsuei
)


# ============================================================
# グレースケール化
# ============================================================

gray1 = cv2.cvtColor(
    touroku,
    cv2.COLOR_BGR2GRAY
)

gray2 = cv2.cvtColor(
    satsuei,
    cv2.COLOR_BGR2GRAY
)


# ============================================================
# 平均値差分フィルタ
# ============================================================

filtered1 = (
    mean_difference_filter(
        gray1,
        radius=20
    )
)

filtered2 = (
    mean_difference_filter(
        gray2,
        radius=20
    )
)


# B-ORBへ実際に渡す8bit画像。微小な差をJPEG圧縮で壊さないようPNGで保存する。
cv2.imwrite(
    "kekka/12_touroku_mean_difference.png",
    filtered1
)

cv2.imwrite(
    "kekka/13_satsuei_mean_difference.png",
    filtered2
)

# 見やすさだけを調整した確認用画像。これはB-ORBへ渡さない。
cv2.imwrite(
    "kekka/12b_touroku_mean_difference_preview.png",
    make_visible_preview(filtered1)
)

cv2.imwrite(
    "kekka/13b_satsuei_mean_difference_preview.png",
    make_visible_preview(filtered2)
)

print_filter_check("登録画像", gray1, filtered1)
print_filter_check("撮影画像", gray2, filtered2)

# 特徴点描画用の背景。計算に使うfiltered1/2と同じ画像をカラー化しただけ。
filtered1_bgr = cv2.cvtColor(filtered1, cv2.COLOR_GRAY2BGR)
filtered2_bgr = cv2.cvtColor(filtered2, cv2.COLOR_GRAY2BGR)


# ============================================================
# B-ORB
# filtered1/2を渡すため、FAST検出・Harris応答・方向・ORB記述子の
# すべてが平均値差分フィルタ後の画像から計算される。
# ============================================================

(
    keypoints1,
    descriptors1,
    stats1
) = balanced_orb_detect_and_compute(
    filtered1,
    nfeatures=NFEATURES,
    scale_factor=SCALE_FACTOR,
    nlevels=NLEVELS,
    bundle_rows=BUNDLE_ROWS,
    bundle_cols=BUNDLE_COLS
)


(
    keypoints2,
    descriptors2,
    stats2
) = balanced_orb_detect_and_compute(
    filtered2,
    nfeatures=NFEATURES,
    scale_factor=SCALE_FACTOR,
    nlevels=NLEVELS,
    bundle_rows=BUNDLE_ROWS,
    bundle_cols=BUNDLE_COLS
)


if (
    descriptors1 is None
    or
    descriptors2 is None
):

    raise RuntimeError(
        "B-ORB特徴量を取得できませんでした"
    )


print_borb_balance_check("登録画像", stats1)
print_borb_balance_check("撮影画像", stats2)


# ============================================================
# B-ORB特徴点のみ
#
# ============================================================

borb_only1 = (
    draw_borb_only(
        filtered1_bgr,
        keypoints1
    )
)

borb_only2 = (
    draw_borb_only(
        filtered2_bgr,
        keypoints2
    )
)


cv2.imwrite(
    "kekka/14_touroku_borb_only.jpg",
    borb_only1
)

cv2.imwrite(
    "kekka/15_satsuei_borb_only.jpg",
    borb_only2
)


# ============================================================
# B-ORB特徴点＋bundleグリッド
#
# 上のborb_onlyと全く同じkeypointsを使用
# ============================================================

borb_grid1, counts1 = (
    draw_borb_grid(
        filtered1_bgr,
        keypoints1,
        BUNDLE_ROWS,
        BUNDLE_COLS
    )
)

borb_grid2, counts2 = (
    draw_borb_grid(
        filtered2_bgr,
        keypoints2,
        BUNDLE_ROWS,
        BUNDLE_COLS
    )
)


cv2.imwrite(
    "kekka/16_touroku_borb_grid.jpg",
    borb_grid1
)

cv2.imwrite(
    "kekka/17_satsuei_borb_grid.jpg",
    borb_grid2
)


# ============================================================
# BFMatcher
# ============================================================

matcher = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=True
)


matches = matcher.match(
    descriptors1,
    descriptors2
)


matches = sorted(
    matches,
    key=lambda match:
    match.distance
)


if (
    len(matches)
    < 4
):

    raise RuntimeError(
        "RANSACに必要な対応点が不足しています"
    )


# ============================================================
# 対応点座標
# ============================================================

points1 = np.float32(
    [
        keypoints1[
            match.queryIdx
        ].pt
        for match
        in matches
    ]
).reshape(
    -1,
    1,
    2
)


points2 = np.float32(
    [
        keypoints2[
            match.trainIdx
        ].pt
        for match
        in matches
    ]
).reshape(
    -1,
    1,
    2
)


# ============================================================
# RANSAC
#
# ※ここはまだOpenCVのfindHomographyを使用
# ============================================================

homography, mask = (
    cv2.findHomography(
        points1,
        points2,
        cv2.RANSAC,
        5.0
    )
)


if (
    homography is None
    or
    mask is None
):

    raise RuntimeError(
        "RANSACによる推定に失敗しました"
    )


# ============================================================
# インライア
# ============================================================

inlier_matches = [
    match
    for (
        match,
        is_inlier
    )
    in zip(
        matches,
        mask.ravel()
    )
    if is_inlier
]


inlier_rate = (
    len(
        inlier_matches
    )
    /
    len(
        matches
    )
    * 100
)


# ============================================================
# RANSAC前
# 背景表示だけ元画像。keypoints1/2とdescriptors1/2はフィルタ後画像由来。
# ============================================================

before_ransac = (
    cv2.drawMatches(
        touroku,
        keypoints1,
        satsuei,
        keypoints2,
        matches[:80],
        None,
        flags=(
            cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
    )
)


# ============================================================
# RANSAC後
# ============================================================

after_ransac = (
    cv2.drawMatches(
        touroku,
        keypoints1,
        satsuei,
        keypoints2,
        inlier_matches[:80],
        None,
        matchColor=(
            0,
            255,
            0
        ),
        flags=(
            cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
    )
)


cv2.imwrite(
    "kekka/18_borb_ransac_mae.jpg",
    before_ransac
)

cv2.imwrite(
    "kekka/19_borb_ransac_go.jpg",
    after_ransac
)


# ============================================================
# 実行結果
# ============================================================

print()

print(
    "================================"
)

print(
    "        B-ORB 実行結果"
)

print(
    "================================"
)

print()

print(
    f"設定特徴点上限："
    f"{NFEATURES}"
)

print(
    f"画像縮小率："
    f"{IMAGE_SCALE}"
)

print(
    f"bundle数："
    f"{BUNDLE_ROWS * BUNDLE_COLS}"
)

print()

print(
    f"登録画像のB-ORB特徴点数："
    f"{len(keypoints1)}"
)

print(
    f"撮影画像のB-ORB特徴点数："
    f"{len(keypoints2)}"
)

print()

print(
    f"RANSAC前の対応点数："
    f"{len(matches)}"
)

print(
    f"インライア数："
    f"{len(inlier_matches)}"
)

print(
    f"インライア率："
    f"{inlier_rate:.1f}%"
)


# ============================================================
# レイヤごとの詳細
# ============================================================

print()

print(
    "===== 登録画像：レイヤ別 ====="
)

for (
    level,
    data
) in stats1.items():

    print(
        f"Level {level}: "
        f"通常上限={data['normal_limit']} / "
        f"2倍上限={data['relaxed_limit']} / "
        f"FAST候補={data['FAST_candidates']} / "
        f"FAST閾値={data['FAST_threshold']} / "
        f"bundle後={data['after_bundle']} / "
        f"最終={data['final']}"
    )


print()

print(
    "===== 撮影画像：レイヤ別 ====="
)

for (
    level,
    data
) in stats2.items():

    print(
        f"Level {level}: "
        f"通常上限={data['normal_limit']} / "
        f"2倍上限={data['relaxed_limit']} / "
        f"FAST候補={data['FAST_candidates']} / "
        f"FAST閾値={data['FAST_threshold']} / "
        f"bundle後={data['after_bundle']} / "
        f"最終={data['final']}"
    )


# ============================================================
# bundle別特徴点数
# ============================================================

print()

print(
    "===== 登録画像：最終bundle別特徴点数 ====="
)

print(
    counts1
)


print()

print(
    "===== 撮影画像：最終bundle別特徴点数 ====="
)

print(
    counts2
)


print()

print(
    "結果をkekkaフォルダへ保存しました"
)

print()

print(
    "B-ORB特徴点のみ："
)

print(
    "kekka/14_touroku_borb_only.jpg"
)

print(
    "kekka/15_satsuei_borb_only.jpg"
)

print()

print(
    "B-ORB + bundle："
)

print(
    "kekka/16_touroku_borb_grid.jpg"
)

print(
    "kekka/17_satsuei_borb_grid.jpg"
)
