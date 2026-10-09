import cv2
import numpy as np
import math
import os


# =========================================================
# 設定
# =========================================================

# 今までのORBとの比較をしやすくするため2000点
# 先行研究の基本実験では10000点を使用
NFEATURES = 2000

# OpenCV ORBの標準設定
# 論文には具体値の記載なし
SCALE_FACTOR = 1.2
NLEVELS = 8

# B-ORBのbundle
# 論文では4～10程度
# 今回は2×4 = 8領域
GRID_ROWS = 2
GRID_COLS = 4


# =========================================================
# 平均値差分フィルタ
# =========================================================
def mean_difference_filter(gray, radius=20):
    """
    平均値差分フィルタ

    対象画素の周囲の平均輝度を求め、
    対象画素の輝度との差分を計算する。

    radius=20の場合、
    上下左右20画素の範囲を使用する。
    """

    ksize = radius * 2 + 1

    # 負の値を扱えるようfloat32へ変換
    gray_float = gray.astype(np.float32)

    # 周囲の平均輝度
    local_mean = cv2.blur(
        gray_float,
        (ksize, ksize),
        borderType=cv2.BORDER_REFLECT
    )

    # 対象画素 - 周囲平均
    diff = gray_float - local_mean

    # 差が0の場所を128とする多階調画像
    filtered = diff + 128.0

    # 0～255に収める
    filtered = np.clip(
        filtered,
        0,
        255
    ).astype(np.uint8)

    return filtered


# =========================================================
# ORBの各ピラミッド層への特徴点数割り当て
# =========================================================
def calculate_orb_level_limits(
    nfeatures,
    scale_factor=1.2,
    nlevels=8
):
    """
    ORBと同様に、
    各ピラミッドレベルへ特徴点数を割り当てる。
    """

    factor = 1.0 / scale_factor

    desired_features = (
        nfeatures
        * (1.0 - factor)
        / (1.0 - factor ** nlevels)
    )

    limits = []
    total = 0

    for level in range(nlevels - 1):

        # OpenCVの丸め方に近づける
        level_features = int(
            math.floor(desired_features + 0.5)
        )

        limits.append(level_features)

        total += level_features

        desired_features *= factor

    # 最終レベルで総数を合わせる
    limits.append(
        max(nfeatures - total, 0)
    )

    return limits


# =========================================================
# 各bundleへの特徴点数の割り当て
# =========================================================
def calculate_region_quotas(
    total_limit,
    rows,
    cols,
    width,
    height
):
    """
    各bundleの面積に比例して、
    使用可能な特徴点数を割り当てる。
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

            areas.append(
                region_width * region_height
            )

    total_area = sum(areas)

    # 面積に比例した理論上の割当数
    raw_quotas = [
        total_limit * area / total_area
        for area in areas
    ]

    # 一旦小数点以下を切り捨て
    quotas = [
        int(math.floor(value))
        for value in raw_quotas
    ]

    # 切り捨てで余った分を配分
    remaining = (
        total_limit
        - sum(quotas)
    )

    # 小数部分の大きい領域から1点ずつ追加
    order = sorted(
        range(len(raw_quotas)),
        key=lambda i:
            raw_quotas[i] - quotas[i],
        reverse=True
    )

    for i in order[:remaining]:
        quotas[i] += 1

    return quotas


# =========================================================
# B-ORB
# =========================================================
def balanced_orb_detect_and_compute(
    gray,
    nfeatures=2000,
    grid_rows=2,
    grid_cols=4,
    scale_factor=1.2,
    nlevels=8
):
    """
    論文記載のB-ORBを再現。

    (1)
    各ピラミッドレイヤの特徴点上限を
    通常ORBの2倍に緩める

    (2)
    各レイヤをbundleに分割し、
    bundleごとに特徴点を選ぶ

    (3)
    全レイヤの特徴点をまとめ、
    Harris応答値の高いものから
    本来の上限特徴点数まで選ぶ
    """

    # -----------------------------------------------------
    # (1) 特徴点上限を2倍に緩めてORBを実行
    # -----------------------------------------------------

    candidate_orb = cv2.ORB_create(
        nfeatures=nfeatures * 2,
        scaleFactor=scale_factor,
        nlevels=nlevels,
        scoreType=cv2.ORB_HARRIS_SCORE
    )

    keypoints, descriptors = (
        candidate_orb.detectAndCompute(
            gray,
            None
        )
    )

    if (
        descriptors is None
        or len(keypoints) == 0
    ):
        return [], None, {}


    # 通常ORBにおける
    # 各レイヤの本来の特徴点上限
    normal_level_limits = (
        calculate_orb_level_limits(
            nfeatures,
            scale_factor,
            nlevels
        )
    )


    height, width = gray.shape[:2]

    selected_indices = []

    level_statistics = {}


    # -----------------------------------------------------
    # (2) 各レイヤをbundleに分割
    # -----------------------------------------------------

    for level in range(nlevels):

        # このピラミッドレベルの特徴点を取得
        level_indices = [
            i
            for i, kp in enumerate(keypoints)
            if kp.octave == level
        ]

        # 論文通り、
        # 各レイヤの上限を2倍
        doubled_limit = (
            normal_level_limits[level] * 2
        )

        # bundleごとの上限数
        region_quotas = (
            calculate_region_quotas(
                doubled_limit,
                grid_rows,
                grid_cols,
                width,
                height
            )
        )

        number_of_regions = (
            grid_rows * grid_cols
        )

        # 各bundle用の配列
        region_indices = [
            []
            for _ in range(
                number_of_regions
            )
        ]


        # -------------------------------------------------
        # 特徴点がどのbundleに属するか判定
        # -------------------------------------------------

        for index in level_indices:

            x, y = keypoints[index].pt

            col = min(
                int(
                    x * grid_cols
                    / width
                ),
                grid_cols - 1
            )

            row = min(
                int(
                    y * grid_rows
                    / height
                ),
                grid_rows - 1
            )

            region_number = (
                row * grid_cols
                + col
            )

            region_indices[
                region_number
            ].append(index)


        level_selected = []


        # -------------------------------------------------
        # bundleごとに上位特徴点を選ぶ
        # -------------------------------------------------

        for region_number in range(
            number_of_regions
        ):

            candidates = (
                region_indices[
                    region_number
                ]
            )

            # Harris応答値の高い順
            candidates.sort(
                key=lambda index:
                    keypoints[
                        index
                    ].response,
                reverse=True
            )

            quota = (
                region_quotas[
                    region_number
                ]
            )

            # 上限数まで使用
            selected = (
                candidates[:quota]
            )

            level_selected.extend(
                selected
            )


        selected_indices.extend(
            level_selected
        )


        # 確認用
        level_statistics[level] = {
            "candidate":
                len(level_indices),

            "normal_limit":
                normal_level_limits[
                    level
                ],

            "doubled_limit":
                doubled_limit,

            "after_bundle":
                len(level_selected)
        }


    # -----------------------------------------------------
    # (3) 全レイヤをまとめてHarris応答値で選択
    # -----------------------------------------------------

    selected_indices.sort(
        key=lambda index:
            keypoints[index].response,
        reverse=True
    )

    # 本来の特徴点上限まで残す
    selected_indices = (
        selected_indices[
            :nfeatures
        ]
    )


    # 最終B-ORB特徴点
    selected_keypoints = [
        keypoints[index]
        for index in selected_indices
    ]


    # 特徴量も同じ順番で取得
    selected_descriptors = (
        descriptors[
            selected_indices
        ].copy()
    )


    return (
        selected_keypoints,
        selected_descriptors,
        level_statistics
    )


# =========================================================
# 保存フォルダ
# =========================================================
os.makedirs(
    "kekka",
    exist_ok=True
)


# =========================================================
# 画像読み込み
# =========================================================
touroku = cv2.imread(
    "gazou/touroku.jpg"
)

satsuei = cv2.imread(
    "gazou/satsuei_real.jpg"
)


if (
    touroku is None
    or satsuei is None
):
    raise FileNotFoundError(
        "画像を読み込めませんでした"
    )


# =========================================================
# グレースケール化
# =========================================================
gray1 = cv2.cvtColor(
    touroku,
    cv2.COLOR_BGR2GRAY
)

gray2 = cv2.cvtColor(
    satsuei,
    cv2.COLOR_BGR2GRAY
)


# =========================================================
# 平均値差分フィルタ
# =========================================================
filtered1 = mean_difference_filter(
    gray1,
    radius=20
)

filtered2 = mean_difference_filter(
    gray2,
    radius=20
)


cv2.imwrite(
    "kekka/10_touroku_mean_difference.jpg",
    filtered1
)

cv2.imwrite(
    "kekka/11_satsuei_mean_difference.jpg",
    filtered2
)


# =========================================================
# B-ORBで特徴点・特徴量抽出
# =========================================================
(
    keypoints1,
    descriptors1,
    stats1
) = balanced_orb_detect_and_compute(
    filtered1,
    nfeatures=NFEATURES,
    grid_rows=GRID_ROWS,
    grid_cols=GRID_COLS,
    scale_factor=SCALE_FACTOR,
    nlevels=NLEVELS
)


(
    keypoints2,
    descriptors2,
    stats2
) = balanced_orb_detect_and_compute(
    filtered2,
    nfeatures=NFEATURES,
    grid_rows=GRID_ROWS,
    grid_cols=GRID_COLS,
    scale_factor=SCALE_FACTOR,
    nlevels=NLEVELS
)


if (
    descriptors1 is None
    or descriptors2 is None
):
    raise RuntimeError(
        "B-ORBで特徴量を取得できませんでした"
    )


# =========================================================
# B-ORB特徴点を描画
# =========================================================
keypoint_image1 = cv2.drawKeypoints(
    touroku,
    keypoints1,
    None,
    flags=(
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
)

keypoint_image2 = cv2.drawKeypoints(
    satsuei,
    keypoints2,
    None,
    flags=(
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
)


cv2.imwrite(
    "kekka/12_touroku_borb_keypoints.jpg",
    keypoint_image1
)

cv2.imwrite(
    "kekka/13_satsuei_borb_keypoints.jpg",
    keypoint_image2
)


# =========================================================
# BFMatcher
# =========================================================
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


if len(matches) < 4:
    raise RuntimeError(
        "RANSACに必要な対応点が不足しています"
    )


# =========================================================
# 対応点座標
# =========================================================
points1 = np.float32(
    [
        keypoints1[
            match.queryIdx
        ].pt
        for match in matches
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
        for match in matches
    ]
).reshape(
    -1,
    1,
    2
)


# =========================================================
# RANSAC
# =========================================================
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
    or mask is None
):
    raise RuntimeError(
        "RANSACによる推定に失敗しました"
    )


# =========================================================
# インライア
# =========================================================
inlier_matches = [
    match
    for match, is_inlier
    in zip(
        matches,
        mask.ravel()
    )
    if is_inlier
]


inlier_rate = (
    len(inlier_matches)
    / len(matches)
    * 100
)


# =========================================================
# RANSAC前
# =========================================================
before_ransac = cv2.drawMatches(
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


# =========================================================
# RANSAC後
# =========================================================
after_ransac = cv2.drawMatches(
    touroku,
    keypoints1,
    satsuei,
    keypoints2,
    inlier_matches[:80],
    None,
    matchColor=(0, 255, 0),
    flags=(
        cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
)


# =========================================================
# 保存
# =========================================================
cv2.imwrite(
    "kekka/14_borb_ransac_mae.jpg",
    before_ransac
)

cv2.imwrite(
    "kekka/15_borb_ransac_go.jpg",
    after_ransac
)


# =========================================================
# 実行結果
# =========================================================
print()
print(
    "===== B-ORB 実行結果 ====="
)

print(
    f"bundle数："
    f"{GRID_ROWS * GRID_COLS}"
)

print(
    f"登録画像のB-ORB特徴点数："
    f"{len(keypoints1)}"
)

print(
    f"撮影画像のB-ORB特徴点数："
    f"{len(keypoints2)}"
)

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


print()
print(
    "===== 登録画像：レイヤ別 ====="
)

for level, data in stats1.items():

    print(
        f"Level {level}: "
        f"候補={data['candidate']} / "
        f"通常上限={data['normal_limit']} / "
        f"2倍上限={data['doubled_limit']} / "
        f"bundle後={data['after_bundle']}"
    )


print()
print(
    "===== 撮影画像：レイヤ別 ====="
)

for level, data in stats2.items():

    print(
        f"Level {level}: "
        f"候補={data['candidate']} / "
        f"通常上限={data['normal_limit']} / "
        f"2倍上限={data['doubled_limit']} / "
        f"bundle後={data['after_bundle']}"
    )


print()
print(
    "B-ORBの結果をkekkaフォルダに保存しました"
)