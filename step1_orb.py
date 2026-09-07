import cv2
import numpy as np


# 画像を読み込む
touroku = cv2.imread("gazou/touroku.jpg")
satsuei = cv2.imread("gazou/satsuei_real.jpg")

if touroku is None or satsuei is None:
    raise FileNotFoundError("画像を読み込めませんでした")

# グレースケール化
gray1 = cv2.cvtColor(touroku, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(satsuei, cv2.COLOR_BGR2GRAY)

# ORBで特徴点と特徴量を取得
orb = cv2.ORB_create(nfeatures=2000)

keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

if descriptors1 is None or descriptors2 is None:
    raise RuntimeError("特徴量を取得できませんでした")

# BFMatcherで特徴点を対応付ける
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(descriptors1, descriptors2)

# ハミング距離が小さい順に並べる
matches = sorted(matches, key=lambda match: match.distance)

if len(matches) < 4:
    raise RuntimeError("RANSACに必要な対応点が不足しています")

# 対応点の座標を取得
points1 = np.float32(
    [keypoints1[match.queryIdx].pt for match in matches]
).reshape(-1, 1, 2)

points2 = np.float32(
    [keypoints2[match.trainIdx].pt for match in matches]
).reshape(-1, 1, 2)

# RANSACで画像間の位置関係を推定
homography, mask = cv2.findHomography(
    points1,
    points2,
    cv2.RANSAC,
    5.0
)

if mask is None:
    raise RuntimeError("RANSACによる推定に失敗しました")

# 位置関係に合う対応点だけを取得
inlier_matches = [
    match
    for match, is_inlier in zip(matches, mask.ravel())
    if is_inlier
]

inlier_rate = len(inlier_matches) / len(matches) * 100

# RANSAC前
before_ransac = cv2.drawMatches(
    touroku,
    keypoints1,
    satsuei,
    keypoints2,
    matches[:80],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# RANSAC後
after_ransac = cv2.drawMatches(
    touroku,
    keypoints1,
    satsuei,
    keypoints2,
    inlier_matches[:80],
    None,
    matchColor=(0, 255, 0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 結果を保存
cv2.imwrite("kekka/01_ransac_mae.jpg", before_ransac)
cv2.imwrite("kekka/02_ransac_go.jpg", after_ransac)

print(f"登録画像の特徴点数：{len(keypoints1)}")
print(f"撮影画像の特徴点数：{len(keypoints2)}")
print(f"RANSAC前の対応点数：{len(matches)}")
print(f"インライア数：{len(inlier_matches)}")
print(f"インライア率：{inlier_rate:.1f}%")
print("RANSAC前後の画像を保存しました")