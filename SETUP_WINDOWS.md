# 学校のWindowsパソコンのセットアップ手順

2026年11月24日から、研究フォルダの置き場所を次のように変えた。

| 内容 | 置き場所 |
| --- | --- |
| コード（.py、HANDOVER.md など） | GitHub（各パソコンでは OneDrive の外に clone） |
| 写真（gazou）・実験結果（kekka） | 大学の OneDrive の `yuka_shimon_data` フォルダ |
| Python の仮想環境 | 各パソコンのユーザーフォルダの `.venvs\yuka_shimon`（OneDrive の外） |

コードフォルダの中の `gazou` と `kekka` は、大学の OneDrive へのリンク（ジャンクション）になっている。
このためコードは変更しておらず、`gazou/...` や `kekka/...` という相対パスのまま動く。
写真と実験結果は Git では管理しない（`.gitignore` で除外）。2026年11月20日以前のものは GitHub の履歴に残っている。

## 0. 最初に：古いフォルダの未 push の変更を確認する

以前は個人用の OneDrive の `デスクトップ\yuka_shimon` で作業していた。このフォルダは後で削除する予定。
その前に、古いフォルダに GitHub に上がっていない変更がないかを確認する。

```powershell
cd "$env:OneDrive\デスクトップ\yuka_shimon"
git status
git log origin/main..main --oneline
```

- `git log` で何か表示されたら、push していないコミットがある。`git push` しておく。
- `git status` で変更されたファイルや新しいファイルが表示されたら、内容を確認して必要なものを残す。
  - コード（.py など）はコミットして push する。
  - 写真や実験結果は、大学の OneDrive の `yuka_shimon_data\gazou` / `yuka_shimon_data\kekka` にコピーする。

## 1. 大学の OneDrive を確認する

エクスプローラーで「OneDrive - Chiba Institute of Technology」が表示され、その中に `yuka_shimon_data\gazou` と `yuka_shimon_data\kekka` があることを確認する。
大学のアカウントで OneDrive にサインインしていない場合は、先にサインインする。
写真をパソコン本体に置かないため、OneDrive の設定で「ファイル オンデマンド」をオンにしておく（開いたファイルだけがダウンロードされる）。

PowerShell では、大学の OneDrive の場所は `$env:OneDriveCommercial` で分かる。

```powershell
$env:OneDriveCommercial
Test-Path "$env:OneDriveCommercial\yuka_shimon_data\gazou"
```

`True` と表示されればよい。

## 2. コードを clone する（OneDrive の外）

```powershell
cd $env:USERPROFILE
git clone --filter=blob:none https://github.com/saitou0627s-pixel/yuka_shimon.git
cd yuka_shimon
```

`--filter=blob:none` を付けると、昔の履歴に入っている写真をダウンロードしない。

## 3. 写真・実験結果へのリンクを作る

```powershell
New-Item -ItemType Junction -Path gazou -Target "$env:OneDriveCommercial\yuka_shimon_data\gazou"
New-Item -ItemType Junction -Path kekka -Target "$env:OneDriveCommercial\yuka_shimon_data\kekka"
dir gazou\satsuei
```

撮影画像の一覧が表示されればよい。

## 4. Python の仮想環境を作る（OneDrive の外）

```powershell
python -m venv $env:USERPROFILE\.venvs\yuka_shimon
& $env:USERPROFILE\.venvs\yuka_shimon\Scripts\pip.exe install numpy==2.5.4 opencv-python==5.0.0.93
```

`python` が見つからない場合は、Python 3.13 を入れる（以前は pyenv-win の 3.13.13 を使っていた）。

## 5. 動作確認

```powershell
cd $env:USERPROFILE\yuka_shimon
& $env:USERPROFILE\.venvs\yuka_shimon\Scripts\python.exe -u step4_auto_grout.py --satsuei "gazou\satsuei\diagonal_angle_5_1.jpg"
```

Mac で同じ画像を実行したときは、タイル候補4個、検出スコア 0.969 だった。結果は `kekka\auto_grout_...`（大学の OneDrive）に保存される。

## 6. VS Code の設定

1. VS Code で `%USERPROFILE%\yuka_shimon` フォルダを開く。
2. Ctrl+Shift+P →「Python: インタープリターを選択」→「インタープリター パスを入力」で
   `%USERPROFILE%\.venvs\yuka_shimon\Scripts\python.exe` を選ぶ。

`.vscode/settings.json` には Mac 用のパスが書いてあるので、Windows では一度だけ手で選ぶ。この選択は settings.json には書き込まれず、そのパソコンの VS Code にだけ保存される。

## 7. 毎回の作業の流れ

- 作業を始めるとき：`git pull`
- 作業を終えるとき：コードを `git add` → `git commit` → `git push`
- 写真と実験結果は大学の OneDrive に自動で同期されるので、Git の操作はいらない。

push を忘れると、もう一方のパソコンにコードの変更が届かない。

## 8. 古いフォルダの削除

手順 0 で古いフォルダの変更を全部移せたら、個人用の OneDrive の `デスクトップ\yuka_shimon` は削除してよい（個人用 OneDrive の容量を空けるため）。
削除すると、OneDrive の同期で Mac 側からも消える。また、OneDrive のごみ箱を空にするまでは容量は空かない。
