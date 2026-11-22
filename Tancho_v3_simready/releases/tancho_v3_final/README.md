# Tancho V3 最終版（2026-11-23）

一個 policy：WASD（最高 1.2 m/s、1.5 rad/s）＋ 原地定點 ＋ LT / RT 收腳 ＋ LT+RT 小跳 ＋ 深蹲（底盤 10.9 cm），輪 Kd 1。部署規格見 `DEPLOY.md`，論文用的 LQR／RL／實機比較表見 `THESIS_TABLE.md`。

## 主要結果

| 測試 | 結果 |
|---|---|
| 論文 4.6 測試（Isaac 標稱） | ±1–10° 全部不倒，最大位移 7.5–7.9 cm；三種速度命令都不倒（0.3 → 0.24 m/s、1.0 → 1.05 rad/s）；50 ms 推力 38 N 過、39 N 失敗（同 LQR）；16.7 Hz 仍穩定 |
| 同上，加雜訊 + 20 ms 延遲 | 全部不倒；推力 32 N 過、34 N 失敗；25 Hz 仍穩定 |
| Isaac 雜訊 + 延遲，站 60 s ×6 | 不倒，漂 6–14 cm |
| Isaac 走路序列、走 0.3 m/s 時與原地做 LT / RT / 小跳 / 蹲下 | 全部不倒；小跳兩輪離地 27–30 mm |
| MuJoCo 邊走邊按 990 種（含小跳、蹲下，雜訊 + 延遲） | **0 / 990 倒** |
| MuJoCo 按鍵中被推 ±0.5 m/s | 11 / 990 倒（1.1%） |
| MuJoCo 最高速 | 命令 1.2 m/s 實跑 0.84 m/s；1.5 rad/s 實跑 1.58 |
| 深蹲 | 底盤最低 10.9 cm（目標 15 cm 以下） |

## 已知限制

- 站立時有 1.4–3.6° 的小幅持續晃動，所以初始偏差的「恢復時間」（±1° 定義）約 4 s，遠比 LQR 的 0.18 s 慢。
- 速度追蹤偏低：大約達到命令的 70–80%。
- 斜坡會溜。
- MuJoCo 站立漂 49–56 cm（有雜訊時輪胎打滑，Isaac 是 6–14 cm）。

影片：`isaac_final_noise_latency.mp4`（0.6 m/s 走路中做 LT、RT、小跳、深蹲 ×2，雜訊 + 20 ms）、`mujoco_final_noise_latency.mp4`（加速到 1.2 m/s、轉向、倒退、停，途中小跳、收腳、深蹲）。

重新匯出：`python scripts/wheel_only/export_policy_cpu.py --checkpoint releases/tancho_v3_final/train/model_1200.pt`。重訓：`python scripts/wheel_only/train.py --headless --task TanchoV3-StationFinal-v0 --num_envs 4096 --max_iterations 1500 --seed 66 --allow-dirty --init_checkpoint releases/tancho_v3_final/train/init_checkpoint.pt --keep_obs_norm`，取第 1200 iter。
