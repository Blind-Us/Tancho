# Tancho V3：WASD + 收左腳 / 收右腳 / **小跳** / 蹲下 + 停下原地保持（輪 Kd 1）

**2026-11-22 晚上的建議版**，五個動作都在同一個 policy 裡。從 `../tancho_v3_walk_squat_kd1/` 開始，把 LT+RT 小跳加回來（hop r3 的 v1 參考動作），一共接續訓練 1700 iter，這版是第 25 次訓練的第 600 iter。

## 部署

跟 `../tancho_v3_walk_squat_kd1/README.md`、`../tancho_v3_squat_kd1_r16_400/DEPLOY.md` 一樣：34 維輸入、輪 kd 1.0、速度命令 vx ±0.6 / yaw ±1、搖桿離開中心時把輪里程歸零、蹲下冷卻 0.3 s。**唯一的差別：LT+RT 同時按 = 小跳**，兩側同步開始，規則同 hop v2 的 `DEPLOY.md`。

## 驗收（全部在感測雜訊 + 10–20 ms 延遲下）

| 測試 | 結果 |
|---|---|
| MuJoCo 邊走邊按 990 種（WASD 隨機命令 + LT / RT / 小跳 / 連跳 / 跳後蹲 / 蹲後跳…，然後停下） | **0 / 990 倒**，最大傾角 33° |
| MuJoCo 原地按 990 種（含小跳） | **0 / 990 倒**，最大傾角 25° |
| MuJoCo 原地按鍵中途被推 ±0.5 m/s | 20 / 990 倒（2.0%；r16_400 是 4.8%） |
| Isaac 原地站 60 s ×3 | 不倒，漂 14–21 cm |
| Isaac 走路序列（0.3 / 0.6 / 倒退 / 轉 / 弧線 / 停） | 不倒，速度誤差 ≤ 0.13 m/s、≤ 0.10 rad/s |
| Isaac 走 0.3 m/s 與原地：LT、RT、小跳、蹲下各 ×2 | 不倒；小跳兩輪離地 31–34 mm |
| MuJoCo 原地站 60 s | 不倒，漂 35–40 cm（MuJoCo 加雜訊時輪胎會打滑） |

影片：`isaac_walk_hop_noise_latency.mp4`（0.3 m/s 走路中做 LT、RT、小跳、蹲下）、`mujoco_walk_hop_noise_latency.mp4`（走路中小跳 → 收左腳 → 原地轉 → 停 → 小跳 → 收右腳 → 蹲下 → 小跳）。

## 已知

- 抬腳或跳的時候被推，還是有 2% 會倒。
- 斜坡會溜。
- 0.6 m/s 大約只跑到 0.5 m/s。
- 停下後的位置精度：Isaac 14–21 cm、MuJoCo 35–40 cm。

訓練：`TanchoV3-StationWalkHop-v0`，run `logs/rsl_rl/tancho_v3_station_skills/2026-11-22_17-36-55_29175a6`，`train/model_600.pt`；起點 `train/init_checkpoint.pt` = 第 24 次 final（動作雜訊 0.02）。重新匯出：`scripts/wheel_only/export_policy_cpu.py --checkpoint train/model_600.pt`。
