# Tancho V3 最終版（tancho_v3_final）：部署規格

同一個 policy 包含：WASD 前進後退與轉向（最高 1.2 m/s、1.5 rad/s）、放開按鍵後原地停住、LT 收左腳、RT 收右腳、LT+RT 小跳、按住 Shift / J 深蹲（底盤離地約 11 cm），放開後站回。

| 項目 | 值 |
|---|---|
| 訓練 | `TanchoV3-StationFinalTrackHold-v0`，run `logs/rsl_rl/tancho_v3_station_skills/2026-11-23_07-51-13_dae8548`，第 200 iter（`train/model_200.pt`）；由 StationFinal 第 1200 iter 接續 |
| 檔案 | `policy.pt`（TorchScript）、`policy.onnx`（輸入 [1, 34]、輸出 [1, 6]，觀測正規化已包含在內），SHA256 在 `SHA256SUMS` |
| 頻率 | policy 50 Hz；馬達迴路照常 |
| **輪馬達** | DM-H3510 速度模式，**Kd = 1.0**，扭力上限 ±0.45 N·m |
| 腿馬達 | DM-J4310 位置模式，Kp 20、Kd 0.2，±12.5 N·m |

## 輸入 obs [34]

| index | 內容 |
|---|---|
| 0–2 | IMU 重力方向（IMU 座標系，直立時 = (0, −1, 0)） |
| 3–5 | IMU 角速度 × 0.25 |
| 6–8 | 速度命令 (vx, 0, yaw)，vx ±1.2 m/s、yaw ±1.5 rad/s（搖桿 / WASD） |
| 9–10 | 輪速 × 0.05 |
| 11–16 | 上一步 action（clip 後，不含偏移量） |
| 17–20 | 腿角 − 標稱 (−0.50, +0.87) |
| 21–24 | 腿速 × 0.05 |
| 25–26 | LT、RT 是否按住（0 / 1） |
| 27–28 | 左、右抬腳開始後經過的時間：min(t, 0.6) / 0.6，閒置時為 1 |
| 29–30 | 輪里程 2·clip(0.03614·(q_wheel − q_hold), ±0.5)；只要命令不為 0，每步都設 q_hold = q_wheel |
| 31–33 | 蹲下：[按住 0/1，深度 level，放開後經過時間 min(t, 0.6)/0.6] |

## 每一步

1. **抬腳 / 小跳**（每 0.02 s 一次；LT/RT 類比 trigger > 0.5 算按下；開機時 phase_L = phase_R = 10 s）：

   ```
   edge[s]  = 這一步按下 且 上一步沒按下         # s = L, R
   phase[s] += 0.02
   busy[s]  = phase[s] < 0.399
   queued[s] |= edge[s] and busy[s]               # 動作中的新按鍵先排隊（最多一個）
   start[s] = (edge[s] or queued[s]) and not busy[s]
   # 小跳同步：兩側都想開始、但有一側還在動作中 → 兩側一起等
   if (edge[L] or queued[L]) and (edge[R] or queued[R]) and (busy[L] or busy[R]):
       queued[L] |= edge[L]; queued[R] |= edge[R]
       start[L] = start[R] = False
   queued[s] &= not start[s]
   if start[s]: phase[s] = 0
   ```

   抬腳偏移 (dthigh, dcalf) 查 `reference_lift_table.csv`（第 k 列 = phase k × 0.02 s；phase ≥ 0.4 s 時為 0）。
2. **蹲下**：放開後 0.3 s 內再按不理會（冷卻）。按住時每步 level += 0.02 / 0.45（上限 1），偏移 = level · (−0.45, +1.10)；放開後偏移 = release_level · (−0.45, +1.10) · max(0, 1 − t / 0.28)。
3. 組成 obs → policy → clip 到 [−1, 1]。
4. 腿目標 = 標稱 + 0.6·a[0:4] + 抬腳偏移 + 蹲下偏移；輪目標速 = 52.36·a[4:6] rad/s。

## 上實機的順序

先架高，確認各腳、輪子的方向，以及輪里程的正負號 → 原地站 60 s → 輕推 → LT / RT → 蹲下 → 小跳 → 一檔 WASD → 再加速。**不要在抬腳或跳的時候推它；斜坡會溜。**

## 自我檢查

- `reference_io.json`：8 組 obs 與對應 actions，`policy.pt` / `policy.onnx` 跑出來誤差要 < 1e-4（2026-11-23 實測：torch 0、onnx 2e-7）。
- `obs[11:17]` 存的是 clip 後的 a，**不含**抬腳 / 蹲下偏移。
- MuJoCo 模擬（需要 `mujoco`、`torch`）：`python evaluate_stand_mujoco.py --checkpoint policy.pt --output-dir out --wheel-kd 1 --duration 20 --crouch=-0.45,1.10 --crouch-s 0.45 --charge-cooldown 0.3 --charge-jump 0 --presses LT:3,RT:6,LT+RT:9,CHG:12`，機器人模型在 `robot/`。
