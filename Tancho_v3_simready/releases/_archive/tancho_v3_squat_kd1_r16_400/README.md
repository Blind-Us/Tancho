# Tancho V3：站立 + 收左腳 / 收右腳 / 蹲下，耐雜訊與延遲版（r16_400，輪 Kd 1）

**2026-11-21 的建議部署版。** 跟 `../tancho_v3_squat_kd1_r8_300/` 同一套動作和輸入格式（34 維輸入、6 維輸出，輪馬達 kd = 1.0，沒有跳）。差別在於：訓練時馬達延遲固定在 10–20 ms，再用「感測雜訊 + 延遲」的壓力測試挑出來，比較接近實機條件。

## 部署條件（同 r8_300）

1. 輪馬達 MIT 模式 **kd = 1.0**。
2. 輸入 **34 維**：29 維 hop r3 輸入，加上輪里程 2 維 `2·clip(0.03614·(q_wheel − 起點), ±0.5)`，再加上蹲下狀態 3 維，見 `config.json`。
3. 這版是用 10–20 ms 延遲訓練的。實機 Pi → CAN → 馬達本來就有延遲，所以正好。在完全沒有延遲的理想模擬裡，它反而漂得比較多（MuJoCo 0 ms：72 cm）。

## 驗收（全部不倒）

| 條件 | Isaac | MuJoCo |
|---|---|---|
| 60 s 站立，雜訊 + 20 ms 延遲 | 18.5 cm | 7–20 cm（5 個 seed） |
| 推 0.5 m/s，雜訊 + 20 ms | 不倒，停在 2 cm 附近 | 不倒 |
| LT / RT / 蹲下，雜訊 + 延遲 | 12 次不倒 | 30 次不倒（10 / 20 ms × 5 seed），按鍵時最大傾角 15.8° |
| 60 s 站立，延遲 0–20 ms，有無雜訊 | 無雜訊無延遲：11.7 cm | 17–72 cm |
| 0–2 cm 凹凸地 | 6–11 cm；凹凸地上被推也不倒 | — |
| 斜坡 ±5° / ±8°（重力傾斜） | 19 cm – **3.2 m**（+5°） | 1.9–3.1 m（MuJoCo 斜坡有接觸滑移問題） |
| 抬腳高度 | 左輪 36 mm、右輪 37 mm | 左輪 47–51 mm、右輪 43–49 mm |

已知弱點：
- **斜坡會溜**：不會倒，但在斜面上停不住。
- MuJoCo 軟接觸（solref 0.02）8.7 s 倒。
- 位置精度在實際條件下大約 10–20 cm，沒有達到 5 cm。

r8_300 跟這版的取捨：r8_300 在理想條件下 MuJoCo 只漂 2.4 cm、斜坡也比較穩（Isaac 7–53 cm）；但加上雜訊和延遲後漂 35–43 cm，按鍵時晃得比較大，在 Isaac 也比較容易出事。實機一定有雜訊和延遲，所以建議用這一版。

## 重現

- 重新匯出：`python scripts/wheel_only/export_policy.py --headless --task TanchoV3-StationSquatKd1Lat-v0 --checkpoint releases/tancho_v3_squat_kd1_r16_400/train/model_400.pt`；不想開 Isaac 的話，用 `scripts/wheel_only/export_policy_cpu.py --checkpoint …`，兩者輸出一樣。
- 重訓：`python scripts/wheel_only/train.py --headless --task TanchoV3-StationSquatKd1Lat-v0 --num_envs 4096 --max_iterations 1000 --seed 54 --allow-dirty --init_checkpoint releases/tancho_v3_squat_kd1_r16_400/train/init_checkpoint.pt --keep_obs_norm`，交付版是第 400 iter。`init_checkpoint.pt` 是 StationSquatKd1Com 第 700 iter，動作雜訊重設為 0.02。
- 篩選：`scripts/wheel_only/screening/stress.sh`（MuJoCo 雜訊 + 延遲）、`terrain.sh`（地形 / 斜坡）、`mj_screen.sh`、`candidates.sh`。

影片：`isaac_noise_latency_LT_RT_crouch.mp4`、`mujoco_noise_latency_LT_RT_crouch.mp4`（兩支都有開雜訊和 20 ms 延遲）。
