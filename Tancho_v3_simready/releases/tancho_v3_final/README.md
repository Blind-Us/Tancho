# Tancho V3 最終版（2026-11-23 09:00 更新）

一個 policy 包含：WASD（最高 1.2 m/s、1.5 rad/s）、原地定點、LT／RT 收腳、LT+RT 小跳、深蹲（底盤離地 10.7 cm）。輪 Kd 1。部署規格見 `DEPLOY.md`；論文的 LQR／RL／實機比較表見 `THESIS_TABLE.md`；論文用圖與 4.6 改寫稿在 `Tancho_v3_simready/docs/thesis/`。原始評估數據（eval/）已移除，要看請到 git commit 2e24695。

這一版是 StationFinal 第 1200 iter 再接續訓練的結果（加強速度追蹤，同時把「停下保持」的權重加倍，第 200 iter）。舊的 r28_1200 放在 `../_archive/tancho_v3_station_candidates/final_r28_1200/`。

## 主要結果（括號內是舊的 r28_1200）

| 測試 | 結果 |
|---|---|
| 論文 4.6（Isaac 標稱） | ±1–10° 全部不倒，最大位移 6.6–12.8 cm；三種速度命令都不倒（0.3 → 0.25 m/s、1.0 → 1.07 rad/s）；50 ms 推力 36 N 通過，38–39 N 短暫碰到 15°，40–42 N 通過，45 N 起倒；16.7 Hz 仍穩定 |
| 同上，加雜訊 + 20 ms 延遲 | 全部不倒；推力分界 36／38 N（32／34 N） |
| Isaac 雜訊 + 延遲，站 60 s ×6 | 不倒，漂 12–18 cm（6–14 cm） |
| Isaac 走路序列、0.3 m/s 走路中與原地的 LT／RT／小跳／蹲下 | 全部不倒；小跳兩輪離地 30–32 mm |
| MuJoCo 最高速（命令 1.2 m/s） | **實跑 1.14 m/s（0.84）** |
| MuJoCo 雜訊 + 延遲，站 60 s | **15–27 cm（49–56 cm）** |
| MuJoCo 邊走邊按全部 1530 種組合（含小跳、蹲下） | 1／1530 倒（0.07%） |
| MuJoCo 按鍵中被推 ±0.5 m/s | **7／990 倒，0.7%（1.1%）** |
| 深蹲 | 底盤最低 10.7 cm |
| 站 5 分鐘（雜訊 + 20 ms） | 不倒；MuJoCo 最遠 18–32 cm（結束時 12–18 cm）、Isaac 26 cm，不會越漂越遠 |
| 0–2 cm 凹凸地（Isaac） | 站 60 s 漂 0.3–6.4 cm；在凹凸地上被推 0.5 m/s 也不倒 |
| 斜坡 ±5° / ±8°（重力傾斜） | 不倒，**但會溜**：Isaac 0.2–2.1 m、MuJoCo 0.7–2.2 m |

## 已知限制

- 站立時有小幅持續晃動，所以 ±1° 定義的恢復時間約 3.4–4.7 s；LQR 是 0.18 s。
- 斜坡會溜。

影片：`isaac_final_noise_latency.mp4`（0.6 m/s 走路中做 LT、RT、小跳、深蹲 ×2，雜訊 + 20 ms）、`mujoco_final_noise_latency.mp4`（加速到 1.2 m/s → 轉向 → 倒退 → 停，途中小跳、收腳、深蹲）。

重新匯出：`python scripts/wheel_only/export_policy_cpu.py --checkpoint releases/tancho_v3_final/train/model_200.pt`。
重訓：`python scripts/wheel_only/train.py --headless --task TanchoV3-StationFinalTrackHold-v0 --num_envs 4096 --max_iterations 500 --seed 69 --allow-dirty --init_checkpoint releases/tancho_v3_final/train/init_checkpoint.pt --keep_obs_norm`，取第 200 iter；`init_checkpoint.pt` 是 StationFinal 第 1200 iter，動作雜訊 0.02。
