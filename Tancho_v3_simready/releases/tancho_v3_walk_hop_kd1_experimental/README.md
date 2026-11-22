# 實驗版：WASD + 收腳 + 蹲下 + **LT+RT 小跳**（輪 Kd 1）——還不建議部署

從 `../tancho_v3_walk_squat_kd1/` 接續訓練 500 iter，把 LT+RT 同時按的小跳加回來（hop r3 的 v1 參考動作）。輸入、輸出格式和部署規則都跟 walk_squat_kd1 一樣，差別只在**可以同時送 LT 和 RT**。

| 測試（感測雜訊 + 20 ms 延遲） | 結果 |
|---|---|
| MuJoCo 原地：LT、RT、小跳 ×3、蹲下 | 不倒；小跳兩輪離地 40–47 mm |
| MuJoCo 0.3 m/s 走路中：小跳 ×2、LT、RT | 不倒；兩輪離地 41–45 mm |
| Isaac 0.3 m/s 走路中：LT、RT、小跳各 ×2 | 不倒；兩輪離地 33–37 mm，機身騰空上升 13 mm（同 hop r3） |
| MuJoCo 原地站 60 s | 不倒，但**漂 80 cm** |

還沒做：990 種按鍵組合測試、Isaac 重複站立測試、地形測試。站立精度也比 walk_squat_kd1 差。**所以只建議在模擬裡展示小跳，實機先不要用。**

影片：`mujoco_walk_hop_noise_latency.mp4`（MuJoCo，雜訊 + 20 ms，走路中小跳 ×2 → 停下 → 小跳 → 蹲下）。
訓練：`TanchoV3-StationWalkHop-v0`，run `logs/rsl_rl/tancho_v3_station_skills/2026-11-22_16-02-36_34bcc45`，`train/model_final.pt`。
