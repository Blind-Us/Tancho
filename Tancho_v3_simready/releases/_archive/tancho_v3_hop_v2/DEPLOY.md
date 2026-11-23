# Tancho V3 hop v2 — 部署規格

Xbox 手把：**LT = 抬左腳、RT = 抬右腳、LT+RT = 小跳**，同時照常用搖桿控制前進/轉向（同 walk / rough）。
policy 負責平衡，抬腳動作本身是 Pi 端播放的一張 0.4 s 腿部偏移表（`reference_lift_table.csv`），兩者相加後送給腿馬達。

| 項目 | 值 |
|---|---|
| 訓練 commit | `3f4fab9`（任務 `TanchoV3-ClimbHop-v0`）；按鍵規則再加上 `81625fc` 的雙鍵同步 |
| 訓練 run | `logs/rsl_rl/tancho_v3_climb_hop/2026-10-08_03-34-23_3f4fab9`，`model_final.pt`（800 iter） |
| 起點 | ClimbHop v1 `2026-10-07_15-30-59_c2728c0/model_final.pt`（`--init_checkpoint --keep_obs_norm`） |
| 檔案 | `policy.pt`（TorchScript）、`policy.onnx`；SHA256 見 `SHA256SUMS` |
| 控制頻率 | **50 Hz** |
| 腿的範圍 | 標稱 ±0.6 rad（rough v1 是 ±0.25） |

## 能做什麼（模擬，標稱物理）

| 測試 | 結果 |
|---|---|
| `evaluate_hop.py`：0.3 m/s 前進，LT / RT / LT+RT 各 2 次，間隔 3 s | 全部不倒；單腳輪子離地 41–43 mm，小跳兩輪 33–34 mm |
| 壓力測試 `evaluate_hop_sweep.py`：618 種組合（單按、連跳 2/3 次、左右交替、先抬左再跳 × 間隔 0.15–1.2 s × 按住 0.1/0.3 s × 靜止/前進 0.3/0.6/後退 0.3/邊走邊轉） | **倒地 0.8%**（5 次）、抬腳不足 2 cm 1.1% |
| 同一壓力測試，v1 模型 + 舊按鍵規則 | 倒地 5.7%；按鍵間隔 < 0.4 s 時 13–27% 會倒 |

剩下會倒的情況：倒車 0.3 m/s 時兩次小跳間隔約 0.5 s；0.6 m/s + 1 rad/s 快轉時快速左右交替。
影片：`hop_mixed.mp4`（0.3 m/s，LT、RT、LT+RT 各兩次）、`hop_fast_double_taps.mp4`（每 0.3 s 點一次 LT+RT，共 4 次，排隊後變成每 0.4 s 跳一次）。

## 輸入 `obs`：float32 [1, 29]

0–24 和 rough v1 完全相同（見 `releases/tancho_v3_rough_v1/DEPLOY.md`），後面多 4 維：

| index | 內容 | 值 |
|---|---|---|
| 0–24 | 同 rough v1（重力、陀螺儀、速度命令、輪速、上一步 action 6 維、腿角、腿速） | |
| 25 | LT 按住 | 0 / 1（類比 trigger > 0.5 算按下） |
| 26 | RT 按住 | 0 / 1 |
| 27 | 左腳抬腳動作開始後經過的時間 | min(phase_L, 0.6) / 0.6；沒在動作中 = 1 |
| 28 | 右腳 | min(phase_R, 0.6) / 0.6 |

開機時 phase_L = phase_R = 10 s（obs 27、28 = 1）。

## 按鍵 → 抬腳動作（Pi 端，每 0.02 s 一次，和模擬的 `climb.py` 相同）

每一側各有 `phase`（秒）和 `queued`（布林）：

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

然後用這一步的 trigger 與 phase 組 obs 25–28，跑 policy。

## 輸出 `actions`：float32 [1, 6]

**先 clip 到 [−1, 1]**，再換算：

| index | 關節 | 目標 |
|---|---|---|
| 0 | thigh_L | −0.50 + **0.6** × a + dthigh(phase_L) |
| 1 | calf_L | +0.87 + **0.6** × a + dcalf(phase_L) |
| 2 | thigh_R | −0.50 + 0.6 × a + dthigh(phase_R) |
| 3 | calf_R | +0.87 + 0.6 × a + dcalf(phase_R) |
| 4 | wheel_L | 52.36 × a rad/s |
| 5 | wheel_R | 52.36 × a rad/s |

- dthigh / dcalf 查 `reference_lift_table.csv`（第 k 步 = phase k × 0.02 s；phase ≥ 0.4 s 時為 0）：0 s 先蹬（+0.275, −0.6），0.02–0.22 s 收腳（−0.3, +0.6），0.22–0.40 s 線性回到 0。
- **obs 11–16 存的是 clip 後的 a，不含偏移表。**
- 馬達參數同 rough v1：腿 DM-J4310 位置模式 Kp 20、Kd 0.2、12.5 N·m；輪 DM-H3510 速度模式 Kd 4、0.45 N·m。

`reference_io.json`：8 組輸入（含剛按下 LT / RT / 雙鍵、按住 0.2 s、放開 0.5 s）與對應輸出，容許誤差 1e-4。

## 沒做到 / 上實機前要注意

- **policy 自己不會抬腳**：試過把偏移表在訓練中慢慢降到 0（`TanchoV3-ClimbHopFree`），policy 選擇不抬腳、只求不倒（無偏移表時 6 次按鍵輪子 0 mm）。所以偏移表是部署的一部分，不能拿掉。
- **上台階仍然不行**（見 PROGRESS.md 的 Climb B1），這版只保證平地與 2 cm 以內凹凸上抬腳/小跳不倒。
- 小跳離地約 33 mm、單腳約 42 mm（模擬）。抬腳時腿馬達的實際扭力和電流我沒有在模擬裡量，實機要先觀察馬達電流與溫度。
- 實機延遲、摩擦、質量差異都還沒驗證。第一次建議架高讓輪子懸空，先按 LT / RT 確認是哪隻腳、往哪個方向動，再放到地上用慢速檔試。
