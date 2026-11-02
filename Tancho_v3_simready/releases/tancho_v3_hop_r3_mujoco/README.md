# Tancho V3 小跳 r3：MuJoCo 驗證包

這是在 Isaac Lab 訓練的二輪腿機器人 policy（tag `hop-push-r3-stable`，commit `c6ca4e7`）。用途：在 MuJoCo 重跑一次，確認換了物理引擎後還能平衡、抬腳、小跳。

## 檔案

| 檔案 | 說明 |
|---|---|
| `policy.onnx` | 輸入 `obs` float32 [1, 29]，輸出 `actions` float32 [1, 6]。觀測正規化已經包在裡面 |
| `policy.pt` | 同一組權重（TorchScript） |
| `config.json` | **所有規格**：觀測順序與 scale、動作換算、馬達 PD、時序、IMU 安裝位置、抬腳規則 |
| `reference_lift_table.csv` | 按下 LT/RT 後要加到腿目標角上的偏移（每 0.02 s 一行） |
| `reference_io.json` | 8 組 obs 與對應的 actions，用來確認 ONNX 載入正確（誤差 < 1e-4） |
| `robot/Tancho_v3.urdf`、`robot/meshes/` | 機器人模型（MuJoCo 可以直接 `compile` URDF，或先轉成 MJCF） |
| `eval_*.json` | Isaac Lab 裡的驗收數字，給你對照 |
| `env.yaml`、`agent.yaml` | 完整訓練設定（Isaac Lab 格式），找不到的細節查這裡 |

## 每一步（50 Hz；物理 200 Hz，同一組目標跑 4 個物理步）

```
1. 讀狀態，照 config.json 的 observation 順序和 scale 組 29 維 obs
   （重力和角速度是 IMU 座標：直立時重力 = (0, -1, 0)）
2. a = clip(onnx(obs), -1, 1)
3. 腿目標角 = 標稱 (-0.50, +0.87) + 0.6 * a[0:4] + 偏移表(各側 phase)
   輪目標速 = 52.36 * a[4:6] rad/s
4. 每個 5 ms 物理步算力矩：
   腿 tau = 20 * (q_target - q) - 0.2 * qdot，限制在 ±12.5 N·m，關節轉速上限 12.5 rad/s
   輪 tau = 4.0 * (qdot_target - qdot)，限制在 ±0.45 N·m
5. 把 clip 後的 a（不含偏移表）存起來，當下一步 obs 的 11–16
```

## 抬腳 / 小跳

obs 25/26 是 LT/RT 有沒有按住（0/1），27/28 是各側動作開始後經過的時間（min(t, 0.6)/0.6，閒置時是 1）。按鍵的上升緣開始一個 0.4 s 動作：先蹬 0.02 s、收腳到 0.22 s、0.40 s 線性回到 0。動作中又按會排隊；LT+RT 一起按就是小跳，兩側同步開始。規則在 `config.json` 的 `reference_lift`。

## 建議的驗證順序

1. 用 `reference_io.json` 確認 ONNX 輸出一致。
2. 不按鍵、命令 0：要能站 20 s 不倒（傾斜 < 15°）。
3. 前進 0.3 m/s，每 3 s 依序按 LT、RT、LT+RT（各按 0.4 s）。Isaac Lab 的結果：都不倒，單腳輪子離地 40–45 mm，小跳兩輪 33–36 mm、離地 0.1 s。
4. 原地連跳 5 次（間隔 3 s）。

## 容易出錯的地方

- **摩擦**：Isaac 的摩擦是兩邊相乘（0.8 × 0.8 = 0.64），MuJoCo 預設取最大值。請把輪胎和地面的有效摩擦設在 0.64 左右。
- **PD 不要用 MuJoCo 預設的 position actuator 參數**：要照上面的公式自己算力矩，或者把 kp/kv 和 forcerange 設成一樣。腿的 Kd 很小（0.2），而且速度上限 12.5 rad/s 會直接限制跳的高度。
- **IMU 座標**：相對 `base_link_root` 位於 (−8.4, 0, −29.4) mm，繞 x 軸 +90°。重力和角速度要轉到 IMU 座標再餵進去。
- **last_action 要用 clip 後、而且不含偏移表的值。**
- 已知差距：依 DM-J4310 V1.2 說明書的 TN 曲線，120 rpm 時實際只剩約 4.5 N·m。模擬給到 12.5 N·m，所以比實機樂觀。若要更貼近實機，可以在 MuJoCo 加上扭矩隨轉速下降的限制。
