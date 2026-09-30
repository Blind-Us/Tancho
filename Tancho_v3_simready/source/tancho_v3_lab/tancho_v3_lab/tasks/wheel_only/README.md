# TanchoV3-WheelOnly-Flat-v0

Tancho V3 的純輪式平衡任務：腿固定在 thigh = −0.50 rad、calf = +0.87 rad，policy 只控制左右兩顆輪子。

這個任務自成一個資料夾，不 import、不繼承 `tasks/direct/tancho_v3` 的任何內容。需要沿用的部分（平地材質、IMU 安裝位置、DM-H3510 輪馬達參數）是直接複製過來的，並在程式碼內註明來源。

| Task ID | 用途 |
|---|---|
| `TanchoV3-WheelOnly-Flat-v0` | 訓練：含質量／摩擦隨機化、推力擾動、觀測雜訊 |
| `TanchoV3-WheelOnly-Flat-Play-v0` | 評估：標稱物理，不推、無雜訊、無隨機化，reset 為正立靜止 |

## 1. 機器人模型

- **URDF**：`assets/robots/Tancho_v3/urdf/Tancho_v3_wheel_only.urdf`，由 `scripts/wheel_only/build_wheel_only_urdf.py` 從 working tree 的 `Tancho_v3.urdf` 產生；來源 sha256 記錄在同名 `.json`。
- **固定腿的方式：剛體合併（不是 fixed joint，也不是高剛性 PD 鎖定）。**
  - 做法：把 base、drawer、pi_case、thigh_L/R、calf_L/R 依標稱角度做 FK，合併成單一剛體 `base_link_root`。合併後的質量、質心、慣性依平行軸定理精確計算：2.5537 kg，COM 在 root 座標 (−14.6, 0, −60.1) mm。
  - 結果：腿在模擬裡**不是自由度**，腿角不可能偏移，也沒有 PD 鎖定帶來的彈性或數值剛性問題。
- **腿角驗證**
  - `scripts/wheel_only/verify_leg_pose.py`：從 URDF 反推四個關節角，誤差 0 rad。
  - `scripts/wheel_only/evaluate_standstill.py`：從模擬中的 USD 碰撞 prim 反推腿角。thigh 誤差 3.8e-8 rad，calf 誤差 4.7e-9 rad，遠小於 0.01 rad 的要求。
- **可動關節**：只有 `joint_wheel_L`、`joint_wheel_R`。body 只有 `base_link_root`、`wheel_L`、`wheel_R`。
- **輪馬達（DM-H3510，未改動）**
  - 速度模式：Kp = 0、Kd = 4.0
  - 力矩上限 0.45 N·m、模擬速度上限 188 rad/s
  - action ±1 對應 ±52.36 rad/s（額定 500 rpm）
- **幾何（標稱姿態，平地）**

  | 量 | 數值 |
  |---|---|
  | root（髖軸）高度 | 260.5 mm |
  | 整機 COM 高度 | 200.4 mm |
  | 底盤碰撞盒底面 | 214.8 mm |
  | 輪軸到 COM | 164.2 mm |
  | COM 在輪軸前方 | 1.9 mm（靜平衡 pitch 約 −0.67°） |

  > 注意：以 thigh = −0.50 / calf = +0.87 計算，沒有任何一個高度定義剛好是 18 cm。本任務以指定的關節角為準。

## 2. 控制頻率：50 Hz（PhysX 200 Hz，decimation 4）

- **倒立擺時間常數**：把機身看成繞輪軸的倒立擺，不穩定極點 ω = √(m·g·l / I_axle) ≈ 6.7 rad/s（約 1.07 Hz，e 倍增時間 149 ms）。
  - 50 Hz 時，一個控制週期內誤差最多放大 1.14 倍，每個 e 倍增時間內約有 7.4 次控制更新，足夠穩定控制。
  - 100 Hz 可以把延遲減半（每步放大 1.07 倍），但對大推力的恢復能力來說，真正的限制是 **0.45 N·m 力矩飽和**，不是取樣率。
- **跟既有基準一致**：`TANCHO_RL_FOUR_ELEMENTS.md` 和 LQR 推力實驗都把 50 Hz 定為硬體對齊的控制週期；推力評估腳本的 50 ms 脈衝在 50 Hz 下剛好可以切成 20 + 20 + 10 ms。
- 目前沒有證據顯示實機控制迴路是 100 Hz，所以不在模擬裡假設更快的控制器。

## 3. Observation

**Policy（10 維）**：只用實機量得到的量。

| 項目 | 維度 | 來源 | 縮放 | 訓練雜訊 |
|---|---|---|---|---|
| `imu_projected_gravity` | 3 | IMU 姿態 | 1 | ±0.02 |
| `imu_ang_vel` | 3 | IMU 陀螺儀 (rad/s) | 0.25 | ±0.05 rad/s |
| `wheel_vel` | 2 | 輪子編碼器 (rad/s) | 0.05 | ±0.2 rad/s |
| `last_action` | 2 | 上一步 action | 1 | — |

- 不使用 `base_pos_z`、`base_lin_vel`、速度命令或輪子角度。速度命令固定為 0，只提供給內建的 tracking reward 使用。
- **Critic（13 維，只在訓練時使用，不會部署）**：真實的 base 線速度（3）、角速度（3）、projected gravity（3）、輪速（2）、上一步 action（2），全部沒有雜訊。這是 asymmetric actor-critic 的特權資訊，用來降低 value 估計的變異。

## 4. Action

2 維：左右輪的速度目標，`JointVelocityActionCfg`，scale 52.36 rad/s，`clip_actions = 1`。

## 5. Reward

全部使用 `isaaclab.envs.mdp` 的內建 term，**沒有 custom reward**。權重單位是「每秒」，reward manager 會自動乘上 step_dt = 0.02。

| Term | 內建函式 | 權重 | 物理意義與權重理由 |
|---|---|---|---|
| `is_alive` | `is_alive` | +1.0 | 存活獎勵，完整 20 s 得 +20 |
| `termination_penalty` | `is_terminated` | −200 | 每次跌倒額外扣 4（不含 timeout），讓跌倒明確比任何存活軌跡差 |
| `upright` | `flat_orientation_l2` | −10 | 懲罰 \|g_xy\|² = sin²(傾角)；傾 5° 時每秒扣 0.076，主要作用是把機身拉回直立附近 |
| `ang_vel_xy` | `ang_vel_xy_l2` | −0.05 | pitch/roll 角速度的阻尼，抑制來回擺動 |
| `lin_vel_xy` | `track_lin_vel_xy_exp`（命令 0，std 0.25） | +1.0 | 原地不動；0.1 m/s 時這項少拿 15% |
| `ang_vel_z` | `track_ang_vel_z_exp`（命令 0，std 0.25） | +0.5 | 不原地自轉 |
| `wheel_vel_l1` | `joint_vel_l1`（輪子） | −0.02 | exp tracking 在速度 0 附近的梯度會消失，L1 在 0 附近仍保持固定拉力，用來消除慢速漂移。1 rad/s 等於 3.6 cm/s |
| `wheel_torque` | `joint_torques_l2`（輪子） | −0.01 / 0.45² | 用峰值力矩正規化的出力成本；兩輪同時飽和時每秒扣 0.02 |
| `action_rate` | `action_rate_l2` | −0.01 | 讓 action 平滑，避免速度目標劇烈跳動 |
| `wheel_vel_limit` | `joint_vel_limits`（soft ratio 0.9） | −1.0 | 輪速接近上限的 90% 時開始懲罰 |

## 6. Termination

- `time_out`：20 s。
- `tilt`：`bad_orientation(limit_angle = 15°)`。這個函式算的是機身 z 軸相對鉛直的傾角；兩輪同軸、roll 幾乎為 0，所以實際上就是 |pitch| > 15°，跟 LQR 推力實驗的門檻相同。
- **不使用 base_contact。** 用實際碰撞幾何繞輪軸旋轉計算（`scripts/wheel_only/ground_clearance.py`），±15° 內只有輪胎會碰到地面。最先碰到地面的非輪子部位是：
  - 前傾：`base_link` 碰撞盒，69.2°
  - 後仰：`thigh_L/R` 碰撞盒，59.6°

  > 對照：舊版 URDF 的 calf 用碰撞盒，後仰 14.7° 就會碰到地面。這正是 09-21 那個 checkpoint 推力評估在約 12° 就觸發 base_contact 的原因。目前的 URDF 已在 commit `aadfcd2` 改成 calf mesh。

## 7. Event（隨機化）

| Term | 訓練 | Play／評估 |
|---|---|---|
| `physics_material` | 機器人摩擦 static/dynamic ∈ [0.6, 1.0]（make_consistent） | 固定 0.8（標稱值） |
| `add_base_mass` | base 質量加 [−0.15, +0.25] kg（約 −6% / +10%） | 關閉 |
| `reset_base` | pitch ±0.05 rad、vx ±0.1 m/s、pitch rate ±0.2 rad/s | 正立靜止 |
| `reset_wheels` | 位置、速度歸零 | 同左 |
| `push_robot` | 每 3–6 s 對底盤加一次 vx ∈ [−0.8, 0.8] m/s（0.8 m/s 相當於 42 N × 50 ms） | 關閉 |
| 觀測雜訊 | 見第 3 節 | 關閉 |

地面摩擦 0.8 與機器人材質相乘（multiply），標稱等效摩擦為 0.64，跟 Flat 系列任務相同。

## 8. 訓練

```bash
# 在 Tancho_v3_simready/ 下，使用 env_isaaclab
python scripts/wheel_only/train.py --headless
```

- `scripts/wheel_only/train.py` 以 Isaac Lab 官方 rsl_rl train 腳本為基礎。
  - 不使用專案的 `scripts/rsl_rl/train.py`：它的舊版相容轉換會丟掉 rsl-rl ≥ 4 的 actor/critic 設定，還會強制 critic 使用 policy 觀測。
  - 如果任務相關檔案有未提交的修改，腳本會拒絕訓練；commit hash 會寫進 run 名稱和 `<log_dir>/git_commit.txt`。
  - 訓練結束時會匯出 `exported/policy.pt` 和 `exported/policy.onnx`。
- **PPO 超參數**（`agents/rsl_rl_ppo_cfg.py`）

  | 項目 | 值 |
  |---|---|
  | 環境數 | 4096 |
  | 每次 rollout 步數 | 24（每個 iteration 約 98k 步） |
  | 最大 iteration | 1500 |
  | Actor MLP | [128, 128, 64]，ELU，觀測正規化，初始 std 0.2 |
  | Critic MLP | [256, 256, 128]，ELU，觀測正規化 |
  | 學習率 | 3e-4，adaptive schedule（desired KL 0.01） |
  | γ / λ | 0.99 / 0.95 |
  | clip | 0.2 |
  | entropy 係數 | 0.001 |
  | learning epoch × mini-batch | 5 × 4 |
  | max grad norm | 1.0 |
  | seed | 42 |

## 9. 評估

```bash
# 20 s 零命令靜置：pitch ±1°、位移 < 5 cm、不失敗、腿角誤差 < 0.01 rad
python scripts/wheel_only/evaluate_standstill.py --headless --checkpoint <run>/exported/policy.pt

# 2–60 N、50 ms 推力掃描（兩份 CSV）與 LQR 座標軸圖（三合一 + 三張單圖）
python scripts/evaluation/evaluate_trained_rl_push.py --headless \
    --task TanchoV3-WheelOnly-Flat-Play-v0 --checkpoint <run>/exported/policy.pt
python scripts/evaluation/plot_rl_push_lqr_axes.py <輸出資料夾>
```

## 10. 結果

（訓練完成後補上。）
