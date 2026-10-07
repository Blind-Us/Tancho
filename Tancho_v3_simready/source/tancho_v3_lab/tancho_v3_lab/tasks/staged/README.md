# Tancho V3 分階段任務（tasks/staged）

三個階段使用同一套四要素，每個要素一個檔案；`env_cfg.py` 只負責組裝。

| 階段 | 訓練任務 | 機器人 | 命令 | 成功條件 |
|---|---|---|---|---|
| 1 | `TanchoV3-WheelOnly-Flat-v0` | 腿固定（剛體合併），只動兩輪 | 0 | 自穩站立 |
| 2 | `TanchoV3-Stand-Flat-v0` | 6-DOF | 0 | 自穩站立，Play 行為正常 |
| 3 | `TanchoV3-Walk-Flat-v0` | 6-DOF，從階段 2 權重開始 | vx、yaw rate | 依命令行走 |
| 4 | `TanchoV3-Walk-Rough-v0` | 6-DOF，從階段 3 權重開始（`--keep_obs_norm`） | vx、yaw rate | 凹凸 0→2 cm、坡 0→8°（`terrain.py`） |
| 5 | `TanchoV3-Walk-Step-v0` | 6-DOF，從階段 4 權重開始（`--keep_obs_norm`） | vx、yaw rate | 金字塔台階上/下 0.5→3 cm |

| 6A | `TanchoV3-ClimbHop-v0` | 6-DOF，腿 ±0.6 rad，觀測 29 維（+ LT/RT、按下後時間） | vx、yaw rate + Trigger | 平地按鍵抬腳/短跳不倒（參考動作疊加） |
| 6B | `TanchoV3-Climb-v0` / `TanchoV3-ClimbFree-v0` | 同上 | 同上 | 按鍵上台階（固定連招 / 全權重）——**未完成** |

階段 4/5 用 `terrain.py` 的課程：撐到 time-out 且速度追蹤 > 60% 升級，跌倒降級。
驗收：`evaluate_walk.py --task TanchoV3-Walk-Rough-Play-v0`、`evaluate_step.py --direction up|down`。
階段 6（`climb.py`）：Xbox LT 抬左腳、RT 抬右腳、LT+RT 短跳。6A 在參考動作疊加下單腳 37–43 mm、短跳 33 mm 不倒；6B 上台階在 2026-10-07 未成功（policy 一律學成「開慢、躲邊緣」），細節見 `logs/rsl_rl/PROGRESS.md`。

2026-10-07 結果：rough 通過（release `releases/tancho_v3_rough_v1`）；上台階受輪徑/摩擦限制只到約 5 mm（準靜態上限 r(1−cos33°) ≈ 5.7 mm），下 3 cm 沒問題。

每個任務都有 `-Play-v0` 版本：標稱物理、無雜訊、無隨機化、無推力、正立靜止 reset、單一機器人。Walk 的 Play 會顯示命令箭頭。

| 檔案 | 內容 |
|---|---|
| `scene.py` | 地面、IMU、兩種機器人 asset、時間步、馬達參數 |
| `observations.py` | ① Observation |
| `actions.py` | ② Action |
| `rewards.py` | ③ Reward |
| `terminations.py` | ④ Termination |
| `env_cfg.py` | 命令、事件（隨機化）與各階段組裝 |

## 0. 共同設定

- PhysX 200 Hz，policy 50 Hz（decimation 4），episode 20 s。
- 標稱腿姿 thigh = −0.50 rad、calf = +0.87 rad。wheel-only asset 固定在這個姿態；6-DOF 以它作為 reset 姿態和零 action 的目標，所以兩個階段的起始物理狀態相同。
- **關節驅動一律是 force 型。** URDF 轉換時如果 `joint_drive=None`，importer 會建出 acceleration 型驅動，PhysX 把 Kp/Kd 乘上連桿等效慣量（約 1e-4 kg·m²）。結果：
  - 腿在零動作下 0.16 s 內就會塌掉，但 log 裡的 `applied_torque` 仍顯示 12.5 N·m。
  - `logs/diagnostics/fixed_gravity_support.summary.json` 的 FAIL 也是這個原因。
  - 改成 force 型以後，6-DOF 零動作時腿部誤差約 0.02 rad、膝力矩約 0.3 N·m，只會因為倒立擺自然倒下而觸發 `tilt`。
  - `tasks/direct` 的任務（包含凍結的 W-RL V1）仍然是 `joint_drive=None`，沒有修改。
- 輪馬達 DM-H3510：速度模式，Kp = 0、Kd = 4.0，峰值 0.45 N·m，模擬速度上限 188 rad/s。
- 腿馬達 DM-J4310：位置模式，Kp = 20、Kd = 0.2，峰值 12.5 N·m。
- 地面摩擦 0.8，與機器人材質相乘（multiply）。

## ① Observation

Actor 只用實機量得到的量，訓練時加均勻雜訊。Critic 拿到同樣的量（無雜訊），再加上特權狀態（asymmetric actor-critic，不部署）。

| 項目 | 維度 | 縮放 | 雜訊 | wheel-only | 6-DOF |
|---|---|---|---|---|---|
| `imu_projected_gravity` | 3 | 1 | ±0.02 | ✓ | ✓ |
| `imu_ang_vel` | 3 | 0.25 | ±0.05 rad/s | ✓ | ✓ |
| `velocity_commands` | 3 | 1 | — | ✓ | ✓ |
| `wheel_vel` | 2 | 0.05 | ±0.2 rad/s | ✓ | ✓ |
| `last_action` | 2 / 6 | 1 | — | ✓ | ✓ |
| `leg_pos`（相對標稱姿態） | 4 | 1 | ±0.01 rad | | ✓ |
| `leg_vel` | 4 | 0.05 | ±0.2 rad/s | | ✓ |
| **合計** | | | | **13** | **25** |

- Critic 額外的輸入：真實 base 線速度、角速度、projected gravity；6-DOF 再加 base 高度。
- 不使用輪子角度（會無界累積）。
- 每個階段都輸入速度命令。站立時命令是 0；這樣階段 2 的 policy 跟階段 3 的輸入格式相同，可以直接接著微調。

## ② Action

| 項目 | 維度 | 意義 |
|---|---|---|
| `leg_pos`（僅 6-DOF） | 4 | 位置目標 = 標稱姿態 + 0.25 rad × action |
| `wheel_vel` | 2 | 速度目標 = 52.36 rad/s（額定 500 rpm）× action |

- `clip_actions = 1`，所以每個腿關節最多偏離標稱姿態 ±0.25 rad，policy 沒辦法把腿折到蹲地。
- 平滑性由 `action_rate` reward 負責，不靠縮小 scale。

## ③ Reward

權重單位是「每秒」，reward manager 會乘上 step_dt = 0.02。

| Term | 函式 | wheel-only | 6-DOF 站立 | 6-DOF 行走 | 說明 |
|---|---|---|---|---|---|
| `is_alive` | `is_alive` | +1 | +1 | +1 | 完整 20 s 得 +20 |
| `termination_penalty` | `is_terminated` | −200 | −200 | −200 | 每次跌倒額外扣 4 |
| `upright` | `flat_orientation_l2` | −10 | −10 | −10 | sin²(傾角) |
| `ang_vel_xy` | `ang_vel_xy_l2` | −0.05 | −0.05 | −0.05 | pitch/roll 阻尼 |
| `lin_vel_xy` | `track_lin_vel_xy_exp` | +1（std 0.25） | +1（std 0.25） | **+2（std 0.5）** | 追蹤速度命令 |
| `ang_vel_z` | `track_ang_vel_z_exp` | +0.5（std 0.25） | +0.5（std 0.25） | **+1（std 0.5）** | 追蹤 yaw rate 命令 |
| `wheel_vel_l1` | `joint_vel_l1`（輪） | −0.02 | −0.02 | **關閉** | 消除站立時的慢速漂移；行走時會跟命令衝突 |
| `wheel_torque` | `joint_torques_l2`（輪） | −0.01/0.45² | 同左 | 同左 | 以峰值正規化的出力成本 |
| `action_rate` | `action_rate_l2` | −0.01 | −0.01 | −0.01 | 平滑 |
| `wheel_vel_limit` | `joint_vel_limits` | −1 | −1 | −1 | 超過速度上限 90% 開始懲罰 |
| `vertical_vel` | `lin_vel_z_l2` | | −1 | −1 | 腿可以讓機身上下彈 |
| `capture_point` | `custom_rewards.wheel_capture_point_l2` | | −0.5 | −0.5 | 全機 capture point 到輪軸的距離，誤差 50 mm 每秒扣 0.5 |
| `mirror` | `custom_rewards.mirror_leg_l2` | | −0.5 | −0.5 | 左右腿對稱 |
| `leg_torque` | `joint_torques_l2`（腿） | | −0.01/12.5² | 同左 | 跟輪子同樣的正規化方式 |
| `leg_acc` | `joint_acc_l2`（腿） | | −2.5e-7 | 同左 | |
| `leg_pos_limits` | `joint_pos_limits`（腿） | | −10 | −10 | |

- 6-DOF 加入 capture point 的原因：腿可以動以後，機身 pitch 不再能代表 COM 的位置，所以直接把平衡寫成「COM 在輪軸上方」（靜止時就退化成 COM 對準輪軸）。
- 不追蹤固定站高，也不追蹤固定腿姿；腿姿的範圍已經由 action 限制住。

## ④ Termination

- `time_out`：20 s，正常結束，不算失敗。
- `tilt`：機身 z 軸偏離鉛直超過 15°（跟 LQR 推力實驗相同）。
- `body_contact`（僅 6-DOF）：base、thigh 或 calf 的接觸力超過 10 N。wheel-only 不需要這一項，因為 ±15° 內只有輪胎會碰到地面（見 `scripts/wheel_only/ground_clearance.py`）。

## 命令與事件

| 項目 | 站立 | 行走 |
|---|---|---|
| 命令 vx | 0 | [−0.6, 0.6] m/s |
| 命令 vy | 0 | 0（兩輪同軸，無法側移） |
| 命令 yaw rate | 0 | [−1.0, 1.0] rad/s |
| 命令重抽間隔 | — | 4–8 s，20% 的環境給 0 命令 |

| 事件 | 訓練 | Play |
|---|---|---|
| 摩擦 | [0.6, 1.0] | 0.8 |
| base 質量 | 加 [−0.15, +0.25] kg | 不變 |
| reset | pitch ±0.05 rad、vx ±0.1 m/s、pitch rate ±0.2 rad/s；關節回到標稱姿態、速度 0 | 正立靜止 |
| 推力 | 每 3–6 s，vx ∈ [−0.8, 0.8] m/s | 關閉 |

## 訓練與觀看

在 `Tancho_v3_simready/` 下執行：

```bash
# 1. wheel-only 站立
python scripts/wheel_only/train.py --headless --task TanchoV3-WheelOnly-Flat-v0
# 2. 6-DOF 站立
python scripts/wheel_only/train.py --headless --task TanchoV3-Stand-Flat-v0
# 3. 6-DOF 行走：從階段 2 的權重開始
python scripts/wheel_only/train.py --headless --task TanchoV3-Walk-Flat-v0 \
    --init_checkpoint logs/rsl_rl/tancho_v3_stand/<run>/model_final.pt

# 觀看（同時匯出 exported/policy.pt、policy.onnx）
cd scripts/rsl_rl
python play.py --task TanchoV3-Stand-Flat-Play-v0 --checkpoint ../../logs/rsl_rl/tancho_v3_stand/<run>/model_final.pt
```

- 如果任務相關檔案有未 commit 的修改，`train.py` 會拒絕訓練；commit hash 會寫進 run 名稱和 `git_commit.txt`。
- `--init_checkpoint` 只載入 actor/critic 權重，optimizer 和 iteration 都從頭開始。
- PPO 設定（`agents/rsl_rl_ppo_cfg.py`）：
  - 4096 個環境，每次 rollout 24 步。
  - Actor [128, 128, 64]，Critic [256, 256, 128]，ELU，觀測正規化，初始 std 0.2。
  - lr 3e-4（adaptive，desired KL 0.01），γ 0.99、λ 0.95。
  - iteration 數：wheel-only 1500、站立 3000、行走 3000。

## Wheel-only asset

`Tancho_v3_wheel_only.urdf` 由 `scripts/wheel_only/build_wheel_only_urdf.py` 產生：

- 把 base、drawer、pi_case 和兩腿依標稱角度做 FK，合併成單一剛體，合併後的質量、質心、慣性都精確計算。合計 2.5537 kg，COM 在 root 座標 (−14.6, 0, −60.1) mm。
- `scripts/wheel_only/verify_leg_pose.py` 和 `evaluate_standstill.py` 會反推腿角，誤差小於 1e-7 rad。
- 標稱姿態的幾何：root 高 260.5 mm、COM 高 200.4 mm、輪軸到 COM 164.2 mm。

```bash
# 20 s 零命令靜置：pitch ±1°、位移 < 5 cm、不失敗
python scripts/wheel_only/evaluate_standstill.py --headless --checkpoint <run>/exported/policy.pt
```

## 結果

（訓練完成後補上。）
