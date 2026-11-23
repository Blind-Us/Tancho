# Tancho V3：站立不漂 + 收左腳 / 收右腳 / 蹲下（r8_300，輪 Kd 1）

同一個 policy 做四件事：零指令原地站（輪里程保持位置）、LT 收左腳、RT 收右腳、按住 Shift/J 兩腳蹲下（放開後 0.28 s 站回，不跳）。沒有跳躍。

## 部署前一定要改的兩件事

1. **輪馬達 MIT 模式 kd = 1.0**（以前是 4）。policy 是用 Kd 1 訓練的。
2. **輸入 34 維**（以前 29）：多了兩個輪里程（29–30）和三個蹲下狀態（31–33），見 `config.json`。輪里程 = 2·clip(0.03614·(q_wheel − 起點角度), ±0.5)，q_wheel 是多圈編碼器角度，開機（或要重設保持點時）歸零。

其餘同 hop r3：50 Hz policy，腿 = 標稱 + 0.6·a + 抬腳參考表（LT/RT 時由 Pi 播放 `reference_lift_table.csv`），輪目標速 = 52.36·a，腿 PD 20/0.2、±12.5 N·m、12.5 rad/s。

## 驗收（全部不倒）

| 測試 | MuJoCo（1 ms，硬接觸） | Isaac |
|---|---|---|
| 60 s 零指令漂移 | **2.4 cm** | 5.5 cm（pitch 偏 2.4°） |
| 摩擦 0.5 / 0.8、輪 Kd 0.8 / 1.2、初始 ±3° | 1.5–4.8 cm | — |
| 推 0.5 m/s（50 ms 約 1.3 N·s） | 不倒，停在 6.7 cm | 不倒（兩輪短暫離地 0.04 s） |
| LT / RT / 蹲下 ×2 | 不倒；左輪 53–56 mm、右輪 46 mm | 不倒；左輪 35 mm、右輪 37–38 mm |

已知弱點：MuJoCo 改成軟接觸（solref 0.02）時會慢慢漂走，約 48 s 倒；dt 2 ms 時漂移大（腳本的顯式腿 PD 在 2 ms 不穩，可能是腳本限制）。按完一輪動作後會離原點 10 cm 左右。

## 檔案

`policy.onnx` / `policy.pt`（輸入 [1,34]、輸出 [1,6]，正規化已包在內）、`config.json`、`reference_io.json`（8 組輸入輸出對照）、`reference_lift_table.csv`、`evaluate_stand_mujoco.py`（MuJoCo 驗證腳本，`--wheel-kd 1.0 --charge-jump 0 --presses LT:3,RT:7,CHG:11`）、`eval/`、兩支影片（Isaac 與 MuJoCo 的 LT/RT/蹲下）。

訓練：run `logs/rsl_rl/tancho_v3_station_skills/2026-11-20_04-01-37_5a1086b-dirty`，第 300 iter（同一次訓練的其他 checkpoint 在 MuJoCo 漂 4–41 cm，這個是用 MuJoCo 篩出來的）。

## 重現 / 重新匯出

在 `Tancho_v3_simready/`、conda 環境 `env_isaaclab`（Isaac Lab）下執行。

**直接重新匯出部署檔**（不重訓）：
```
python scripts/wheel_only/export_policy.py --headless --task TanchoV3-StationSquatKd1Com-v0 \
    --checkpoint releases/tancho_v3_squat_kd1_r8_300/train/model_300.pt
```
會在 `train/exported/` 產生 `policy.pt` 和 `policy.onnx`。

**重訓這一段**：任務 `TanchoV3-StationSquatKd1Com-v0`，程式在 `source/tancho_v3_lab/tancho_v3_lab/tasks/staged/station_skills.py`。
```
python scripts/wheel_only/train.py --headless --task TanchoV3-StationSquatKd1Com-v0 --num_envs 4096 \
    --max_iterations 1500 --seed 45 --allow-dirty \
    --init_checkpoint releases/tancho_v3_squat_kd1_r8_300/train/init_checkpoint.pt --keep_obs_norm
```
交付版是這次訓練的第 300 iter。PhysX 在 GPU 上不是完全決定性的，重訓的第 300 iter 不保證一模一樣。各 checkpoint 在 MuJoCo 的漂移差很多（2–41 cm），所以要再用篩選腳本挑：
```
scripts/wheel_only/screening/candidates.sh <run 資料夾> <標籤> TanchoV3-StationSquatKd1Com-Play-v0 "200 250 300 350 ..."
scripts/wheel_only/screening/robust.sh <policy.pt> <輸出資料夾>
```

**`init_checkpoint.pt` 的來源**（各段都已 commit 在 git 裡）：HopCharge（hop r3 + 蹲下）→ StationSkills（補輪里程輸入，`pad_checkpoint_obs.py --actor-at 29 --critic-at 73`）→ StationSquat（拿掉跳）→ StationSquatHold → StationSquatKd1 ×2（輪 Kd 1）→ 動作雜訊重設為 0.03。

**驗收**：
- Isaac：`scripts/wheel_only/evaluate_stability.py`（60 s 站立 / 推 0.5）、`scripts/wheel_only/evaluate_hop.py --guidance 1 --vx 0 --schedule LT,RT,CHG`
- MuJoCo：`evaluate_stand_mujoco.py --wheel-kd 1.0 --charge-jump 0`
