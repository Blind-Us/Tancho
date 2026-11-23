# Tancho V3 staged 訓練進度

## 2026-10-10 23:28 深蹲（機身底部15cm目標）

使用者試玩淺蹲後授權更深的蹲姿並繼續訓練；db21190新增DeepSquat獨立任務，保持馬達上限。URDF近似IK估髖-.864、膝1.764rad，reference相對標稱(-.36,+.89)，0.6s下蹲、1s保持、0.6s回站。這是policy+reference，不是獨立policy蹲。

8env/200步zero通過，run tancho_v3_deep_squat/2026-10-10_23-03-01_db21190從淺蹲model_final接續400iter、4096env、keep_obs_norm，989.6秒完成，已匯出TorchScript/ONNX。三次驗收不倒，機身下降6.83cm，base_link碰撞框最低角距平地13.69/13.73/13.74cm（考慮機身旋轉，目標15±1.5cm），最大傾角6.90度，回站四關節誤差最大0.0285rad（1.63度）。輪地接觸力<1N連續最長左0.10、右0.14秒，故嚴格持續接地pass_all=false。只有三次固定序列驗證，未做廣泛穩健性測試。

成果outputs/deep_squat_20261010：summary.json、videos/squat-step-0.mp4、REVIEW.md。原淺蹲/rough模型均保留。驗收script scripts/wheel_only/evaluate_deep_squat.py。

## 2026-10-10 22:30 平地下蹲驗收

接觸力補測：左／右輪低於1N最長0.10／0.16秒。三次下蹲回站可重現，但兩輪始終保持有效支撐的標準未過。

使用者新增授權：先練平地下蹲，不跳、不爬階，期限22:30。commit0b4644d，TanchoV3-Squat-v0：兩側髖膝協同reference，0.35s到蹲姿（髖-0.25、膝+0.50相對標稱），保持到0.95s，1.4s回站；actor沿用29維硬體訊號及命令/相位。下蹲是policy+reference共同完成，不宣稱policy獨立學會下蹲。

zero8環境200步通過（mean29.69步）。22:08啟動220iter/4096環境，從HopHip第三輪model600接續並保留正規化，22:18完成（558.3s），模型logs/rsl_rl/tancho_v3_squat/2026-10-10_22-08-42_0b4644d/model_final.pt及exported/policy.pt、policy.onnx。

三次下蹲測試16s沒有倒：保持期間機身下降3.30、3.34、3.34cm，回站後最大四關節誤差0.0338、0.0340、0.0342rad，最大傾角5.70度。輪軸相對起始上升最多11.8mm，超過原先設定10mm門檻，因此pass_all=false，不能宣稱嚴格接地通過；另以接觸力量測核實，見outputs/squat_20261010/contact_summary.json。

影片outputs/squat_20261010/videos/squat-step-0.mp4，summary.json與contact_summary.json。原rough/hop模型不覆寫；馬達硬體參數不變。正式訓練拖到22:08才啟動，原先21:37檢查已通過卻未立刻接訓練，屬執行疏漏，已向使用者說明。

## 2026-10-10 髖膝協同抬腳／樓梯（20:00 驗收）

### 驗收地形更正（18:29，優先於下方初次記錄）

evaluate_hop.py / evaluate_hop_sweep.py 原本未強制平地；StairHip Play預設樓梯，故第一/四輪原「平地」結果實際是在樓梯。修正兩腳本明確建立flat場景並重跑。第二/三輪HopHip預設flat，其壓測數字有效。第一輪76.7%及第四輪66.5%只作錯誤場景歷史，不作平地指標。

真正平地：Claude原StairV2单腳3.79/3.17cm、雙跳1.38/1.12cm，單次序列沒倒但跳高未過；第四輪单腳7.32/5.83cm、雙跳1.21/2.97cm，8.58s向後倒。618組平地倒39.97%、抬升不足32.2%。LT大腿角偏移0.301→0.451rad（約+50%），峰值力矩0.677→1.676Nm；小腿首次回正0.38→0.42s，雙腳未回正即倒，不能宣稱解決回正問題。

樓梯掃描仍有效：2cm13/32、3cm4/32各爬一階不倒；3cm指定0.4m/s/30度案例在單環境錄影重測4.64s倒，並未穩定重現掃描中的成功。真正平地影片：outputs/hip_20261010_stairs/trueflat_video/；總結與原始JSON在同目錄REVIEW.md及trueflat_*.json。整體未通過。

使用者授權自主修改訓練，固定實際馬達上限；D 槽只讀，所有成果寫 /home，不需圖表。四輪均完成註冊/語法與 8 環境 200 步 zero-agent 檢查，最近五輪 timeout 未達 50% 於約600 iter 停止並匯出。沿用 Claude staged train.py，保持觀測正規化；GPU 同時只有一個 Isaac Sim。

| 輪次 | commit / run | 改動 | 驗收 |
|---|---|---|---|
| 1 | bab63e1 / stair_hip 15-38-40 | v5 髖膝參考、支撐腿協同；從 Claude StairV2 最終開始 | 單抬 7.2/4.7 cm；10.48 s 倒；618 組倒76.7%；2cm爬上不倒6/32、3cm0/32 |
| 2 | aa9a539 / hop_hip 16-21-14 | 改成 flat/rough 課程學回穩 | 單抬7.4/5.0cm且全段不倒；雙跳1.1/3.9cm未過；壓測倒47.09% |
| 3 | 51baf5f / hop_hip 17-00-56 | 動作期姿態-3、角速-0.08、alive3、倒地-100 | 单抬7.0/5.8cm不倒；双跳1.3/2.8cm未過；壓測倒44.17%；rough双跳8.72s倒 |
| 4 | 6a872c3 / stair_hip 17-41-53 | v6 膝伸縮各0.12s（舊0.08s），保留回穩reward、轉回樓梯 | 2cm13/32爬上且不倒、3cm4/32；平地壓測倒66.5%；rough單抬6.95/3.84cm，雙跳未過 |

第二、三輪 HopHip 的自動楼梯觸發與階梯任務不同，這兩輪 stairs 結果不作公平樓梯比較。第四輪樓梯樣本只爬上一階；不能宣稱連續爬樓梯完成。

實測大腿參與：相同 LT 測試，Claude StairV2 的大腿偏移0.299rad、峰值力矩0.822Nm；第四輪0.452rad、1.712Nm，支撐側大腿偏移0.096→0.218rad。角度與峰值力矩證明參與增加，未量測正向機械作功，不能宣稱推進功份額已增加。
小腿回正為「reference結束後進入標稱+0.87±0.1rad並維持0.1s的首次時刻」；Claude LT為按鍵後0.38s，第四輪LT/RT為0.42s，rough雙腳為1.0s。首次回正不等於持續回正。原模型小腿曾回正但仍3.3s倒，說明也需處理殘留角動量、重心及輪子平衡，不應只歸因於小腿角度。

四輪上限已達，不再新訓練。整體未通過；保留 Claude rough/hop 基準不覆寫。詳細結果與模型指向：outputs/hip_20261010_stairs/status.json，第三輪單腳回穩可參考 outputs/hip_20261010_balance。影片與驗收指令見 outputs/hip_20261010_stairs/REVIEW.md。

## 2026-10-06

| 時間 | 階段 | run | commit | 改動 | 結果 |
|---|---|---|---|---|---|
| 04:05 | wheel-only | `tancho_v3_wheel_only/2026-10-06_04-05-18` | dbcfc93 | 誤用 `scripts/rsl_rl/train.py`，PPO 設定被換掉 | 靜站通過：pitch 0.85°、漂移 2.1 cm。不能部署（設定不對） |
| 08:12 | 6-DOF 站立 | `tancho_v3_stand/2026-10-06_08-12-47_535555f` | 535555f | 原始設定 | 不倒、推 0.5 m/s 能恢復；左右腿差 0.058 / 0.112 rad，用 0.12 rad 門檻算通過 |
| 09:04 | wheel-only | `tancho_v3_wheel_only/2026-10-06_09-04-08_30c2ae5` | 30c2ae5 | 改用正確腳本 | 沒過：pitch −0.67° 不變，輪速 0.74 rad/s 等速前進，20 s 漂 53 cm |
| 09:22 | 6-DOF 站立（第 2 輪） | `tancho_v3_stand/2026-10-06_09-22-02_996ea2d` | 996ea2d | mirror −0.5→−20，新增 leg_deviation −1.0 | **通過**（不推 / 推 0.5 m/s）：pitch 0.93°、漂移 2.8 / 1.4 cm、左右腿差 0.008 rad、腿偏離 0.023 rad、推後恢復 |
| 10:13 | wheel-only（第 2 輪） | `tancho_v3_wheel_only/2026-10-06_10-13-21_4dfa686` | 4dfa686 | wheel_vel_l1 −0.02→−0.2 | **通過**：pitch 0.95°（上限 1°）、漂移 2.7 cm、最大輪扭力 0.38 Nm |
| 14:02 | 6-DOF 行走（第 1 輪） | `tancho_v3_walk/2026-10-06_14-02-39_0985f04` | 0985f04 | 從站立第 2 輪 `model_final.pt` 開始（--init_checkpoint） | 第 300 iter 檢查點沒過，停掉：episode 147 步、tilt 99%。命令 0.3 m/s 跑到 0.98 m/s 後往前倒。原因：沿用站立的觀測正規化，命令維度 std≈0 且樣本數 2.9e8，命令被放大約 5 倍 |
| 14:10 | 6-DOF 行走（第 2 輪） | `tancho_v3_walk/2026-10-06_14-10-31_04428c5` | 04428c5 | init_checkpoint 時把觀測正規化的樣本數歸零 | 三個檢查點都過，不倒；yaw 追得準（+1.0→+1.018），但 vx 只到命令的約 60%（0.3→0.17、0.6→0.37）。原因：capture point 項在等速時懲罰速度 |
| 15:02 | 6-DOF 行走（第 3 輪） | `tancho_v3_walk/2026-10-06_15-02-25_5b6a164` | 5b6a164 | walk 的 capture_point 權重設 0 | **通過**：每段 vx 誤差 ≤ 0.005 m/s、yaw 誤差 ≤ 0.016 rad/s，不倒，停下 2 mm/s，左右腿差 0.018 rad；後退段最大 pitch 11°（終止門檻 15°） |

## 2026-10-07（rough terrain → 3 cm 台階）

新任務（commit 254db26）：`TanchoV3-Walk-Rough-v0`（隨機凹凸 0→2 cm 峰對峰、坡 0→~8°、15% 平地）與 `TanchoV3-Walk-Step-v0`（金字塔台階上/下 0.5→3 cm，踏面 0.6 m，另含 20% rough）。觀測/動作與 flat walk 相同（盲走，無高度掃描，可部署），用 `--init_checkpoint` 從 walk v1 接續。課程：撐到 time-out 且速度追蹤 > 60% 升級，跌倒降級（內建 `terrain_levels_vel` 量直線距離，對 yaw-rate 命令不適用）。

| 時間 | 階段 | run | commit | 改動 | 結果 |
|---|---|---|---|---|---|
| 09:09 | rough（第 1 輪） | `tancho_v3_walk_rough/…_254db26` | 254db26 | 從 walk v1 `model_final.pt` 開始 | 200 iter 主動停掉：一開始 0.2 s 就倒，200 iter 後 episode 仍只有 ~110 步、地形全在 level 0。原因：`--init_checkpoint` 會把觀測正規化歸零（為 stand→walk 設計），walk→rough 觀測分布相同，歸零反而把會走的 policy 打壞 |
| 09:22 | rough（第 2 輪） | `tancho_v3_walk_rough/…_a1bf9fc` | a1bf9fc | train.py 新增 `--keep_obs_norm`，保留 walk 的觀測統計 | 1200 iter 提前停（800 iter 後已持平：timeout 97–98%、地形在最高等級循環、vel 誤差 0.17）。用 `model_1200.pt`。**Rough 驗收通過**（Play：單片 16 m、2 cm 峰對峰凹凸）：不倒，各段 vx 誤差 ≤ 0.046 m/s、yaw 誤差 ≤ 0.091 rad/s，停下 3 mm/s，左右腿差 0.016 rad，後退段最大 pitch 9.9°。平地回歸也通過，但 vx 偏慢約 7–11%（0.3→0.268、0.6→0.557；v1 幾乎零誤差）。對照：walk v1 在同一片 rough 上 13.6 s 倒（0.6 m/s 段跑到 0.85 m/s 後失控）。影片 `videos/play/rl-video-step-0.mp4` |
| 10:36 | step（第 1 輪） | `tancho_v3_walk_step/…_a1bf9fc` | a1bf9fc | 從 rough `model_1200.pt` 開始，`--keep_obs_norm` | **1000 iter 檢查點沒過，停掉**：timeout 63%（門檻 80%）、等級 700 iter 後卡在 3.4–3.5、body_contact 升到 12%。驗收（`model_1000`）：上台階只過得了 5 mm（0.6 m/s 連上 6 階），1/1.5/2/3 cm 在 0.4 和 0.6 m/s 下全部撞第一個邊緣後倒；下 3 cm 連下 5 階不倒。平地/rough 退步：vx 只到命令約 65%（0.6→0.39），學會慢開以減少在邊緣跌倒。不採用 |

### 台階結論（11:50）

- 下台階 3 cm：rough `model_1200` 就做得到（1/2/3 cm 都連下 6 階，pitch 5.5°）。
- 上台階：rough policy 連 1 cm 都過不去；step 訓練後也只到 5 mm。
- 原因是物理限制，不是 reward 的問題。輪半徑 36.1 mm、有效摩擦 0.64（摩擦錐半角 33°）：
  - 準靜態可爬高度上限 = r(1−cos33°) ≈ 5.7 mm，和實測的「5 mm 過、10 mm 倒」吻合。
  - 再高就得靠撞擊動量。3 cm 時接觸法線離鉛直 80°，撞上去時大部分速度會損失，需要的速度遠超過命令上限 0.6 m/s。
  - policy 是盲走的，不知道台階在哪，沒辦法提前加速或跳。
- 要上 3 cm 至少需要改下列其中一項（都會動到部署規格，要使用者決定）：
  1. 加感知（高度掃描 / 深度），讓 policy 能提前起跳或加速；
  2. 放寬腿的動作範圍（目前 ±0.25 rad）讓它能跳；
  3. 提高速度命令上限；
  4. 換大輪子。
- 交付建議：用 **rough `model_1200`**（rough + 坡 + 下 3 cm），step `model_1000` 不採用。

### Climb：操作者 Trigger 抬腳 / 短跳（使用者 12:40 決定）

使用者的方案：Xbox 手把操作。**LT = 抬左腳、RT = 抬右腳、LT+RT = 短跳**，腿的範圍放寬到標稱姿態 ±0.6 rad（使用者同意變更部署規格）。

- 新任務 `TanchoV3-Climb-v0`（commit 29affcf、8ecd778）。
  - actor 27 維：原本 25 維，最後加 LT、RT 各 1 維（0/1）。
  - 訓練時模擬操作者按鍵：用特權高度掃描（只在模擬裡用，不部署）找出輪子前方 > 1.2 cm 的台階邊緣。邊緣進入隨機前視距離（8–30 cm）時，按下該側的鍵：正面接近時兩鍵同時按（短跳），斜著接近時只按先到的那一側。
  - 另外有隨機誤按，90% 的時間操作者是專心的。
  - reward：按下那一側的輪子離地高度（5 cm 拿滿分）。按鍵期間和放開後 0.6 s 內暫停 mirror、腿偏離、垂直速度三項懲罰。
  - 地形：上台階 1→3 cm 佔 50%、下台階 25%、rough 15%、平地 10%。
- `expand_climb_checkpoint.py`：從 rough model_1200 擴充權重。新輸入欄位補 0，腿輸出乘 0.25/0.6，正規化的 eps 也補償，起始行為與 rough 完全等價（差異 1e-7）。

| 時間 | run | 改動 | 結果 |
|---|---|---|---|
| 12:58 | climb 試跑 | 腿 std 0.15、輪 std 0.05、正規化 count 降到 1e6 | 10 iter 時 episode 才 35 步，停掉。離線驗證權重等價，問題出在 std 太大和正規化漂移 |
| 13:00 | climb（第 1 輪） | 保留 count，腿 std 0.05，輪 std 不變 | 300 iter 停：wheel_lift 每 episode 約 0.001（完全沒抬腳）、動作雜訊縮到 0.02、vel 誤差 0.43（卡在台階前） |
| 13:18 | climb（第 2 輪） | 加連續的收腳 reward（權重 2）；按鍵改成每個邊緣只按 0.4 s + 冷卻 0.8 s，防止停在邊緣刷分 | 200 iter 停：收腳 reward 只拿到上限約 1.5%，訊號只佔總 reward 約 0.1%。（第一次啟動因殘留程序佔 GPU 而 CUDA 錯誤，kill -9 後重開） |
| 13:29 | climb（第 3 輪） | 抬腳/收腳權重 2→10，腿初始 std 0.15 | 收腳 reward 升到 0.10，但離地沒變。model_400 實測：1/2/3 cm 都不倒但 0 階；命令 0.4 m/s 只開 0.16；在邊緣前 10 cm 停下晃。學會「按鍵就蹲」並避開台階 = reward hacking |
| 13:53 | climb（第 4 輪，最後一輪） | 新增 climb_progress：episode 內輪下地面創新高才給分（3 cm 一階 = 3），收腳權重降回 2 | 400 iter 停：爬高 reward 持平約 0.02、離地沒變、動作雜訊 0.01。model_400 實測：1 cm 和 3 cm 都是 0 階，開到 x≈0.70 m（邊緣在 0.75 m）停下，vx 0.135（命令 0.4）。4 輪用完，停止迭代 |

### Climb 結論（14:25）

- 4 輪都沒學會上台階。每輪都在約 300 iter 內收斂到同一個局部解：在台階前減速停下，避開邊緣（跟 Walk-Step 一樣），動作雜訊在發現抬腳/跳之前就縮到 0.01–0.02。
- 還沒驗證的關鍵問題：在 ±0.6 rad、0.4 s 按鍵內，這台機器人物理上能不能把輪子抬 3 cm 放上台階？RL 一直在盲目探索一個可能根本做不到的動作。
- 建議下一步（等使用者決定）：
  1. 先寫一個腳本化的抬腳/短跳測試：輪子照常平衡，腿跑固定軌跡，量最大輪子離地高度與能否上 1/2/3 cm。先確認物理上做得到，也找出可行的動作。
  2. 做得到的話，用這個動作當參考軌跡（模仿 reward），讓 RL 從「會跳」開始學時機和平衡，不靠隨機探索。
  3. 訓練時先只放上台階地形和按鍵，不放隨機誤按，等學會再加回來。


### 物理可行性測試（14:30–15:05，使用者：沒有相機，維持人按 Trigger）

`scripts/wheel_only/leg_lift_feasibility.py`：rough policy 控制輪子平衡，腿跑固定的「（蹲）→ 蹬 → 收」軌跡。結果在 `logs/rsl_rl/feasibility/`。

- **運動學（URDF，大腿 0.15 m、小腿 0.10 m）**：輪軸保持在機身下方 1 cm 內時，±0.25 rad 只能收 16 mm（這是一開始腿「完全沒抬」的原因）；±0.6 rad 能收 42 mm、往下蹬 21 mm。幾何上夠上 3 cm。
- **平地動態**：
  - 最佳組合（蹬 0.02 s 後收 0.15 s）輪子離地 26.6 mm，但 20 種組合全部倒下。
  - 倒的原因：大腿快速擺動的反作用力讓機身 0.15 s 內俯仰到 15°，觸發 tilt。rough policy 沒學過這種擾動。
  - 先蹲再蹬的組合，蹲的時候就倒了。
  - Kp 20→50：25 mm，沒有改善，瓶頸不在馬達增益。
- **台階（每個高度 40 種組合：速度 0.3/0.5、觸發距離 0–10 cm、蹬/收時間）**：
  - 1 cm：1/40 成功（0.5 m/s、距邊緣 10 cm 開始收腿 0.25 s）。
  - 2 cm：0/40。
  - 3 cm：0/40。撞到邊緣時輪子最高到 40 mm，但都倒下。
- **結論**：抬腿的幾何和力量大致夠（離地 2.7–4 cm），卡住的是動作過程中的平衡：需要輪子在收腿的同時主動抵消俯仰。這正是 RL 要學的部分，但純探索找不到。3 cm 在這台機器人的能力邊緣。

### Climb 第 5 輪：參考動作（使用者選方案 1，15:00–16:25）

| 時間 | run | 改動 | 結果 |
|---|---|---|---|
| 14:49 | climb 5a | 參考抬腳軌跡（蹬 0.02 s → 收到 0.22 s → 0.40 s 回標稱）：按鍵上升緣啟動，疊加到腿輸出，300 iter 全量後在 1500 iter 前退到 0；加模仿 reward；actor 加「按下後經過時間」（29 維）；台階 1→2 cm、不誤按 | model_300：0.115 m/s 慢開，按鍵在邊緣前約 1 s 觸發（距離窗口 × 慢車速），抬腳做在空地上，學成一按鍵就倒車。停掉 |
| 15:09 | climb 5b | 改用 time-to-contact 按鍵（碰到前 0.1–0.3 s）；evaluate_step 改看最後高度（之前有一次 4.9 m 的接觸數值暴衝灌水） | model_300：0.13 m/s，開到 x≈0.69 m 停，有無疊加都 0 階。避開台階的局部解在學會抬腳平衡之前就出現。停掉 |
| 15:30 | **步驟 A**：`TanchoV3-ClimbHop-v0` | 平地 + rough，隨機按 LT/RT/雙鍵（約每 4 s 一次），參考動作全程疊加 | 800 iter 跑完：timeout 0 → 95%。驗收 `evaluate_hop.py`（0.3 m/s，LT/RT/雙鍵各 2 次）：**有疊加時**單腳 37–43 mm、雙腳 33 mm，全程不倒，pass；**無疊加時**腿不動（還沒學自己產生動作，這是步驟 B 的工作）。直接上台階（疊加，0.3/0.5 m/s，1/2/3 cm）：都在邊緣倒下（沒看過台階） |

使用者指示：這輪報告完 git，不做下一輪。

下一步（等使用者決定）：
- 步驟 B：從步驟 A 權重接續，上台階 1→2→3 cm。
- 參考動作的疊加有兩種處理方式：
  1. 照原計畫退到 0，部署規格不變；
  2. 保留疊加：Pi 按鍵後播放固定的 0.4 s 腿部偏移表，policy 只負責平衡。比較好學，但 Pi 端要多一段程式。

### Climb B1（固定連招版，19:50–20:55）

| 時間 | run | 改動 | 結果 |
|---|---|---|---|
| 19:50 | B1 第 1 次 | 從步驟 A 接續，參考動作固定疊加，台階 1→3 cm，30% 隨機按鍵 | 700 iter 停：爬高 reward 持平 0.03、等級卡在約 2.5、body_contact 升到 18%。model_300 追蹤：會開到邊緣、會按鍵、輪子抬 21 mm，但車速只有 0.2 m/s（命令 0.4），抬腳在邊緣前約 3 cm 就落回地面 |
| 20:28 | B1 第 2 次 | 命令改成以前進為主（vx 0.3–0.6、yaw ±0.3） | 400 iter 停：速度追蹤誤差升到 0.59。model_400：命令 0.5 只開 0.16 m/s；3 cm 碰到邊緣倒，1 cm 在邊緣前停下（沒按鍵）。又是同一個「開慢、躲台階」的局部解 |

結論：
- 輪子要在空中往前走 4–6 cm 才能落上台階，騰空時間約 0.1 s，所以接近速度要 ≥ 0.4–0.6 m/s。
- 在台階地形上 policy 一律選擇不理命令、開慢避開。
- 停止迭代，等使用者決定。
| 21:07 | B1 第 3 次 | 使用者的「舉起來往前摔」：按鍵窗口（按下到放開後 0.6 s）內 tilt 門檻 15°→35°、暫停 upright 和 ang_vel_xy 懲罰 | 400 iter 停：3 cm 時有往前倒（最大 pitch 32°），比之前多往前 7 cm（x 0.82 m），但還是倒下沒上去；1 cm 在邊緣前停下；車速 0.18 m/s；爬高 reward 持平 |

使用者指示：這輪無效就停，整理能交付的東西，不再開新訓練。

### 交付（21:50）

`releases/tancho_v3_rough_v1/`：rough `model_1200`。
- 內容：`DEPLOY.md`、`reference_io.json`（與 walk v1 同一組輸入）、`SHA256SUMS`、驗收 summary、`play_rough.mp4`。
- I/O 與 walk v1 完全相同，Pi 端只要換模型檔。
- 模型檔被 gitignore，和 walk v1 一樣要另外上傳到 GitHub release。

上台階（Climb）實驗的程式碼、工具和紀錄都已 commit，可以之後接著做。

### 使用者 GUI 試玩 ClimbHop（22:10）

- 遙控腳本 `teleop_keyboard.py`：Q / E / J（或 Q+E）= LT / RT / 短跳。
- 試玩時修了兩個問題：
  1. 按住按鍵時算移動命令出現 KeyError；
  2. 空白鍵是 Isaac Sim 的播放/暫停鍵，短跳改用 J。
- 結果：單腳抬、單次短跳 OK；**連續兩次短跳（double jump）會倒**。訓練時每次按鍵後有 0.8 s 冷卻，沒練過連跳。

## 2026-10-08（夜間自主：LT/RT 抬腳、LT+RT 小跳的穩健性）

使用者授權：可自由改 reward / 課程，邊界是機器人本身與 IMU/編碼器量得到的訊號；Isaac Sim 同時只開一個；早上前交出結果，做完可以教新東西。

### 連跳會倒的原因（03:00）

新壓力測試 `scripts/wheel_only/evaluate_hop_sweep.py`：一次跑 618 個情境（單按、連跳 2/3 次、左右交替、先抬左再跳）× 按鍵間隔 0.15–1.2 s × 按住 0.1/0.3 s × 6 種行駛命令（含倒車、快速轉彎），參考動作疊加 1.0。

| 模型 / 規則 | 倒地率 | 備註 |
|---|---|---|
| v1（c2728c0），舊規則 | 5.7% | 間隔 < 0.4 s 時 13–27% 會倒；≥ 0.5 s 為 0 |
| v1，按鍵排隊規則 | 1.8% | 只改規則，不重訓 |
| v2（3f4fab9 訓練），排隊 + 雙鍵同步 | **0.8%** | 抬腳不足 1.1% |

- 原因：第二次按鍵在前一個 0.4 s 參考動作做到一半時重啟（腿從收腳瞬間跳回蹬腳）。GUI 用鍵盤快速點兩下 J 就是這個情況。
- 新規則（commit 3f4fab9、81625fc；**Pi 端要照做**）：
  1. 某一側的抬腳動作還沒做完（相位 < 0.4 s）時，該側的新按鍵先排隊（最多一個），做完立刻接上。
  2. LT+RT 按下時若有一隻腳還在抬，另一隻也等它，兩腳一起開始。
- v2 訓練（`tancho_v3_climb_hop/2026-10-08_03-34-23_3f4fab9`，從 v1 接續、`--keep_obs_norm`、800 iter）：一半的隨機按鍵放開後 0.1–0.8 s 再按一次（可串連），LT+RT 兩鍵最多差 60 ms。結束時 timeout 88%（含 rough 地形等級 ~6）。
- v2 剩下的倒地：倒車 0.3 m/s 時第二跳間隔 0.5 s，以及 0.6 m/s + 1 rad/s 轉彎時快速左右交替。

### ClimbHopFree：拿掉參考動作疊加（04:19 開始）

`TanchoV3-ClimbHopFree-v0`：從 v2 接續，疊加在 100 iter 後線性降到 0（1100 iter），保留模仿 reward，1500 iter。目標是 Pi 不用播放腿部偏移表，policy 自己做出抬腳。

ClimbHopFree 結果（05:10 停在 iter ~1150）：疊加降到 0 的過程中，抬腳 reward 從 0.10 掉到 0.0035，policy 選擇「不抬腳、穩穩站著」而不是自己接手抬腳（timeout 98%）。model_1100 無疊加：6 次按鍵 0 mm。模仿 reward（權重 3）抵不過抬腳帶來的倒地風險。**結論：交付 v2 + Pi 端參考動作表**。參考表只依按鍵與按下後經過的時間，是純前饋，不需要額外感測器，在部署邊界內。

### Recover：摔倒後站起來（05:17 開始）

`TanchoV3-Recover-v0`（commit 435ecc1），從 v2 接續：reset 後把機身前後傾斜 U(±最大傾角) 再從 3 cm 落下；只有 10 s timeout 會結束；站好（傾角 < 15°、高度 > 85%）每秒 +3，機身/大腿/小腿著地每個每秒 −0.5。最大傾角課程：20° 起，timeout 時站好的比例（EMA）> 80% 就 +10°，到 90°。

Recover 第 1 次（05:17）：課程每次 reset 都更新 EMA，60 iter 內就升到 90°，停掉重開。
Recover 第 2 次（05:21，commit 3c8fc5d）：EMA 依完成數加權、每級至少 50 iter。卡在 30°（timeout 時站好 63%，升級門檻 80%），600 iter 停。`evaluate_recover.py`（0–90° 每 5°，前/後倒，落下 3 cm，看 8–10 s 是否站好）：前 10°、後 15°，和 hop v2 本身（前 15°、後 20°）差不多。
RecoverWide（05:54，commit 4bf2d44）：一開始就 0–90° 均勻隨機倒地，加 cos(傾角) 連續 reward，1000 iter 跑完。站好時間只佔約 18%，評估前後都只到 15°。

站起來的結論：
- 這晚的三次嘗試都只學會「15–20° 以內接住」，平躺站起來完全沒有出現。
- 物理上的限制（推算，沒有另外做腳本驗證）：兩輪合計峰值扭力 0.9 N·m，在 36 mm 輪徑上最多約 25 N 水平力，摩擦 0.64 又把它限到約 16 N（≈ 0.64 mg），所以靠輪子把機身「推回來」的上限大約是 tan θ ≈ 0.64 → 30° 左右；機身躺平時重力對輪軸的力矩約 3.9 N·m，輪子做不到，只能靠腿撐。腿目前只能離標稱 ±0.6 rad，撐地的姿勢可能根本到不了。
- 要繼續做的話需要使用者決定：放寬腿的範圍（例如用完整的 URDF 關節範圍）讓腿能當支架，或是接受「倒了就人扶」。

### 早上交付

- `releases/tancho_v3_hop_v2/`（commit ed687fa）：DEPLOY.md、腿部偏移表、reference I/O、壓力測試結果、兩段影片。模型檔被 gitignore，要另外上傳。
- GPU 已空，沒有訓練在跑。

## 2026-10-08 白天：領域隨機化 + 3 cm 上樓梯（使用者給 3–4 小時）

### ClimbHopDR（11:43，commit 61eb369）

從 hop v2 接續，加：腿 Kp/Kd、輪 Kd × U(0.8, 1.2)，機身重心 ±1 cm，控制延遲 0–20 ms（每個 policy 步後的前 0–4 個 5 ms 物理子步仍用上一個目標）。800 iter，結束時 timeout 63%（還在上升）。

壓力測試（618 種按法，`evaluate_hop_sweep.py --delay-substeps / --gain-scale`）倒地率：

| 情況 | v2 | DR |
|---|---|---|
| 標稱 | 0.8% | 0.8% |
| 延遲 10 ms | 55% | 19% |
| 延遲 20 ms | 100% | 8% |
| 延遲 20 ms + 增益 0.8 | 100% | 25% |

**v2 不能在有 ≥ 10 ms 延遲的實機上用。**

### 台階時機掃描（12:20，`evaluate_step_sweep.py`，DR 模型 + 參考疊加）

車速 0.3–0.6 × 提前按 0–0.35 s（time-to-contact）：1 cm 全倒（舊的觸發門檻 1.2 cm，沒按鍵）；2 cm 32 格中 9 格上去 1–2 階（2 格不倒）；3 cm 2 格上去 1 階但落地後 0.1–0.2 s 倒。輪軸最高 4.6 cm，高度夠，問題在落地平衡。

### ClimbStepDR（12:34，commit 47bab28）

從 ClimbHopDR 接續，B1 設定 + DR + 延遲；按鍵提前 0.15–0.25 s、觸發門檻 8 mm；跌倒懲罰 −200 → −20、爬高 reward 100 → 300（讓「開慢躲台階」不再划算）。1200 iter。

ClimbStepDR 第 1 次：300 iter 後持平（爬高 reward ~0.35、地形等級 ~1.7、timeout 41%），600 iter 停。model_600 時機掃描（DR 延遲不開、參考疊加）：**1 cm 穩定上**（0.6 m/s、提前 0.05–0.15 s 按，連上 5 階不倒）；2 cm 32 格中 7 格上 1–4 階；3 cm 0 格。最佳按鍵時機是提前 0.05–0.15 s，比訓練用的 0.15–0.25 s 晚。
第 2 次（13:20）：從 model_600 接續，按鍵改提前 0.05–0.2 s，700 iter。
第 2 次結果：一樣在 ~300 iter 持平（爬高 ~0.33、地形 ~1.7、timeout 36%）。掃描：1 cm 21/32 格上 2–4 階不倒（提前 0.1–0.25 s 都行），2 cm 0/32（第 1 次 model_600 有 5/32），3 cm 0/32。平地抬腳壓力測試：標稱倒地 3.7%、延遲 20 ms 9.1%。

### 驗收（14:00，使用者指定 14:30 驗收、不再開新一輪）

`releases/tancho_v3_dr_v1/`：hop_dr（平地抬腳，抗延遲）、step_dr（1 cm 上台階）、壓力測試與台階掃描結果、1 cm 上台階影片。2 / 3 cm 上台階沒做到。

### 參考動作 v2：一腳抬一腳蹬、蹲→蹬→收的小跳（使用者提議，14:03–15:10）

`climb.reference_leg_offsets_v2`（commit 4354366）：單按時另一腳往下伸（REF_EXTEND），兩側同時開始時改成 0–0.10 s 半蹲、0.10–0.18 s 蹬、0.18–0.30 s 收、0.40 s 回正。
- 不重訓直接套在 step_dr：單腳抬起的輪子 39 mm（v1 40 mm），另一側輪子被蹬得彈起 25 mm；小跳只離地 9–10 mm（v1 34–38 mm），8.5 s 倒。腿往下伸的行程只有約 2 cm，0.08 s 蹬完機身往上的速度很小，離地主要靠收腳。
- `TanchoV3-ClimbStepDRv2`，從 step_dr 接續 600 + 300 iter：timeout 34%（v1 版 36%），地形等級 ~1.6。
- 台階掃描：1 cm 14/32 格上去不倒（v1 21/32），2 cm 0/32（v1 0/32），3 cm 沒測。
- 結論：在目前 ±0.6 rad 的腿範圍下，v2 沒有比 v1 好；step_dr（v1）仍是交付版。

## 2026-10-09 3 cm 上樓梯（使用者 04:53：自由訓練與修改，邊界只剩 URDF 與實機規格，12:00 驗收）

計畫文件：https://claude.ai/code/artifact/ed97c5b2-36bb-4e80-b984-1532e6a20497 。參考 CTBC（arXiv 2509.02986，接觸觸發 + 前饋抬腳，Tron1 上 20 cm）與 Ascento 盲爬台階（arXiv 2402.06143，位置式任務 + 停滯懲罰，一輪先上、另一輪跟上）。

- URDF FK（腿平面，x 前、z 上）：大腿 +0.3 rad 讓輪軸往前 68–80 mm（模擬實測 80），小腿 +0.6 往前 49、往上 37 mm。輪軸離機身下方 1 cm 內的最大收腳：±0.6 rad 42 mm、±1.0 rad 78 mm。
- `TanchoV3-Stair-v0`（commit 69fa67f）：腿 ±1.0 rad；v3 參考動作（0.10 s 抬到 +55 mm、0.20 s 往前 +50 mm、0.28 s 放在 +30 mm、0.40 s 回標稱）；模擬操作者用 1 cm 細掃描，輪子離邊緣 < 1.5 cm 就按那一側，兩側同時碰到時先按近的、另一側等 0.3 s；停滯懲罰 −1/s；摩擦 0.4–1.2、重心 ±2 cm。從 step_dr 換算到 1.0 rad 腿比例後接續。
- 訓練前（未訓練）掃描 3 cm：0/32，碰到邊緣 0.1 s 內就倒或來不及觸發。
- Stair 第 1 輪 05:01 開始，800 iter。
- Stair 第 1 輪結果（05:48）：地形等級仍停在 ~1.8。掃描（車速 × 靠近角度）：**1 cm 32/32 不倒**、**2 cm 15/32 不倒**（之前最好 5/32，斜著靠近 +10–30° 最好）、3 cm 0/32（按下後 0.15 s 內倒，輪軸只抬 1–2 cm，單輪支撐時側倒）。
- Stair 第 2 輪（05:55 起，從第 1 輪 model_final 接續，800 iter）：地形課程改成「爬上 2 階就升級、爬不到半階又跌倒才降級」。
- Stair 第 2 輪結果（06:40）：地形等級升到 ~2.35，但速度追蹤從 1.08 掉到 0.45（又變慢）。掃描：1 cm 27/32（只上 1–3 階，開得慢）、2 cm 8/32、3 cm 0/32。**比第 1 輪差**。
- Stair 第 3 輪（13:55，`TanchoV3-StairFast-v0`，從第 1 輪接續 800 iter）：v3 參考動作快 40%（0.06 s 抬起、0.18 s 放上，縮短單輪支撐時間）；升級改成「爬上 2 階且速度追蹤 > 50%」。
- Stair 第 3 輪結果（14:36）：地形等級升到 2.71（之前天花板 ~1.8–2.35），速度追蹤 0.38。掃描：1 cm 32/32、**2 cm 18/32 不倒**（23 格至少上 1 階）、**3 cm 4/32 不倒**（16 格至少上 1 階）。影片：0.4 m/s 斜 10° 連上 4 階 3 cm 不倒。平地 618 種按法倒地 84%（只適合上台階用）。
- 使用者 14:53「不開下一輪」：整理 `releases/tancho_v3_stair_v1/`。

### 使用者試玩 stair v1（20:30–20:45）與 StairV2

- 使用者回饋：單腳上台階會觸發 15° 重置；小跳像「軟腳再接住」，沒有真正蹬地；平地會漂移。建議只在機身撞地時重置。
- `TanchoV3-StairV2-v0`（20:44，從 stair v1 接續 800 iter）：移除 tilt 終止（只剩機身/大腿/小腿碰地），傾斜只靠 reward；小跳 v4 = 0.12 s 蹲到收腳姿勢（髖–輪軸 189 mm）→ 0.20 s 伸到 248 mm → 0.30 s 收腳 → 0.40 s 回正；站立命令 5% → 20%。teleop 加 `--max-tilt-deg`。
- StairV2 結果（21:32，run `tancho_v3_stair/2026-10-09_20-44-43_18c3b9b`）：地形等級 2.49。台階掃描（Play 也沒有 tilt 終止，只算碰地）：1 cm 32/32、**2 cm 20/32 不倒**（28 至少上 1 階）、**3 cm 9/32 不倒**（20 至少上 1 階）；stair v1 是 18/32、4/32。影片：0.5 m/s 斜 −10° 連上 4 階 3 cm。平地 618 種按法倒地 93%（小跳 v4 在平地幾乎都倒；不按鍵、靜止時 58%）。
- 使用者 21:58–22:01：小腿碰地不算倒（深蹲小跳會讓小腿著地），重新訓練；跳的時候腿的關節速度不要扣太重。StairV2 第 2 次（22:05，從 StairV2 接續 800 iter）：body_contact 只看機身與大腿；按鍵期間（按住到放開後 0.6 s）腿加速度、腿扭力、action rate 懲罰只扣 10%。
- StairV2 第 2 次結果（22:50，run `tancho_v3_stair/2026-10-09_22-02-49_7faf7b4`）：地形等級 2.09、速度追蹤 0.60（第 1 次 0.47）。掃描：**2 cm 27/32 不倒**（第 1 次 20）、3 cm 4/32 不倒（第 1 次 9，但 22 格至少上 1 階）。平地 618 種按法倒地 91%，小跳在平地仍幾乎 100% 倒。

### 使用者試玩 StairV2 第 2 次的觀察（22:56，不再開訓練）

- **跳躍幾乎沒有用到大腿，這是主要原因**：蹬地主要靠小腿（膝）發力。
- **因為沒用到大腿，所以**只靠小腿發力的滯空時間不夠，來不及回到平衡點；單腳時尤其明顯，一抬就馬上傾斜（使用者：「因為一所以二」）。
- 下次方向：參考動作與 reward 要讓大腿（髖）參與蹬地與擺腿，拉長滯空、讓單腳時機身有時間回正。注意 URDF FK：只動大腿主要是把輪子前後擺（±0.3 rad ≈ ±70 mm），腿長由膝決定，所以「用大腿」要配合把機身重心帶到支撐輪上方，而不只是加大髖的角度。

## 2026-10-11 RoughSquat 第 900 輪試玩（只記錄，不訓練）

- 試玩模型：`tancho_v3_rough_squat/2026-10-10_23-53-28_efa4f8f/model_900.pt`，rough 地圖，reference guidance 1.0。
- 使用者實測：**單獨按住 Shift 會向後加速**，不是原地保持深蹲。
- 使用者實測：**Shift 與 W（前進）同時按時，仍會加速向後並翻倒**。
- 判定：目前合併模型的 Shift 深蹲與速度控制互相干擾，尚不可視為可玩的「WASD + 原地長按深蹲」版本。
- 本次依使用者要求僅記錄觀察；未修改參數、未啟動新訓練。

## 2026-10-17 無大腿 4 關節版（NoThigh）

- 資產：`Tancho_v3_no_thigh.urdf`（`Tancho_v3/make_no_thigh_urdf.py`）：拿掉大腿 link/joint，小腿裝在原髖電機位置（兩電機外寬約 154 mm），calf ±90°，全機 1.8625 kg。任務 `TanchoV3-NoThigh-Stand/Walk-v0`（commit f935e37、ea367ac；動作 4 維、actor 觀測 19 維，標稱 calf 0 rad）。因工作區有別人未 commit 的 env_cfg.py/commands.py，用 `--allow-dirty`。
- 站立（17:23，`tancho_v3_nothigh_stand/2026-10-17_17-23-18_ea367ac-dirty`）：3000 iter、44 分鐘，timeout 97%，reward 31.4。動作 std 收斂到 ~0。
- 行走第 1 次（18:08，從站立 `model_final.pt` 接續）：300 iter 檢查點沒過，episode 14 步、tilt 100%；站立 policy 的 std ~0，換任務後探索不動。停掉。
- 行走第 2 次（18:12，從頭，`tancho_v3_nothigh_walk/2026-10-17_18-12-42_ea367ac-dirty`）：300 iter timeout 99%；使用者 18:41「先停」，停在 iter 2250：timeout 99.3%、reward 71.2、vel 誤差 xy 0.24 / yaw 0.12。最新 `model_2250.pt`，未匯出、未驗收、未錄影。

## 2026-11-01 HopPush：平地主動往上跳且不倒（使用者 12:09 授權開訓）

- 基準（hop v2 + v1 參考，平地）：雙跳時兩輪同時離地 0.1 s，騰空時機身比按鍵前高 12.3 mm，向上速度峰值 0.53 m/s；靜止時也是 12.4 mm、不倒。v1 本來就有一點主動蹬地（0.02 s），只是不高。
- v7 參考（commit 3ff897c，`TanchoV3-HopPush-v0`）：輪軸沿著標稱位置的垂直線伸縮（FK：dthigh = −0.41 × dcalf），慢蹲到 −185 mm，然後給步階目標蹬到 −245 mm，空中收腳。線性 0.11 s 的伸腿太軟（Kp 20），機身向上 0.85–1.0 m/s 但輪子不離地；改成步階後，舊權重可以兩輪離地 40 mm，但會倒。新 reward `hop_rise`：只在兩輪都離地時，按機身高於站姿的高度給分。前 400 iter 從 v1 逐步換成 v7。
- 第 1 輪 run `tancho_v3_hop_push/2026-11-01_20-17-18_3ff897c-dirty`（從 hop v2 接續，keep_obs_norm，1000 iter）：timeout 90% →（換動作時）31% → 71%，仍在上升；iter 1000 的門檻是 80%，沒過。
- 驗收（`outputs/hop_push_20261101/`，guidance 1）：0.3 m/s 下 LT/RT/雙跳序列 ×2、連續雙跳 ×5、間隔 1 s 的雙跳 ×4 都沒倒。雙跳兩輪離地 30–36 mm、0.08 s，騰空時機身上升 8–10 mm（門檻 10 mm，沒過），向上速度峰值 0.95 m/s。**靜止時第一次雙跳就在 2.9 s 倒**。結論：沒有比 hop v2 跳得高，靜止反而更差。影片 `videos/video_seq-step-0.mp4`。
- 第 2 輪（commit 7fbd33e，`TanchoV3-HopPushV1-v0`，21:20 開始）：參考動作維持 v1，只加 `hop_rise` 與放寬腿部懲罰，從 hop v2 接續。
- 第 2 輪結果（run `tancho_v3_hop_push/2026-11-01_21-20-43_7fbd33e-dirty`）：訓練全程 timeout 89–92%。驗收全部不倒、pass_all=true：行進與靜止的雙跳、間隔 1 s 的快速雙跳都是兩輪離地 33–37 mm、0.1 s，機身上升 12.6–13.1 mm，向上速度 0.54 m/s。**跟 hop v2 幾乎一樣（12.4 mm）**：hop_rise 每個 episode 只有約 0.035，總 reward 約 82，太小，推不動 policy。1 cm 門檻定得太低，hop v2 本來就超過了；之後以 ≥ 2 cm 當作「明顯跳得更高」。
- 第 3 輪（commit c6ca4e7，`TanchoV3-HopPushV1Hi-v0`，22:16 開始）：hop_rise 權重 20 → 200、上限 4 → 6 cm，從第 2 輪接續。
- 第 3 輪結果（run `tancho_v3_hop_push/2026-11-01_22-16-40_c6ca4e7-dirty`）：timeout 91–94%；驗收不倒，雙跳機身上升 12.8 mm、向上速度 0.54 m/s，**跟第 2 輪一模一樣**。hop_rise 的數字只是因為權重和上限換算而變大，實際行為沒變。判斷：跳高由注入的參考動作加上 Kp 20 的 PD 決定，policy 的動作雜訊從 hop v2 起就幾乎是 0，加大 reward 也探索不到更用力的蹬地。v7（第 1 輪）蹬得更快（0.95 m/s），但機身反而只上升 8–10 mm，代表在 Kp 20、12.5 Nm 的條件下，伸腿的速度大多被腿和輪子的質量吃掉了。
- 3 / 4 輪已用完，暫停，等使用者決定第 4 輪的方向。
- 🚩 使用者 11-02 試玩第 3 輪：「動作蠻標準的」。標記為穩定的一般跳：git tag `hop-push-r3-stable`（c6ca4e7），模型 `logs/rsl_rl/tancho_v3_hop_push/2026-11-01_22-16-40_c6ca4e7-dirty/exported/policy.pt`，任務 `TanchoV3-HopPushV1Hi-Play-v0`。

## 2026-11-02 HopCharge：按住蹲下蓄力、放開起跳（使用者 01:02–01:06）

- 使用者更正：Shift 和 J 都是蓄力跳（每一次跳都要先蓄力），收腳時要用到大腿。
- 任務 `TanchoV3-HopCharge-v0`（commit 7f49159，teleop 0ff675f）：新增 `charge` 指令，按住時沿垂直線蹲到輪軸 −172 mm（0.35 s 蹲到最深），放開後給步階目標蹬到 −245 mm、維持 0.06 s，再收腳到 (−0.305, +0.75)，最後回到標稱。觀測 29 → 32 維。從第 3 輪接續（`pad_checkpoint_obs.py` 補零，腿的動作雜訊重設為 0.1）。
- run `tancho_v3_hop_push/2026-11-02_09-10-30_0ff675f-dirty`，1000 iter：timeout 1% → 59%（iter 600 檢查 54% 過），charge_rise 0.03 → 0.018。
- 驗收（`outputs/hop_charge_20261102/`）：每一項都沒倒。完整蓄力（0.6 s）兩輪離地 33–47 mm、0.1 s，大腿收 0.26–0.27 rad（有用到大腿），向上速度 1.0 m/s，**但騰空時機身只比站姿高 1–5 mm**；短按蓄力（0.15 s）反而高 9–10 mm。單腳抬腳正常。結論：蹲得越深，起跳時越低；蓄力沒有變成高度。這跟 v7 的結果一樣，瓶頸應該在 Kp 20 的軟 PD 與 12.5 rad/s 的速度上限。影片 `videos/video_chg-step-0.mp4`。
- 11-02 使用者試玩：下蹲 OK，提議下一步改 PD，12.5（扭力 / 速度上限）不能改。PD 掃描（不重訓，HopCharge 模型，原地，`outputs/hop_charge_20261102/pd_sweep/`）：Kp/Kd 20/0.2、40/0.4、60/0.6 都不倒，蓄力跳機身上升 1–7 mm、向上速度 1.02 m/s，數字幾乎一樣；100/1.0 在 2.96 s 倒。**Kp 20 時蹬地的膝關節就已經頂到 12.5 Nm 和 12.5 rad/s**，所以真正的瓶頸是 12.5 rad/s 的轉速上限，不是 PD（之前我判斷是軟 PD，那是錯的）。FK：膝每 rad 讓輪軸移動的距離，蹲姿時約 84 mm、接近伸直時只剩 21 mm；轉速上限下，起跳速度約 1 m/s，但機身得先從蹲姿回到站姿高度，所以蹲越深，超出站姿的高度越少。同一個模型的標準雙跳（LT+RT）反而上升 15 mm。

### TODO（使用者 2026-11-02 03:02）
1. 優化小跳（不蓄力的標準跳；HopCharge 模型裡的 LT+RT 目前機身上升 15 mm，r3 是 12.8 mm）。
2. 保留下蹲動作（Shift/J 按住蹲，使用者試玩說 OK）。
3. 抑制關節雜訊（腿的抖動）。

## 2026-11-02 穩定優先重訓（使用者授權）
- 保留 HopPush 29維觀測/6動作/腿±0.6rad及原馬達上限、增益、200/50Hz；新增 Stability 任務。flat/零命令/無按鍵/無reference；FullStand rewards、15°終止。從頭訓練3000iter，seed42。
- 新任務自有檔案commit 5dd6ff3；既有使用者dirty檔案保留，使用--allow-dirty且保存完整before_diff.patch。
- 註冊及8環境200步zero-agent通過；mean episode22.39steps，67次tilt，零動作診斷不代表policy驗收。
- 完成：run `/home/azul/Tancho/Tancho_v3_simready/logs/rsl_rl/tancho_v3_stability/2026-11-02_19-43-59_5dd6ff3-dirty`；訓練exit=0，early_stop=None；Isaac站立/推力驗收 pass_all=False。詳見 `/home/azul/Tancho/Tancho_v3_simready/logs/stability_retrain_20261102/acceptance.json`。

## 2026-11-03 RL v2 第 1 階段：穩定站立 + 原地保持（使用者 11-03 設為目標）

- 問題：Stability run（`tancho_v3_stability/2026-11-02_19-43-59_5dd6ff3-dirty`）60 s 不倒，但零命令往前漂 1.85 m；29 維觀測裡沒有任何位置資訊。
- 任務 `TanchoV3-Station-v0`（commit a22a61c，`tasks/staged/station.py`）：actor 觀測 29 → 31，後面加兩個輪里程 2·clip(r·q_wheel, ±0.5 m)（編碼器多圈位置，reset 時歸零；**部署介面改變，要告訴使用者**）；critic 再加真實位移。reward：pos_hold exp(−(d/0.1)²) ×1、pos_l1 ×−2、wheel_saturation ×−0.2、輪扭力 ×10、wheel_vel_l1 0.2 → 0.05。DR：腿 Kp/Kd 與輪 Kd ×U(0.8,1.2)、COM ±1 cm、0–20 ms 延遲、摩擦 0.4–1.2、彈性 0–0.1。從 Stability model_final 補零接續（std 重設 0.05，keep_obs_norm），1500 iter。
- 新 MuJoCo 檢查 `scripts/wheel_only/evaluate_stand_mujoco.py`（URDF → MjSpec，1 ms，自己算 PD）。接觸 solref 0.02 時舊 Stability 1.8 s 倒；0.005 時能站但 20 s 漂 2.1 m。預設 0.005。
- 第 1 輪結果（run `tancho_v3_station/2026-11-03_00-05-26_a22a61c-dirty`）：timeout 95–98%。Isaac 60 s：不倒，漂到 12–13 cm 後停住（門檻 5 cm，沒過），pitch 有 ±1.5° 的極限環（最大偏 5°）；推 0.5 m/s 不倒、恢復。MuJoCo 60 s：不倒但漂 2.6 m。
- 診斷：輪扭力 86% 時間頂在 0.45 N·m = 12.5 N 牽引力，但每輪正向力約 13 N × 摩擦 0.64 ≈ 8 N，輪胎打滑，輪里程不再代表位移（訓練中 odom 的 std ≈ 0.4 m，實際離原點平均 0.15 m）。MuJoCo 裡看到右輪在摩擦極限上空轉。
- 第 2 輪（commit b495a5b，`TanchoV3-StationV2-v0`，01:16 開始）：加 wheel_slip（接地點平面滑動速度）×−2，pos_hold σ 0.1 → 0.05、權重 1 → 2。從第 1 輪接續（std 0.03），1500 iter。
- 第 2 輪結果（run `tancho_v3_station/2026-11-03_01-16-47_b495a5b-dirty`）：timeout 95%。Isaac 60 s 零命令：**漂移 1.8 cm（門檻 5 cm，過）**、不倒；pitch 帶 4.5° 沒過，但最大值出現在 t=2.1 s 的起始暫態（重生 pitch 0，平衡點約 −1.2°），10–60 s 只在 −1.8 ~ −0.65° 之間。推 0.5 m/s：不倒、恢復。LQR 測試組：±10° 全救回；50 ms 推力不倒上限 **45 N**（Stability 是 40 N）；命令追蹤 3 項都倒（站立 policy 沒學過，屬於第 2 階段）。
- MuJoCo 腳本修正：顯式 1 ms 的輪 Kd 在很輕的輪子上會顫振（dt·Kd/I ≈ 200），改用 MuJoCo 伺服器（implicitfast 隱式積分）、加 noslip 10 次。修正後 r2 20 s 漂 0.59 m、Stability 1.44 m，都不倒，輪扭力飽和 81%（跟 Isaac 86% 接近）。量到輪胎接地點平均滑動 0.15–0.17 m/s（Isaac 訓練中約 0.03）：同樣 μ 下 MuJoCo 的摩擦撐不住打滿的扭力。dt 5 ms + solref 0.005 會不穩（1.3 s 倒），dt 2 ms 能站。
- 第 3 輪（commit fc5b7c1，`TanchoV3-StationV3-v0`，02:55 開始）：訓練摩擦 0.4–1.2 → 0.25–1.2（有效 0.2–0.96）、wheel_slip −2 → −10。從第 2 輪接續（std 0.03），1500 iter。
- 第 3 輪結果（run `tancho_v3_station/2026-11-03_02-55-29_fc5b7c1-dirty`）：Isaac 60 s 漂 2.2 cm、推 0.5 不倒；MuJoCo 60 s 漂 1.35 m（r2 1.67 m、Stability 2.43 m），但 MuJoCo 推 0.5 m/s 在 13.8 s 倒，不採用。
- 07:30 驗收整理：`logs/station_20261103/ACCEPTANCE.md`。交付候選 r2。Isaac 漂移/推擠過；MuJoCo sim2sim 沒過（站得住、推不倒，但漂 1–2 m，軟接觸會倒）。4 輪用了 3 輪。

### TODO（使用者 2026-11-03 07:58 試玩 Station r2）
- 機身俯仰會撞到 20°，所以目前不能 WASD（第 2 階段速度追蹤要處理）。
- 最後一輪（第 4 輪）用來修 MuJoCo 打滑。
- 第 4 輪（commit 0e2f3a6，`TanchoV3-StationV4-v0`，輪速度環 Kd 4 → 1，從 r2 接續）08:00 開始；使用者 08:01「筆記好就關掉」，停在約 iter 30，未驗收。開訓時輪扭力飽和從 ~0.35 降到 ~0.14（獎勵單位），timeout 暫時掉到 39%。
- TODO：修 MuJoCo 打滑（輪扭力飽和 → 輪胎滑）。可從第 4 輪的 Kd 1 方案接著做（部署時輪馬達 kd 也要改成 1.0）。

## 2026-11-19 StationSkills：站立不漂 + 收左腳 / 收右腳 / 蹲下（使用者 18:38–18:40）
- 使用者：時間不足，MuJoCo 打滑先放 TODO；下一步是同一個模型做到站立不漂、LT/RT 收腳、Shift/J 兩腳蹲下；跳躍可以訓練但先不部署；之後做 MuJoCo sim2sim。持續追進度，沒追到就改了重訓。
- 任務 `TanchoV3-StationSkills-v0`（commit 80c3b71，`tasks/staged/station_skills.py`）：Station r2 + hop r3 的按鍵（v1 參考，guidance 1）+ HopCharge 的 charge 指令；按鍵窗口內姿勢懲罰關閉、腿部懲罰 10%、tilt 上限 35°。actor 34 維（29 + 輪里程 2 + charge 3）。
- 從 r2 model_final 補零（`logs/station_20261103/skills_init.pt`，腿 std 0.1、輪 0.03，keep_obs_norm），1500 iter，18:41 開始，PID 17005，log `logs/rsl_rl/skills_train_20261119_184134.log`。
- 驗收計畫：Isaac `evaluate_stability.py --task TanchoV3-StationSkills-Play-v0`（60 s 漂移、推 0.5）+ `evaluate_hop.py --schedule LT,RT,CHG --vx 0 --guidance 1` + LQR 測試組；MuJoCo `evaluate_stand_mujoco.py`（目前只支援 31 維，34 維要補 charge 三維 = 0,0,1）。
- 結果：沒過 300 iter 檢查。iter 0 起 episode 就只有 12–50 步（0.2–1 s），timeout 0%，body_contact 50–75%、tilt 25–37%，到 iter 430 都沒改善。使用者 19:00 要切到 Windows，停在 iter 430（run `tancho_v3_station_skills/2026-11-19_18-41-40_80c3b71-dirty`）。
- 待查：iter 0 就倒，表示是設定問題，不是學不會（r2 本身能站 60 s）。懷疑：按鍵頻率（random 0.8 + burst 0.5）加上 charge 0.5 同時開、DR 延遲版的腿動作加參考、或 calf 接觸終止。下一步先用 Play 單獨測 r2 權重 + 不按鍵 / 只按 LT / 只按 charge，找出哪個造成倒地。
- 19:45 排查（`logs/station_20261103/skills_diag/`，用第 1 次的 model_0）：不按鍵站 20 s 漂 1.6 cm 不倒；LT 按下後左輪抬 30 mm、右輪也彈起 12 mm，2.14 s 倒。結論：r2 沒學過抬腳，而訓練中一開局就按鍵，每局都在 1 s 內倒，學不起來。
- 第 2 次（19:49，PID 7150，log `logs/rsl_rl/skills2_train_20261119_194905.log`）：改從 HopCharge model_final（會抬腳、蹲下、跳）開始，`pad_checkpoint_obs.py --actor-at 29 --critic-at 73` 在中間插入輪里程（actor +2）與 critic +4，腿 std 0.05、輪 0.03。iter 100：episode 379 步、timeout 18%、tilt 78%，持續上升。
- 第 2 次結果：iter 600 timeout 35%（門檻 50%），iter 400 起停滯，tilt 為主。20:15 停。
- MuJoCo 腳本（commit 7cff39d）：加 LT/RT 參考表與 charge 按鍵、支援 32/34 維；腿改成顯式 1 ms PD + 12.5 rad/s 轉速上限（沒有上限時 HopCharge 的 LT 抬到 84 mm 就倒）。HopCharge 在 MuJoCo：LT 58 mm / RT 50 mm 不倒，蓄力跳 8.6 s 倒。
- 第 3 次 `TanchoV3-StationSquat-v0`（commit 925697c，20:17，PID 9167，log `logs/rsl_rl/squat_train_20261119_201711.log`）：拿掉 LT+RT 小跳（只隨機按 L 或 R）、蹲下放開後 0.28 s 線性站回（不跳），hop_rise / charge_rise 權重 0。從第 2 次 model_600 接續。iter 60 timeout 78%。
- 第 3 次結果（run `tancho_v3_station_skills/2026-11-19_20-17-17_925697c-dirty`）：timeout 82%。Isaac：60 s 零命令不倒但漂 14.5 cm（✗）；推 0.5 不倒；LT 32 mm、RT 43 mm、蹲下 calf 0.73–0.75 rad 完整蹲到底再站回，兩輪重複都不倒。MuJoCo（`logs/station_20261103/squat/`）：站 60 s 不倒漂 1.0 m、推 0.5 不倒、LT/RT 53–56 mm、蹲下、全部不倒。
- 第 4 次 `TanchoV3-StationSquatHold-v0`（commit 68a310e，21:25，PID 12578，log `logs/rsl_rl/squathold_train_20261119_212452.log`）：pos_hold 2 → 4、pos_l1 −2 → −5，從第 3 次 model_final 接續 1000 iter。
- 第 4 次結果（run `tancho_v3_station_skills/2026-11-19_21-25-01_68a310e-dirty`）：**Isaac 全過**：60 s 漂 1.4 cm、pitch 0.5°、pass_all ✓；推 0.5 ✓；LT 32–34 mm、RT 40 mm、蹲下 calf 0.71–0.75 rad 站回，不倒。MuJoCo：全部不倒，但 60 s 漂 1.68 m。整理 `logs/station_20261103/SQUATHOLD_ACCEPTANCE.md`。
- 第 5 次 `TanchoV3-StationSquatKd1-v0`（commit cff4849，22:12，PID 15365，log `logs/rsl_rl/squatkd1_train_20261119_221207.log`）：同上 + 輪 Kd 4 → 1（部署變更），從第 4 次接續 1500 iter，目標修 MuJoCo 打滑。
- 第 5 次結果（run `tancho_v3_station_skills/2026-11-19_22-12-15_cff4849-dirty`，輪 Kd 1）：MuJoCo 輪扭力飽和 80% → 28%，60 s 漂移 1.68 → 0.64 m（等速爬約 1 cm/s），推 0.5 後只漂 4 cm、LT/RT/蹲下不倒；軟接觸 53 s 倒。但 Isaac 退步：60 s 漂 11.4 cm、pitch 最大偏 7.7°（pass_all ✗），LT 34–36 mm、RT 37–38 mm 不倒。判斷 Kd 換掉後還沒練夠。
- 第 6 次：23:19 第一次啟動立刻當掉（動作 std 有一維塌到 3e-4，PPO 更新後變負，`normal expects all elements of std >= 0`）。23:30 重開：std 全部重設 0.03（`logs/station_20261103/kd1_final_std.pt`），log `logs/rsl_rl/squatkd1b_train_20261119_232958.log`，PID 20619，1500 iter（seed 43）。交付版本仍是第 4 次 StationSquatHold（Kd 4）。
- 第 6 次結果（run `tancho_v3_station_skills/2026-11-19_23-30-07_cff4849-dirty`，Kd 1，timeout 88%）：**MuJoCo 大幅改善**：60 s 漂 11.8 cm（Kd 4 時 1.68 m）、輪扭力飽和 3%、推 0.5 不倒、LT 56–58 mm / RT 45 mm / 蹲下不倒。Isaac：60 s 漂 20 cm（每秒約 3 mm 的慢爬，伴隨 0.06 m/s 的前後晃）、pitch 偏 3.7°、推 0.5 不倒、LT 36–37 mm / RT 32–37 mm / 蹲下不倒。兩個引擎的行為已經接近，剩下位置精度。
- 第 7 次 `TanchoV3-StationSquatKd1Hold-v0`（9e27520，00:42，logs/rsl_rl/squatkd1hold_train_20261120_004221.log）：pos_l1 −5 → −10，加寬核 pos_hold_wide（std 0.2 m，權重 2），從第 6 次接續（std 0.03），1500 iter。
- 第 7 次結果（run `tancho_v3_station_skills/2026-11-20_00-42-30_9e27520-dirty`）：Isaac 60 s 漂 8.2 cm、pitch 1.4°、推 0.5 pass_all ✓、LT 35 / RT 34 mm、蹲下不倒；MuJoCo 不倒但來回晃，最大 42 cm（結束 11.5 cm），推 / 動作不倒。比較表 `logs/station_20261103/SKILLS_SUMMARY_20261120.md`。7 次嘗試，暫停等使用者選版本。
- 使用者 02:13：目標 = MuJoCo 60 s 零命令漂移 < 5 cm 就完全 OK，繼續練。第 8 次 `TanchoV3-StationSquatKd1Com-v0`（commit 5a1086b，02:14，log `logs/rsl_rl/squatkd1com_train_20261120_021421.log`）：Kd1Hold 的獎勵 + 機身 COM x 隨機化 ±1 → ±2 cm（MuJoCo 的 URDF 重心比 Isaac 零姿勢檢查偏後約 8 mm），從第 6 次 model_final 接續（std 0.03）。同時用 `logs/station_20261103/ckpt_scan/scan.sh` 把第 6、7 次的中途 checkpoint（300/600/900/1200/1450）逐一匯出跑 MuJoCo 60 s，挑漂移最小的。
- 02:15 使用者要切 Windows：第 8 次與 checkpoint 掃描都停掉（第 8 次只跑到開頭）。使用者睡覺時再跑：重開 `TanchoV3-StationSquatKd1Com-v0`（init `logs/station_20261103/kd1b_final_std.pt`）+ `logs/station_20261103/ckpt_scan/scan.sh`（要先刪掉 ckpt_scan 裡未完成的子資料夾）。

### 2026-11-20 夜間計畫（使用者 04:00 授權，10:30 交報告，中途不用回報）
目標：部署站立 / LT / RT / 蹲下；MuJoCo 60 s 零命令漂移 < 5 cm 且動作不倒。達標後再考慮把跳加回來。
1. [進行中] 第 8 次 `TanchoV3-StationSquatKd1Com-v0`（04:01，log `logs/rsl_rl/squatkd1com_train_20261120_040126.log`）。
2. [進行中] `logs/station_20261103/ckpt_scan/scan.sh`：第 6、7 次中途 checkpoint 的 MuJoCo 60 s 漂移。
3. 第 8 次完成後：同樣掃它的 checkpoint（MuJoCo），挑最好的跑完整驗收（Isaac：evaluate_stability 60 s / push 0.5、evaluate_hop LT,RT,CHG；MuJoCo：stand60 / push0.5 / presses，`--wheel-kd 1.0 --charge-jump 0`）。
4. 若都 > 5 cm：下一步想法 (a) 用「checkpoint 以 MuJoCo 漂移挑選」作主要選版方法；(b) pos 獎勵加速度項（離原點時要求往回的速度）；(c) 訓練時加 MuJoCo 式的接觸軟度 / 腿轉速上限差異。
5. 10:30 前寫報告 `logs/station_20261103/REPORT_20261120.md` 並貼到 thread。
- 04:05 checkpoint 掃描（MuJoCo 60 s 硬接觸）：r6 300/600/900/1200/1450 = 28 / 18 / 23 / **5.6** / 21 cm；r7 300/600/900/1200/1450 = **4.1** / 20 / 57 / 61 / 69 cm。同一次訓練裡不同 checkpoint 差很多。
- r7_300 細查：MuJoCo 摩擦 0.5/0.8、輪 Kd 0.8/1.2、初始 pitch ±3° 都在 3.3–5.2 cm；推 0.5 不倒；LT/RT/蹲下 ×2 不倒（LT 53–54、RT 42 mm）。但 MuJoCo 軟接觸（solref 0.02）36 s 倒、dt 2 ms 漂 4.5 m（dt 2 ms 時輪扭力飽和 77–81%，疑似顯式腿 PD 在 2 ms 不穩，屬腳本限制）。**Isaac：60 s 漂 11 cm，推 0.5 在 10.26 s 倒 ✗** → r7_300 不合格。
- 準備第 9 次 `TanchoV3-StationSquatKd1Soft-v0`（commit c395346）：地面 compliant contact 5000 N/m、120 N·s/m。checkpoint 篩選腳本 `logs/station_20261103/ckpt_scan/candidates.sh`（MuJoCo 硬 / 軟 / 推 + Isaac 60 s / 推），結果累積在 `candidates_table.txt`。
- 第 8 次結果（run `tancho_v3_station_skills/2026-11-20_04-01-37_5a1086b-dirty`，timeout 76%）checkpoint 篩選（`ckpt_scan/candidates_table.txt`）：MuJoCo 60 s 硬接觸 300/600/900/1200/1499 = **2.4** / 21 / 13 / 29 / 8 cm；軟接觸全部漂 2.5–5.9 m（3 個倒）；Isaac 60 s = 5.5 / 23 / 23 / 4.8 / 10 cm；Isaac / MuJoCo 推 0.5 全部不倒。
- **候選 r8_300**（`logs/station_20261103/ckpt_scan/r8_300/exported/policy.pt`）：MuJoCo 60 s 2.4 cm ✓，摩擦 0.5/0.8、Kd ±20%、初始 ±3° 都在 1.5–4.8 cm ✓；MuJoCo 推 0.5 不倒、停在 6.7 cm；MuJoCo LT/RT/蹲下 ×2 不倒（LT 53–56、RT 46 mm）。Isaac：60 s 5.5 cm、pitch 2.4°；推 0.5 不倒（兩輪短暫離地 0.04 s，20 s 結束時還在 0.06 m/s 慢動）；LT 35、RT 37–38 mm、蹲下 ×2 不倒。弱點：MuJoCo 軟接觸會漂走並倒。
- 第 9 次結果（Kd1Soft，run `2026-11-20_05-10-02_c395346-dirty`）：MuJoCo 硬接觸 60 s = 16 / 10 / 16 / 15 / 16 / 18 cm（300/500/800/1000/1200/1450），軟接觸仍漂 0.6–7.7 m，多數倒；Isaac pitch 偏 4.8–8.7°。**不採用**。
- r8 其他 checkpoint 的 MuJoCo 60 s：150/200/250/350/400/450/500/650/700/750 = 25 / 11 / 41 / 8.6 / 20 / 14 / 36 / 20 / **4.4** / 28 cm。分散很大，位置迴路在兩個引擎都偏軟。
- 第 10 次 `TanchoV3-StationSquatKd1Quiet-v0`（commit 1d40c55，06:34，log `logs/rsl_rl/squatkd1quiet_train_20261120_063418.log`）：推的間隔 3–6 s → 8–14 s（讓靜止站立佔大部分時間，慢爬才會被扣分），從 r8_300 接續（std 0.02），1000 iter。
- 第 10 次結果（Kd1Quiet，run `2026-11-20_06-34-27_1d40c55-dirty`）：MuJoCo 60 s 隨訓練變差（300 iter 7 cm → 900 iter 136 cm），Isaac 950 iter 漂 14 cm、pitch 11.6°。**不採用**。
- r8_700 完整篩選：MuJoCo 4.4 cm，但 Isaac 14.5 cm；軟接觸 5.8 m。
- **交付 release `releases/tancho_v3_squat_kd1_r8_300/`**（commit b3d41a6）：policy.onnx/pt、config.json（34 維、輪 kd 1.0）、reference_io、抬腳表、MuJoCo 腳本、eval、Isaac 與 MuJoCo 的 LT/RT/蹲下影片。MuJoCo 腳本加 `--video`（MUJOCO_GL=egl）。
- 第 11 次（試驗）`TanchoV3-StationHopKd1-v0`（commit f04bf07，07:27，log `logs/rsl_rl/stationhop_train_20261120_072718.log`）：把 LT+RT 小跳加回來（v1 參考 + hop_rise 200），蹲下仍不跳，從 r8_300 接續 1000 iter。
- 第 11 次結果（StationHopKd1，run `2026-11-20_07-27-26_f04bf07-dirty`，timeout 61%）：小跳在 MuJoCo / Isaac 都不倒（兩輪 24–42 mm、機身上升 12.5 mm），但 60 s 站立漂移 MuJoCo 19 cm–1.5 m、Isaac 23 cm。
- 報告 `logs/station_20261103/REPORT_20261120.md`。
- 第 12 次（08:17，logs/rsl_rl/stationhop2_train_20261120_081542.log）：第 11 次 model_final 再接續 1000 iter（std 0.02），看加跳版的站立漂移能否壓下來。
- 第 12 次結果（run `2026-11-20_08-15-50_f04bf07-dirty`，timeout 65%）：加跳版 r11+r12 共 38 個 checkpoint 的 MuJoCo 60 s 漂移，最好 r12_750 8.6 cm，其餘 10–50 cm；跳都不倒。未達 5 cm，停。

## 2026-11-21 穩定度 + 地形（使用者 15:32：晚上 10–10:30 交報告，中途不用回報；穩定度和地形都 OK 再試跳）
- 新驗收：`scripts/wheel_only/screening/terrain.sh`（Isaac 0–2 cm 凹凸 難度 0.5 / 1、重力傾斜 ±5° / ±8° 模擬斜坡、凹凸上推 0.5；MuJoCo 重力傾斜 ±5° / ±8°）。
- r8_300 基準（`logs/station_20261103/terrain/r8_300`）：凹凸 3.4 / 8.3 cm、凹凸上推 0.5 不倒；Isaac 斜坡 +5/+8/−5/−8° = 8.3 / 6.9 / 53 / 17 cm；MuJoCo 斜坡 +5/+8/−5/−8° = 117 / 179 / 38 / 151 cm；全部不倒。斜坡是弱點。
- 第 13 次 `TanchoV3-StationSquatKd1Terrain-v0`（6f7c101，15:42，logs/rsl_rl/terrain_train_20261121_154231.log）：HOP 地形（一半平地、一半 0–2 cm 凹凸）+ 每個 episode 機身 x 方向固定推力 U(−4, 4) N（30% 沒有），從 r8_300 接續（std 0.02），1500 iter。
- 16:05：±4 N 版到 iter 308 timeout 從 49% 掉到 20%（tilt 78%），停掉。改 ±2 N、一半 episode 沒有推力，重開（logs/rsl_rl/terrain2_train_20261121_155914.log）。
- 新增 `scripts/wheel_only/export_policy_cpu.py`（不開 Isaac 的匯出，跟 export_policy.py 輸出誤差 0；訓練佔 GPU 時用，地形訓練時 Isaac 匯出會 CUDA OOM）。
- MuJoCo 斜坡漂移查證（r8_300，重力傾斜 5°，30 s）：機身沿坡下滑 0.58 m 但輪里程只有 −3 / +17 cm → 輪胎在 MuJoCo 打滑。改參數：noslip 100 → 61 cm、無 noslip 64 cm、球形輪 47 cm、摩擦 1.2 → 27 cm、solref 0.002 → 19 cm。MuJoCo 斜坡結果對接觸參數很敏感，主要是 MuJoCo 圓柱輪接觸的滑移，實機橡膠輪在 2 N 切向力下不該滑。Isaac 斜坡數字較可信。
- r13（±2 N）早期 MuJoCo：200 / 300 iter 平地 44 / 48 cm、斜坡 0.3–1.8 m，比 r8_300 差。
- 第 13 次結果（Kd1Terrain ±2 N，run `2026-11-21_15-59-19_2fac3c9`，timeout 63%）：MuJoCo 平地 20–64 cm（比 r8_300 差）、斜坡 0.15–2.4 m。Isaac（1450 iter）：凹凸 7–13 cm、**斜坡 ±5/±8° 7–11 cm**（r8_300 是 7–53 cm）、全部不倒。斜坡在 Isaac 變好，平地精度變差。
- 第 14 次（17:28，logs/rsl_rl/odom01_train_20261121_172821.log）：Kd1Com（平地，r8 的設定），從 r8_300 開始，但輪里程輸入的正規化 std 0.78 → 0.1（`rescale_obs_norm.py`，第一層權重同比例縮小，起始行為不變），讓幾公分的位置誤差對網路的訊號放大約 8 倍。1000 iter。
- 第 14 次結果（odom 正規化 std 0.1，run `2026-11-21_17-28-27_e3a846b`，timeout 75%）：MuJoCo 平地 60–153 cm、斜坡 0.7–7 m，**明顯變差**，不採用（輪里程在 MuJoCo 被打滑汙染，放大後 policy 更依賴它）。
- 加跳版 r12_750 地形：凹凸地上 LT/RT/LT+RT/蹲下 ×2 都不倒（跳在凹凸地兩輪離地 13–27 mm），凹凸地站 5.5–17 cm；**Isaac 斜坡 −5° / −8° 滑走 2.2 / 8.4 m**（不倒但等於溜走），不適合部署。r8_300 在凹凸地（難度 0.5 / 1）做 LT/RT/蹲下 ×2 也都不倒。
- r13 其餘 checkpoint（350–1400）MuJoCo 平地最好 26 cm，全部 > 20 cm。
- 第 15 次（18:30，logs/rsl_rl/r13flat_train_20261121_183033.log）：r13_1450（斜坡好）回平地 Kd1Com 接續 500 iter，看能否把平地精度練回來、保留斜坡能力。
- 第 15 次結果（r13 回平地 500 iter）：MuJoCo 平地 19–140 cm、斜坡更差，不採用。結論：r8_300 維持交付；之後改做部署條件壓力測試（延遲、感測雜訊）。

### 部署壓力測試（19:00 起）
- MuJoCo 加訓練用的感測雜訊 + 20 ms 馬達延遲（`screening/stress.sh`：站 60 s ×3 seed、推 0.5、LT/RT/蹲下 ×2 輪 ×3 seed）。**r8_300：3 輪按鍵倒 2 次**，站立漂 35–43 cm（理想條件 2.4 cm）；只加 10 ms 延遲也漂 56 cm。
- 其他候選：**r8_700 全部不倒、站立 11–20 cm**；r8_150 不倒 15 cm、r8_650 不倒 25 cm、r13_1350 不倒 29 cm；r13_1450、r6_1200、r13_450 都有倒。
- Isaac 雜訊 + 20 ms 延遲：r8_700 站 60 s 漂 30 cm、推不倒、LT/RT/蹲下 ×3 不倒；r8_300 按鍵測試 0.18 s 就倒。
- 第 16 次 `TanchoV3-StationSquatKd1Lat-v0`（commit 9202dff，19:13，log `logs/rsl_rl/lat_train_20261121_191312.log`）：延遲 0–20 → 10–20 ms，從 r8_700 接續 1000 iter。
- 更多 seed（MuJoCo 雜訊 + 延遲 10/20 ms，按鍵 ×5 seed）：r8_150 倒 3/10、r8_700 倒 4/10、r8_650 倒 2/10，多半倒在 RT（7.1 s、19.1 s）。沒有雜訊時從沒倒過 → 感測雜訊是 MuJoCo 按鍵倒地的主因。
- Isaac 雜訊 + 20 ms：r8_700 按鍵 2×12 次都不倒、站立漂 30 cm；r8_150 按鍵 2×12 次不倒、但站立漂 53 cm。
- 第 16 次結果：r16_400（`2026-11-21_19-13-17_9202dff` iter 400）在 MuJoCo 雜訊 + 延遲下按鍵 30 輪都不倒、站立 7–20 cm；Isaac 雜訊 + 20 ms 站 18.5 cm、推、按鍵 12 次都不倒；凹凸地 6–11 cm；斜坡 Isaac +5° 溜 3.2 m。MuJoCo 0 延遲 72 cm（用 10–20 ms 延遲訓練的結果）。
- 修正：延遲動作重置後第一步送出 0 rad 腿目標（climb.py `_fresh_prev`，commit 973a1bc）；MuJoCo 倒地判定按鍵期間改用 35°。
- **新交付 `releases/tancho_v3_squat_kd1_r16_400/`**＋報告 REPORT_20261121.md。
- 第 17 次（20:26，logs/rsl_rl/latterrain_train_20261121_202631.log）：Kd1LatTerrain（延遲 10–20 ms + 凹凸地 + ±2 N 負載），從 r16_400 接續 700 iter，看能否補斜坡。
- 第 17 次結果（run `2026-11-21_20-26-36_cfa4d06`）：MuJoCo 雜訊 + 延遲站立有 1/3 倒（200/500/600 iter）、按鍵傾角 22–26°、斜坡 5° 仍 1.8–2.1 m。不採用，交付維持 r16_400。

## 2026-11-22 夜間（使用者 00:43：繼續到 10:00）
- 第 18 次 `TanchoV3-StationInt-v0`（652d4e6，00:44，logs/rsl_rl/int_train_20261122_004436.log）：actor 加輪里程的洩漏積分（每輪，τ 5 s，clip ±0.25 m·s，×4），34 → 36 維；Kd1Lat（10–20 ms 延遲）+ 平地 ±2 N 固定負載。從 r16_400 補兩維接續，1500 iter。目的：給位置迴路積分作用，消除穩態偏移與斜坡下滑。
- 第 18 次結果：iter 600 timeout 41%（tilt 59%），未過 50% 門檻，停。MuJoCo 雜訊+延遲站立 44–65 cm。第 19 次（01:11，logs/rsl_rl/intnl_train_20261122_011119.log）：只加積分、不加固定負載（StationIntNoLoad），從 r16_400 補維接續 1200 iter。
- 第 19 次結果（StationIntNoLoad）：timeout 67%（同 r16）；Isaac 雜訊+延遲站 20.6 cm、斜坡 +5° 14.9 cm（積分在 Isaac 對斜坡有幫助）；但 MuJoCo 雜訊+延遲站 67–108 cm（r16_400 7–20 cm），摩擦 1.0 時也 40–42 cm。停在 ~500 iter，不採用。
- 查證：MuJoCo 加雜訊後 r16_400 輪扭力飽和 41%（無雜訊 0.3%），輪里程讀 −0.57 m 但機身只動 5–12 cm → 雜訊經 policy 變成輪扭力尖峰、輪胎打滑、里程失真；積分把失真累積起來，所以更糟。
- 第 20 次 `TanchoV3-StationSquatKd1QuietTorque-v0`（ad0d050，01:39，logs/rsl_rl/qtorque_train_20261122_013904.log）：wheel_saturation ×10、wheel_torque ×5，從 r16_400 接續 1000 iter。
- 第 20 次結果（扭力懲罰 ×10）：飽和率沒降（0.38），timeout 掉到 44%，停。另：r16_400 在 MuJoCo 0 雜訊、20 ms 延遲時飽和 11%、漂 30 cm → 漂移不是雜訊造成。
- **MuJoCo 按鍵組合壓力測試** `screening/press_sweep.py`（990 種：11 種按法 × 間隔 0.15–1.5 s × 按住 0.1–1 s × 延遲 10/20 ms × 3 seed，加雜訊）：r16_400 倒 6/990，全是「蹲下放開 0.15 s 內又按蹲下」；加 Pi 端 0.3 s 蹲下冷卻規則後 **0/990**、最大傾角 28°。r16 其他 checkpoint 0–3 次/240。
- 同測試 + 序列中隨機 ±0.5 m/s 推一下：r16_400 48/990、r16_600 75、r16_450 61、r8_700 42。抬腳中被推是主要倒地原因。
- 寫好 `releases/tancho_v3_squat_kd1_r16_400/DEPLOY.md`（34 維輸入、蹲下冷卻規則、驗證表）。
- 第 21 次 `TanchoV3-StationSquatKd1Push-v0`（55ba838，03:19，logs/rsl_rl/push_train_20261122_031929.log）：推的間隔 3–6 s → 1.5–3 s，從 r16_400 接續 1000 iter。
- 05:55 發現第 20 次（QuietTorque）訓練當時沒被真正停掉（只殺到 conda 外層），一直跑到現在，已停。
- 第 21 次結果（StationSquatKd1Push，run `2026-11-22_03-19-39_55ba838`，timeout 53%）：推入按鍵的 MuJoCo sweep 26–39/990（r16_400 48）；r21_400 MuJoCo 站 6–26 cm、不推 sweep 0/990。但 **Isaac 雜訊+20 ms 站 60 s ×4：倒 2 次（42.4 s、23.6 s）** → 不採用。
- r16_400 加驗：Isaac 雜訊+20 ms 站 60 s ×10 全不倒（漂 12–22 cm）；按鍵 ×48 全不倒。交付維持 r16_400。
- r16_400 長時間：雜訊+20 ms 站 300 s，MuJoCo ×3 不倒（最遠 30–47 cm、結束 9–40 cm），Isaac 不倒（24 cm）。不會發散。
- 第 22 次 `TanchoV3-StationSquatKd1Push2-v0`（06:50，logs/rsl_rl/push2_train_20261122_065022.log）：推的間隔 2–4 s，從 r16_400 接續 500 iter，每 50 iter 篩（MuJoCo 推入按鍵 sweep + Isaac 雜訊延遲站立 ×4）。
- 第 22 次（Push2，推 2–4 s，run `2026-11-22_06-50-27_f372598`）gate（`screening/gate.sh`，`logs/station_20261103/gate/gate_table.txt`）：100 = 按鍵 5/990、Isaac 站立漂 1.2–1.8 m；200 = 0/990、Isaac 7–17 cm、MuJoCo 35–47 cm、推入按鍵 51/990；300 = 2/990。沒有比 r16_400 好。之後程序中斷，11:30 報告延誤到 14:55。
- **最終交付維持 r16_400**。

## 2026-11-22 第 2 階段：WASD（使用者 14:58 試玩 r16_400：一按 WASD / 三檔就超過 15° 倒，要 WASD 也要小跳）
- 任務 `TanchoV3-StationWalk-v0`（`tasks/staged/station_walk.py`，8357daf）：vx ±0.6、yaw ±1（400 iter 內從 30% 放大到 100%），25% 站立；保持點在有命令時跟著機器人走（輪角、位置），命令歸零時固定 → 輪里程輸入與位置獎勵只在停下時有作用（Pi 端：搖桿離開中心時把輪里程歸零）；追蹤獎勵 std 0.5、權重 2/1；capture point 關掉。其他同 Kd1Lat。從 r16_400（std 0.05）接續 1500 iter，15:00，logs/rsl_rl/walk_train_20261122_150012.log。
- 第 23 次結果（StationWalk，run `2026-11-22_15-00-17_8357daf`，timeout 85%）：Isaac 命令序列（站→0.3→0.6→倒退→轉→弧→停）有無雜訊延遲都不倒，誤差 vx ≤ 0.12、yaw ≤ 0.13；MuJoCo 不倒、速度約命令 85%、最大傾角 6.8°；走路中 LT/RT/蹲下不倒；原地按鍵 990 種 0 倒；原地站 Isaac 7–9 cm、MuJoCo 95 cm。→ 交付 `releases/tancho_v3_walk_squat_kd1/`（10533c5）。
- 第 24 次（StationWalkHop，從 walk 接續 500 iter，run `2026-11-22_16-02-36_34bcc45`，timeout 60%）：MuJoCo / Isaac 原地與 0.3 m/s 走路中小跳都不倒，兩輪 33–47 mm、機身上升 13 mm；原地站 MuJoCo 80 cm。→ 實驗版 `releases/tancho_v3_walk_hop_kd1_experimental/`（29175a6）。
- 第 25 次（17:36，logs/rsl_rl/walkhop2_train_20261122_173650.log）：StationWalkHop 從第 24 次 final 再接續 1200 iter（std 全部 0.02）。
- 邊走邊按的 sweep（`press_sweep.py --walk [--hop]`）：walk_squat 0/990；walk_hop 實驗版 3/990（HOP_CHG、HOPHOP、CHG_HOP 各 1）。
- MuJoCo 停下後漂 ~1 m 的原因：雜訊下輪胎打滑（機身前移 58 cm、輪里程 −15 cm），同 11-22 夜間的發現；Isaac 7–9 cm。
- 第 25 次結果（WalkHop 接續，run `2026-11-22_17-36-55_29175a6`）：邊走邊按＋小跳 sweep 200/400/600/800/1000 iter = 1/0/0/0/0 倒（990 次）。Isaac gate（`screening/isaac_walk_gate.sh`）：400 站 30 cm、600 站 14–21 cm、1000 站 12–33 cm，走路與小跳都不倒。MuJoCo 站：400 55–80、600 **35–40**、1000 53–61 cm。選 **r25_600**：原地按鍵含小跳 0/990、推入按鍵 20/990（2%）。→ `releases/tancho_v3_walk_hop_kd1/`。
- 第 26 次（19:27，logs/rsl_rl/walkhoppush_train_20261122_192732.log）：WalkHopPush（推 2–4 s），從 r25_600 接續 500 iter。
- 第 26 次結果（WalkHopPush，推 2–4 s，run `2026-11-22_19-27-*`）：邊走邊按＋小跳 200 / 350 iter = 0 / 1 倒（990）；推入按鍵 20 / 17（r25_600：20）。沒有明顯改善，程序在 450 iter 篩選時中斷。不採用。
- **11-22 最終交付**：WASD + 收腳 + 小跳 + 蹲下 = `releases/tancho_v3_walk_hop_kd1/`（r25_600）；只要原地站 = `releases/tancho_v3_squat_kd1_r16_400/`；WASD 不含跳 = `releases/tancho_v3_walk_squat_kd1/`。

### TODO（使用者 2026-11-22 21:54 試玩 walk_hop_kd1）
- 蹲下要更低，最好 15 cm 以下（目前蹲下參考：輪軸 −172 mm 等級的 CROUCH (−0.305, +0.75)；要確認 15 cm 指的是機身離地高度，並檢查腿 ±0.6 rad 範圍是否夠）。
- 其他地形（粗糙地、斜坡、台階）的表現要測、要能玩；粗糙地先開給使用者玩。
- 使用者 21:57：粗糙地不會倒，但太慢、三檔都像一檔 → 第 27 次 `TanchoV3-StationWalkHopFast-v0`（919a5b3，21:57，logs/rsl_rl/walkhopfast_train_20261122_215757.log）：命令範圍 0.6/0.75 → 1.2 m/s / 1.5 rad/s（400 iter 放大），從 r25_600 接續 1200 iter。teleop 加 `--fast`（三檔 0.4/0.8/1.2 m/s）。
- 21:58 第一次啟動 GPU 記憶體不足（使用者的 teleop 還開著），改成等 teleop 關掉後自動開始。
- 使用者 21:58「都先列 TODO，先不要做」：第 27 次（加速）取消，沒有開始訓練。TODO：(1) 蹲更低，目標 15 cm 以下；(2) 斜坡、台階等其他地形；(3) 速度加快（三檔都像一檔；任務 `TanchoV3-StationWalkHopFast-v0` 與 teleop `--fast` 已寫好、commit 919a5b3，未訓練）。

## 2026-11-23 最終版（使用者 03:24：做到 10:30；TODO 做進最終版 → 用專題論文的測試複測 → 交付可部署）
- 蹲下深度 FK（MuJoCo、機身水平、輪接地）：底盤最低點離地 (0,0) 21.5 cm；HopCharge 蹲 (−0.305, +0.75) 16.2 cm；(−0.36, +0.89) 15.0；(−0.40, +1.0) 14.0；**(−0.45, +1.10) 12.9 cm**；(−0.5, +1.2) 11.9。重心都在輪軸上方 ±1.2 cm。
- 第 28 次 `TanchoV3-StationFinal-v0`（85dfd84，03:25，logs/rsl_rl/final_train_20261123_032543.log）：WalkHop + 速度 1.2 m/s / 1.5 rad/s（400 iter 放大）+ 蹲下 (−0.45, +1.10)、0.45 s + HOP 地形（一半 0–2 cm 凹凸）。從 r25_600 接續 1500 iter。
- 論文複測工具 `scripts/wheel_only/thesis/summarize_suite.py`：用舊 Stability run1 驗證，+10° 恢復 1.38 s、−10° 4.10 s、推力 40 / 42 N，與論文 4.6 一致。
- 第 28 次中途 iter 300：MuJoCo 邊走邊按＋小跳 1/990 倒；蹲下底盤最低 11.2 cm（目標 < 15 cm ✓）。
- 第 28 次 iter 600（timeout 56%）：MuJoCo 邊走邊按＋小跳 0/990；快速命令 0.6/1.0/1.2 m/s 實際 0.44/0.73/0.89、yaw 1.5 → 1.68，不倒、最大傾角 10°。
- 第 28 次 iter 900：MuJoCo 邊走邊按＋小跳 0/990；快速命令 0.6/1.0/1.2 → 0.42/0.71/0.84、yaw 1.5 → 1.56。
- 第 28 次結束（timeout 60%）。篩選：1200 / 1450 MuJoCo 邊走邊按＋小跳 0 / ? 倒；Isaac gate（雜訊+20 ms）：900 ×2 全不倒（站 7–23 cm）、1200 ×2 全不倒（站 6–14 cm）、1450 原地按鍵 11.46 s 倒。MuJoCo 站：900 67–75、1200 49–56、1450 21–22 cm。1200 按鍵中被推 11/990。**選 r28_1200 為最終版。**
- 論文 4.6 測試（Isaac 標稱，r28_1200，`logs/station_20261103/thesis_final/suite`）：±1–10° 全不倒，±10° 最大位移 7.9 / 7.5 cm（LQR 16.9 cm），恢復 4.6 / 4.1 s（以 ±1° 定義，50 Hz 有 1.4–3.6° 的小幅持續振盪拉長了恢復時間；LQR 0.176 s）；速度命令三項都不倒：0.3 m/s → 0.24、1.0 rad/s → 1.05、0.2 + 0.5 → 0.18 / 0.31；推力 38 N 通過 / 39 N 失敗（同 LQR）；降頻：16.7 Hz 不倒、12.5 Hz 倒。
- 06:02 交付 `releases/tancho_v3_final/`（r28_1200，commit 6d117cd）＋ THESIS_TABLE.md。下一步：壓站立晃動 / 恢復時間。
- 第 29 次 `TanchoV3-StationFinalCalm-v0`（06:01，logs/rsl_rl/calm_train_20261123_060131.log）：ang_vel_xy ×5、upright ×2，從最終版接續 600 iter。
- 第 29 次結果（Calm）：timeout 44–54%（原 60%）、MuJoCo 邊走邊按 1–2/990、+10° 後晃動 2.0–2.5°（最終版 1.16°）→ 更差，不採用。
- 第 30 次 `TanchoV3-StationFinalTrack-v0`（06:38，logs/rsl_rl/track_train_20261123_063822.log）：追蹤 std 0.5→0.3、權重 2/1→3/1.5，從最終版接續 600 iter。
- 第 30 次結果（Track）：MuJoCo 速度 1.2 → 1.20–1.22 m/s（原 0.84），邊走邊按 1/0 倒；但**定點保持壞掉**：MuJoCo 站 90–105 cm、Isaac（r30_450，雜訊延遲）站 94–102 cm（最終版 6–14 cm）。不採用。
- 第 31 次 `TanchoV3-StationFinalTrackHold-v0`（07:51，logs/rsl_rl/trackhold_train_20261123_075108.log）：追蹤 std 0.4 權重 2.5/1.25、站立保持項 ×2，從最終版接續 500 iter。
- 第 31 次 iter 200：MuJoCo 1.2 → 1.14 m/s、站立 15–27 cm（最終版 49–56）、邊走邊按 1/990。
- 第 31 次 iter 200（r31_200）：Isaac gate ×2 全不倒，站 12–18 cm；MuJoCo 站 15–27 cm、1.2 → 1.14 m/s、邊走邊按 1/990、按鍵中被推 7/990。iter 400/450 MuJoCo 站 59–81 cm。**r31_200 取代 r28_1200 成為最終版。**
- 08:53 最終版換成 r31_200（commit e768acf），論文表 / 圖 / 4.6 稿 / 影片已重做。舊 r28_1200 → releases/tancho_v3_station_candidates/final_r28_1200。
- 最終版補測：凹凸地 0.3–6.4 cm、凹凸地被推不倒；斜坡不倒但溜 0.2–2.2 m；5 分鐘站 MuJoCo 18–32 cm、Isaac 26 cm。
- 最終版全部 1530 種邊走邊按（含小跳）：1 倒（0.07%）。
- MuJoCo（LQR 同模擬器）論文測試：±1–10° 全不倒、位移 3.4–9.0 cm；等效衝量推力 60 N 通過 / 65 N 失敗（雜訊延遲 60 N 仍過）。
- MuJoCo 速度命令：0.3 → 0.26 m/s（傾角 4.6°）、1.0 → 1.19 rad/s、0.2+0.5 → 0.18 / 0.58，皆不倒。
- MuJoCo 降頻：50 Hz 晃 1.5° pp、25 Hz 13°、16.7 Hz 22–23°（未倒）、12.5 Hz 倒。論文 MuJoCo 各項都已加入 THESIS_TABLE。
