---
name: tancho-training-monitor
description: 啟動、監看、診斷與驗收 Tancho V3 分階段（wheel-only → 6-DOF 站立 → 6-DOF 行走）Isaac Lab / RSL-RL 訓練。使用者說「訓練」「train」「跑一下」「看訓練」「驗收」「play」，或要求沒過就自己改參數重訓時使用。
---

# Tancho 訓練監看（V3 staged）

## 位置與硬性規定

- 專案根目錄：`/home/azul/Tancho/Tancho_v3_simready`（ext4）。log、checkpoint、評估結果、影片全部寫在這裡。
- D 槽 `/media/azul/861896C11896B023` 只能讀，任何東西都不寫進去。要同步到 D 槽先問使用者。
- conda 環境：`env_isaaclab`。
- 回報一律用繁體中文。

## 任務

| 階段 | 訓練任務 | Play 任務 | iter | 驗收腳本 |
|---|---|---|---|---|
| 1 | `TanchoV3-WheelOnly-Flat-v0` | `TanchoV3-WheelOnly-Flat-Play-v0` | 1500 | `scripts/wheel_only/evaluate_standstill.py` |
| 2 | `TanchoV3-Stand-Flat-v0` | `TanchoV3-Stand-Flat-Play-v0` | 3000 | `scripts/wheel_only/evaluate_stand.py`（推力 0 和 0.5 m/s 各跑一次） |
| 3 | `TanchoV3-Walk-Flat-v0` | `TanchoV3-Walk-Flat-Play-v0` | 3000 | `scripts/wheel_only/evaluate_walk.py`（固定命令序列，量追蹤誤差） |

預設只註冊上面 6 個 staged 任務。舊的 Flat / Rough / Fixed-Flat 任務（`tasks/direct/tancho_v3/register.py`）要設 `TANCHO_LEGACY_TASKS=1` 才會註冊，`scripts/diagnostics`、`scripts/evaluation` 裡的舊腳本需要它。

程式碼在 `source/tancho_v3_lab/tancho_v3_lab/tasks/staged/`，四要素各一檔：`observations.py`、`actions.py`、`rewards.py`、`terminations.py`；`env_cfg.py` 負責組裝，`agents/rsl_rl_ppo_cfg.py` 是 PPO 設定。

## 修改環境的規矩

- 改之前先看 `git diff`，使用者自己的修改優先，不要覆蓋。
- 不要的 reward 把權重設成 0，不刪除；不要的 event 改成無作用（例如把推力範圍設成 0），不刪除。這樣前後 run 的 log 欄位才能對得上。
- 一次只改一類東西（reward 權重、PPO 參數、隨機化範圍、終止條件），才知道是哪個改動有效。
- 不能動部署相關的東西，除非使用者同意：actor 觀測只能用硬體量得到的訊號（IMU、編碼器、命令、上一個動作），不能加 base 線速度、高度等特權資訊；動作的意義與比例（輪速 ±52.36 rad/s、腿 ±0.25 rad）、馬達增益與扭力上限、200 Hz / 50 Hz 時序也不能改。
- 新的 asset 設定一定要用 `JointDriveCfg(drive_type="force")`；`joint_drive=None` 會變成 acceleration drive，腿會在自身重量下塌掉。

## 訓練前檢查

1. `python -m py_compile` 檢查改過的 Python 檔。
2. `python scripts/list_envs.py` 確認任務有註冊。
3. 跑 zero-agent，8 個環境、200 步：
   `python scripts/zero_agent.py --headless --task <訓練任務> --num_envs 8 --max_steps 200`
   必須印出 `ZERO_AGENT_GATE status=passed`。二輪平衡車零動作本來就會倒，倒得快不算錯；平均 episode 不到 10 步，或出現 shape、selector、CUDA、reward 的錯誤才擋下。
4. 把改動 commit。`scripts/wheel_only/train.py` 遇到 task 檔案未 commit 時會拒絕訓練，commit hash 會寫進 run 名稱。

## 啟動訓練

- **只能用 `scripts/wheel_only/train.py`**。`scripts/rsl_rl/train.py` 的 `to_compatible_rsl_rl_cfg` 會偷偷換掉 PPO 設定（actor 變成 [256,256,128]、沒有觀測正規化、init std 1.0、critic 拿不到特權觀測），而 `params/agent.yaml` 照樣記錄原本的設定，看不出來。
- 永遠從頭訓練，不接續舊 checkpoint。唯一例外：階段 3 用 `--init_checkpoint` 從階段 2 的權重開始。觀測或動作維度不同的階段之間不能轉移權重。
- 同一時間只跑一個訓練，不要並行（筆電 GPU，並行會大幅拖慢兩邊）。有排隊的工作，等前一個的 PID 結束再開始。
- 背景執行，log 每個 run 各自一份：
  ```bash
  cd /home/azul/Tancho/Tancho_v3_simready
  nohup python scripts/wheel_only/train.py --headless --task <訓練任務> \
      > logs/rsl_rl/<stage>_train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
  echo $!   # 記下 PID
  ```
- 使用者在場且要看終端機時，才另開 80x46 的 GNOME Terminal 跑同一個指令。

## 監看

- 等訓練結束用 PID：`while kill -0 <PID>; do sleep 30; done`。**不要用 `pgrep -f "<pattern>"`**，等待指令本身的命令列也含有同一段字，會一直找到自己，永遠不結束。
- 每 100 iter 回報一次：iter、Mean reward、Mean episode length（步數和秒數，1000 步 = 20 s）、`Episode_Termination/time_out` 比例、比例最高的失敗原因（`tilt` 或 `body_contact`）、ETA。
- 監看程式要能抓到失敗：`Traceback`、`Error`、`NVRM`、`Killed`、`CUDA`，以及程式消失但沒有印出 `Exported policy`。

## 中途檢查點（看最近 5 個 iter 的平均）

數值是 2026-10-06 的觀察值推來的初始門檻，之後依實際資料調整。

| iter | 停掉的條件 |
|---|---|
| 300 | episode < 300 步（6 s），而且同一個失敗原因 > 95% |
| 600 | time_out < 50% |
| 1000 | time_out < 80%，或 reward 從 600 iter 以來沒進步，而且 episode 也沒變長 |

沒過就停掉訓練，寫下原因，進入「迭代」。

## 訓練結束後驗收

1. 確認 run 資料夾是這次新建的，用它的 `exported/policy.pt`（`scripts/wheel_only/train.py` 結束時會自動匯出）。
   舊 run 或缺匯出檔時用：`python scripts/wheel_only/export_policy.py --headless --task <訓練任務> --checkpoint <model.pt>`
2. 跑該階段的驗收腳本，看 `pass_all`。
3. 不要只看 reward 判斷成功，也要看：
   - episode 長度接近 1：reset 姿勢有問題；
   - timeout 很高但姿勢怪（蹲、斜站、單腳歪）：reward 有漏洞；
   - 單一失敗原因獨大：門檻、碰撞體或 reset 設定有問題；
   - reward 上升但驗收指標變差：reward hacking。
4. 錄影給使用者看：
   在 `scripts/rsl_rl/` 下執行 `python play.py --headless --enable_cameras --video --video_length 1000 --task <Play 任務> --checkpoint <model.pt>`，錄完 20 s 會自己結束，影片在 `<run>/videos/play/`。Play 任務的鏡頭跟著機器人，距離約 1.1 m。**不要和訓練同時錄影**：8 GB GPU 會被拖到幾十分鐘，還可能因記憶體不足失敗。timeout 要用 `timeout -s KILL`，Isaac Sim 會忽略一般的 SIGTERM，留下佔著 GPU 的 python 程序。
5. 使用者在場時，在 80x46 終端機填好 GUI Play 指令，不按 Enter，讓使用者自己執行。

## 迭代（使用者授權「沒過就自己改參數重訓」時）

- 每輪只改一類東西，commit 訊息寫清楚改了什麼、為什麼。
- 同一階段最多迭代 4 輪；4 輪都沒過就停下，整理結果等使用者決定。
- 通過後才進下一階段。
- 每輪都記錄在 `logs/rsl_rl/PROGRESS.md`，用繁體中文寫：時間、run 名稱、commit、改動、訓練結果、驗收數字、下一步。早上的報告直接從這份整理。
