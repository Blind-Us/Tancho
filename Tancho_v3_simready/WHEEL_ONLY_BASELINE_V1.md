# Tancho V3 僅輪強化學習基準 V1

## 狀態

**W-RL V1 — 可用成果，已凍結**

## 對應實驗組

- 控制算法：PPO 強化學習
- 可控自由度：`joint_wheel_L`、`joint_wheel_R`
- 腿部狀態：固定於標稱姿態
- 標稱姿態：thigh = -0.50 rad、calf = +0.87 rad
- 輪子扭矩限制：±0.45 Nm

## 凍結模型

- 任務：`TanchoV3-Fixed-Flat-v0`
- 訓練輪次：3000
- 訓練資料夾：`logs/rsl_rl/tancho_v3_fixed/2026-09-21_02-35-57`
- 凍結 checkpoint：`logs/rsl_rl/tancho_v3_fixed/2026-09-21_02-35-57/model_final.pt`

## 此版本包含

- 零移動與零旋轉命令
- `ang_z` 角速度懲罰
- 第1500輪啟用平面 XY 速度脈衝 curriculum
- 每次推力事件隨機選取50%訓練環境
- 其餘50%環境保留純站立資料
- 訓練推力間隔16秒
- 物理模型、重生與碰撞設定沿用目前凍結基準

## 用途

- W-LQR 與 W-RL 的算法比較
- W-RL 與 F-RL 的自由度提升比較
- 全機控制實驗的 wheel-only 參考基準

## 凍結規則

此 checkpoint 不再續訓或覆寫。後續 reward、curriculum、物理參數或控制自由度變更，必須建立新的實驗版本與資料夾。
