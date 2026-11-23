# Tancho V3 RL 四要素基準

本基準只服務 Tancho V3 的物理控制問題，不以其他機器人的 reward 配方作為權威。除 Tancho 專屬幾何條件外，優先使用 Isaac Lab 官方 MDP term。

## 1. Observation（狀態）

- 機身線速度、角速度與高度
- 重力在機身座標系的投影
- 零速度站立命令
- 腿部相對位置／速度與輪速（不輸入會無界累積的 wheel angle）
- 上一拍 action

這些量足以描述倒立擺姿態、腿部構形、輪速與控制器記憶。Flat baseline 不加入噪聲；rough terrain 的地形高度觀測另立實驗，不混入本基準。

## 2. Action（控制輸入）

- 四個 DM-J4310 腿關節：位置目標增量，scale 固定為 `0.25 rad`
- 兩個 DM-H3510 輪關節：速度目標，scale 為額定 `500 rpm = 52.36 rad/s`
- 腿部 actuator：`Kp=20`、`Kd=0.2`、峰值 `12.5 Nm`
- 輪部 actuator：速度迴路、峰值 `0.45 Nm`
- PhysX `dt=0.005 s`（200 Hz），policy decimation `4`（50 Hz）

Action scale 不拿來掩蓋不連續控制；平滑性由官方 `action_rate_l2` 約束。輪子不使用無增益的直接力矩 action。

## 3. Reward（最佳控制目標）

主要狀態成本：

- 直立誤差、roll/pitch 角速度
- 零命令下的 XY 線速度與 yaw 角速度
- 垂直速度
- Tancho 專屬的全機動態 capture point 到輪軸線距離；靜止時自然退化為 COM 對輪軸
- Tancho 左右腿鏡像誤差

控制成本：

- 腿／輪實際力矩平方
- action 一階變化率
- 關節加速度
- 腿部位置限制與輪速限制

存活與跌倒：每秒存活獎勵為正；非 timeout 跌倒有明確負成本。Reward 不追固定站高、不追固定腿姿，也不鼓勵蹲地接觸。

## 4. Termination（失敗集合）

- episode 到達 20 秒：正常 timeout
- 6-DOF：base、thigh 或 calf 發生超過門檻的非法接觸
- fixed-leg：合併後 base assembly 發生非法接觸

Reset 使用姿態相依 FK 計算輪底支撐高度；正式 `t=0` 前清除速度與 buffer，不把人工落地脈衝當作學習問題。

## 實驗邊界

- Formal baseline：平地、零速度命令、相同 friction、相同 50 Hz policy rate。
- Push curriculum 只改外擾，不自動加入旋轉或移動命令。
- Rough terrain 必須先接入左右輪 local terrain height／ray observation 才能視為有效基準。
- DM-H3510 韌體 damping factor 與 Isaac damping 的單位尚未由實機辨識；目前模擬值是可說明的初始近似，不宣稱已完成 hardware identification。
