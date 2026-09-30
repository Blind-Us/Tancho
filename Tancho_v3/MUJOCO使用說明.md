# Tancho V3 — MuJoCo standalone 使用方式

本資料夾已是可攜式 standalone 模型：

- `Tancho_v3.urdf`：原始 URDF 模型
- `Tancho_v3.xml`：MuJoCo MJCF 模型，建議拖入 Viewer
- `meshes/`：URDF 引用的全部 STL

URDF 內使用 `meshes/檔名.stl` 相對路徑，不依賴 ROS `package://`。

## 使用 GUI 開啟

1. 雙擊桌面的 **MuJoCo Viewer**。
2. 在 MuJoCo 視窗選擇載入模型。
3. 選取本資料夾中的 `Tancho_v3.xml`。

也可以把 `Tancho_v3.xml` 直接拖曳到 MuJoCo Viewer 視窗。XML 會自動從同資料夾的 `meshes/` 載入 STL。

## 搬移或分享

請整個複製 `Tancho_v3` 資料夾，不要只複製 URDF。URDF 與 `meshes/` 的相對位置必須保持不變，否則模型會缺少外觀。

## 指令開啟指定模型

```bash
conda activate env_isaaclab
python -m mujoco.viewer --mjcf /media/azul/861896C11896B023/Tancho/Tancho_v3/Tancho_v3.xml
```

## 已驗證內容

MuJoCo 3.13.0 可直接載入此 URDF；目前解析結果為 6 joints、7 dynamic bodies、11 geoms。
