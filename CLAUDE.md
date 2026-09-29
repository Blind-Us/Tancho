
## 工作位置規則（2026-09-30）
- 主要工作區是 ext4：/home/azul/Tancho。所有寫入（程式碼、git、訓練 log、checkpoint、tensorboard、評估 CSV 和圖）都只能在這裡。
- /media/azul/861896C11896B023（D 槽，NTFS）只能讀取。Linux 的 ntfs3 驅動在 2026-09-30 寫檔時觸發過 kernel BUG。
- 要同步到 D 槽時，先問使用者。
- 回覆使用繁體中文。
