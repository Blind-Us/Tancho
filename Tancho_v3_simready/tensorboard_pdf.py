import os
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tbparse import SummaryReader


def main():
    # ---------------------------------------------------------
    # 用法檢查
    # ---------------------------------------------------------
    if len(sys.argv) < 2:
        print("用法：")
        print("  python tensorboard_pdf.py <訓練資料夾>")
        print()
        print("也可以直接把訓練資料夾拖進 Terminal。")
        sys.exit(1)

    # 支援路徑中有空格
    log_dir = os.path.abspath(sys.argv[1])

    if not os.path.isdir(log_dir):
        print(f"錯誤：找不到資料夾")
        print(log_dir)
        sys.exit(1)

    # ---------------------------------------------------------
    # PDF 輸出位置
    # 直接放在訓練資料夾裡
    # ---------------------------------------------------------
    run_name = os.path.basename(os.path.normpath(log_dir))
    output_pdf = os.path.join(
        log_dir,
        f"{run_name}.pdf"
    )

    print("=" * 60)
    print("TensorBoard → PDF")
    print("=" * 60)
    print(f"訓練資料夾：{log_dir}")
    print(f"輸出 PDF：  {output_pdf}")
    print()

    # ---------------------------------------------------------
    # 讀取 TensorBoard
    # ---------------------------------------------------------
    print("正在讀取 TensorBoard 資料...")

    try:
        reader = SummaryReader(log_dir)
        df = reader.scalars
    except Exception as e:
        print("讀取 TensorBoard 失敗：")
        print(e)
        sys.exit(1)

    if df.empty:
        print("找不到任何 TensorBoard Scalar 資料！")
        print("請確認資料夾內含有 events.out.tfevents... 檔案。")
        sys.exit(1)

    tags = df["tag"].unique()

    print(f"找到 {len(tags)} 個指標。")
    print("開始生成 PDF...")
    print()

    # ---------------------------------------------------------
    # 產生 PDF
    # ---------------------------------------------------------
    with PdfPages(output_pdf) as pdf:

        for index, tag in enumerate(tags, start=1):

            tag_df = df[df["tag"] == tag]

            print(f"[{index}/{len(tags)}] {tag}")

            plt.figure(figsize=(8, 5))

            plt.plot(
                tag_df["step"],
                tag_df["value"],
                linewidth=1.5
            )

            plt.title(
                tag,
                fontsize=14,
                fontweight="bold"
            )

            plt.xlabel("Step", fontsize=11)
            plt.ylabel("Value", fontsize=11)

            plt.grid(
                True,
                linestyle="--",
                alpha=0.6
            )

            plt.tight_layout()

            pdf.savefig()
            plt.close()

    # ---------------------------------------------------------
    # 完成
    # ---------------------------------------------------------
    print()
    print("=" * 60)
    print("完成！")
    print(f"PDF：{output_pdf}")
    print("=" * 60)


if __name__ == "__main__":
    main()