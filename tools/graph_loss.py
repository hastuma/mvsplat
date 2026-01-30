import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_total_loss(input_csv, output_dir):
    # 1. 讀取 CSV
    try:
        df = pd.read_csv(input_csv)
        df.columns = df.columns.str.strip()  # 清除欄位名稱空格
    except Exception as e:
        print(f"❌ 讀取錯誤: {e}")
        return

    # 檢查是否有 loss/total 欄位
    target_col = 'loss/total'
    if target_col not in df.columns:
        print(f"❌ 找不到欄位: '{target_col}'。請檢查 CSV 標題。")
        print(f"現有欄位包含: {list(df.columns)}")
        return

    # 確保輸出資料夾存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. 繪圖設定
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-muted') # 乾淨的配色風格

    # 3. 繪製 Total Loss
    # 使用 step 作為 X 軸，loss/total 作為 Y 軸
    plt.plot(df['step'], df[target_col], color='#1f77b4', linewidth=2, label='Total Loss')

    # 設定圖表資訊
    plt.title('Training Total Loss Trend', fontsize=14, pad=15)
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # 4. 儲存圖片
    output_path = os.path.join(output_dir, 'total_loss_plot.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ 成功繪製 Total Loss！圖片儲存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='只繪製 CSV 中的 Total Loss')
    
    # 設定參數
    parser.add_argument('--input', type=str, required=True, help='輸入的 .csv 檔案路徑')
    parser.add_argument('--out_dir', type=str, default='./output_plots', help='儲存圖片的資料夾')

    args = parser.parse_args()

    plot_total_loss(args.input, args.out_dir)