import pandas as pd
import matplotlib.pyplot as plt

# Đường dẫn tới file results.csv sau khi train classification
csv_path = r"/runs/classify/train8/results.csv"

# Đọc dữ liệu
df = pd.read_csv(csv_path)

# Epoch làm trục x
epochs = df['epoch']

plt.figure(figsize=(12, 6))

# Vẽ Top-1 và Top-5 accuracy
plt.plot(epochs, df['metrics/accuracy_top1'], label='Top-1 Accuracy', linewidth=2)
plt.plot(epochs, df['metrics/accuracy_top5'], label='Top-5 Accuracy', linewidth=2)

# Vẽ train loss và val loss với 2 màu khác nhau
plt.plot(epochs, df['train/loss'],    label='Train Loss', linestyle='--', linewidth=2, color='tab:orange')
plt.plot(epochs, df['val/loss'],      label='Val Loss',   linestyle='--', linewidth=2, color='tab:red')

plt.title('Biểu đồ Top-1, Top-5 Accuracy và Loss theo Epoch', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Giá trị', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("loss_raf.png", dpi=300)
plt.show()
