import os
import re
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl

def setup_paper_style():
    """Cấu hình matplotlib chuẩn bài báo khoa học"""
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 11
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.bbox'] = 'tight'

def parse_log_file(log_path):
    """Đọc file log và trích xuất các epoch, loss, f1, auroc"""
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Không tìm thấy file log: {log_path}")

    # Regex patterns
    train_pattern = re.compile(r"Epoch (\d+) Train\s+\|\s+Avg Loss:\s+([\d.]+)")
    # Group 1: Epoch, Group 2: Val Loss, Group 3: F1, Group 4: AUROC (Optional)
    val_pattern = re.compile(r"Epoch (\d+) Val\s+\|\s+Loss:\s+([\d.]+).*?F1:\s+([\d.]+)(?:.*?AUROC:\s+([\d.]+))?")

    data = {}

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse Train Loss
            train_match = train_pattern.search(line)
            if train_match:
                ep = int(train_match.group(1))
                if ep not in data: data[ep] = {}
                data[ep]['train_loss'] = float(train_match.group(2))
                continue
            
            # Parse Validation Metrics
            val_match = val_pattern.search(line)
            if val_match:
                ep = int(val_match.group(1))
                if ep not in data: data[ep] = {}
                data[ep]['val_loss'] = float(val_match.group(2))
                data[ep]['val_f1'] = float(val_match.group(3))
                if val_match.group(4):
                    data[ep]['val_auroc'] = float(val_match.group(4))

    # Lọc bỏ các epoch chưa chạy xong (thiếu số liệu) và sắp xếp theo epoch
    epochs = sorted([ep for ep in data.keys() if 'train_loss' in data[ep] and 'val_loss' in data[ep]])
    
    train_losses = [data[ep]['train_loss'] for ep in epochs]
    val_losses = [data[ep]['val_loss'] for ep in epochs]
    val_f1s = [data[ep]['val_f1'] for ep in epochs]
    val_aurocs = [data[ep].get('val_auroc', None) for ep in epochs]

    has_auroc = any(a is not None for a in val_aurocs)
    
    return epochs, train_losses, val_losses, val_f1s, val_aurocs, has_auroc

def plot_training_curves(log_path, output_path=None):
    epochs, train_losses, val_losses, val_f1s, val_aurocs, has_auroc = parse_log_file(log_path)
    
    if len(epochs) == 0:
        print("LỖI: Không tìm thấy dữ liệu hợp lệ trong file log. Mô hình đã chạy xong Epoch 0 chưa?")
        return

    setup_paper_style()
    
    # Khởi tạo Figure với 2 khung hình phụ (1 hàng, 2 cột)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Plot 1: Learning Curve (Loss) ----
    ax1.plot(epochs, train_losses, label='Train Loss', color='#1f77b4', marker='o', linewidth=2, markersize=5)
    ax1.plot(epochs, val_losses, label='Validation Loss', color='#ff7f0e', marker='s', linewidth=2, markersize=5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary Cross-Entropy Loss')
    ax1.set_title('Model Learning Curve')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Đặt giới hạn trục X là số nguyên
    ax1.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

    # ---- Plot 2: Performance Metrics (F1 & AUROC) ----
    ax2.plot(epochs, val_f1s, label='F1-Score', color='#2ca02c', marker='^', linewidth=2, markersize=5)
    
    if has_auroc:
        # Lọc ra các epoch có AUROC để vẽ (đề phòng log bị lai tạp)
        a_epochs = [ep for i, ep in enumerate(epochs) if val_aurocs[i] is not None]
        a_vals = [a for a in val_aurocs if a is not None]
        ax2.plot(a_epochs, a_vals, label='AUROC', color='#d62728', marker='D', linewidth=2, markersize=5)
        ax2.set_title('Validation Performance (F1 & AUROC)')
    else:
        ax2.set_title('Validation Performance (F1-Score)')

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Đặt giới hạn trục Y từ 0.7 đến 1.0 cho đẹp (hoặc tự động nới lỏng nếu điểm thấp hơn)
    min_score = min(min(val_f1s), min([a for a in val_aurocs if a is not None] if has_auroc else [1.0]))
    ax2.set_ylim(bottom=max(0.0, min_score - 0.05), top=1.02)
    ax2.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

    # Căn chỉnh bố cục và lưu ảnh
    plt.tight_layout()
    
    if output_path is None:
        log_name = os.path.splitext(os.path.basename(log_path))[0]
        output_path = f"{log_name}_curves.png"
        
    plt.savefig(output_path)
    print(f"Đã vẽ xong! Biểu đồ được lưu tại: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ huấn luyện từ file Log")
    parser.add_argument("log_file", type=str, help="Đường dẫn tới file .log (ví dụ: runs/ex6_0319/train_detector_xxx.log)")
    parser.add_argument("--out", type=str, default=None, help="Tên file ảnh đầu ra (mặc định: tự động sinh)")
    
    args = parser.parse_args()
    plot_training_curves(args.log_file, args.out)