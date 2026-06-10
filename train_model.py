import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mnist_model import MNISTNet, MODEL_DIR, MODEL_PATH
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# Đánh giá model — trả về accuracy, tất cả nhãn thật & dự đoán
# ============================================================
def evaluate(model, device, test_loader, return_preds=False):
    model.eval()
    correct = 0
    total = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)
            if return_preds:
                all_targets.extend(target.cpu().numpy())
                all_preds.extend(pred.cpu().numpy())

    model.train()
    acc = 100. * correct / total
    if return_preds:
        return acc, np.array(all_targets), np.array(all_preds)
    return acc


# ============================================================
# Biểu đồ 1 & 2: Training Loss + Accuracy theo epoch
# ============================================================
def plot_training_history(history, epochs, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Lịch sử huấn luyện', fontsize=14, fontweight='bold')

    ep = range(1, epochs + 1)

    # Loss
    axes[0].plot(ep, history['train_loss'], color='#D85A30', marker='o', markersize=4, linewidth=2, label='Train Loss')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Accuracy
    axes[1].plot(ep, history['train_acc'], color='#185FA5', marker='o', markersize=4, linewidth=2, label='Train Accuracy')
    axes[1].plot(ep, history['test_acc'],  color='#0F6E56', marker='s', markersize=4, linewidth=2,
                 linestyle='--', label='Test Accuracy')
    axes[1].set_title('Accuracy theo epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [saved] training_history.png")


# ============================================================
# Biểu đồ 3: Confusion Matrix
# ============================================================
def plot_confusion_matrix(y_true, y_pred, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = [str(i) for i in range(10)]
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Dự đoán', fontsize=12)
    ax.set_ylabel('Nhãn thật', fontsize=12)
    ax.set_title('Confusion Matrix (Test Set)', fontsize=13, fontweight='bold')

    thresh = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center', fontsize=8,
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [saved] confusion_matrix.png")


# ============================================================
# Biểu đồ 4: Precision / Recall / F1 theo từng lớp
# ============================================================
def plot_classification_report(y_true, y_pred, save_dir):
    report = classification_report(y_true, y_pred, output_dict=True)
    classes = [str(i) for i in range(10)]

    precision = [report[c]['precision'] * 100 for c in classes]
    recall    = [report[c]['recall']    * 100 for c in classes]
    f1        = [report[c]['f1-score']  * 100 for c in classes]

    x = np.arange(10)
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, precision, width, label='Precision', color='#185FA5', alpha=0.9)
    ax.bar(x,         recall,    width, label='Recall',    color='#0F6E56', alpha=0.9)
    ax.bar(x + width, f1,        width, label='F1-score',  color='#EF9F27', alpha=0.9)

    ax.set_xlabel('Chữ số')
    ax.set_ylabel('Giá trị (%)')
    ax.set_title('Precision / Recall / F1 theo từng lớp', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(96, 100.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # Ghi macro avg lên biểu đồ
    macro_f1 = report['macro avg']['f1-score'] * 100
    ax.text(0.99, 0.97, f"Macro F1: {macro_f1:.2f}%",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, color='#333',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(save_dir, 'classification_report.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [saved] classification_report.png")


# ============================================================
# Biểu đồ 5: Phân bố confidence score trên test set
# ============================================================
def plot_confidence_distribution(model, device, test_loader, save_dir):
    model.eval()
    correct_conf = []
    wrong_conf   = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.exp(output)
            conf, pred = probs.max(dim=1)
            mask_correct = pred.eq(target)
            correct_conf.extend(conf[mask_correct].cpu().numpy())
            wrong_conf.extend(conf[~mask_correct].cpu().numpy())

    model.train()

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 50)
    ax.hist(correct_conf, bins=bins, alpha=0.7, color='#185FA5', label=f'Đúng ({len(correct_conf):,})')
    ax.hist(wrong_conf,   bins=bins, alpha=0.8, color='#D85A30', label=f'Sai ({len(wrong_conf):,})')
    ax.axvline(x=0.80, color='#EF9F27', linewidth=2, linestyle='--', label='Ngưỡng 0.80')
    ax.set_xlabel('Confidence score')
    ax.set_ylabel('Số lượng mẫu')
    ax.set_title('Phân bố confidence score trên test set', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'confidence_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [saved] confidence_distribution.png")


# ============================================================
# Hàm train chính
# ============================================================
def train_model(epochs=15, batch_size=64, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = MNISTNet().to(device)

    os.makedirs(MODEL_DIR, exist_ok=True)
    data_dir = os.path.join(MODEL_DIR, 'data')

    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("Downloading MNIST dataset...")
    train_dataset = datasets.MNIST(data_dir, train=True,  download=True, transform=train_transform)
    test_dataset  = datasets.MNIST(data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Starting training...")
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)

        train_acc = 100. * correct / total
        avg_loss  = total_loss / len(train_loader)
        test_acc  = evaluate(model, device, test_loader)

        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        print(f"  Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} - Train: {train_acc:.1f}% - Test: {test_acc:.1f}%")

    # Lưu model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Lấy predictions cuối để vẽ biểu đồ
    final_acc, y_true, y_pred = evaluate(model, device, test_loader, return_preds=True)
    print(f"Final Test Accuracy: {final_acc:.2f}%")

    # In classification report ra console
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    # Xuất 5 biểu đồ
    print("\nGenerating evaluation charts...")
    plot_training_history(history, epochs, MODEL_DIR)
    plot_confusion_matrix(y_true, y_pred, MODEL_DIR)
    plot_classification_report(y_true, y_pred, MODEL_DIR)
    plot_confidence_distribution(model, device, test_loader, MODEL_DIR)
    print(f"\nAll charts saved to: {MODEL_DIR}/")


if __name__ == '__main__':
    train_model(epochs=15)