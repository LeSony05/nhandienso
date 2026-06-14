# Nhận Dạng Chữ Số Viết Tay

## Cách Chạy

### 1. Cài đặt thư viện

đối với Window

```bash
pip install -r requirements.txt
```

đối với mac

```bash
pip3 install -r requirements.txt
```

### 2. Chạy ứng dụng

đối với win

```bash
python app.py
```

đối với mac

```bash
python3 app.py
```

### 3. Truy cập giao diện

Mở trình duyệt và vào: **http://localhost:5000**

## 📖 Sử Dụng

### Chạy ứng dụng

```bash
python3 app.py
```

Ứng dụng sẽ khởi động Flask server tại `http://localhost:5000`

### Sử dụng giao diện web

1. Mở trình duyệt và truy cập `http://localhost:5000`
2. Click nút **"Chọn Ảnh"** để chọn ảnh chứa chữ số
3. Hoặc **kéo thả ảnh** vào vùng được chỉ định
4. Click **"Nhận Dạng"** để bắt đầu quá trình nhận dạng
5. Xem kết quả với:
   - Ảnh gốc được xử lý
   - Danh sách chữ số được nhận dạng
   - Độ tin cậy (xác suất) của từng chữ số

## Cấu Trúc Project

```
nhandienso/
├── app.py                    # Ứng dụng Flask chính
├── cnn_engine.py             # Module xử lý ảnh và nhận dạng CNN
├── mnist_model.py            # Định nghĩa mô hình CNN + DigitRecognizer
├── train_model.py            # Script huấn luyện model (lần đầu)
├── requirements.txt          # Danh sách thư viện phụ thuộc
├── README.md                 # Tài liệu này
│
├── static/                   # Các file tĩnh (CSS, JS)
│   ├── css/
│   │   └── style.css         # Kiểu dáng giao diện
│   ├── js/
│   │   └── app.js            # Logic frontend (xử lý upload, gọi API)
│   └── uploads/              # Thư mục tạm lưu ảnh upload
│
├── templates/
│   └── index.html            # Giao diện web chính
│
├── models/
│   ├── mnist_cnn.pth         # Model CNN đã huấn luyện (PyTorch)
│   └── data/
│       └── MNIST/            # Bộ dữ liệu MNIST (tự động download)
│           └── raw/          # File MNIST gốc
│
└── README.md
```

## Chi Tiết Mô Hình

### Kiến Trúc CNN

- **Input**: Ảnh xám 28x28 pixel
- **Layer 1**: Conv2d (1→32, 3×3) + ReLU + Conv2d (32→64, 3×3) + ReLU + MaxPool(2×2) + Dropout(0.25)
- **Layer 2**: Conv2d (64→128, 3×3) + ReLU + MaxPool(2×2) + Dropout(0.25)
- **Fully Connected**: FC(128×7×7 → 256) + ReLU + Dropout(0.5) → FC(256 → 10)
- **Output**: 10 lớp (chữ số 0-9)

### Thông Số

- **Framework**: PyTorch
- **Bộ dữ liệu**: MNIST (70,000 ảnh)
- **Độ chính xác**: ~99% trên test set
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 50

## Xử Lý Ảnh

Quá trình xử lý ảnh để nhận dạng:

1. **Đọc ảnh** → Chuyển sang grayscale
2. **Blur** → Giảm nhiễu (Gaussian Blur 5×5)
3. **Nhị phân hóa** → Adaptive Thresholding
4. **Morphological Operations** → Closing + Opening để làm sạch
5. **Tìm Contour** → Phát hiện các chữ số trong ảnh
6. **Chuẩn hóa** → Cắt, resize về 28×28
7. **Dự đoán** → Chạy qua CNN model

## Gỡ Lỗi

### Lỗi "No module named 'flask'"

```bash
pip install flask
```

### Lỗi "ModuleNotFoundError: No module named 'torch'"

```bash
pip install torch torchvision
```

### Server không khởi động

- Kiểm tra cổng 5000 có bị chiếm dụng không
- Thử chạy với port khác: chỉnh sửa trong `app.py`

### Model không tải được

- Kiểm tra file `models/mnist_cnn.pth` tồn tại
- Huấn luyện lại model: `python3 train_model.py`

## Huấn Luyện Lại Model

Nếu muốn huấn luyện model từ đầu:

```bash
python3 train_model.py
```

Script sẽ:

- Tải dataset MNIST
- Huấn luyện CNN
- Lưu model vào `models/mnist_cnn.pth`
- Hiển thị biểu đồ training history

## Lời Khuyên Sử Dụng

- **Ảnh sạch nhất**: Chữ số viết tay rõ ràng trên nền trắng/sáng
- **Độ sáng**: Tránh ảnh quá tối hoặc quá sáng
- **Định dạng**: PNG hoặc JPG cho kết quả tốt nhất
- **Kích thước**: Ảnh quá nhỏ có thể ảnh hưởng độ chính xác

## License

Project này được tạo cho mục đích học tập.

## Tác Giả

**Hoàn chỉnh**: 2024

---

**Ghi chú**: Nếu gặp vấn đề, hãy đảm bảo tất cả thư viện đã được cài đặt đúng và Python version 3.8+
