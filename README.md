# GRU model for IoT-23 NIDS

Huấn luyện và đánh giá mô hình GRU cho bài toán phát hiện xâm nhập mạng (NIDS) trên dữ liệu IoT-23.

## Yêu cầu

- Python 3.10 trở lên
- PyTorch
- NumPy
- scikit-learn

Cài các thư viện:

```bash
pip install -r requirements.txt
```

## Huấn luyện

Chuẩn bị một tệp `.npz` gồm các mảng `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`; có thể kèm `class_names`.

```bash
python train_gru.py --data data/iot23_seq.npz --out runs/gru_h64
```

Thư mục dữ liệu, kết quả huấn luyện và checkpoint được bỏ qua khỏi Git để tránh tải nhầm các tệp dung lượng lớn lên GitHub.
