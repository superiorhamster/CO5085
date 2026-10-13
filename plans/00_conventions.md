# Quy ước chung cho notebook CO5085

## Mục tiêu

Năm notebook phải tự chứa, chạy tuần tự từ kernel sạch trên Colab/Kaggle và tạo artifact có thể truy vết. Notebook không import module Python riêng của repository.

## Cấu hình chung

- `run_mode`: `smoke` hoặc `full`. Smoke giới hạn batch/epoch/step và ghi artifact vào namespace riêng.
- `experiment_ids`: danh sách thí nghiệm cần chạy. ID không hợp lệ phải gây lỗi sớm.
- `seed=42`; ghi seed và phiên bản thư viện vào `environment.json`.
- `data_root`, `output_root`: đọc từ biến môi trường, có fallback tương thích Colab/Kaggle/local.
- `resume_from`: `None`, `"last"`, hoặc đường dẫn checkpoint cụ thể.
- `evaluate_only`: đọc từ `CO5085_EVALUATE_ONLY=1`; tải checkpoint và chạy lại phần đánh giá mà không cập nhật trọng số.
- Full run chọn checkpoint bằng validation set; test chỉ chạy sau khi checkpoint đã được chọn.

## Split dùng chung E1–E3

Fashion-MNIST official train được chia phân tầng thành 54.000 train và 6.000 validation bằng `StratifiedShuffleSplit(random_state=42)`. Notebook ghi `fashion_mnist_split_seed42.npz` và SHA-256 của hai mảng index. E1–E3 dừng nếu file split hiện hữu không khớp hash tính lại.

## Artifact contract

Mỗi run lưu tại `<output_root>/<assignment>/<run_mode>/<experiment_id>/`:

- `config.json`, `environment.json`, `history.csv`, `metrics.json`;
- `last.pt` và `best.pt` cho E1–A1; A2 dùng `step_*.pt` và `last.pt`;
- `curves.png`, `confusion_matrix.png`, `failures.png` khi phù hợp;
- dự đoán hoặc manifest cần thiết để tái tạo bảng báo cáo.

Checkpoint E1–A1 chứa model, optimizer, scheduler, AMP scaler, epoch, best metric và RNG state. A2 lưu thêm EMA, global step và trạng thái sampler. Checkpoint được ghi qua file tạm rồi thay thế để giảm nguy cơ hỏng file khi runtime ngắt.

## Đo hiệu năng

Đo trên cùng device, precision và batch size. GPU được warm-up, đồng bộ trước/sau phép đo; data loading không tính vào inference latency. Ghi peak allocated memory sau khi reset CUDA statistics. Không xếp hạng tốc độ của các run ở GPU khác nhau.

## Quy trình chạy

1. Chạy smoke toàn notebook và kiểm tra artifact.
2. Chạy pilot một cấu hình để ước lượng thời gian/VRAM.
3. Chạy full theo từng experiment ID và lưu checkpoint bền vững.
4. Chạy cell tổng hợp sau khi đủ experiment; chỉ số smoke bị loại khỏi bảng báo cáo.
5. Lưu notebook đã chạy cùng artifact. Không ghi kết luận trước khi có số liệu full.

