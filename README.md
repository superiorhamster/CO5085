# CO5085 — Training notebooks

Repository này chứa 5 notebook tự chứa cho E1, E2, E3, A1 và A2. Mỗi notebook có hai chế độ:

- `smoke`: kiểm tra pipeline bằng ít batch/step; kết quả không dùng trong báo cáo.
- `full`: chạy đủ ma trận thí nghiệm và xuất artifact để tổng hợp báo cáo.

## Notebook

| Bài | Notebook | Dataset | Số cấu hình |
|---|---|---|---:|
| E1 | `E1/e1_fashion_mnist_baselines.ipynb` | Fashion-MNIST | 3 |
| E2 | `E2/e2_attention_tokenization.ipynb` | Fashion-MNIST | 6 |
| E3 | `E3/e3_sequence_models.ipynb` | Fashion-MNIST | 4 |
| A1 | `A1/a1_cnn_vs_transformer.ipynb` | CIFAR-100 | 8 |
| A2 | `A2/a2_diffusion_schedules.ipynb` | CelebA | 2 |

Chi tiết cách chạy, lưu checkpoint, quy tắc so sánh và định dạng artifact nằm trong [quy ước chung](plans/00_conventions.md). Các quyết định riêng của từng bài nằm trong `plans/`.

## Chạy trên Colab/Kaggle

1. Mở notebook tương ứng và chọn GPU runtime.
2. Sửa cell `CONFIG`; giữ `run_mode="smoke"` cho lần chạy đầu.
3. Chạy `Restart and run all` để xác nhận notebook không phụ thuộc trạng thái ẩn.
4. Đặt `run_mode="full"`, chọn `experiment_ids`, rồi chạy các thí nghiệm chính.
5. Sao chép thư mục artifact sang Google Drive hoặc tạo Kaggle Dataset trước khi phiên GPU kết thúc.

Để chỉ đánh giá lại `best.pt`/`last.pt` mà không train, đặt biến môi trường `CO5085_EVALUATE_ONLY=1` trước khi chạy notebook. `resume_from="last"` tiếp tục một run bị ngắt; experiment đã có `metrics.json` sẽ được bỏ qua theo mặc định.

Có thể đặt `CO5085_DATA_ROOT` và `CO5085_OUTPUT_ROOT` để đổi vị trí dữ liệu/artifact mà không sửa code. A2 yêu cầu người dùng gắn CelebA Align&Cropped vào runtime và trỏ `CO5085_CELEBA_ROOT` tới thư mục ảnh.

## Kiểm tra cục bộ

Notebook được sinh từ `tools/build_notebooks.py` để giữ cấu trúc nhất quán. Sau khi sửa generator, chạy:

```bash
python tools/build_notebooks.py
python tools/validate_notebooks.py
```

Validation chỉ kiểm tra JSON, thứ tự cell và cú pháp Python; train thực tế cần môi trường có GPU và các dependency được ghi trong notebook.

## GitHub Pages & Đồng bộ Báo cáo

Mã nguồn giao diện GitHub Pages nằm tại `docs/` (kèm `index.html` chuyển hướng ở thư mục gốc để tương thích cả 2 chế độ cấu hình `/` và `/docs`):

- `docs/index.html`: Trang chủ nhóm, thông tin thành viên, giảng viên hướng dẫn, tiến độ W1–W11 và bản đồ 23 cấu hình thực nghiệm.
- `docs/exercises.html`: Trang chi tiết chuỗi bài tập thực hành E1, E2, E3 trên Fashion-MNIST.
- `docs/a1.html`: Trang chi tiết Bài tập lớn A1 (ResNet-18 vs DeiT-Tiny trên CIFAR-100, sơ đồ Freeze Strategies tương tác).
- `docs/a2.html`: Trang chi tiết Bài tập lớn A2 (Ablation lịch nhiễu Linear vs Cosine cho DDPM trên CelebA 32×32 & phân tích bài báo ICML 2021).
- `docs/data/site_config.json`: File cấu hình tập trung duy nhất cho thông tin nhóm (Họ tên, MSSV, URL GitHub Repo, URL Video Demo/Thuyết trình).

Sau khi tải thư mục `artifacts/` từ phiên chạy `full` về máy hoặc cập nhật báo cáo PDF trong `reports/`, chạy lệnh sau để đồng bộ dữ liệu và kiểm tra trang web:

```bash
python tools/sync_site_data.py --artifacts-dir ./artifacts
python tools/validate_site.py
```

Lệnh sync từ chối ghi đè manifest khi không tìm thấy full run hợp lệ. Tùy chọn `--allow-empty` chỉ dùng một lần khi khởi tạo website chưa có kết quả.

Commit cả `docs/data/experiments_manifest.json`, `docs/assets/js/fallback_data.js` và các hình trong `docs/assets/artifacts/` sau khi kiểm tra thay đổi. Workflow `.github/workflows/deploy-pages.yml` chỉ kiểm tra và triển khai nội dung `docs/` đã commit; nó không tự chạy sync trên runner rỗng để tránh ghi đè kết quả đã xuất ở local.

Trước lần deploy đầu tiên, mở **Settings → Pages → Build and deployment → Source** của repository và chọn **GitHub Actions**. URL dự kiến của website là <https://superiorhamster.github.io/CO5085/>.

Trước mốc nộp chính thức, chạy kiểm tra nghiêm ngặt để phát hiện MSSV còn thiếu, báo cáo placeholder và ma trận thí nghiệm chưa đủ:

```bash
python tools/validate_site.py --strict-release
```


