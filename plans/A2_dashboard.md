# Dashboard A2 từ export Kaggle

Trang A2 có5 tab: Tổng quan, Dữ liệu, Paper, Huấn luyện, Kết quả & thảo luận.
Chúng bao phủ năm nhóm nội dung của đề; PDF/video được theo dõi riêng và không
được đánh dấu hoàn tất theo trạng thái train. Giữ hệ giao diện HCMUT và A1/E1–E3.

## Nhập lại dữ liệu

```bash
python tools/sync_site_data.py --artifacts-dir artifacts
```

Flag này nhập riêng A2 và giữ nguyên snapshot/hình đã công bố của bài khác.
Export cần manifest.json và full/<variant> với metrics, config, history, protocol,
trained, evaluation/benchmark status và benchmark. Importer đối chiếu SHA256
manifest, worker tái dựng từ sáu cell writefile đã chạy (có newline IPython),
fingerprint cấu hình, cùng metric protocol và đủ20.000 dòng history mỗi nhánh.
Không thực thi notebook khi import. Input lỗi bị từ chối trước khi ghi website.
Không cần checkpoint/source images trên website.

Giữ cách nhập canonical hiện tại bằng --artifacts-dir. Từ export A2, website chỉ
chép các figure được whitelist và dữ liệu công khai đã loại path local; không
chép log, checkpoint hay manifest đầy đủ chứa filename. History hiển thị400
điểm, mỗi điểm trung bình50 updates; full history nguồn không bị thay đổi.

Nguồn hiện tại: A2/dcgan-vs-augmenteddcgan-dl-assignment-2.ipynb và bundle artifacts/A2; export gốc đã chuyển vào archive ngoài repository.
EDA41.522 unique, train34.522/val2.000/test5.000;22.043 duplicates,0 corrupt.
FID298,51→41,84 nhưng recall0→0,0584: kết luận có giới hạn ở seed42, không khẳng
định significance hoặc tái hiện toàn bộ paper. Metric parameters là tham số G.

## Tài nguyên còn thiếu

Báo cáo thật reports/assignment2.pdf đã được build và xác minh; giữ bản ghi hash khi sync.
Đặt URL HTTPS demo/presentation trong docs/data/site_config.json rồi sync lại
để fallback file:// cũng nhận cùng dữ liệu. URL rỗng và PDF placeholder không
tạo liên kết tải giả. Không tự cập nhật timeline sang completed.

## Kiểm tra

```bash
python -m unittest tools.test_a2_web_export tools.test_a2_site_contract
node tools/test_a2_dashboard.cjs
node tools/test_a1_dashboard.cjs
python tools/validate_site.py
```

Tab hỗ trợ mũi tên/Home/End và URL #a2-overview/data/paper/training/results.
Các anchor cũ #paper-analysis/#pipeline-results/#a2-videos được ánh xạ tới tab.
SVG EDA/loss không phụ thuộc CDN; số liệu và ảnh hoạt động từ fallback khi file://
không fetch JSON được. Desktop hai cột; mobile xếp dọc, bảng cuộn trong container.
Phóng to ảnh bằng button/Enter, đóng bằng Escape/nút Đóng và trả focus về ảnh.

## Kết quả kiểm tra triển khai

- Sáu test Python về export/website contract đạt; renderer kiểm tra0/1/2 full
  runs, các tab/anchor/Back, PDF/video và khởi tạo fallback khi fetch bị từ chối.
- Validator liên kết/cấu trúc, kiểm tra cú pháp JavaScript và renderer A1 đạt.
- Quan sát được desktop/dark và mobile/light; kiểm tra tab bằng phím End, EDA
  và bảng metric thật. Viewport390px không có tràn ngang toàn trang. Lần xác
  nhận browser cuối bị gián đoạn do kết nối preview localhost; chưa xác nhận
  trực tiếp hành vi Escape/focus của lightbox sau thay đổi cuối.
- Snapshot E1–E3/A1 và29 hình công bố của các bài đó giữ nguyên so với lúc bắt
  đầu thay đổi dashboard. Tổng full run sau nhập là15/23; A2 có2/2, không đánh
  dấu PDF/video hoàn tất. Không train, tạo PDF/video, push hoặc deploy.
