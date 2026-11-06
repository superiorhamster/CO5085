# Báo cáo CO5085 · HK261 · 2026–2027

Ba báo cáo tiếng Việt dùng bản dẫn xuất của `HCMUT_RP_template/SoftEng_252`.
Template gốc giữ nguyên. Bìa, logo, header/footer, article/vntex và mục lục được giữ;
metadata chung nằm trong `shared/metadata.tex`. Hai thành viên: **Phạm Duy Anh - 2310139** và **Nguyễn Phan Tuấn Duy - 2310491**.

## Chuẩn bị môi trường

Cần Python 3.11+, pdfLaTeX, BibTeX, latexmk và Poppler (`pdfinfo`, `pdftotext`,
`pdftoppm`). Không cần torch, GPU, dataset hoặc checkpoint để build báo cáo.

Ubuntu/Debian (nếu máy chưa có TeX):

```bash
sudo apt-get install latexmk texlive-latex-extra texlive-lang-other texlive-fonts-recommended poppler-utils
```

Với TeX Live/tlmgr, cần các gói `vntex latexmk booktabs caption lastpage fancyhdr
listings geometry hyperref enumitem xurl`, cùng LaTeX/base fonts. Trong phiên triển
khai này, TeX Live 2026 được cài riêng ở `/tmp/co5085-texlive`, không thay môi trường
hệ thống. Thư mục `/tmp` không phải dependency được đóng gói vào Git; hãy cài TeX ở
máy đích hoặc thêm thư mục `bin` của bản cài hiện có vào `PATH`.

## Build và đối chiếu

Chạy từ repository root:

```bash
python tools/build_reports.py                         # cả ba PDF
python tools/build_reports.py --report exercises     # chỉ E1–E3
python tools/build_reports.py --report assignment1
python tools/build_reports.py --report assignment2
python tools/build_reports.py --check                # kiểm tra bản đã build, không cần TeX
python tools/sync_site_data.py --artifacts-dir artifacts
python tools/validate_site.py
```

Builder tự chạy `prepare_reports.py`, rồi latexmk/pdfLaTeX/BibTeX. Lần đầu cần các
artifact nhẹ A1/A2 và các metrics/config/hình E1–E3 đã lưu. Nó không tải dữ liệu,
không train, không đọc checkpoint pickle, không lấy preview trên website.

- `generated/`: bảng `.tex`, hình sao chép và `provenance.json` có SHA-256 nguồn.
- `.build/`: log, aux và text extraction; bị Git ignore.
- `../exercises.pdf`, `../assignment1.pdf`, `../assignment2.pdf`: PDF nguồn công bố.
- `../../docs/reports/`: bản sao công bố tương ứng.
- `../report_manifest.json`: hash PDF/dependency, số trang và kiểm tra text/ref/overflow.

Builder từ chối citation/reference chưa giải quyết, missing glyph, overfull box,
placeholder template hoặc thiếu các nội dung chính. Nếu nguồn LaTeX/hình/tool thay
đổi sau build, importer đánh dấu PDF chưa xác minh cho tới khi build lại. CI chỉ
kiểm tra bản ghi và hash của PDF đã có, không cài TeX hoặc chạy ML.

Để sinh ảnh toàn bộ trang và contact sheets, chạy `python tools/render_reports.py`
(cần Pillow và Poppler; output mặc định ở `/tmp/co5085-pdf-review`).

Sau mỗi lần sửa nội dung, cần render PDF và kiểm tra tất cả trang, đặc biệt bảng,
công thức và hình. Có thể chạy `pdftoppm -r 90 -png reports/assignment1.pdf /tmp/a1-page`
để kiểm tra trực quan. Không suy ra chất lượng bố cục chỉ từ text extraction.

## Project độc lập cho Prism / Overleaf

```bash
python tools/package_overleaf.py
# hoặc: python tools/package_overleaf.py --report assignment1
```

Ba project độc lập, mỗi thư mục có `main.tex`, `latexmkrc`, README riêng,
`shared/` (metadata, style, logo, bibliography) và `generated/` (bảng/hình cần dùng):

- `reports/projects/exercises/` và `reports/projects/exercises.zip`
- `reports/projects/assignment1/` và `reports/projects/assignment1.zip`
- `reports/projects/assignment2/` và `reports/projects/assignment2.zip`

ZIP không có thư mục bọc bên ngoài: `main.tex` nằm ngay ở gốc khi giải nén.
Mỗi project tự chứa, không phụ thuộc hai báo cáo còn lại, repo, Python hoặc artifact
training. Upload từng ZIP thành project riêng; chọn **main.tex / pdfLaTeX**.
Chỉnh MSSV trong `shared/metadata.tex`. Sau khi dọn repository, chỉ giữ một bộ ZIP
tại `reports/projects/`; lệnh đóng gói không tạo thêm bản sao `reports/overleaf/`.

Prism có hỗ trợ import folder/ZIP theo [hướng dẫn chính thức](https://help.openai.com/en/articles/20001050-troubleshooting-prism).
OpenAI đã thông báo [ngừng Prism](https://help.openai.com/en/articles/20001615-prism-to-codex-migration-guide);
nếu không truy cập được dịch vụ, dùng Overleaf hoặc LaTeX local với cùng bộ nguồn.

Thư mục project có thể chỉnh riêng. Lệnh đóng gói sẽ từ chối ghi đè nếu phát hiện
file đã sửa hoặc file bổ sung. Chỉ dùng `--force` sau khi đã sao lưu các chỉnh sửa.
Các ZIP là output sinh lại được; thư mục nguồn project có thể đưa lên Git.

## Nguồn và giới hạn

E1–E3 dùng 13 cấu hình full/20 epoch; A1 dùng study CIFAR-100 full/25 epoch với
8 main +2 controls; A2 dùng 2 variant/20.000 updates. Hash bundle A1/A2 được kiểm
tra lại. Kiểm tra checkpoint và predictions là **bằng chứng đã ghi ở lần pack gốc**,
không phải kiểm tra lại những file đã bị loại khỏi bundle nhẹ.

E2 không có history CSV trong bundle: đường cong PNG và output notebook là bằng
chứng, không tạo CSV hoặc sai số attention không được log. E1–E3 không xuất macro-F1.
A1 giữ12 nhóm duplicate cross-split; chưa có bằng chứng đồng ý đổi dataset sau A1.1.
A2 recall còn thấp; KID subset std/IS split std không thay thế nhiều seed.

Cả ba báo cáo khai báo AI hỗ trợ biên soạn. Tác giả cần rà soát, hiểu và bảo vệ
nội dung trước khi nộp. **URL video và mã nhóm chính thức vẫn chưa được cung cấp**; PDF có nội dung không đồng nghĩa mọi thủ tục nộp bài đã hoàn tất.
