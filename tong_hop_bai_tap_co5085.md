# ĐẠI HỌC BÁCH KHOA – ĐHQG TP.HCM
## KHOA KHOA HỌC VÀ KỸ THUẬT MÁY TÍNH
### MÔN HỌC: HỌC SÂU VÀ ỨNG DỤNG TRONG THỊ GIÁC MÁY TÍNH (CO5085)
**Học kỳ & Năm học:** Học kỳ 261 (2026–2027)  
**Giảng viên phụ trách:** ThS./TS. Lê Thành Sách, TS. Nguyễn Quốc Minh  

---

# TỔNG HỢP YÊU CẦU BÀI TẬP VÀ BÀI TẬP LỚN (E1–E3, A1, A2)

---

## I. QUY ĐỊNH CHUNG VÀ CƠ CẤU ĐÁNH GIÁ

### 1. Cơ cấu điểm phần Coursework (40% điểm học phần)
Phần bài tập quá trình (Coursework) chiếm tổng cộng **40% điểm tổng kết môn học**, được phân bổ chi tiết như sau:
- **Bài tập E1 – E3 (Bài tập thực hành):** Chiếm **40%** của phần Coursework (chia đều cho 3 bài E1, E2, E3).
- **Bài tập lớn A1 (CNN vs Transformer):** Chiếm **30%** của phần Coursework.
- **Bài tập lớn A2 (Chủ đề tự chọn + Paper Analysis):** Chiếm **30%** của phần Coursework.

### 2. Hình thức làm việc
- **Nhóm sinh viên:** Tối đa **02 sinh viên/nhóm** (đăng ký chính thức trên LMS).
- **Tính nhất quán:** Giữ nguyên nhóm cho toàn bộ chuỗi bài tập E1–E3, A1 và A2.
- Cho phép sinh viên làm việc độc lập (nhóm 1 người).

### 3. Quy định nộp bài & Nền tảng quản lý
- **Không nộp file trực tiếp lên LMS** cho cả bài tập E1–E3, A1 và A2. Mọi sản phẩm nộp thông qua **GitHub Repository** và **GitHub Pages**.
- **Quy định trễ hạn:** Trừ **20% điểm** của mốc/mục đó cho mỗi tuần nộp trễ (tối đa trễ 7 ngày; quá 7 ngày không nhận bài).
- **Hạn chót các mốc nộp:** Mặc định lúc **23:59 GMT+7** của ngày tương ứng. Giảng viên chấm điểm dựa trên commit/tag hợp lệ cuối cùng trước thời điểm đóng hạn.
- **Liêm chính học thuật:** Tuân thủ quy định của nhà trường. Công cụ AI chỉ được dùng hỗ trợ tra cứu, gợi ý cú pháp, sửa lỗi ngữ pháp và phải trích dẫn/khai báo rõ ràng. Tuyệt đối không nộp sản phẩm do AI sinh ra phần lớn; sinh viên phải hiểu sâu và bảo vệ được mã nguồn cùng toàn bộ số liệu thực nghiệm.

---

## II. LỊCH TRÌNH VÀ CÁC MỐC THỜI GIAN QUAN TRỌNG (TIMELINE)

| Tuần | Mốc thời gian | Hạng mục nộp | Nội dung chi tiết |
| :---: | :---: | :---: | :--- |
| **W1** | 07/09/2026 | Công bố đề bài | Phát hành đề E1–E3, A1, A2 |
| **W7** | 19/10/2026 | **GitHub Pages** | Hoàn thành khung trang nhóm (Landing Page) trên GitHub Pages |
| **W8** | 26/10/2026 | **Mốc A1.1** | Đăng ký & báo cáo Dataset + Protocol cho Bài tập lớn A1 |
| **W9** | 02/11/2026 | **Bài tập E1 & E2** | Nộp mã nguồn + Báo cáo PDF phần E1 và E2 |
| **W9** | 02/11/2026 | **Mốc A2.1** | Đăng ký Topic + Paper + Dataset + Metric cho Bài tập lớn A2 |
| **W10** | 09/11/2026 | **Bài tập E3** | Bổ sung phần E3 vào Báo cáo PDF chung + Mã nguồn E3 |
| **W10** | 09/11/2026 | **Mốc A1.2** | Báo cáo hoàn chỉnh A1, so sánh kết quả cuối, hoàn thiện code |
| **W11** | 16/11/2026 | **Mốc A2.2** | Hoàn tất Pipeline A2, phân tích Paper, hoàn thiện code & trang web |

---

## III. CHI TIẾT BÀI TẬP THỰC HÀNH E1 – E3

### 1. Quy định chung cho E1–E3
- **Tập dữ liệu:** Bắt buộc dùng **cùng một tập ảnh nhỏ** xuyên suốt cả 3 bài E1, E2, E3.
  - Các tập chuẩn: **MNIST** ($28 \times 28$, xám, 10 lớp), **Fashion-MNIST** ($28 \times 28$, 10 lớp), **CIFAR-10** ($32 \times 32$, màu, 10 lớp).
  - Tùy chọn tương đương: **KMNIST, SVHN, EMNIST** (phải giải trình lý do trong báo cáo). Không dùng tập quá lớn.
- **Quy chuẩn mã nguồn:** Tự viết vòng lặp huấn luyện chuẩn PyTorch:
  $$\text{Forward} \rightarrow \text{Loss} \rightarrow \text{backward}() \rightarrow \text{optimizer.step}()$$
  *Tuyệt đối không chỉ gọi API trừu tượng hóa mức cao (như PyTorch Lightning `trainer.fit()` hay Keras).*
- **Hình thức báo cáo:** **01 file PDF duy nhất** cho toàn bộ E1–E3 (gợi ý đường dẫn: `reports/exercises.pdf`). Cập nhật lũy tiến: Tuần 9 nộp E1 + E2; Tuần 10 cập nhật bổ sung E3.

---

### 2. Bài tập E1: Phân loại ảnh cơ bản (Softmax, MLP, CNN)
- **Hạn nộp:** 02/11/2026 (W9)
- **Mục tiêu:** Xây dựng, huấn luyện và so sánh 3 bộ phân loại cơ bản trên cùng một tập ảnh nhỏ.
- **Yêu cầu kỹ thuật bắt buộc:**
  1. **Softmax Classifier:** Làm phẳng ảnh (flatten) thành vector $1\text{D}$, qua phép biến đổi tuyến tính (Linear mapping) và Softmax.
  2. **MLP Classifier:** Mạng truyền thẳng nhiều tầng fully connected (tầng ẩn + hàm kích hoạt phi tuyến tính), đầu ra Softmax; đầu vào là vector flattened.
  3. **CNN Classifier:** Gồm các khối Convolution / Pooling (có thể kết hợp Batch Normalization) + classification head, kiến trúc tối ưu cho ảnh nhỏ.
  4. **Training loop:** Tự viết vòng lặp huấn luyện; theo dõi và đánh giá chi tiết trên tập Validation/Test.
  5. **Đánh giá & So sánh:**
     - So sánh Accuracy, số lượng tham số / dung lượng mô hình (model capacity).
     - Đồ thị học tập: Loss curve và Accuracy curve qua các epoch.
     - Phân tích lỗi (Error analysis): Ma trận nhầm lẫn (Confusion Matrix) và trực quan hóa một số mẫu dự đoán sai tiêu biểu.
- **Phạm vi lưu ý:** Không đưa Transformer/ViT, LSTM/GRU vào E1.

---

### 3. Bài tập E2: Transformer – Multi-Head Self-Attention (MSA) và Tokenization
- **Hạn nộp:** 02/11/2026 (W9)
- **Mục tiêu:** Hiện thực cơ chế Multi-Head Self-Attention từ các phép toán nền tảng, khảo sát các phương pháp trích xuất token (tokenization) trên ảnh nhỏ.
- **Yêu cầu kỹ thuật bắt buộc:**
  1. **MSA tự viết (From scratch):** Cài đặt từ các toán tử cơ bản (Linear layer, phép nhân ma trận / `torch.einsum`, Softmax tỉ lệ $\frac{QK^T}{\sqrt{d_k}}$, ghép head). Tuyệt đối không dùng `nn.MultiheadAttention` hay `nn.TransformerEncoder*` cho bản này.
  2. **MSA chuẩn PyTorch:** Sử dụng `nn.MultiheadAttention` (hoặc `TransformerEncoderLayer`) để đối chiếu, kiểm chứng độ chính xác và hiệu năng.
  3. **Mô hình hoàn chỉnh:** Tích hợp MSA vào mô hình phân loại ảnh nông (shallow encoder + classification head). Có thể bổ sung LayerNorm, FFN, Positional Encoding.
  4. **Chiến lược Tokenization:** Cài đặt **ít nhất 2 cách** (khuyến nghị 3 cách), ví dụ:
     - *Patch-based:* Chia lưới ô vuông (grid patch), có thể gắn kèm token `[CLS]`.
     - *Pixel / Hàng:* Mỗi hàng hoặc mỗi điểm ảnh đóng vai trò một token.
     - *CNN-stem:* Dùng CNN trích xuất feature map, mỗi vị trí không gian (spatial location) là một token.
     - *Biến thể khác:* Patch chồng lấn (overlapping), patch đa tỉ lệ, tokenize theo trục kênh $C$.
  5. **Thực nghiệm & So sánh:**
     - So sánh chi tiết: MSA tự viết vs. MSA PyTorch (trên cùng cơ chế tokenize).
     - So sánh chéo giữa các chiến lược Tokenization (độ chính xác, thời gian huấn luyện/suy luận, tài nguyên bộ nhớ).

---

### 4. Bài tập E3: LSTM / GRU trên chuỗi ảnh
- **Hạn nộp:** 09/11/2026 (W10)
- **Mục tiêu:** Mô hình hóa ảnh dưới dạng dữ liệu chuỗi tuần tự bằng mạng hồi quy (RNNs), so sánh hiệu quả với các baseline không hồi quy.
- **Yêu cầu kỹ thuật bắt buộc:**
  1. **Kiến trúc mô hình:** Cài đặt cả **LSTM** và **GRU** (được phép dùng `nn.LSTM` và `nn.GRU` của PyTorch).
  2. **Chuyển đổi ảnh thành chuỗi (Sequence Modeling):**
     - Cách tiếp cận: Mỗi hàng (hoặc cột) là một bước thời gian ($T$ time steps); hoặc chuỗi các image patch quét theo thứ tự hàng/cột.
     - Triển khai **ít nhất 2 cách biểu diễn chuỗi** hoặc so sánh chuyên sâu LSTM vs. GRU trên 1 cách biểu diễn.
  3. **So sánh với Baseline:** Đối chiếu trực tiếp với mô hình không hồi quy (MLP và/hoặc CNN từ E1) trên cùng split dữ liệu.
  4. **Đánh giá:** So sánh Accuracy, số lượng tham số, khả năng mô hình hóa sự phụ thuộc không gian/thời gian so với CNN/MLP.

---

## IV. BÀI TẬP LỚN A1: CNN VS TRANSFORMER TRÊN TẬP ẢNH LỚN

### 1. Mục tiêu & Định hướng
So sánh toàn diện giữa hai trường phái kiến trúc **CNN** và **Vision Transformer** trên một tập dữ liệu ảnh quy mô lớn, qua nhiều chế độ huấn luyện:
- Huấn luyện từ đầu (*train from-scratch*) vs. Tận dụng mô hình nạp sẵn trọng số (*pretrained*).
- Khảo sát các mức độ can thiệp trọng số: Đóng băng toàn bộ phần trích xuất đặc trưng (*freeze backbone*), đóng băng một phần (*partial freeze*), và tinh chỉnh toàn bộ (*full finetune*).

### 2. Yêu cầu về tập dữ liệu
- **Quy mô & Tính chất:**
  - Tập dữ liệu công khai, dùng cho mục đích học thuật: **ImageNet-subset, CIFAR-100, Tiny-ImageNet, Food-101, Stanford Cars, iNaturalist subset**, hoặc các bộ dữ liệu chất lượng cao tương đương từ Kaggle/Hugging Face.
  - **Số lượng nhãn:** Ưu tiên $\ge 10$ lớp.
  - **Kích thước dữ liệu:** Ưu tiên từ vài nghìn đến hàng chục nghìn ảnh trở lên.
  - *Tuyệt đối không dùng MNIST, Fashion-MNIST, CIFAR-10 làm tập chính.*
- **Tính minh bạch:** Khai báo rõ bản quyền (license), phân chia tập train/val/test, và lý do lựa chọn bộ dữ liệu.

### 3. Yêu cầu kỹ thuật & Thực nghiệm
Được phép sử dụng mô hình pretrained từ các thư viện chuẩn: `timm`, `torchvision`, `Hugging Face`. Nhóm cần thực hiện đầy đủ ma trận thí nghiệm:
1. **Kiến trúc đối chiếu:**
   - Ít nhất 01 mô hình CNN: ResNet, ConvNeXt, EfficientNet,...
   - Ít nhất 01 mô hình Transformer: ViT, DeiT, Swin Transformer,...
2. **From-scratch vs. Pretrained:** Thực nghiệm trên cùng một phân chia dữ liệu (tối thiểu trên 1 họ mô hình; khuyến khích làm trên cả hai).
3. **Ba chiến lược đóng băng (Freeze strategies) trên mô hình pretrained:**
   - *Freeze toàn bộ backbone:* Chỉ huấn luyện classification head mới.
   - *Freeze một phần:* Đóng băng các khối đầu, chỉ mở các block cuối của backbone + head.
   - *Full Finetune:* Cập nhật trọng số của toàn bộ mô hình với learning rate phù hợp.
4. **Chỉ số đánh giá (Metrics):**
   - Độ chính xác (Accuracy), F1-score trên tập Validation và Test.
   - Thời gian huấn luyện/epoch, thời gian suy luận (latency), dung lượng bộ nhớ GPU, số lượng trainable parameters.
   - Phân tích lỗi chi tiết (Confusion Matrix, trực quan hóa failure cases). Nhận xét rõ khi nào cơ chế chú ý (attention) phát huy tác dụng, khi nào ViT phụ thuộc vào pretraining dữ liệu lớn.

### 4. Các mốc đánh giá & Tiêu chí chấm điểm (Chiếm 30% Coursework)
- **Mốc A1.1 (25% điểm A1) – Hạn: 26/10/2026 (W8):**
  - Khóa Dataset, mô tả chi tiết quy trình chuẩn bị dữ liệu (Data Augmentation, Train/Val/Test Split).
  - Bản thiết kế kế hoạch thực nghiệm (Danh sách CNN/Transformer cụ thể, kế hoạch from-scratch / freeze / finetune).
- **Mốc A1.2 (75% điểm A1) – Hạn: 09/11/2026 (W10):**
  - Hoàn tất toàn bộ các thực nghiệm theo yêu cầu kỹ thuật.
  - Bảng tổng hợp số liệu, đồ thị minh họa, phân tích thảo luận chuyên sâu.
  - Mã nguồn hoàn chỉnh, cấu trúc sạch sẽ, có tài liệu hướng dẫn tái lập (`README.md`).

---

## V. BÀI TẬP LỚN A2: CHỦ ĐỀ TỰ CHỌN + PHÂN TÍCH PAPER

### 1. Mục tiêu & Định hướng
Mở rộng kiến thức sang các bài toán thị giác máy tính nâng cao (**không làm bài toán phân loại ảnh thuần túy như A1**). Mỗi nhóm thực hiện 2 nhiệm vụ song hành:
1. Xây dựng một **pipeline thực thi hoàn chỉnh** (huấn luyện/fine-tune + đánh giá) so sánh ít nhất 2 giải pháp.
2. **Phân tích một bài báo khoa học đỉnh cao (Paper Analysis)** liên quan trực tiếp đến đề tài, kết nối lý thuyết của bài báo với thực nghiệm của nhóm.

### 2. Các chủ đề gợi ý, Dữ liệu và Metric tương ứng
Nhóm chọn 1 trong các hướng chuyên sâu sau (nếu chọn chủ đề ngoài danh mục phải được giảng viên phê duyệt trước mốc A2.1):

| Hướng nghiên cứu | Kiến trúc / Mô hình gợi ý | Tập dữ liệu đề xuất | Metric đánh giá cốt lõi |
| :--- | :--- | :--- | :--- |
| **Phân đoạn ảnh (Segmentation)** *(Semantic / Instance / Panoptic)* | U-Net, FCN, DeepLab (v3/v3+), PSPNet, SegFormer, Mask R-CNN; so sánh backbone CNN vs. ViT | Pascal VOC, COCO-Stuff, ADE20K, Cityscapes, CamVid | mIoU, Dice Score |
| **Phát hiện đối tượng (Object Detection)** | YOLO (các phiên bản), Faster R-CNN, DETR; CNN-based vs. Transformer-based detector | COCO, Pascal VOC, Open Images (subset), KITTI | mAP@0.5, mAP@0.5:0.95, FPS |
| **Định danh / Nhận diện (Identification / ReID)** | ArcFace, CosFace, mạng trích xuất đặc trưng metric learning; Person/Vehicle ReID | LFW, CelebA, Market-1501, MSMT17 | Rank-1 / Recall@k, mAP, Verification Accuracy |
| **Tạo sinh hình ảnh (Generative Models)** | So sánh GAN vs. VAE vs. Diffusion Models (hoặc 2 biến thể trong cùng một họ) | CelebA, LSUN, COCO Captions (chỉ dùng MNIST/CIFAR nếu paper yêu cầu) | FID (Fréchet Inception Distance), IS (Inception Score) |

### 3. Yêu cầu phân tích Paper khoa học (Bắt buộc)
Nhóm chọn 01 bài báo uy tín (ưu tiên các hội nghị hàng đầu: **CVPR, ICCV, ECCV, NeurIPS, ICLR**). Nội dung báo cáo phải tự diễn giải (tuyệt đối không dịch hoặc chép lại Abstract):
1. **Phương pháp đề xuất:** Phân tích ý tưởng cốt lõi, sơ đồ kiến trúc, hàm mất mát (loss formulation), các giả thiết bài toán.
2. **Thực nghiệm của bài báo:** Phân tích bộ dữ liệu tác giả dùng, các baseline đối chiếu, metric và các phát hiện quan trọng.
3. **Phân tích hạn chế (Limitations):** Chi phí tính toán, độ nhạy siêu tham số, các failure cases, sự không tương thích của giả thiết khi áp dụng vào bài toán thực tế của nhóm.
4. **Đối chiếu thực nghiệm:** Phân tích rõ pipeline của nhóm đã tái hiện/áp dụng được những gì từ paper, điểm khác biệt là gì và nhóm rút ra được bài học kỹ thuật nào.

### 4. Các mốc đánh giá & Tiêu chí chấm điểm (Chiếm 30% Coursework)
- **Mốc A2.1 (25% điểm A2) – Hạn: 02/11/2026 (W9):**
  - Chốt đề tài nghiên cứu; cung cấp citation và đường dẫn paper.
  - Mô tả tập dữ liệu, phương pháp tiền xử lý và các metrics lựa chọn.
  - Bản thảo thiết kế pipeline và kế hoạch chạy thử nghiệm.
- **Mốc A2.2 (75% điểm A2) – Hạn: 16/11/2026 (W11):**
  - Pipeline chạy trơn tru; hoàn thành so sánh $\ge 2$ mô hình/biến thể (kiến trúc, backbone, pretrained vs scratch, loss function).
  - Báo cáo phân tích bài báo chất lượng cao gắn liền với kết quả thực nghiệm.
  - *Điểm cộng khuyến khích (Bonus):* Trực quan hóa giải thích mô hình (Grad-CAM, Attention Map), web/app demo tương tác, kiểm thử cross-dataset, phân tích triệt để (ablation study).

---

## VI. QUY CHUẨN NỘP BÀI: GITHUB REPO & GITHUB PAGES

### 1. Cấu trúc chuẩn của GitHub Repository
Toàn bộ mã nguồn và báo cáo được lưu trữ tập trung trong 01 repository duy nhất của nhóm với cấu trúc thư mục đề xuất:

```text
├── E1/                     # Mã nguồn bài tập E1
├── E2/                     # Mã nguồn bài tập E2
├── E3/                     # Mã nguồn bài tập E3
├── A1/                     # Mã nguồn Bài tập lớn A1 (DataLoader, Model, Training script)
├── A2/                     # Mã nguồn Bài tập lớn A2 (Pipeline, Inference, Demo)
├── reports/
│   ├── exercises.pdf       # 01 Báo cáo PDF duy nhất cho E1-E3 (cập nhật dần)
│   ├── assignment1.pdf     # Báo cáo PDF hoàn chỉnh cho A1
│   └── assignment2.pdf     # Báo cáo PDF hoàn chỉnh cho A2
├── index.html / docs/      # Mã nguồn trang GitHub Pages (Landing Page)
└── README.md               # Hướng dẫn thiết lập môi trường, lệnh chạy (Reproducibility)
```

### 2. Thiết kế GitHub Pages (Landing Page)
- **Hạn hoàn thiện khung trang:** **19/10/2026 (W7)**.
- **Trang chủ (Main Landing Page):**
  - Tên nhóm, danh sách thành viên, MSSV, giảng viên hướng dẫn.
  - Menu điều hướng liên kết trực tiếp tới các mục: Bài tập E1–E3, Bài tập lớn A1, Bài tập lớn A2.
- **Trang chi tiết Bài tập lớn A1:**
  - Mô tả bài toán, tập dữ liệu đã sử dụng (kèm EDA - Exploratory Data Analysis).
  - Pipeline xử lý, kỹ thuật tăng cường dữ liệu (Data Augmentation).
  - Bảng so sánh kết quả chi tiết, biểu đồ trực quan, phân tích trường hợp lỗi.
  - Liên kết tải mã nguồn (thư mục `A1/`) và báo cáo `assignment1.pdf`.
  - *(Khuyến nghị)* Video demo ngắn.
- **Trang chi tiết Bài tập lớn A2:**
  - Tên đề tài, bài toán cụ thể và bài báo phân tích (kèm trích dẫn và liên kết paper gốc).
  - Phân tích chi tiết phương pháp trong paper và bài học rút ra.
  - Chi tiết kiến trúc pipeline, so sánh các biến thể và kết quả đo lường (mIoU, mAP, FID,...).
  - Liên kết mã nguồn (`A2/`), báo cáo `assignment2.pdf`.
  - **Liên kết Video Demo & Video thuyết trình:** Giảng viên có thể xem video để chấm điểm nếu chưa tổ chức báo cáo trực tiếp.

---

## VII. CHECKLIST HÀNH ĐỘNG DÀNH CHO NHÓM SINH VIÊN

- [ ] **Tuần 1–7 (trước 19/10/2026):**
  - [ ] Thành lập nhóm (tối đa 2 SV), đăng ký chính thức trên LMS.
  - [ ] Khởi tạo GitHub repo, dựng khung trang GitHub Pages hoàn chỉnh.
  - [ ] Chọn 1 tập ảnh nhỏ (MNIST / Fashion-MNIST / CIFAR-10) để sẵn sàng làm E1–E3.
- [ ] **Tuần 8 (trước 26/10/2026):**
  - [ ] Chọn tập ảnh lớn cho A1, thiết kế protocol, nộp mốc **A1.1**.
- [ ] **Tuần 9 (trước 02/11/2026):**
  - [ ] Hoàn thành code và huấn luyện E1 (Softmax, MLP, CNN).
  - [ ] Hoàn thành code MSA tự viết vs PyTorch và các cách tokenize cho E2.
  - [ ] Nộp code E1, E2 và file PDF `reports/exercises.pdf` (chương E1 và E2).
  - [ ] Chốt chủ đề, paper và dataset cho A2, nộp mốc **A2.1**.
- [ ] **Tuần 10 (trước 09/11/2026):**
  - [ ] Hoàn thành mô hình LSTM/GRU trên ảnh, viết chương E3 vào file PDF `reports/exercises.pdf`, hoàn tất toàn bộ **E1–E3**.
  - [ ] Chạy xong ma trận thí nghiệm A1 (from-scratch vs pretrained, 3 kiểu freeze), nộp code, file `assignment1.pdf`, cập nhật trang A1, hoàn tất **A1.2**.
- [ ] **Tuần 11 (trước 16/11/2026):**
  - [ ] Hoàn tất pipeline A2 (so sánh $\ge 2$ biến thể), hoàn thành bài phân tích Paper.
  - [ ] Quay video demo/thuyết trình cho A2, nộp mã nguồn, file `assignment2.pdf`, cập nhật trang A2 trên GitHub Pages, hoàn tất toàn bộ **A2.2**.