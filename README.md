# Đồ án 2: Thực nghiệm Chủ đề Phân hạng (Chapter 9: Ranking)

Thư mục này chứa toàn bộ mã nguồn cài đặt từ đầu (from scratch) và các thực nghiệm minh họa cho chủ đề **Ranking** thuộc môn học *Nhập môn Học máy* (CSC14005), trường Đại học Khoa học Tự nhiên - ĐHQG TP.HCM.

---

## 1. Môi trường cài đặt & Phiên bản thư viện

Các thực nghiệm được phát triển và kiểm tra tính ổn định trên hệ điều hành **Windows / Linux** với cấu hình môi trường đề xuất như sau:
* **Phiên bản Python**: `3.11.x` (hỗ trợ tối thiểu từ `3.8+`)
* **Thư viện cài đặt (trong `requirements.txt`)**:
  * `numpy==1.24.3` (hỗ trợ tính toán ma trận và đại số tuyến tính)
  * `scipy==1.11.4` (hỗ trợ các phép toán khoa học bổ trợ)
  * `matplotlib==3.8.2` (hỗ trợ vẽ đồ thị và trực quan hóa kết quả)
  * `pandas==2.1.3` (hỗ trợ lưu trữ bảng dữ liệu kết quả ra file CSV)

### Hướng dẫn cài đặt thư viện:
Mở terminal tại thư mục dự án và chạy lệnh sau để tự động cấu hình môi trường:
```bash
pip install -r requirements.txt
```

---

## 2. Cấu trúc thư mục mã nguồn

Mã nguồn dự án được tổ chức khoa học nhằm phân tách rõ ràng giữa thuật toán lõi, kịch bản thực nghiệm và kết quả đầu ra:

```text
lab2-intro2ML/
├── README.md                     # Hướng dẫn chạy và thông tin tái lập (file này)
├── requirements.txt              # Khai báo thư viện và phiên bản cụ thể
├── run_all_experiments.py        # Script chạy tự động toàn bộ 4 thực nghiệm
├── EQUATION_TO_CODE_MAPPING.md   # Mapping giữa phương trình trong sách và code
├── report/
│   └── sections/
│       └── 09_experiments.tex    # File nội dung báo cáo LaTeX phần thực nghiệm
├── src/                          # Thư mục mã nguồn cốt lõi (Huấn luyện từ đầu)
│   ├── __init__.py               # Package initializer
│   ├── data.py                   # Sinh dữ liệu nhân tạo bipartite và tạo các cặp
│   ├── metrics.py                # Các độ đo cài đặt từ đầu (AUC, ROC, Pairwise Error, Margin Loss)
│   ├── rankboost.py              # Thuật toán RankBoost & Soft Decision Stump
│   ├── rank_svm_sgd.py           # [MỞ RỘNG] Thuật toán RankSVM tối ưu bằng SGD
│   └── visualization.py          # Các hàm vẽ đồ thị trực quan hóa dữ liệu và kết quả
├── experiments/                  # Các kịch bản chạy thực nghiệm
│   ├── exp_01_pairwise_ranking_demo.py  # Thực nghiệm 1: Trực quan hóa dữ liệu & score contours
│   ├── exp_02_rankboost_demo.py         # Thực nghiệm 2: Đường cong huấn luyện RankBoost & Edge
│   ├── exp_03_auc_roc_demo.py           # Thực nghiệm 3: Dựng đường cong ROC và chỉ số AUC
│   └── exp_04_margin_vs_error.py        # Thực nghiệm 4: Đánh đổi giữa Margin rho và Margin Loss
└── outputs/                      # Thư mục lưu trữ kết quả đầu ra
    ├── figures/                  # Thư mục lưu trữ các hình ảnh đồ thị dạng PNG
    └── results_summary.csv       # Bảng tổng hợp số liệu của tất cả các thực nghiệm
```

---

## 3. Các thành phần chính cài đặt từ đầu (From Scratch)

Nhóm cam kết tuân thủ nghiêm ngặt yêu cầu không sử dụng các thư viện thuật toán có sẵn (như `scikit-learn` hay `libsvm`). Các thành phần sau đây hoàn toàn được tự tay lập trình:

1. **Thuật toán RankBoost (`src/rankboost.py`):**
   * Huấn luyện ensemble phân hạng dạng: $g(x) = \sum_t \alpha_t h_t(x)$.
   * Cập nhật phân phối trọng số cặp $D_{t+1}(i)$ bằng hàm mũ (exponential loss).
   * **[MỞ RỘNG] Soft Decision Stump:** Cải tiến bộ phân hạng yếu stump nhị phân $\{0, 1\}$ truyền thống bằng hàm Sigmoid liên tục để làm mịn biên phân chia và ổn định phân bổ trọng số:
     $$h_{soft}(x) = \frac{1}{1 + \exp\left(-\frac{x_j - \theta}{\tau}\right)}$$

2. **Thuật toán RankSVM (`src/rank_svm_sgd.py`):**
   * **[MỞ RỘNG]** Cài đặt thuật toán phân hạng SVM tối ưu hóa Hinge Loss bằng phương pháp hạ gradient ngẫu nhiên (SGD) trên hiệu các cặp điểm dữ liệu.

3. **Hệ thống độ đo phân hạng (`src/metrics.py`):**
   * *Pairwise Misranking Error:* Tỷ lệ các cặp bị phân hạng ngược.
   * *AUC (Area Under ROC Curve):* Tính toán xác suất xếp đúng thứ tự ngẫu nhiên bằng thuật toán từ đầu.
   * *ROC Curve:* Thu thập các điểm (FPR, TPR) thông qua việc duyệt ngưỡng threshold.
   * *Margin Loss ($L_\rho$):* Tính sai số vi phạm biên phân hạng dựa trên tham số biên $\rho$.

---

## 4. Hướng dẫn chạy các thực nghiệm

Bạn có thể chạy độc lập từng thực nghiệm hoặc chạy tất cả cùng lúc bằng script tự động.

### Cách 1: Chạy tự động toàn bộ 4 thực nghiệm (Khuyên dùng)
```bash
python run_all_experiments.py
```
Script này sẽ tuần tự thực thi cả 4 thực nghiệm, in ra các thông số đánh giá trên màn hình console, tự động lưu biểu đồ vào `outputs/figures/` và tổng hợp kết quả vào file `outputs/results_summary.csv`.

### Cách 2: Chạy độc lập từng thực nghiệm
```bash
# Thực nghiệm 1: Minh họa dữ liệu phân hạng 2D và Score Contours
python experiments/exp_01_pairwise_ranking_demo.py

# Thực nghiệm 2: Quá trình huấn luyện RankBoost (Error, Loss, Edge)
python experiments/exp_02_rankboost_demo.py

# Thực nghiệm 3: Dựng đường cong ROC và so sánh AUC với Random Baseline
python experiments/exp_03_auc_roc_demo.py

# Thực nghiệm 4: Phân tích sự ảnh hưởng của tham số biên margin rho
python experiments/exp_04_margin_vs_error.py
```

---

## 5. Kết quả thực nghiệm và Minh họa đầu ra

Sau khi chạy xong, các tệp tin sau sẽ được tạo ra tại thư mục `outputs/`:

* **`outputs/figures/pairwise_ranking_demo.png`:** Cho thấy phân bố dữ liệu thô và các đường đồng mức điểm số (score contours) vuông góc với vector trọng số thực tế.
* **`outputs/figures/rankboost_training_curves.png`:** Thể hiện sai số cặp và exponential loss giảm dần theo số vòng lặp $T$ (Kiểm chứng Định lý 9.2), đi kèm biểu đồ cột giá trị Edge $\gamma_t > 0$ của bộ học yếu ở mỗi vòng.
* **`outputs/figures/roc_auc_analysis.png`:** So sánh trực quan đường cong ROC và chỉ số AUC giữa mô hình học được (AUC $\approx 0.9789$) và mô hình ngẫu nhiên (AUC $\approx 0.50$).
* **`outputs/figures/margin_vs_error.png`:** Phân phối giá trị margin trên tập dữ liệu cặp (Mean margin $\approx 5.2586$) và sự đánh đổi biên margin loss $L_\rho$ khi tham số $\rho$ tăng dần từ $0.0$ đến $3.0$ (Kiểm chứng Hệ quả 9.2).
* **`outputs/results_summary.csv`:** Bảng lưu trữ đầy đủ các chỉ số đo đạc thu được từ thực nghiệm để kiểm chứng độ chính xác.

---

## 6. Tính tái lập kết quả (Reproducibility)

Để đảm bảo kết quả trùng khớp hoàn toàn với báo cáo LaTeX và file CSV:
* Giá trị **Random Seed = 42** đã được gán cố định cho các module sinh dữ liệu và huấn luyện.
* Không được thay đổi độ lệch chuẩn noise (`noise_std=0.5`) trong dữ liệu thô để duy trì đúng kết quả phân tách lớp.