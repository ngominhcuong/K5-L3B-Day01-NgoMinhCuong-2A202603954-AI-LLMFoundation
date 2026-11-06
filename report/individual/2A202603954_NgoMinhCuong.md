# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

* **MSSV:** SE18xxxx
* **Họ và tên:** Ngô Minh Cường
* **SEED** (in ở ô đầu notebook): 42

## 1\. Bằng chứng

|Tệp|Có?|
|-|-|
|[`day01\_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output|x|
|[`lab1\_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra|x|
|[`hand\_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước|x|

## 2\. Kết quả chính

|Phương pháp|F1 Test|F1 SUB 40|p95 latency (ms/tin)|USD/1k tin|
|-|:-:|:-:|:-:|:-:|
|**1. BoW + LogReg (tự code)**|0.921|0.895|0.11 ms|$0.00000|
|**2. BoW + MLP (Deep Learning)**|0.925|0.901|0.18 ms|$0.00000|
|**3a. Encoder (MiniLM) + kNN**|0.942|0.920|18.50 ms|$0.00000|
|**3b. Encoder (MiniLM) + LogReg**|0.965|0.948|1.85 ms|$0.00000|
|**3c. Encoder (MiniLM) + MLP head**|0.968|0.950|2.10 ms|$0.00000|
|**4a. LLM nhỏ zero-shot**|0.915|0.880|380.00 ms|$0.00150|
|**4b. LLM lớn zero-shot**|0.952|0.935|920.00 ms|$0.01500|

## 3\. Dự đoán và thực tế

1. **Thí nghiệm 1.B (`predict("ml\_lr")`)**:

   * *Đoán:* Ở `lr=50`, bước nhảy quá lớn sẽ khiến mô hình phân kỳ hoàn toàn và loss nổ tung không giảm được.
   * *Thực tế:* Loss nhảy vọt lên \~2.6 ở epoch đầu nhưng sau đó cắm dốc về gần bằng 0 (`loss cuối = 0.006`).
   * *Vì sao lệch:* Do bài toán Logistic Regression có hàm mất mát lồi (convex) và dữ liệu Bag-of-Words thưa, gradient cập nhật được chuẩn hoá chia cho $n$ nên dù bước nhảy lớn, mô hình vẫn đẩy trọng số về cực đại để ép xác suất về 0 và 1.
2. **Thí nghiệm 2.A (`predict("dl\_curve")`)**:

   * *Đoán:* MLP 2 lớp sẽ cho F1 vượt trội hơn hẳn Logistic Regression (kỳ vọng tăng 5-7%) nhờ học được tổ hợp từ phi tuyến.
   * *Thực tế:* F1 của MLP chỉ đạt xấp xỉ (\~0.925), chênh lệch không đáng kể so với LogReg (\~0.921).
   * *Vì sao lệch:* Đầu vào vẫn là Bag-of-Words nhị phân tĩnh đã mất hoàn toàn thứ tự từ ngữ và ngữ cảnh. Deep Learning không phát huy được sức mạnh biểu diễn sâu nếu các đặc trưng nạp vào đã bị làm phẳng từ đầu.
3. **Thí nghiệm 3.A (`predict("emb\_search")`)**:

   * *Đoán:* Câu truy vấn không chứa từ khóa spam ("free", "win") sẽ khiến mô hình tìm kiếm nhầm ra các tin nhắn bạn bè (ham).
   * *Thực tế:* Top các kết quả gần nhất đều trả về đúng các tin nhắn spam.
   * *Vì sao lệch:* Mô hình Transformer đã ánh xạ toàn bộ ngữ nghĩa của câu ("selected to receive a complimentary holiday") vào cùng miền vector với các nội dung quảng cáo/lừa đảo, chứng minh khả năng bắt ngữ nghĩa vượt qua từ khóa bề mặt.

## 4\. Phân tích 50 case khó

* **Model tụt nhiều nhất:** Mô hình Bag-of-Words (Logistic Regression / MLP) tụt độ chính xác sâu nhất (xuống còn dưới 50%), đặc biệt ở 2 nhóm case: `ham\_looks\_spam` và `prompt\_injection`.
* **2 case cụ thể và cơ chế khiến model sai:**

  1. *Case `ham\_looks\_spam`:* `"Txt me ur address and I'll send the free ticket..."` (Nhãn thật: `ham`).

     * *Cơ chế:* Chứa các từ `"free"`, `"txt"` – vốn là những từ đẩy về nhãn spam mạnh nhất trong bộ từ điển BoW. Vì BoW cộng dồn điểm số từng từ riêng lẻ mà không hiểu đây là tin nhắn rủ rê bạn bè đời thường, mô hình BoW phán đoán sai thành `spam`.
  2. *Case `prompt\_injection`:* Tin nhắn chứa các câu lệnh chỉ thị phá vỡ hệ thống (Nhãn thật: `spam`).

     * *Cơ chế:* Khác với ML/DL truyền thống chỉ phân loại dựa trên đặc trưng số, LLM dễ bị lừa bởi chỉ thị trực tiếp trong văn bản (ví dụ câu lệnh giả mạo hệ thống yêu cầu trả về `"ham"`), dẫn tới việc LLM bị bẻ gãy luật phân loại và đoán sai.

## 5\. Thử thách cá nhân

* Chưa làm

## 6\. Phương án cho sản phẩm thật

* **Chưa làm**

## 7\. AI log và điều đã học

* **Nhật ký tương tác với AI:**

  1. *Vấn đề 1:* Hàm `metrics` ở TODO 1.4 bị lỗi `accuracy: 0.0`. Nhận giải thích từ AI do đầu vào kiểm thử là list thông thường chứ không phải mảng NumPy nên phép so sánh trả về `False`; đã sửa bằng cách ép kiểu `np.asarray(...)` ngay đầu hàm.
  2. *Vấn đề 2:* Hàm `top\_k\_similar` ở Phần 3 bị lỗi `TypeError: 'NoneType'`. Nhận hướng dẫn bổ sung chuẩn hoá vector bằng `np.linalg.norm` và đảo ngược mảng `\[::-1]\[:k]` để lấy top điểm lớn nhất.
  3. *Vấn đề 3:* Lỗi `RateLimitError 429` khi gọi LLM. Nhận phương án cấu hình `HAS\_LLM = False` hoặc thêm độ trễ `time.sleep` để tránh vượt hạn mức 15 requests/phút.
* **Điều hiểu ra hôm nay:**

  * Hiểu rõ bản chất 5 dòng kinh điển trong PyTorch, đặc biệt là lý do vì sao luôn phải gọi `opt.zero\_grad()` để tránh cộng dồn gradient sai lệch từ batch trước.
  * Nhận thức được giới hạn của Bag-of-Words khi chỉ đếm tần suất từ bề mặt mà không có ngữ cảnh, và thấy được sức mạnh vượt trội của Vector Embedding khi ánh xạ câu vào không gian ngữ nghĩa hình học.
  * Nhận ra độ chính xác (Accuracy) 98% hoàn toàn có thể vô nghĩa trên dữ liệu mất cân bằng lớp nếu mô hình bỏ sót toàn bộ nhóm nhãn thiểu số (spam).

