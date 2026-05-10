# Tối ưu hóa giá sàn trong quảng cáo hiển thị bằng Deep Reinforcement Learning

Dự án triển khai mô hình Deep Reinforcement Learning cho bài toán tối ưu hóa giá sàn (Reserve Price Optimization) trong hệ thống quảng cáo hiển thị thời gian thực (RTB - Real-Time Bidding).

Mô hình được xây dựng dựa trên ý tưởng từ bài báo:

> **Learning Adaptive Reserve Price in Display Advertising**  
> Kun Hu et al., KDD 2025

Mục tiêu của hệ thống:
- tối đa hóa doanh thu quảng cáo,
- duy trì fill-rate hợp lý,
- đồng thời kiểm soát rủi ro không đạt KPI doanh thu.

---

# 1. Tổng quan bài toán

Trong hệ thống RTB, mỗi impression quảng cáo sẽ kích hoạt một phiên đấu giá thời gian thực giữa các advertiser.

Publisher cần quyết định:
- nên đặt giá sàn bao nhiêu cho mỗi phiên đấu giá,
- dựa trên ngữ cảnh hiện tại của impression.

Hai vấn đề lớn xuất hiện:

- Nếu giá sàn quá cao:
  - phiên đấu giá dễ thất bại,
  - impression không được bán.

- Nếu giá sàn quá thấp:
  - impression được bán rẻ,
  - gây thất thoát doanh thu.

Dự án mô hình hóa bài toán này thành:
- một bài toán ra quyết định tuần tự (Sequential Decision Making),
- và giải bằng Deep Reinforcement Learning.

---

# 2. Môi trường mô phỏng RTB

Môi trường `OfflineRTBEnv` được xây dựng theo chuẩn Gymnasium để mô phỏng lại cơ chế đấu giá quảng cáo offline.

## State Space

Tại thời điểm \( t \), trạng thái được biểu diễn bởi:

\[
s_t = (t, b, x_t)
\]

Trong đó:

- \( t \): số phiên đấu giá còn lại,
- \( b \): doanh thu KPI còn thiếu,
- \( x_t \): vector đặc trưng ngữ cảnh của impression.

Các feature hiện tại gồm:

- hour,
- region,
- city,
- slot width,
- slot height,
- slot visibility,
- slot format.

---

## Action Space

Agent PPO sinh action liên tục:

\[
a_t \in [-1,1]
\]

Sau đó được ánh xạ thành reserve price thực:

\[
r_t \in [r_{\min}, r_{\max}]
\]

theo công thức:

\[
r_t =
r_{\min}
+
\frac{a_t + 1}{2}
(r_{\max} - r_{\min})
\]

Cơ chế này giúp agent học reserve price liên tục thay vì discrete pricing.

---

# 3. Hàm phần thưởng

Nếu phiên đấu giá thành công:

\[
reward_t =
\left(
\frac{revenue_t}{1000}
-
\frac{|bid_t - r_t|}{300}
\right)\beta_t
\]

Nếu phiên đấu giá thất bại:

\[
reward_t =
-1.5
\cdot
\frac{|bid_t - r_t|}{1000}
-
\frac{\zeta}{1000}
\]

Trong đó:

- \( bid_t \): historical winning bid,
- \( r_t \): reserve price,
- \( \beta_t \): hệ số quản trị rủi ro,
- \( \zeta \): penalty cho failed auction.

Reward được thiết kế để:
- tối đa hóa doanh thu,
- giảm chênh lệch giữa bid và reserve,
- tránh reserve price quá cao gây mất fill-rate.

---

# 4. Quản trị rủi ro động

Mô hình sử dụng hệ số beta động:

\[
\beta_t =
\frac{\kappa}
{
1 + v \tanh(\Delta_t)
}
\]

với:

\[
\Delta_t =
\frac{
U_{t,b} - \bar U
}
{
\bar U
}
\]

Trong đó:

\[
U_{t,b} = \frac{b}{t}
\]

và:
- \( b \): KPI doanh thu còn thiếu,
- \( t \): số phiên còn lại,
- \( \bar U \): doanh thu trung bình mục tiêu mỗi phiên.

Cơ chế này giúp:
- agent tự điều chỉnh mức độ “aggressive” của reserve price,
- ưu tiên fill-rate khi KPI còn thiếu nhiều,
- tránh mất toàn bộ impression ở cuối episode.

---

# 5. Mô phỏng đấu giá offline

Do chỉ có dữ liệu lịch sử, môi trường sử dụng replay-based simulation.

Logic đấu giá:

\[
success =
\begin{cases}
1 & \text{if } bid_t \ge r_t \\
0 & \text{otherwise}
\end{cases}
\]

Nếu thành công:
- publisher thu được reserve price.

Nếu thất bại:
- impression bị mất.

Environment reset tại vị trí ngẫu nhiên trong dataset nhằm:
- tăng tính đa dạng trạng thái,
- giảm overfitting.

---

# 6. Pipeline huấn luyện PPO

Mô hình được huấn luyện bằng:

- PPO (Proximal Policy Optimization),
- Stable-Baselines3,
- PyTorch backend.

## Hyperparameters chính

| Hyperparameter | Value |
|---|---|
| Learning Rate | \(1 \times 10^{-4}\) |
| n\_steps | 256 |
| batch\_size | 64 |
| n\_epochs | 10 |
| gamma | 0.99 |
| gae\_lambda | 0.95 |
| clip\_range | 0.2 |
| ent\_coef | 0.01 |

Policy network sử dụng:
- MLP Actor-Critic Architecture.

---

# 7. Dữ liệu
Bộ dữ liệu của iPinYou được public:
- Mỗi bản ghi chứa ba loại thông tin: 
    - Đặc trưng đấu giá và quảng cáo (tất cả các cột ngoại trừ 3, 20 và 21). Các đặc trưng này được gửi đến công cụ đấu thầu để đưa ra phản hồi thầu.
    - Giá thắng thầu (cột 21), tức là giá thầu cao nhất từ các đối thủ cạnh tranh. Nếu công cụ đấu thầu phản hồi một mức giá thầu cao hơn giá thắng đấu giá, DSP sẽ thắng cuộc đấu giá này và nhận được lượt hiển thị quảng cáo. 
    - Phản hồi của người dùng (nhấp chuột và chuyển đổi) trên lượt hiển thị quảng cáo (cột 3).

- Một dòng trong data = một lần đấu giá thắng quảng cáo mà một DSP tham gia, gồm context + giá bid của nó + giá phải trả + giá sàn + kết quả thắng thua.

  | Col # | Description              | Diễn giải |
|------|--------------------------|----------|
| 1    | Bid ID                  | Mã định danh duy nhất cho tất cả các nhật ký sự kiện; dùng để nối bid, impression, click, conversion |
| 2    | Timestamp               | Định dạng thời gian: yyyyMMddHHmmssSSS |
| 3    | Log type               | 1: hiển thị, 2: nhấp chuột, 3: chuyển đổi |
| 4    | iPinYou ID             | ID người dùng nội bộ do iPinYou thiết lập |
| 5    | User-Agent             | Mô tả thiết bị, hệ điều hành và trình duyệt của người dùng |
| 6    | IP                     | Địa chỉ IP của người dùng |
| 7    | Region                 | Mã khu vực |
| 8    | City                   | Mã thành phố |
| 9    | Ad exchange            | Sàn giao dịch quảng cáo |
| 10   | Domain                 | Tên miền trang web chứa quảng cáo (đã được băm) |
| 11   | URL                    | URL trang web chứa quảng cáo (đã được băm) |
| 12   | Anonymous URL ID       | Dùng khi URL không có sẵn; do ad exchange cung cấp |
| 13   | Ad slot ID             | ID vị trí quảng cáo |
| 14   | Ad slot width          | Chiều rộng vị trí quảng cáo |
| 15   | Ad slot height         | Chiều cao vị trí quảng cáo |
| 16   | Ad slot visibility     | Vị trí hiển thị: FirstView, SecondView... hoặc Na |
| 17   | Ad slot format         | Fixed, Pop, Background, Float hoặc Na |
| 18   | Ad slot floor price    | Giá sàn; bid thấp hơn sẽ không thắng; đã chuẩn hóa |
| 19   | Creative ID            | ID nội dung quảng cáo |
| 20   | Bidding price          | Giá thầu từ iPinYou |
| 21   | Paying price           | Giá thanh toán (giá thị trường/giá thắng) |
| 22   | Key page URL           | URL chính của trang (đã băm) |
| 23   | Advertiser ID          | ID nhà quảng cáo |
| 24   | User Tags              | Thẻ người dùng (segment), chỉ cung cấp một phần |

# 8. Một số kết quả bước đầu
# 10. Visualization

## PPO Training Reward

<div align="center">

### Figure 1 — PPO Training Reward Curve



# 9. Hạn chế hiện tại
- Tại các mức giá bid cao nhất (300), hành vi của agent không đặt theo giá mà lựa chọn đặt ngược lại - thấp hơn rất nhiều, dải đặt floor của agent khá đều, nên nghiên cứu phương pháp chia các nhóm giá trị quảng cáo.
