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

Tại thời điểm $t$, trạng thái được biểu diễn bởi:

$$
s_t = (t, b, x_t)
$$

Trong đó:

- $t$: số phiên đấu giá còn lại
- $b$: doanh thu KPI còn thiếu
- $x_t$: vector đặc trưng ngữ cảnh của impression

Các feature hiện tại gồm:

- hour
- region
- city
- slot width
- slot height
- slot visibility
- slot format

---

## Action Space

Agent PPO sinh action liên tục:

$$
a_t \in [-1,1]
$$

Sau đó được ánh xạ thành reserve price thực:

$$
r_t \in [r_{min}, r_{max}]
$$

theo công thức:

$$
r_t =
r_{min}
+
\frac{a_t + 1}{2}
(r_{max} - r_{min})
$$

Cơ chế này giúp agent học reserve price liên tục thay vì discrete pricing.

---

# 3. Hàm phần thưởng

Nếu phiên đấu giá thành công:

$$
reward_t =
\left(
\frac{revenue_t}{1000}
-
\frac{|bid_t - r_t|}{300}
\right)\beta_t
$$

Nếu phiên đấu giá thất bại:

$$
reward_t =
-1.5
\cdot
\frac{|bid_t - r_t|}{1000}
-
\frac{\zeta}{1000}
$$

Trong đó:

- $bid_t$: historical winning bid
- $r_t$: reserve price
- $\beta_t$: hệ số quản trị rủi ro
- $\zeta$: penalty cho failed auction

Reward được thiết kế để:
- tối đa hóa doanh thu
- giảm chênh lệch giữa bid và reserve
- tránh reserve price quá cao gây mất fill-rate

---

# 4. Quản trị rủi ro động

Mô hình sử dụng hệ số beta động:

$$
\beta_t =
\frac{\kappa}
{
1 + v \tanh(\Delta_t)
}
$$

với:

$$
\Delta_t =
\frac{
U_{t,b} - \bar U
}
{
\bar U
}
$$

Trong đó:

$$
U_{t,b} = \frac{b}{t}
$$

và:
- $b$: KPI doanh thu còn thiếu
- $t$: số phiên còn lại
- $\bar U$: doanh thu trung bình mục tiêu mỗi phiên

Cơ chế này giúp:
- agent tự điều chỉnh mức độ aggressive của reserve price
- ưu tiên fill-rate khi KPI còn thiếu nhiều
- tránh mất toàn bộ impression ở cuối episode

---

# 5. Mô phỏng đấu giá offline

Do chỉ có dữ liệu lịch sử, môi trường sử dụng replay-based simulation.

Logic đấu giá:

$$
success =
\begin{cases}
1 & \text{if } bid_t \ge r_t \\
0 & \text{otherwise}
\end{cases}
$$

Nếu thành công:
- publisher thu được reserve price

Nếu thất bại:
- impression bị mất

Environment reset tại vị trí ngẫu nhiên trong dataset nhằm:
- tăng tính đa dạng trạng thái
- giảm overfitting

---

# 6. Pipeline huấn luyện PPO

Mô hình được huấn luyện bằng:

- PPO (Proximal Policy Optimization)
- Stable-Baselines3
- PyTorch backend

## Hyperparameters chính

| Hyperparameter | Value |
|---|---|
| Learning Rate | $1 \times 10^{-4}$ |
| n_steps | 256 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.01 |

Policy network sử dụng:
- MLP Actor-Critic Architecture

---

# 7. Dữ liệu

Bộ dữ liệu iPinYou được public và sử dụng để mô phỏng môi trường đấu giá quảng cáo.

Mỗi bản ghi chứa ba loại thông tin:

- Đặc trưng đấu giá và quảng cáo  
  (tất cả các cột ngoại trừ 3, 20 và 21)

- Giá thắng thầu (cột 21)  
  tức là giá thầu cao nhất từ các đối thủ cạnh tranh

- Phản hồi của người dùng  
  (click và conversion)

Một dòng dữ liệu tương ứng với:
- một lần đấu giá quảng cáo mà DSP tham gia,
- bao gồm context + bidding price + paying price + floor price + kết quả đấu giá.

| Col # | Description | Diễn giải |
|---|---|---|
| 1 | Bid ID | Mã định danh duy nhất cho các sự kiện đấu giá |
| 2 | Timestamp | Định dạng thời gian yyyyMMddHHmmssSSS |
| 3 | Log type | 1: impression, 2: click, 3: conversion |
| 4 | iPinYou ID | ID người dùng nội bộ |
| 5 | User-Agent | Thông tin thiết bị và trình duyệt |
| 6 | IP | Địa chỉ IP |
| 7 | Region | Mã khu vực |
| 8 | City | Mã thành phố |
| 9 | Ad exchange | Sàn giao dịch quảng cáo |
| 10 | Domain | Domain website |
| 11 | URL | URL website |
| 12 | Anonymous URL ID | URL ID ẩn danh |
| 13 | Ad slot ID | ID vị trí quảng cáo |
| 14 | Ad slot width | Chiều rộng slot |
| 15 | Ad slot height | Chiều cao slot |
| 16 | Ad slot visibility | Mức độ hiển thị |
| 17 | Ad slot format | Loại định dạng quảng cáo |
| 18 | Ad slot floor price | Giá sàn |
| 19 | Creative ID | ID creative |
| 20 | Bidding price | Giá bid từ DSP |
| 21 | Paying price | Giá thanh toán thực tế |
| 22 | Key page URL | URL chính |
| 23 | Advertiser ID | ID advertiser |
| 24 | User Tags | User segments |

---

# 8. Một số kết quả bước đầu

Trong quá trình huấn luyện, agent dần học được cách điều chỉnh reserve price theo context thay vì đặt giá cố định.

Một số hành vi đáng chú ý:

- Reserve price thường cao hơn tại:
  - peak-hour traffic
  - slot visibility tốt
  - context có market bid cao

- Khi KPI còn thiếu nhiều nhưng số phiên còn lại ít:
  - agent có xu hướng giảm reserve price
  - ưu tiên fill-rate thay vì giữ giá quá cao

Gap penalty:

$$
|bid_t - r_t|
$$

giúp:
- giảm reserve price quá cực đoan
- làm reserve trajectory ổn định hơn
- hạn chế oscillation của policy

Ngoài ra:
- entropy giảm dần trong quá trình train
- reward hội tụ ổn định hơn theo thời gian
- policy update tương đối ổn định

## PPO Training Reward

<div align="center">

<img src="Screenshot%202026-05-10%20211229.png" width="750"/>

<br>
<b>Figure 1.</b> So sánh giữa giá bid thực tế và giá sàn agent đặt

</div>

---

## Revenue vs Fill-rate

<div align="center">

<img src="Screenshot%202026-05-10%20211259.png" width="750"/>

<br>
<b>Figure 2.</b> Phân phối của giá bid thực tế và giá sàn agent đặt

</div>

---

## Adaptive Beta Factor

<div align="center">

<img src="Screenshot%202026-05-10%20211308.png" width="750"/>

<br>
<b>Figure 3.</b> Phân phối gap giữa giá sàn và giá bid thực tế

</div>

---

# 10. Hạn chế hiện tại

- Environment vẫn là offline replay-based simulation
- Chưa mô phỏng bidder dynamics đầy đủ
- Chưa xử lý hoàn toàn censored auction feedback
- Chưa có online deployment evaluation

Ngoài ra:
- tại các mức bid rất cao (ví dụ quanh 300), agent đôi khi không lựa chọn reserve price tương ứng mà có xu hướng đặt thấp hơn đáng kể
- reserve distribution hiện vẫn còn khá đồng đều giữa nhiều context khác nhau

Các hướng cải thiện tiếp theo:
- inventory value segmentation
- context clustering
- Transformer-based state encoder
- SAC / TD3 comparison
- Distributional Reinforcement Learning
