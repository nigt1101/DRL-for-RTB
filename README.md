# RL-armRP: Tối ưu hóa giá sàn trong quảng cáo hiển thị bằng Reinforcement Learning

Dự án này triển khai thuật toán RL-armRP dựa trên bài báo tại hội nghị KDD 2025 với mục tiêu học cách đặt giá sàn thích ứng trong hệ thống quảng cáo hiển thị. Mục tiêu là giúp publisher tối đa hóa doanh thu đồng thời kiểm soát rủi ro về doanh số.

## 1. Mô tả bài toán: tối ưu hóa giá sàn

Trong hệ thống đấu giá thời gian thực, mỗi khi người dùng truy cập trang web, một phiên đấu giá sẽ diễn ra ngay lập tức.

Cơ chế đấu giá sử dụng đấu giá giá thứ hai. Người thắng cuộc sẽ trả mức giá bằng giá thầu cao thứ hai hoặc giá sàn nếu giá sàn cao hơn.

Bài toán đặt ra hai tình huống khó:

- Nếu giá sàn quá cao thì phiên đấu giá dễ thất bại, dẫn đến không bán được lượt hiển thị
- Nếu giá sàn quá thấp thì publisher có thể bị mất doanh thu vì giá thị trường cao hơn nhiều

Giải pháp là sử dụng học tăng cường sâu để học chiến lược đặt giá sàn thích ứng dựa trên ngữ cảnh của phiên đấu giá và mục tiêu doanh thu.

## 2. Các cơ chế trong môi trường

Môi trường OfflineRTBEnv được xây dựng theo chuẩn Gymnasium để mô phỏng lại quá trình đấu giá.

### Cấu trúc trạng thái

Trạng thái tại thời điểm t bao gồm ba thành phần:

- t: số phiên đấu giá còn lại trong một chu kỳ
- b: doanh thu mục tiêu còn thiếu
- x_t: đặc trưng ngữ cảnh của phiên hiện tại như người dùng, địa điểm, thời gian và thiết bị

### Quản trị rủi ro

Hệ số beta được sử dụng để điều chỉnh phần thưởng.

Khi số phiên còn lại ít nhưng doanh thu còn thiếu nhiều, hệ số này sẽ khiến tác nhân ưu tiên bán được hàng thay vì giữ giá cao.

### Logic đối sánh giả định

Do sử dụng dữ liệu lịch sử, môi trường cần xử lý hai trường hợp:

- Dữ liệu thành công: so sánh giá sàn mới với giá thầu cao nhất trong lịch sử để xác định thắng hay thua
- Dữ liệu bị che: xử lý các trường hợp không có giá thầu vượt qua giá sàn cũ, giúp mô hình học được giới hạn thị trường

## 3. Mô phỏng dữ liệu và huấn luyện

### Giả lập dữ liệu

Hàm generate_rtb_history_data tạo ra 5000 dòng dữ liệu với các đặc điểm:

- Bao gồm giờ trong ngày, điểm địa điểm, điểm người dùng và loại thiết bị
- Giá thầu có tương quan với ngữ cảnh, ví dụ thiết bị PC và giờ cao điểm sẽ có giá cao hơn
- Sử dụng phân phối log normal để mô phỏng hành vi đặt giá

### Huấn luyện mô hình

Mô hình được huấn luyện bằng thuật toán PPO từ thư viện Stable Baselines 3.

Các tham số chính:

- Learning rate 1e-4 giúp mô hình hội tụ ổn định
- Entropy coefficient 0.01 giúp tăng khả năng khám phá
- Mạng neural dạng MLP phù hợp với dữ liệu dạng bảng

## 4. Kết quả thực nghiệm

Sau 20000 bước huấn luyện trên dữ liệu giả lập, mô hình đạt được:

- Average revenue khoảng 48.8657 cho mỗi phiên
- Deal rate đạt khoảng 86 phần trăm

Kết quả cho thấy mô hình học được cách cân bằng giữa việc tối đa hóa doanh thu và đảm bảo tỷ lệ khớp lệnh. Mô hình có xu hướng tăng giá sàn ở các phiên có giá trị cao và giảm giá sàn ở các phiên có giá trị thấp để duy trì hiệu quả tổng thể.
