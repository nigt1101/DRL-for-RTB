import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class OfflineRTBEnv(gym.Env):
    def __init__(self, 
                 dataframe, 
                 target_revenue=1000, 
                 max_auctions_per_episode=100, #T
                 kappa=1.0, 
                 v=0.5, 
                 zeta=1.0, 
                 esp=1e-2): # sai số khi tính giá sàn mới dựa trên giá bid cũ
        
        super(OfflineRTBEnv, self).__init__()
        
        self.df = dataframe
        self.total_rows = len(dataframe)
        self.T = max_auctions_per_episode
        self.B = target_revenue
        self.kappa = kappa
        self.v = v
        self.zeta = zeta
        self.U_bar = target_revenue / self.T # Giả định episode dài T bước
        self.esp = esp
        
        # Feature dim dựa trên dữ liệu thực
        self.feature_dim = len(self.df.iloc[0]['features'])
        
        # Action và Observation space giữ nguyên như bản trước
        self.action_space = spaces.Box(low=0, high=150, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0, 0] + [-np.inf] * self.feature_dim, dtype=np.float32),
            high=np.array([self.T, self.B] + [np.inf] * self.feature_dim, dtype=np.float32)
        )

        self.current_idx = 0

    def reset(self, seed=None):
        super().reset(seed=seed)
        # Bắt đầu tại một điểm ngẫu nhiên trong data
        self.current_idx = np.random.randint(0, self.total_rows - self.T)
        self.steps_left = self.T
        self.unrealized_revenue = float(self.B)
        
        row = self.df.iloc[self.current_idx]
        x_t = np.array(row['features'], dtype=np.float32)
        
        self.state = np.concatenate([[self.steps_left, self.unrealized_revenue], x_t])
        return self.state, {}

    def _calculate_risk_factor(self, t, b):
        """
        Triển khai Equation (10): Risk-aware factor beta(t, b)
        """
        if t <= 0: return 1.0
        U_tb = b / t # Remaining expected average auction revenue
        
        # Beta: tính toán theo công thức đã cho, với U_bar là giá trị trung bình 
        diff = (U_tb - self.U_bar) / (self.U_bar + 1e-6)
        beta = self.kappa / (1 + self.v * np.tanh(diff))
        return beta
    
    def step(self, action):
        a_t = action[0] # Giá sàn mới do Agent đặt
        row = self.df.iloc[self.current_idx]    
        
        # Lấy giá bid từ dữ liệu lịch sử
        # Chú ý: Nếu bid < original_reserve, bid_1 thường là NaN hoặc 0
        h_bid_1 = row['bid_1']
        h_bid_2 = row['bid_2'] if not np.isnan(row['bid_2']) else 0
        
        # --- Logic Đối sánh Giả định 
        revenue = 0
        success = False
        
        # Trường hợp 1: Dữ liệu cũ là thắng (có bid_1)
        if not np.isnan(h_bid_1):
            if a_t <= h_bid_1:
                # Agent đặt giá sàn mới vẫn thấp hơn người cao nhất -> Thắng!
                # Doanh thu mới = max(giá người thứ hai cũ, giá sàn mới)
                revenue = max(h_bid_2 + self.esp, a_t)
                success = True
            else:
                # Agent đặt giá sàn quá cao so với giá người ta đã trả -> Thất bại!
                revenue = 0
                success = False
        # Trường hợp 2: Dữ liệu cũ đã là thất bại (bid_1 < original_reserve)
        else:
            # Chúng ta không biết bid thực sự là bao nhiêu, nhưng chắc chắn nó < original_reserve
            # Để an toàn, Agent đặt giá sàn mới cao hơn hoặc bằng giá cũ thì chắc chắn vẫn trượt
            revenue = 0
            success = False

        # --- Reward Shaping (Theo Section 4.3 & Eq 9-10) ---
        # Sử dụng h_bid_1 (nếu có) để phân loại Value Bucket
        # Nếu h_bid_1 là NaN, coi như thuộc Low-value bucket
        beta = self._calculate_risk_factor(self.steps_left, self.unrealized_revenue)
        
        if success:
            reward = revenue * beta
            self.unrealized_revenue = max(0, self.unrealized_revenue - revenue)
        else:
            reward = -self.zeta # Phạt vì không bán được hàng

        # Cập nhật trạng thái
        self.current_idx += 1
        self.steps_left -= 1
        
        next_row = self.df.iloc[self.current_idx]
        next_x_t = np.array(next_row['features'], dtype=np.float32)
        self.state = np.concatenate([[self.steps_left, self.unrealized_revenue], next_x_t])
        
        done = self.steps_left <= 0
        return self.state, float(reward), done, False, {"real_bid": h_bid_1, "revenue": revenue}
    
