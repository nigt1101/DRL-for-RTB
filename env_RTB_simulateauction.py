import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

class RTBReservePriceEnv(gym.Env):
    """
    Mô phỏng môi trường tối ưu Reserve Price theo bài báo 'Learning Adaptive Reserve Price in Display Advertising' (KDD '25).
    """
    def __init__(self, 
                 max_auctions_per_episode=100, #T
                 target_revenue=500, #B
                 feature_dim=10,#x_t
                 kappa=1.0, 
                 v=0.5, 
                 penalty_zeta=1.0):
        super(RTBReservePriceEnv, self).__init__()

        # --- Hyperparameters từ Paper ---
        self.T_max = max_auctions_per_episode
        self.B_target = target_revenue # Doanh thu kỳ vọng (Expected Revenue)
        self.feature_dim = feature_dim
        self.kappa = kappa             # Tham số dương cho Risk Factor (Eq 10)
        self.v = v                     # Tham số dương cho Risk Factor (Eq 10)
        self.zeta = penalty_zeta       # Penalty cho đấu thầu thất bại (Eq 9)
        self.U_bar = self.B_target / self.T_max # Doanh thu trung bình mục tiêu mỗi round

        # --- Action Space ---
        # Action là reserve price a_t. Giả sử chuẩn hóa trong [0, 100]
        self.action_space = spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)

        # --- Observation Space ---
        # State s = (t, b, x_t) theo Section 3.1
        # t: Số phiên đấu thầu còn lại
        # b: Doanh thu chưa đạt được (unrealized revenue)
        # x_t: Feature vector của ad inventory hiện tại
        low = np.array([0, 0] + [-np.inf] * self.feature_dim, dtype=np.float32)
        high = np.array([self.T_max, self.B_target] + [np.inf] * self.feature_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Khởi tạo trạng thái
        self.state = None
        self.steps_left = 0
        self.unrealized_revenue = 0

    def _get_multi_dsp_bids(self, x_t, num_dsps=3):
        """
        Mô phỏng nhiều DSP cùng tham gia đấu thầu (Section 5.3).
        """
        bids = []
        intrinsic_value = np.mean(x_t) * 50 + 20
        
        for i in range(num_dsps):
            # Mỗi DSP có một mức độ 'máu chiến' (aggressiveness) khác nhau
            aggressiveness = np.random.uniform(0.8, 1.2)
            # Tạo bid với nhiễu riêng cho từng DSP
            bid = np.random.lognormal(mean=np.log(intrinsic_value * aggressiveness), sigma=0.15)
            bids.append(bid)
            
        return bids

    def _identify_value_bucket(self, bid):
        """
        Mô phỏng module (b) - Value Bucket Identification.
        Trả về trọng số interval (w_i) dựa trên giá trị inventory.
        """
        if bid > 80: # High-value bucket
            return {"weights": [1.0, 2.0, 3.0], "intervals": [0, 60, 80, 100]}
        else:        # Low-value bucket
            return {"weights": [1.0, 5.0, 3.0, 1.0], "intervals": [0, 20, 40, 60, 100]}

    def _calculate_risk_factor(self, t, b):
        """
        Triển khai Equation (10): Risk-aware factor beta(t, b)
        """
        if t <= 0: return 1.0
        U_tb = b / t # Remaining expected average auction revenue
        
        # Công thức: beta = kappa / (1 + v * tanh((U_tb - U_bar) / U_bar))
        diff = (U_tb - self.U_bar) / (self.U_bar + 1e-6)
        beta = self.kappa / (1 + self.v * np.tanh(diff))
        return beta

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_left = self.T_max
        self.unrealized_revenue = float(self.B_target)
        
        # Khởi tạo feature ngẫu nhiên cho ad inventory x_t
        x_t = np.random.uniform(0, 1, size=self.feature_dim).astype(np.float32)
        
        self.state = np.concatenate([[self.steps_left, self.unrealized_revenue], x_t])
        return self.state, {}

    def step(self, action):
        a_t = action[0] # Reserve price do Agent đặt ra
        t = self.state[0]
        b = self.state[1]
        x_t = self.state[2:]

        # 1. DSP đặt giá bid delta dựa trên inventory x_t
        delta = self._get_bid_price(x_t)

        # 2. Xác định bucket và trọng số interval (Section 4.2)
        bucket_info = self._identify_value_bucket(delta)
        w_i = 1.0
        for i in range(len(bucket_info["intervals"]) - 1):
            if bucket_info["intervals"][i] <= a_t < bucket_info["intervals"][i+1]:
                w_i = bucket_info["weights"][i]
                break

        # 3. Tính toán Risk Factor (Equation 10)
        beta = self._calculate_risk_factor(t, b)
        
        # 1. Lấy danh sách bid từ nhiều DSP
        bids = self._get_multi_dsp_bids(x_t, num_dsps=5)
        bids.sort(reverse=True) # Sắp xếp từ cao xuống thấp
        
        bid_1st = bids[0]
        bid_2nd = bids[1] if len(bids) > 1 else 0
        
        # 2. Logic đấu giá mức giá thứ hai (Section 3.2)
        revenue = 0
        if bid_1st >= a_t:
            # Người cao nhất thắng, nhưng trả giá của người thứ hai hoặc giá sàn
            revenue = max(bid_2nd, a_t)
            
            # Tính reward dựa trên revenue thực tế thu được
            # (Áp dụng beta và w_i như cũ nhưng dùng revenue thay vì a_t)
            beta = self._calculate_risk_factor(t, b)
            reward = w_i * revenue * beta
            self.unrealized_revenue -= revenue
        else:
            # Thất bại vì giá sàn đặt quá cao
            reward = -self.zeta
            revenue = 0

        # 5. Cập nhật State
        self.steps_left -= 1
        done = self.steps_left <= 0
        
        # Sinh inventory mới cho bước tiếp theo
        next_x_t = np.random.uniform(0, 1, size=self.feature_dim).astype(np.float32)
        self.state = np.concatenate([[self.steps_left, self.unrealized_revenue], next_x_t])

        # Episode-level penalty nếu không đạt mục tiêu doanh thu (Section 4.3)
        if done and self.unrealized_revenue > 0:
            reward -= 10.0 

        return self.state, float(reward), done, False, {"bid": delta, "revenue": revenue}
    
    def _print_terminal_stats(self):
        """
        Thay thế Sharpe Ratio bằng các chỉ số trong Section 5.1.3 của Paper
        """
        ar = np.mean(self.episode_revenue)
        ad = np.mean(self.episode_deal_prices) if self.episode_deal_prices else 0
        dr = self.success_count / self.T_max
        
        print("\n=== KDD '25 RTB Terminal Stats ===")
        print(f"Average Revenue (AR): {ar:.4f}")
        print(f"Average Deal Price (AD): {ad:.4f}")
        print(f"Deal Rate (DR): {dr:.2%}")
        print(f"Remaining Goal: {self.unrealized_revenue:.2f}")
        print("===================================\n")

    def render(self, mode='human'):
        if mode == 'human':
            t = int(self.state[0])
            unrealized_b = self.state[1]
            
            # Giả sử chúng ta lưu lại giá bid và action gần nhất để xem
            last_bid = self.last_info.get('bid', 0)
            last_action = self.last_info.get('action', 0)
            last_reward = self.last_info.get('reward', 0)
            
            print(f"Step {self.T_max - t}/{self.T_max} | "
                f"Target Left: {unrealized_b:.2f} | "
                f"Agent Set Action (a_t): {last_action:.2f} | "
                f"DSP Bid (delta): {last_bid:.2f} | "
                f"Reward: {last_reward:.4f}")

    def get_sb_env(self):
        """
        Hàm tiện ích để wrap vào Stable Baselines 3
        """
        # Lưu ý: Stable Baselines 3 tự handle việc wrap environment
        e = DummyVecEnv([lambda: self])
        return e

# # --- Demo sử dụng với một Agent ngẫu nhiên ---
# if __name__ == "__main__":
#     env = RTBReservePriceEnv()
#     obs, _ = env.reset()
    
#     total_reward = 0
#     for _ in range(100):
#         action = env.action_space.sample() # Thay bằng PPO policy của bạn ở đây
#         obs, reward, done, _, info = env.step(action)
#         total_reward += reward
#         if done:
#             break
            
#     print(f"Hoàn thành episode. Tổng thưởng: {total_reward:.2f}")
#     print(f"Doanh thu còn thiếu: {obs[1]:.2f}")