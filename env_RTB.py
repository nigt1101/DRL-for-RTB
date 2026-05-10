import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class OfflineRTBEnv(gym.Env):
    def __init__(self, 
                 dataframe, 
                 target_revenue=1000, 
                 max_auctions_per_episode=1024, 
                 kappa=1.0, 
                 v=3.0, 
                 zeta=10, 
                 terminal_penalty=0.0,
                 reserve_low=0.0,
                 reserve_high=None,
                 esp=1e-2):
        
        super(OfflineRTBEnv, self).__init__()
        
        self.df = dataframe
        self.total_rows = len(dataframe)
        self.T = max_auctions_per_episode
        self.B = target_revenue
        self.kappa = kappa
        self.v = v
        self.zeta = zeta
        self.terminal_penalty = terminal_penalty
        self.U_bar = target_revenue / self.T 
        self.max_bid = float(dataframe['bidprice'].max())
        self.reserve_low = float(reserve_low)
        self.reserve_high = float(self.max_bid + 100 if reserve_high is None else reserve_high)
        if self.reserve_high <= self.reserve_low:
            raise ValueError("reserve_high must be greater than reserve_low")
        self.feature_cols = [
        'hour', 'region', 'city', 'slotwidth', 'slotheight', 'slotvisibility','slotformat'
        ]
        
        self.feature_dim = len(self.feature_cols)
        
        # PPO samples a Gaussian action on [-1, 1], then the env maps it to
        # a non-negative reserve price in [reserve_low, reserve_high].
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=np.array([0, 0] + [-np.inf] * self.feature_dim, dtype=np.float32),
            high=np.array([self.T, self.B] + [np.inf] * self.feature_dim, dtype=np.float32)
        )

        self.current_idx = 0

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_idx = np.random.randint(0, self.total_rows - self.T)
        self.steps_left = self.T
        self.unrealized_revenue = float(self.B)
        
        row = self.df.iloc[self.current_idx]
        x_t = row[self.feature_cols].values.astype(np.float32)
        
        self.state = np.concatenate([[self.steps_left/self.T, self.unrealized_revenue/self.B], x_t]).astype(np.float32)
        return self.state, {}

    def _calculate_risk_factor(self, t, b):
        if t <= 0: return 1.0
        U_tb = b / t 
        # Beta: tính toán theo công thức đã cho, với U_bar là giá trị trung bình 
        diff = (U_tb - self.U_bar) / (self.U_bar + 1e-6)
        beta = self.kappa / (1 + self.v * np.tanh(diff))
        return beta

    def _action_to_reserve(self, action):
        raw_action = float(np.asarray(action, dtype=np.float32)[0])
        clipped_action = np.clip(raw_action, -1.0, 1.0)
        reserve = self.reserve_low + (clipped_action + 1.0) * 0.5 * (self.reserve_high - self.reserve_low)
        return float(reserve), raw_action
    
    def step(self, action):
        a_t, raw_action = self._action_to_reserve(action)
        row = self.df.iloc[self.current_idx]    
        
        # Lấy giá thầu cao nhất của thị trường (Winning Bid)
        h_bid_1 = row['bidprice']

        revenue = 0
        success = False
        beta = self._calculate_risk_factor(self.steps_left, self.unrealized_revenue)

        gap = abs(h_bid_1 - a_t)

        if h_bid_1 >= a_t:
            revenue = a_t
            success = True

            reward = (
                revenue / 1000
                - gap / 300
            ) * beta 

            self.unrealized_revenue = max(0, self.unrealized_revenue - revenue)

        else:
            revenue = 0
            success = False

            reward = (
                -1.5 * gap / 1000
                - self.zeta / 1000
            )

        # Cập nhật trạng thái cho phiên tiếp theo
        self.current_idx += 1
        self.steps_left -= 1
        
        # Kiểm tra nếu hết dữ liệu
        if self.current_idx >= self.total_rows:
            return self.state, float(reward), done, False, {
                "raw_action": raw_action,
                "reserve_price": a_t,
                "real_bid": h_bid_1,
                "gap": gap,
                "beta": beta,
                "revenue": revenue,
                "success": success,
                "unrealized_revenue": self.unrealized_revenue,
                "required_avg_remaining": self.unrealized_revenue / max(self.steps_left, 1),
            }
        
        next_row = self.df.iloc[self.current_idx]
        next_x_t = next_row[self.feature_cols].values.astype(np.float32)
        self.state = np.concatenate([[self.steps_left/self.T, self.unrealized_revenue/self.B], next_x_t]).astype(np.float32)
        
        done = self.steps_left <= 0
        if done:
            # Penalize missing the KPI at episode end. Keep this on the same
            # revenue scale as the step reward so the KPI has real weight.
            reward -= self.terminal_penalty * self.unrealized_revenue

        return self.state, float(reward), done, False, {
            "raw_action": raw_action,
            "reserve_price": a_t,
            "real_bid": h_bid_1,
            "gap": gap,
            "beta": beta,
            "revenue": revenue,
            "success": success,
            "unrealized_revenue": self.unrealized_revenue,
            "required_avg_remaining": self.unrealized_revenue / max(self.steps_left, 1),
        }