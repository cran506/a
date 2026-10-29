"""
================================================================================
《智能流体传动与控制》 第八章 深度强化学习 (SAC) 训练与边缘推理基准代码
文件名: train_sac_hydraulic.py

【实验目的与操作说明】
本代码旨在引导学生完成从“虚拟环境训练 (Offline Training)”到“物理边缘部署 
(Online Inference)”的深度强化学习完整闭环。

核心流程：
  1. 训练阶段 (Training)：
     代码将实例化一个基于 Gymnasium 的非线性液压数字孪生环境 (HydraulicGymEnv)。
     利用 Soft Actor-Critic (SAC) 算法框架，智能体将在该环境中进行交互探索。
     * 教学提示：为适应课堂/作业机器的算力限制，本脚本已配置为“极速收敛模式”
       (总步数降至 10,000 步，增大学习率)。在实际前沿科研中，为达到绝对的
       控制平滑度与泛化边界，通常需执行 100 万步以上的大规模集群训练。
  2. 部署与推理阶段 (Inference)：
     训练完成后，提取并冻结最优参数矩阵 (sac_hydraulic_model.zip)。在面对 
     t=0.5s 突加 5000N 负载及刚度骤降 50% 的极端复合失配工况时，验证该模型
     在线前向推理的鲁棒恢复能力。

环境依赖：
    pip install stable-baselines3[extra] gymnasium matplotlib
================================================================================
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import SAC
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("\n[环境缺失] 请先安装必需的深度强化学习基础库：")
    print("pip install stable-baselines3[extra] gymnasium\n")
    import sys
    sys.exit(1)

# 配置 Matplotlib 中文字体，确保图表渲染正常
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 模块一：液压数字孪生动力学推演与 Gym 交互环境
# ==============================================================================
class HydraulicTwinCore:
    """真实物理演化内核：四阶非线性伺服阀控缸动力学模型"""
    def __init__(self, default_m=250.0, default_beta=1000e6, default_Fc=1200.0):
        self.default_m    = default_m
        self.default_beta = default_beta
        self.default_Fc   = default_Fc
        self.update_physics({'m': default_m, 'beta_e': default_beta, 'F_c': default_Fc})
        self.reset()

    def update_physics(self, params):
        self.m      = params.get('m', self.default_m)
        self.beta_e = params.get('beta_e', self.default_beta)
        self.F_c    = params.get('F_c', self.default_Fc)

    def reset(self):
        # 状态向量: [位移误差 e(mm), 速度误差 de(mm/s), 加速度 dde(mm/s^2), 负载压力 PL(MPa), 供油压力 Ps(MPa)]
        self.state = np.array([0.0, 0.0, 0.0, 4.5, 21.0], dtype=np.float32)
        return self.state.copy()

    def step(self, u_volts, f_ext=0.0, dt=0.001):
        """执行单步物理域推进，包含外部机械负载扰动 f_ext 的时域耦合"""
        e, de, dde, PL, Ps = self.state
        stiff_ratio = self.beta_e / self.default_beta
        
        # 微分方程离散化推演
        PL_next  = np.clip(PL + 0.15 * stiff_ratio * u_volts - 0.001 * f_ext, 0.5, Ps)
        dde_next = -18.0 * de + 12.5 * stiff_ratio * u_volts - 0.008 * f_ext
        de_next  = de + dde_next * dt
        e_next   = e + de_next * dt
        
        self.state = np.array([e_next, de_next, dde_next, PL_next, Ps], dtype=np.float32)
        return self.state.copy(), {'u_volts': u_volts, 'beta_e': self.beta_e}


class HydraulicGymEnv(gym.Env):
    """强化学习标准的 Gym 交互环境，集成域随机化 (Domain Randomization) 与复合奖励机制"""
    def __init__(self):
        super().__init__()
        self.twin = HydraulicTwinCore()
        
        # 定义动作空间: 归一化指令 [-1.0, 1.0]，后续映射为实际伺服阀电压 [-10V, 10V]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # 定义状态空间: 涵盖位移、速度、加速度及腔室压力的全可观测五维特征
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        # 状态输入正则化 (Z-score 标准化参数)，消除不同物理量级的梯度发散问题
        self.obs_mean = np.array([0.0, 0.0, 0.0, 10.0, 21.0], dtype=np.float32)
        self.obs_std  = np.array([10.0, 50.0, 500.0, 8.0, 2.0], dtype=np.float32)
        
        # 教学版加速收敛限制
        self.max_steps = 500 
        self.step_count = 0
        self.last_u = 0.0

    def normalize_obs(self, obs):
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.last_u = 0.0

        # 【核心理论：域随机化 Domain Randomization】
        # 在每一回合起始，注入随机的参数摄动。这迫使神经网络无法对单一工况产生“过拟合”，
        # 从而习得从状态斜率中隐式感知系统刚度变化的泛化能力 (Sim2Real 的底层基石)。
        rand_params = {
            'm':      250.0   * np.random.uniform(0.85, 1.15),   
            'beta_e': 1000e6  * np.random.uniform(0.70, 1.30),   
            'F_c':    1200.0  * np.random.uniform(0.80, 1.20)    
        }
        self.twin.update_physics(rand_params)
        return self.normalize_obs(self.twin.reset()), {}

    def step(self, action_norm):
        # 动作反归一化输出
        u_volts = float(np.clip(action_norm[0], -1.0, 1.0) * 10.0)
        s_next, info = self.twin.step(u_volts)

        # 复合目标奖励计算 (Reward Shaping)
        e_t = float(s_next[0])
        # 教学版调整：强化稳态靶心奖励，加快收敛速度
        r_err    = -50.0 * (e_t ** 2)
        r_energy = -0.05 * (u_volts ** 2)
        r_smooth = -0.5  * ((u_volts - self.last_u) ** 2)
        r_bonus  = 5.0 if abs(e_t) < 0.3 else 0.0
        
        total_reward = r_err + r_energy + r_smooth + r_bonus

        self.last_u = u_volts
        self.step_count += 1
        
        truncated = bool(self.step_count >= self.max_steps)
        terminated = False
        
        return self.normalize_obs(s_next), float(total_reward), terminated, truncated, info


# ==============================================================================
# 模块二：模型训练 (Training) 与边缘前向推理 (Inference) 验证
# ==============================================================================
def train_sac_agent_fast():
    """驱动 PyTorch 内核执行梯度下降与反向传播 (极速教学模式)"""
    print("\n" + "="*80)
    print(">> 启动 SAC (Soft Actor-Critic) 深度神经网络交互训练...")
    print(">> 注意：当前已开启【极速收敛模式】(总步数: 10,000 步)。")
    print(">> 预计训练耗时 15 ~ 30 秒，请观察控制台 ep_rew_mean (平均奖励) 的攀升过程。")
    print("="*80 + "\n")
    
    env = HydraulicGymEnv()
    
    # 实例化 SAC 网络，配置激进的学习率与较小的缓存池以实现秒级收敛
    model = SAC(
        "MlpPolicy", 
        env, 
        learning_rate=1e-3,  # 增大初始学习率
        buffer_size=10000, 
        batch_size=128, 
        ent_coef='auto',     # 自动调节温度系数，实现最大熵探索
        gamma=0.98, 
        verbose=1            
    )
    
    t0 = time.time()
    # 执行强化学习闭环
    model.learn(total_timesteps=10000, log_interval=5)
    train_time = time.time() - t0
    
    print(f"\n>> 训练阶段执行完毕！总耗时: {train_time:.1f} 秒。")
    print(">> 正在固化权重矩阵并保存模型文件至 sac_hydraulic_model.zip...")
    model.save("sac_hydraulic_model")
    return model


def evaluate_sim2real_robustness():
    """加载冻结参数，执行鲁棒性边缘推理 (Online Inference) 与动态绘图"""
    print("\n>> 加载已收敛的预训练模型权重...")
    model = SAC.load("sac_hydraulic_model")
    
    twin = HydraulicTwinCore()
    env = HydraulicGymEnv() 
    
    dt = 0.001
    t_seq = np.arange(0.400, 1.000 + dt, dt)
    n_steps = len(t_seq)
    
    ev_arr = np.zeros(n_steps)
    u_arr  = np.zeros(n_steps)
    
    # 获取初始系统状态
    obs = env.normalize_obs(twin.reset())
    
    print(">> 正在推演极端复合失配工况：t=0.5s 时突加 5000N 负载，且油液混气致刚度骤降 50% ...")
    for k, t in enumerate(t_seq):
        # 故障突变注入
        if t >= 0.500:
            f_ext = 5000.0
            twin.beta_e = 500e6  # 有效体积模量骤降
        else:
            f_ext = 0.0
            twin.beta_e = 1000e6
            
        # ----------------------------------------------------------------------
        # 【模型前向推理 (Forward Pass)】
        # 实际部署于嵌入式控制器中的核心逻辑，本质上是矩阵乘加运算
        action_norm, _states = model.predict(obs, deterministic=True)
        # ----------------------------------------------------------------------
        
        # 映射为实际伺服阀输出电压
        u_volts = float(np.clip(action_norm[0], -1.0, 1.0) * 10.0)
        
        # 物理孪生体状态演化
        s_next, info = twin.step(u_volts, f_ext=f_ext, dt=dt)
        obs = env.normalize_obs(s_next)
        
        ev_arr[k] = s_next[1]  # 记录速度偏差
        u_arr[k]  = u_volts
        
    print(">> 推演完成！正在生成模型抗干扰性能响应图表...")
    
    # 可视化结果渲染
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('SAC 智能体抗扰动性能测试 (基于 10k 步极速预训练权重)', fontsize=14, fontweight='bold')
    
    axs[0].axvspan(0.50, 1.00, color='gray', alpha=0.12, label='复合突变区 (5000N 负载 + 刚度减半)')
    axs[0].axvline(0.50, color='gray', linestyle='--', linewidth=1.4)
    axs[0].plot(t_seq, ev_arr, color='royalblue', linewidth=2.5, label='SAC 智能体输出：速度跌落与自适应恢复')
    axs[0].set_title('速度跟踪误差 $e_v(t)$', fontweight='bold')
    axs[0].set_ylabel('速度误差 (mm/s)', fontweight='bold')
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].legend(loc='lower right')
    
    axs[1].axvspan(0.50, 1.00, color='gray', alpha=0.12)
    axs[1].axvline(0.50, color='gray', linestyle='--', linewidth=1.4)
    axs[1].plot(t_seq, u_arr, color='darkgreen', linewidth=2.0, label='SAC 网络决策输出电压 $u(t)$')
    axs[1].set_title('伺服阀控制电压 $u(t)$ (捕捉动态频率特征)', fontweight='bold')
    axs[1].set_xlabel('时间 $t$ (s)', fontweight='bold')
    axs[1].set_ylabel('控制电压 (V)', fontweight='bold')
    axs[1].grid(True, linestyle=':', alpha=0.6)
    axs[1].legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    if not HAS_SB3:
        pass
    else:
        # 工作流控制：检测是否已存在预训练权重
        if not os.path.exists("sac_hydraulic_model.zip"):
            train_sac_agent_fast()
        else:
            print("\n>> 检测到本地已存在最优权重 sac_hydraulic_model.zip，将自动跳过训练过程。")
            print(">> (如需重新观察训练过程，请先删除该 zip 文件)")
            
        # 提取权重执行部署推理验证
        evaluate_sim2real_robustness()