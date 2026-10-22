import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 字体设置，确保中文与负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class HydraulicTwin:
    """
    阀控单出杆非对称液压缸四阶状态空间仿真模型 (对应第3章机理模型)
    
    状态向量定义: x = [xp, vp, P1, P2]^T
        - xp : 活塞位移 (m)，以缸底死点为 0，向右伸出为正
        - vp : 活塞运动速度 (m/s)
        - P1 : 无杆腔(大腔)绝对压力 (Pa)
        - P2 : 有杆腔(小腔)绝对压力 (Pa)
    """
    def __init__(self, config):
        # 深拷贝参数字典，保证单次仿真周期内静态物理参数的独立性
        self.p = config.copy()
        
        # 外泄漏系数默认置零（实际工程中合格压盖动密封的外泄漏极小，常忽略）
        if 'C_ep' not in self.p:
            self.p['C_ep'] = 0.0  
            
        # 状态向量索引常量
        self.IDX_POS = 0  # 活塞位移 xp
        self.IDX_VEL = 1  # 活塞速度 vp
        self.IDX_P1  = 2  # 无杆腔压力 P1
        self.IDX_P2  = 3  # 有杆腔压力 P2
        
        self.t = 0.0      
        self.reset()

    def reset(self, x0=None):
        """
        重置系统状态与仿真时钟
        """
        self.t = 0.0
        if x0 is not None:
            self.state = np.array(x0, dtype=float)
        else:
            self.state = np.zeros(4)
            # 初值设定注意：静止时两腔初始压力需设为回油背压 Pr (0.1 MPa)，不可设为 0。
            # 若设为 0 (绝对真空)，起动首步计算压差 (P2 - Pr) 时会出现负压差，引发数值异常。
            pr_val = self.p.get('Pr', 1e5)
            self.state[self.IDX_P1] = pr_val 
            self.state[self.IDX_P2] = pr_val
        return self.state

    def _friction(self, v):
        """
        平滑化 Stribeck 摩擦模型 (对应课件 3.2.1 节)
        公式: F_f(v) = [Fc + (Fs - Fc) * exp(-(v/vs)^2)] * tanh(gamma * v) + sigma * v
        """
        p = self.p
        # 边界润滑(静摩擦 Fs)向混合润滑(库伦摩擦 Fc)的指数衰减过渡项
        f_mixed = p['Fc'] + (p['Fs'] - p['Fc']) * np.exp(-(v / p['vs'])**2)
        
        # 数值平滑：用连续可导的 tanh(gamma * v) 替代不连续的符号函数 sgn(v)，
        # 避免隐式积分器在 v=0 附近因导数突变(雅可比矩阵奇异)而发生步长死锁。
        return f_mixed * np.tanh(p['gamma'] * v) + p['sigma'] * v

    def _env_force(self, x, v):
        """
        外部负载与机械行程限位模型 (对应课件 3.2.3 节 Kelvin-Voigt 接触模型)
        """
        F_env = 0.0
        p = self.p
        
        # 1. 工艺接触负载：当活塞伸出超过工件位置 x_work (0.4m) 时，开始压制工件，
        #    产生与压入深度成正比的弹性抗力及阻尼力。
        if x > p.get('x_work', 1.0):
            F_env += p['K_env'] * (x - p['x_work']) + p['B_env'] * v
            
        # 2. 机械行程上下限位 (惩罚函数法)：当位移超出物理行程 [0, L] 时，
        #    触发高刚度虚拟端盖反力，约束活塞不越界。
        k_wall, b_wall = p.get('K_wall', 1e10), p.get('B_wall', 1e6)
        if x >= p['L']:        # 前端盖限位
            F_env += k_wall * (x - p['L']) + b_wall * v 
        elif x <= 0:           # 后端盖限位
            F_env += k_wall * x + b_wall * v
            
        return F_env

    def _get_beta(self, P):
        """
        低压气穴区的有效体积模量衰减模型 (对应课件 3.3.2 节)
        原理：当腔室绝对压力 P 跌至大气压 (1e5 Pa) 以下时，油液中游离气体析出膨胀，
        有效体积模量 beta_e 急剧下降。由 dP/dt = (beta_e / V) * Q_net 可知，
        beta_e 减小会使压力下降速率自然放缓，在不破坏导数连续性的前提下防止压力跌破零点。
        """
        beta_oil = self.p['beta_e']
        # S型平滑过渡系数：高压区趋近于 1.0，低压趋近真空时衰减至 0.001
        scale = 0.001 + 0.999 * (0.5 * (np.tanh((P - 1e5) / 5e4) + 1.0))
        return beta_oil * scale

    def _dynamics(self, t, x_vec, u):
        """
        四阶状态空间微分方程组: dx/dt = f(t, x, u) (对应课件 3.1 节与 3.5.2 节)
        """
        xp, vp, P1, P2 = x_vec
        p = self.p
        
        # --- 1. 伺服阀死区非线性 (对应课件 3.2.2 节) ---
        # 正遮盖伺服阀在控制电压 |u| 小于死区阈值 deadzone 时无流量输出
        u_eff = 0.0
        if abs(u) > p['deadzone']:
            u_eff = np.sign(u) * (abs(u) - p['deadzone'])
        xv_real = u_eff * p['xv_max']  # 折算为实际阀芯位移 (m)
        
        # --- 2. 阀口代数流量方程 (对应课件 3.1.1 节) ---
        # 节流窗口面积仅与阀芯位移绝对值 |xv_real| 有关
        A_valve = p['Cd'] * p['w'] * abs(xv_real)
        sgn_flow = np.sign(u_eff) if u_eff != 0 else 0
        
        # 代数流量符号约定：流入液压缸腔室为正(+)，流出腔室为负(-)
        # 根号内加 np.abs() 防止数值试探步中出现短暂负压差导致开方报错
        if sgn_flow >= 0:
            # 正向伸出 (xv >= 0)：无杆腔接高压油源 Ps 进油(+)，有杆腔接油箱 Pr 排油(-)
            q1 =  A_valve * np.sqrt(2 / p['rho'] * np.abs(p['Ps'] - P1))
            q2 = -A_valve * np.sqrt(2 / p['rho'] * np.abs(P2 - p['Pr']))
        else:
            # 反向缩回 (xv < 0)：无杆腔接油箱 Pr 排油(-)，有杆腔接高压油源 Ps 进油(+)
            q1 = -A_valve * np.sqrt(2 / p['rho'] * np.abs(P1 - p['Pr']))
            q2 =  A_valve * np.sqrt(2 / p['rho'] * np.abs(p['Ps'] - P2))
            
        # --- 3. 流体连续性方程/压力动态方程 (对应课件 3.1.2 节) ---
        # 两腔瞬时有效容积 (初始死区容积 + 活塞行程容积)
        V1 = p['V0_1'] + p['A1'] * xp
        V2 = p['V0_2'] + p['A2'] * (p['L'] - xp)
        
        # 当前压力下的有效体积模量
        beta1 = self._get_beta(P1)
        beta2 = self._get_beta(P2)
        
        # 内外泄漏流量计算
        leak_int  = p['C_ip'] * (P1 - P2)  # 活塞间隙串腔内泄漏
        leak_ext1 = p['C_ep'] * P1         # 无杆腔广义外泄漏
        leak_ext2 = p['C_ep'] * P2         # 有杆腔外泄漏
        
        # 压力变化率 dP/dt = (beta / V) * 腔室净流入流量
        dP1_dt = (beta1 / V1) * (q1 - p['A1'] * vp - leak_int - leak_ext1)
        dP2_dt = (beta2 / V2) * (q2 + p['A2'] * vp + leak_int - leak_ext2)
        
        # --- 4. 牛顿-欧拉力学平衡方程 (对应课件 3.1.3 节) ---
        # m * dv/dt = P1*A1 - P2*A2 - F_friction - F_env
        F_drive = P1 * p['A1'] - P2 * p['A2']
        F_res   = self._friction(vp) + self._env_force(xp, vp)
        dv_dt   = (F_drive - F_res) / p['mass']
        
        return [vp, dv_dt, dP1_dt, dP2_dt]

    def step(self, u, dt):
        """ 
        单步离散仿真接口 (对应课件 3.5.3 节)
        :param u: 当前控制周期的输入电压指令 [-1.0, 1.0]
        :param dt: 控制采样周期 (s)
        """
        # 零阶保持器 (ZOH)：在 [t, t+dt] 积分区间内保持控制量 u 恒定
        fun_dynamics = lambda t, y: self._dynamics(t, y, u)
        
        # 采用具备 L-稳定性的隐式 Radau 算法求解刚性微分方程 (对应课件 3.4.3 节)
        sol = solve_ivp(
            fun=fun_dynamics,
            t_span=[self.t, self.t + dt], 
            y0=self.state,
            method='Radau',      
            rtol=1e-5,           
            atol=1e-8
        )
        
        self.state = sol.y[:, -1]
        self.t += dt
        return self.state


# ==============================================================================
# 主程序：通过一次完整的“起动 -> 空载快进 -> 接触工件保压”过程，测试模型的三项核心特性
#   - 测试阶段 1 (0 ~ 0.10s) : 起动瞬态 —— 验证 Stribeck 低速摩擦与低压防气穴特性
#   - 测试阶段 2 (0.10 ~ 0.32s): 空载稳态 —— 验证非对称缸两腔面积比 (2:1) 下的压力分配规律
#   - 测试阶段 3 (0.32 ~ 1.00s): 接触保压 —— 验证撞击高刚度工件时的压力跃升与 Radau 算法稳定性
# ==============================================================================
if __name__ == '__main__':
    # ----------------- 1. 配置系统物理参数 -----------------
    phys_config = {
        # 几何参数：注意 A1 : A2 = 2 : 1，这是理解测试阶段 2 两腔压力关系的关键
        'A1': 0.01, 'A2': 0.005,           # 无杆腔面积 0.01 m^2，有杆腔面积 0.005 m^2
        'L': 1.0,                          # 液压缸总行程 1.0 m
        'V0_1': 0.001, 'V0_2': 0.001,      # 两腔初始死区容积 (含连接管路容积) 1 L
        
        # 负载与油源参数
        'mass': 500.0,                     # 运动部分等效质量 500 kg
        'Ps': 21e6, 'Pr': 1e5,             # 供油压力 21 MPa，回油背压 0.1 MPa
        'beta_e': 1.0e9,                   # 油液标称有效体积模量 1000 MPa
        'rho': 870.0,                      # 液压油密度 870 kg/m^3
        
        # 伺服阀与泄漏参数
        'Cd': 0.62, 'w': 0.02,             # 阀口流量系数 0.62，面积梯度 0.02 m
        'xv_max': 0.005, 'deadzone': 0.05, # 最大阀芯位移 5 mm，死区阈值 5%
        'C_ip': 1e-12,                     # 内泄漏系数 (m^3/(s·Pa))
        
        # Stribeck 摩擦参数 (用于测试阶段 1)
        'Fs': 800.0, 'Fc': 500.0,          # 最大静摩擦力 800 N，库伦摩擦力 500 N
        'vs': 0.05, 'sigma': 1000.0,       # Stribeck 特征速度 0.05 m/s，粘性阻尼 1000 N·s/m
        'gamma': 500.0,                    # tanh 符号平滑系数
        
        # 工件接触环境参数 (用于测试阶段 3)
        'x_work': 0.4,                     # 工件摆放位置：在 xp = 0.4 m 处发生接触碰撞
        'K_env': 5e7, 'B_env': 1e5         # 工件接触刚度 5×10^7 N/m，接触阻尼 10^5 N·s/m
    }
    
    # ----------------- 2. 初始化仿真与时间网格 -----------------
    twin = HydraulicTwin(phys_config)
    
    dt = 0.002        # 采样步长 2 ms (对应工业控制器 500 Hz 采样率)
    sim_time = 1.0    # 仿真总时长 1.0 s
    steps = int(sim_time / dt)
    
    t_log = np.linspace(0, sim_time, steps)
    x_log = np.zeros(steps)
    v_log = np.zeros(steps)
    p1_log = np.zeros(steps)
    p2_log = np.zeros(steps)
    
    print(">> 开始运行四阶液压系统仿真 (求解器: Radau)...")
    
    # 用一阶惯性环节 (时间常数 tau_v = 15 ms) 模拟伺服阀阀芯的机械响应延迟，
    # 避免在 t=0 时刻直接输入理想方波导致初始加速度突变。
    u_real = 0.0
    tau_v = 0.015  
    
    # ----------------- 3. 主仿真循环 -----------------
    for i in range(steps):
        u_cmd = 0.4  # 给定 40% 正向阶跃开度指令，驱动液压缸向右伸出
        
        # 计算经过一阶惯性延迟后的实际阀指令
        u_real += (u_cmd - u_real) * (1 - np.exp(-dt / tau_v))
        
        # 推进一个控制周期 dt
        state = twin.step(u_real, dt)
        
        # 记录四阶状态变量
        x_log[i]  = state[twin.IDX_POS]
        v_log[i]  = state[twin.IDX_VEL]
        p1_log[i] = state[twin.IDX_P1]
        p2_log[i] = state[twin.IDX_P2]
        
    print(">> 仿真计算完成，正在生成三阶段验证图表...")

    # ----------------- 4. 绘制三阶段动力学响应曲线 -----------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 9.5), sharex=True)
    fig.suptitle('阀控非对称液压缸四阶模型仿真：起动、空载与接触保压全过程验证', fontsize=14, fontweight='bold')
    
    # [子图 1] 活塞位移曲线：观察从空载推进到 x = 0.4m 处接触工件停止的过程
    axs[0].plot(t_log, x_log, 'b', linewidth=2.5, label='活塞位移 $x_p$')
    axs[0].axhline(y=phys_config['x_work'], color='r', linestyle='--', linewidth=2, label='工件接触位置 (x = 0.4m)')
    axs[0].set_ylabel('活塞位移 (m)', fontweight='bold')
    axs[0].legend(loc='lower right', framealpha=0.9)
    axs[0].grid(True, linestyle=':', alpha=0.6, color='gray')
    
    # [子图 2] 活塞速度曲线：重点观察测试阶段 1 的起动波动与阶段 2 的匀速稳态
    axs[1].plot(t_log, v_log, 'k', linewidth=2, label='活塞速度 $v_p$')
    axs[1].set_ylabel('活塞速度 (m/s)', fontweight='bold')
    axs[1].grid(True, linestyle=':', alpha=0.6, color='gray')
    
    # 标注测试阶段 1：起动阶段的 Stribeck 摩擦非线性表现
    axs[1].annotate('【测试1: 起动瞬态】\n克服最大静摩擦 Fs 后进入负阻尼区，\n引起起动速度超调波动',
                    xy=(0.04, 0.4), xycoords='data',
                    xytext=(0.12, 0.18), textcoords='data',
                    arrowprops=dict(facecolor='gray', shrink=0.05, width=1.5, headwidth=5),
                    fontsize=9.5, color='black')
    
    # [子图 3] 双腔绝对压力曲线：重点观察测试阶段 2 的稳态压力比与阶段 3 的碰撞升压
    axs[2].plot(t_log, p1_log / 1e6, 'r', label=r'无杆腔压力 $P_1$ (大腔 $A_1=0.01\,\mathrm{m}^2$)', linewidth=2)
    axs[2].plot(t_log, p2_log / 1e6, 'g', label=r'有杆腔压力 $P_2$ (小腔 $A_2=0.005\,\mathrm{m}^2$)', linewidth=2)
    axs[2].set_ylabel('腔室绝对压力 (MPa)', fontweight='bold')
    axs[2].set_xlabel('仿真时间 (s)', fontweight='bold')
    axs[2].legend(loc='upper left', framealpha=0.9)
    axs[2].grid(True, linestyle=':', alpha=0.6, color='gray')
    
    # 标注测试阶段 2：空载匀速阶段的非对称面积匹配关系
    axs[2].annotate('【测试2: 空载稳态面积匹配】\n因 $A_1 = 2A_2$，力平衡决定 $P_1(\\approx 2.4\\mathrm{MPa}) \\approx 0.5 P_2(\\approx 4.8\\mathrm{MPa})$',
                    xy=(0.45, 4.8), xycoords='data',
                    xytext=(0.28, 9.5), textcoords='data',
                    arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=5),
                    fontsize=9.5, color='darkgreen', fontweight='bold')

    # 标注测试阶段 3：接触工件瞬间的刚度突变与压力跃升（移至曲线左侧，防止超出右边框）
    contact_idx = np.argmax(x_log > phys_config['x_work'])
    if contact_idx > 0:
        contact_t = t_log[contact_idx]
        axs[2].annotate('【测试3: 刚性接触保压】\n接触高刚度工件瞬间，活塞受阻减速，\n$P_1$ 迅速升至供油压力 21MPa，$P_2$ 降至回油背压',
                        xy=(contact_t, p1_log[contact_idx]/1e6 + 2.0), xycoords='data',
                        xytext=(contact_t - 0.38, 13.5), textcoords='data',
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                        fontsize=9.5, fontweight='bold', color='red',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=1.2, alpha=0.9))
                        
    stats_text = (
        "【仿真结果物理规律导读（对应课程三大测试点）】\n"
        "1. 起动与防气穴测试 (0 ~ 0.10s)：起动瞬间两腔压力平稳建立且始终保持正值 (P > 0)，验证了初始背压 Pr 设定与气穴体积模量模型的有效性。\n"
        "2. 空载稳态面积匹配测试 (0.10 ~ 0.91s)：匀速阶段加速度为 0，力平衡满足 P1*A1 - P2*A2 = F_f。因摩擦力 F_f 较小且 A1 = 2*A2，\n"
        "   故稳态时必然出现 P1(约 2.4MPa) ≈ 0.5 * P2(约 4.8MPa) 的现象，体现了非对称缸“大腔低压、小腔高压”的面积匹配关系。\n"
        "3. 刚性碰撞与数值稳定性测试 (0.91 ~ 1.00s)：在 x=0.4m 处接触高刚度工件，系统刚性比剧增；Radau 隐式算法有效抑制了非物理高频振荡。"
    )
    fig.text(0.04, 0.015, stats_text, fontsize=9, color='navy', va='bottom',
             bbox=dict(facecolor='lightblue', alpha=0.2, boxstyle='round,pad=0.5', edgecolor='navy'))

    plt.tight_layout(rect=[0.0, 0.15, 1.0, 0.96])
    plt.show()