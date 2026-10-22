"""第六章课堂参考轨迹：0.2 m/s² 起步，0.2 m/s 巡航，1 s 停车。

分段速度多项式及其导数、积分均解析计算；没有用仿真输出构造参考。
首样本 a_d=0.2 m/s²，故教材示例 m*a_d=100 N（m=500 kg）。
"""
import numpy as np
from trajectory import time_grid


def generate_textbook_trajectory(dt=0.001, duration=1.2):
    if duration < 1.0:
        raise ValueError("总时长至少1.0s")
    t = time_grid(duration, dt)
    x, v, a = [np.zeros_like(t) for _ in range(3)]
    # 0~0.16 s: 从静止以 a=0.2 m/s² 启动，对应图6.4约0.032m/s。
    m = t < .16 - 1e-12
    h = t[m]
    x[m], v[m], a[m] = .1*h*h, .2*h, .2
    x16, v16, a16 = .1*.16**2, .2*.16, .2
    # 0.16~0.35 s: 五次 Hermite 速度多项式，v/a/jerk 在右端归零。
    # 左端速度与加速度连续；选取起始jerk=0，避免轨迹内部拼接尖峰。
    t1, t2 = .16, .35
    width = t2-t1
    c1 = a16*width
    delta = .2-v16-c1
    slope = -c1
    c3, c4, c5 = 10*delta-4*slope, 7*slope-15*delta, 6*delta-3*slope
    m = (t >= t1-1e-12) & (t < t2-1e-12)
    s = (t[m]-t1)/width
    v[m] = v16+c1*s+c3*s**3+c4*s**4+c5*s**5
    a[m] = (c1+3*c3*s**2+4*c4*s**3+5*c5*s**4)/width
    x[m] = x16+width*(v16*s+c1*s*s/2+c3*s**4/4+c4*s**5/5+c5*s**6/6)
    x35 = x16+width*(v16+c1/2+c3/4+c4/5+c5/6)
    # 0.35~0.72 s: 0.2 m/s 匀速巡航。
    t3 = .72
    m = (t >= t2-1e-12) & (t < t3-1e-12)
    x[m], v[m] = x35+.2*(t[m]-t2), .2
    x72 = x35+.2*(t3-t2)
    # 0.72~1.0 s: 速度用五次 smootherstep 平滑收至零，位置解析积分。
    stop_width = 1.0-t3
    m = (t >= t3-1e-12) & (t < 1.0-1e-12)
    s = (t[m]-t3)/stop_width
    v[m] = .2*(1-10*s**3+15*s**4-6*s**5)
    a[m] = .2*(-30*s**2+60*s**3-30*s**4)/stop_width
    x[m] = x72+.2*stop_width*(s-2.5*s**4+3*s**5-s**6)
    final_position = x72+.2*stop_width*.5
    m = t >= 1.0-1e-12
    x[m], v[m], a[m] = final_position, 0., 0.
    if not np.all(np.isfinite(x)) or np.min(v) < -1e-10 or not 0 < final_position < .3:
        raise RuntimeError("参考轨迹超出行程或非有限")
    return t, x, v, a
