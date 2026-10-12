"""七段式 S 曲线的解析积分，避免时间网格偏移和末端残差。"""
import numpy as np


def time_grid(duration, dt):
    if not np.isfinite(duration) or not np.isfinite(dt) or dt <= 0 or duration <= 0:
        raise ValueError("时长与采样周期必须为有限正数")
    steps = int(round(duration/dt))
    if not np.isclose(steps*dt, duration, rtol=0, atol=1e-10):
        raise ValueError("总时长必须是采样周期的整数倍")
    return np.arange(steps+1, dtype=float)*dt


def generate_scurve(dist=.1, T_move=1., T_total=1.4, dt=.001):
    """与 Word 相同接口，返回 t、位置、速度、加速度（均为 SI）。

    七段时长比例为 .10/.15/.10/.30/.10/.15/.10；逐段解析积分。
    跃度有界，加速度连续；跃度本身允许在分段边界跳变。
    """
    if not np.isfinite(dist) or not np.isfinite(T_move) or T_move <= 0 or T_total < T_move:
        raise ValueError("位移必须有限，0<T_move<=T_total")
    t = time_grid(T_total, dt)
    durations = np.array([.1, .15, .1, .3, .1, .15, .1])*T_move
    tj, tca = durations[0:2]
    ta = 2*tj+tca
    jerk = dist/(tj*(tj+tca)*(T_move-ta))
    signs = (1, 0, -1, 0, -1, 0, 1)
    r, v, a = (np.zeros_like(t) for _ in range(3))
    start = x0 = v0 = a0 = 0.
    for duration, sign in zip(durations, signs):
        mask = (t >= start-1e-12) & (t < start+duration-1e-12)
        h = t[mask]-start
        j = sign*jerk
        r[mask] = x0+v0*h+a0*h*h/2+j*h**3/6
        v[mask] = v0+a0*h+j*h*h/2
        a[mask] = a0+j*h
        x0 += v0*duration+a0*duration**2/2+j*duration**3/6
        v0 += a0*duration+j*duration**2/2
        a0 += j*duration
        start += duration
    hold = t >= T_move-1e-12
    # 解析末端自然满足 x=dist、v=a=0；仅消去浮点舍入量。
    if not np.allclose([x0, v0, a0], [dist, 0, 0], rtol=0, atol=1e-10):
        raise RuntimeError("S 曲线解析末端不满足边界条件")
    r[hold], v[hold], a[hold] = dist, 0., 0.
    return t, r, v, a


def generate_process_scurve(dist=.13, T_move=1., T_total=1.2, dt=.001):
    """课堂六的十二段工艺 S 轨迹：低速接近、加速、巡航、减速、持位。

    前0.06s采用0.1m/s²柔和接近，再连续升降加速度，到0.16s为0.032m/s；
    0.35s进入0.20m/s巡航，末端停车段时长由目标位移解析求出。
    返回 t,x,v,a,j；这些参考量逐段解析积分，绝不使用仿真输出。
    T_move 缩放整段时间，距离参数必须允许正的巡航与减速时长。
    """
    if not np.isfinite(dist) or dist<=0 or not np.isfinite(T_move) or T_move<=0 or T_total<T_move:
        raise ValueError('工艺轨迹要求正位移且0<T_move<=T_total')
    t=time_grid(T_total,dt)
    scale=T_move
    acceleration=.1/scale
    approach_peak=(.032-.075*scale*acceleration-.003)/(.065*scale)
    peak=(.2-.032-.2*.07/2)/((.07/2+.08+.04/2)*scale)
    specifications=[(.01*scale,acceleration/(.01*scale)),(.05*scale,0.),
                    (.04*scale,(approach_peak-acceleration)/(.04*scale)),
                    (.03*scale,0.),(.03*scale,(.2/scale-approach_peak)/(.03*scale)),
                    (.07*scale,(peak-.2/scale)/(.07*scale)),
                    (.08*scale,0.),(.04*scale,-peak/(.04*scale))]
    x0=v0=a0=0.
    for duration,jerk in specifications:
        x0+=v0*duration+a0*duration**2/2+jerk*duration**3/6
        v0+=a0*duration+jerk*duration**2/2
        a0+=jerk*duration
    brake_duration=(x0+.2*(T_move-.35*scale)-dist)/.1
    brake_ramp=.04*scale
    cruise_duration=T_move-brake_duration-.35*scale
    if brake_duration<2*brake_ramp or cruise_duration<0:
        raise ValueError('位移与速度、工艺段时长不相容')
    brake_acceleration=.2/(brake_duration-brake_ramp)
    specifications += [(cruise_duration,0.),(brake_ramp,-brake_acceleration/brake_ramp),
                       (brake_duration-2*brake_ramp,0.),(brake_ramp,brake_acceleration/brake_ramp)]
    r,v,a,j=(np.zeros_like(t) for _ in range(4))
    start=x0=v0=a0=0.
    for duration,jerk in specifications:
        mask=(t>=start-1e-12)&(t<start+duration-1e-12)
        h=t[mask]-start
        r[mask]=x0+v0*h+a0*h**2/2+jerk*h**3/6
        v[mask]=v0+a0*h+jerk*h**2/2
        a[mask]=a0+jerk*h
        j[mask]=jerk
        x0+=v0*duration+a0*duration**2/2+jerk*duration**3/6
        v0+=a0*duration+jerk*duration**2/2
        a0+=jerk*duration
        start+=duration
    if not np.allclose([start,x0,v0,a0],[T_move,dist,0.,0.],rtol=0,atol=1e-10):
        raise RuntimeError('工艺S轨迹不满足解析末端条件')
    hold=t>=T_move-1e-12
    r[hold],v[hold],a[hold],j[hold]=dist,0.,0.,0.
    return t,r,v,a,j
