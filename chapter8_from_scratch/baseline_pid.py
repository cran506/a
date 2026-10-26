"""固定参数传统 PID：只读取位置/速度误差，不读取故障标签。

控制器接口的误差方向是 reference-actual，单位分别是 mm 与 mm/s。
参考 Gym 的五维原始观测采用 actual-reference，所以调用时显式取反。
kp=1 V/mm、ki=40 V/(mm·s)、kd=.01 V/(mm/s)，全部工况固定。
"""
from __future__ import annotations
import numpy as np


class Controller:
    def __init__(self,initial_voltage,dt=.001,kp=1.,ki=40.,kd=.01,
                 derivative_alpha=.85):
        if not np.isfinite([initial_voltage,dt,kp,ki,kd,derivative_alpha]).all():
            raise ValueError('PID settings must be finite')
        if dt<=0. or not 0.<=derivative_alpha<1.:
            raise ValueError('Invalid sample time or derivative filter')
        self.initial_voltage=float(initial_voltage);self.dt=float(dt)
        self.kp,self.ki,self.kd=float(kp),float(ki),float(kd)
        self.derivative_alpha=float(derivative_alpha)
        self.integral_mm_s=0.;self.filtered_derivative_mm_s=0.

    def update(self,position_error_mm,velocity_error_mms):
        error=float(position_error_mm);velocity_error=float(velocity_error_mms)
        if not np.isfinite([error,velocity_error]).all():
            raise ValueError('PID feedback must be finite')
        a=self.derivative_alpha
        self.filtered_derivative_mm_s=(a*self.filtered_derivative_mm_s+
                                       (1.-a)*velocity_error)
        trial=self.integral_mm_s+error*self.dt
        proposal=(self.initial_voltage+self.kp*error+self.ki*trial+
                  self.kd*self.filtered_derivative_mm_s)
        # Conditional integration removes additional accumulation into a
        # saturated actuator; it does not give the PID a load observer.
        if abs(proposal)<=10. or np.sign(error)!=np.sign(proposal):
            self.integral_mm_s=trial
        voltage=(self.initial_voltage+self.kp*error+self.ki*self.integral_mm_s+
                 self.kd*self.filtered_derivative_mm_s)
        return float(np.clip(voltage,-10.,10.))
