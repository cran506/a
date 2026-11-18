"""第二章：自主递推中值+Butterworth SOS级联，不使用filtfilt偷看未来。"""
import numpy as np
from scipy.signal import butter
from physics import njit


@njit(cache=True)
def causal_median(values,window=5):
    x=np.asarray(values);result=np.empty_like(x)
    for i in range(len(x)):
        start=max(0,i-window+1)
        for j in range(x.shape[1]):result[i,j]=np.median(x[start:i+1,j])
    return result


@njit(cache=True)
def sos_recursion(values,sos):
    """直接II型转置SOS；状态按首样本稳态初始化，不把常压当成零。"""
    out=values.copy()
    for section in sos:
        b0,b1,b2,a0,a1,a2=section
        b0/=a0;b1/=a0;b2/=a0;a1/=a0;a2/=a0
        dc=(b0+b1+b2)/(1+a1+a2)
        for j in range(values.shape[1]):
            first=out[0,j];y0=dc*first
            z1=y0-b0*first;z2=b2*first-a2*y0
            for i in range(len(values)):
                xi=out[i,j];yi=b0*xi+z1
                z1=b1*xi-a1*yi+z2;z2=b2*xi-a2*yi
                out[i,j]=yi
    return out


class CascadeFilter:
    def __init__(self,fs=1000.,cutoff=120.,window=5,order=2):
        if window<1 or window%2==0 or not 0<cutoff<fs/2:raise ValueError('中值窗口需奇数，截止频率须低于Nyquist频率')
        self.window=int(window);self.sos=butter(order,cutoff,fs=fs,output='sos')
        self.fs=fs;self.cutoff=cutoff
    def __call__(self,values):
        x=np.asarray(values,dtype=np.float64)
        if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('输入须为有限二维数组')
        return sos_recursion(causal_median(x,self.window),self.sos)
