"""静摩擦阈值与有向 Stribeck 滑动分支，单位为 N、m/s。

静止摩擦不是 ``F(v=0)=0``：它由力平衡取 [-Fs, Fs] 内的值。
本模块只给出滑动分支；粘住/释放由 twin_adapter 的混合积分处理。
"""
import numpy as np


def validate_friction(parameters):
    values = [parameters[key] for key in ('Fs', 'Fc', 'vs', 'sigma')]
    if (not all(np.isfinite(value) for value in values)
            or not parameters['Fs'] >= parameters['Fc'] >= 0
            or parameters['vs'] <= 0 or parameters['sigma'] < 0):
        raise ValueError('摩擦需满足 Fs>=Fc>=0、vs>0、sigma>=0，且参数有限')


def _direction(velocity, direction):
    if direction not in (-1, 0, 1):
        raise ValueError('滑动方向必须为 -1、0 或 1')
    return float(direction) if direction else float(np.sign(velocity))


def sliding_friction(velocity, parameters, direction=0):
    """滑动摩擦；direction=±1 可取 v=0 的左右极限 ±Fs。

    事件积分时固定 direction，避免隐式求解器在尚未定位的零速附近
    切换不连续分支。direction=0 则按当前速度取符号；其零值不是静摩擦。
    """
    sign = _direction(velocity, direction)
    magnitude = parameters['Fc']+(parameters['Fs']-parameters['Fc'])*np.exp(
        -(velocity/parameters['vs'])**2)
    return float(sign*magnitude+parameters['sigma']*velocity)


def sliding_friction_slope(velocity, parameters, direction=0):
    """固定滑动分支 dF/dv；不包含 v=0 的符号跳变。"""
    sign = _direction(velocity, direction)
    return float(parameters['sigma']-2*sign*(parameters['Fs']-parameters['Fc'])
                 *velocity/parameters['vs']**2*np.exp(-(velocity/parameters['vs'])**2))


def breakaway_friction(direction, parameters):
    """运动意图为 ±1 时的静摩擦极限；没有运动意图时为零。"""
    return float(_direction(0., direction)*parameters['Fs'])
