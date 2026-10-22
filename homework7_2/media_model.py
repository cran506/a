"""同一个公开含气油模型供被控对象和控制器使用，保留上传Twin源码。"""
import math
from hydraulic_digital_twin import HydraulicTwin


class AeratedHydraulicModel(HydraulicTwin):
    """绝热气泡压缩+油液串联柔度，低压有柔度，高压趋近油液模量。

    α0 是回油绝对压力Pr下的气体体积分数，κ=1.4。
    这是课堂模型假设；不修改位置、速度或积分后的压力。
    α0=0时精确恢复原Twin的有效模量。
    """
    def __init__(self,config):
        alpha=config.get('air_fraction_at_pr',0.)
        kappa=config.get('gas_polytropic_exponent',1.4)
        if not math.isfinite(alpha) or not 0<=alpha<1 or not math.isfinite(kappa) or kappa<=0:
            raise ValueError('含气率和气体多变指数无效')
        super().__init__(config)

    def _get_beta(self,pressure):
        oil_beta=HydraulicTwin._get_beta(self,pressure)
        alpha0=self.p.get('air_fraction_at_pr',0.)
        if alpha0==0:
            return oil_beta
        kappa=self.p.get('gas_polytropic_exponent',1.4)
        guarded=max(float(pressure),2000.)
        gas_volume=alpha0*(self.p['Pr']/guarded)**(1/kappa)
        fraction=gas_volume/(1-alpha0+gas_volume)
        mixture=1/((1-fraction)/oil_beta+fraction/(kappa*guarded))
        return max(.001*self.p['beta_e'],mixture)
