import unittest
import numpy as np
from observation_normalization import EmpiricalZScore,fold_actor_first_layer,actor_forward


class ZScoreTests(unittest.TestCase):
    def test_fitted_statistics_and_constant_channel(self):
        samples=np.array([[1.,100.,7.],[2.,200.,7.],[6.,900.,7.]])
        norm=EmpiricalZScore.fit(samples);z=norm.transform(samples)
        np.testing.assert_allclose(z.mean(axis=0),0.,atol=1e-14)
        np.testing.assert_allclose(z[:,:2].std(axis=0),1.)
        np.testing.assert_array_equal(z[:,2],0.)
        self.assertEqual(norm.scale[2],1.)

    def test_coordinate_change_preserves_held_out_nonlinear_actor(self):
        rng=np.random.default_rng(8081)
        raw=rng.normal(size=(80,3))*[.1,20.,3.]+[5.,100.,7.]
        old_mean=np.array([1.,40.,3.]);old_scale=np.array([2.,30.,4.])
        w=[rng.normal(size=(5,3)),rng.normal(size=(1,5))];b=[rng.normal(size=5),rng.normal(size=1)]
        norm=EmpiricalZScore.fit(raw[:50])
        new_w,new_b=fold_actor_first_layer(w,b,old_mean,old_scale,norm)
        np.testing.assert_allclose(actor_forward((raw[50:]-old_mean)/old_scale,w,b),
                                   actor_forward(norm.transform(raw[50:]),new_w,new_b),atol=1e-13)


if __name__=='__main__':unittest.main()
