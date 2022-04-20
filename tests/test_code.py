from dsec import DSEC

from events import *

import time

def test_volume() :
    D = DSEC()
    sample = D[1]
    E = sample['events']

    def volume_from_interp(E, res, bins, *args, **kwargs) :
        E_interp = interp_polarity_sparse(E, res, bins, *args, **kwargs)
        V = np.zeros((15, *np.flip(sample['res'])))

        V[E_interp[:, 2].astype(np.int64),
          E_interp[:, 1].astype(np.int64),
          E_interp[:, 0].astype(np.int64)] = E_interp[:, 3]

        return V

    V_erafted = interp_volume(E, D.res, 15, 0, sample['dt'], False)
    V_customized = volume_from_interp(E, sample['res'], 15,
                              sample['dt'], 0, return_ts=False)
    error = np.abs(V_erafted - V_customized) > 1e-6

    V_e_values = V_erafted[error]
    V_c_values = V_customized[error]
    error_idx = np.argwhere(error)

    print(np.abs(V_erafted -
           V_customized).mean())
    print(np.abs(V_erafted - V_customized).max())

    t = time.time()
    for i in range(10) :
        V = interp_volume(E, D.res, 15, 0, sample['dt'], False)
    print(time.time() - t)

    t = time.time()
    for i in range(10):
        V_interp = volume_from_interp(E, sample['res'], 15,
                                      sample['dt'], t_begin=0, return_ts=False)
    print(time.time() - t)

    t = time.time()
    for i in range(10):
        V, mask = interp_volume(E, D.res, 15, 0, sample['dt'], False, True)
        idx = np.argwhere(mask)
        Ep = V[idx[:, 0], idx[:, 1], idx[:, 2]]
        E_vd = np.concatenate([idx[:, 2].reshape(-1, 1),
                           idx[:, 1].reshape(-1, 1),
                           idx[:, 0].reshape(-1, 1),
                           Ep.reshape(-1,1)], axis=1)
    print(time.time() - t)
    t = time.time()
    for i in range(10) :
        E_csr = interp_polarity_sparse(E, sample['res'], 15,
                               sample['dt'], t_begin=0, return_ts=False)
    print(time.time() - t)
    print(len(E_vd))
    print(len(E_csr))
    print((E_vd - E_csr).sum())

def test_timestamps() :
    d = DSEC(append_backward=True)


    t0_begin, t0_end = d.get_ts(0, 0, False, False, 1.)
    t1_begin, t1_end = d.get_ts(0, 1, False, False, 1.)

    assert t0_end == t1_begin

    t0back_begin, t0back_end = d.get_ts(0, 0, True, False, 1.)
    t1back_begin, t1back_end = d.get_ts(0, 1, True, False, 1.)

    assert t0back_end == t1back_begin

    assert t0_begin == t0back_end
    assert t1_begin == t1back_end

    assert t0_end == t1back_end

    t0back_prev = d.get_ts(0, 0, True, True, 1.)
    t0_prev = d.get_ts(0, 0, False, True, 1.)

    assert t0back_prev == t0_prev


if __name__ == "__main__" :
    test_volume()

