"""
Tests for the `pulsation` feature (linear spheroidal mode with a temperature map).

The light-curve tests compare against the semi-analytic disk-integrated
amplitude of a mode on a spherical, non-rotating star (Watson 1988;
Daszynska-Daszkiewicz et al. 2002; Dupret et al. 2003), in the feature's
convention dF/F = Re[C exp(i Phi)] with Ys = sqrt(4 pi) Y_l^m:

    C = Ys(i, 0) [teffamp e^{i teffphase} alpha_T b_l + radamp (2 + l)(1 - l) b_l]

using blackbody intensities and a fixed linear limb-darkening law, so that the
gravity term vanishes and b_l is known exactly.
"""

import numpy as np
import pytest
import phoebe
from scipy.integrate import quad
from scipy.special import eval_legendre, sph_harm_y

from phoebe.backend import asteroseismo

U = 0.6
FREQ = 5.0


def _b_l(l):
    h = lambda mu: 1 - U*(1 - mu)
    return quad(lambda mu: h(mu)*eval_legendre(l, mu)*mu, 0, 1)[0]/quad(lambda mu: h(mu)*mu, 0, 1)[0]


def _star(incl, times, pblum_mode='component-coupled', ntriangles=8000):
    b = phoebe.default_star()
    b.set_value('ntriangles', component='starA', value=ntriangles)
    b.set_value('period', component='starA', value=1e4)
    b.set_value('incl', component='starA', value=incl)
    b.set_value('distortion_method', value='sphere')
    b.add_dataset('lc', compute_times=list(times), dataset='lc01', passband='Johnson:V', pblum_mode=pblum_mode)
    b.set_value_all('atm', 'blackbody')
    for suffix in ('', '_bol'):
        b.set_value_all('ld_mode' + suffix, 'manual')
        b.set_value_all('ld_func' + suffix, 'linear')
        b.set_value_all('ld_coeffs' + suffix, [U])
    return b


def _complex_amplitude(t, y):
    A = np.vstack([np.cos(2*np.pi*FREQ*t), np.sin(2*np.pi*FREQ*t), np.ones_like(t)]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return complex(c[0], -c[1])


def _alpha_T():
    b = _star(90, [0.0], pblum_mode='absolute')
    b.run_compute(irrad_method='none', model='m0')
    b.set_value('teff', component='starA', value=6000*1.001)
    b.run_compute(irrad_method='none', model='m1')
    return np.log(b.get_value('fluxes', model='m1', context='model')[0]/b.get_value('fluxes', model='m0', context='model')[0])/np.log(1.001)


def _lc_amplitude(l, m, incl, radamp, teffamp, teffphase=0.0):
    t = np.linspace(0, 1/FREQ, 16, endpoint=False)
    b = _star(incl, t)
    b.add_feature('pulsation', component='starA', feature='puls01', l=l, m=m, freq=FREQ,
                  radamp=radamp, teffamp=teffamp, teffphase=teffphase)
    b.run_compute(irrad_method='none')
    f = b.get_value('fluxes', dataset='lc01', context='model')
    return _complex_amplitude(t, f/f.mean() - 1)


def test_harmonics_pole_regular():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 3))
    x = np.vstack([x, [[0, 0, 1], [0, 0, -1]]])     # include both poles
    xhat = x/np.linalg.norm(x, axis=1)[:, None]
    theta = np.arctan2(np.hypot(xhat[:, 0], xhat[:, 1]), xhat[:, 2])
    phi = np.arctan2(xhat[:, 1], xhat[:, 0])
    h = 1e-6
    for l in range(5):
        for m in range(-l, l+1):
            assert np.allclose(asteroseismo.cart_Y(l, m, xhat), sph_harm_y(l, m, theta, phi), atol=1e-12)
            g = asteroseismo.cart_grad_Y(l, m, xhat)
            assert np.all(np.isfinite(g))
            # finite differences of the 0-homogeneous Y(x/|x|) give the surface gradient
            fd = np.stack([(asteroseismo.cart_Y(l, m, xhat + h*e) - asteroseismo.cart_Y(l, m, xhat - h*e))/(2*h)
                           for e in np.eye(3)], axis=-1)
            assert np.allclose(g, fd, atol=1e-6)
            assert np.allclose(np.einsum('ij,ij->i', g, xhat), 0, atol=1e-10)


def test_sectoral_dYdtheta():
    # regression: as_dYdtheta returned 0 for |m| == l
    theta, phi, h = np.linspace(0.3, 2.8, 7), 0.4, 1e-6
    for l in range(1, 5):
        for m in (-l, l):
            fd = (asteroseismo.as_Y(l, m, theta + h, phi) - asteroseismo.as_Y(l, m, theta - h, phi))/(2*h)
            assert np.allclose(asteroseismo.as_dYdtheta(l, m, theta, phi), fd, atol=1e-6)


def test_radial_geometric_term():
    C = _lc_amplitude(0, 0, 90, radamp=1e-3, teffamp=0.0)
    assert abs(abs(C) - 2e-3)/2e-3 < 1e-3
    assert abs(np.angle(C)) < np.radians(0.5)


@pytest.mark.parametrize('l,m,incl,teffphase', [(0, 0, 90, -40.0), (1, 0, 0, 60.0), (2, 0, 60, -120.0)])
def test_temperature_and_geometric_terms(l, m, incl, teffphase):
    alpha_T = _alpha_T()
    radamp, teffamp = 1e-3, 5e-3
    C = _lc_amplitude(l, m, incl, radamp, teffamp, teffphase)
    Ys = np.sqrt(4*np.pi)*sph_harm_y(l, m, np.radians(incl), 0.0)
    C_semi = Ys*_b_l(l)*(teffamp*np.exp(1j*np.radians(teffphase))*alpha_T + radamp*(2 + l)*(1 - l))
    assert abs(C - C_semi)/abs(C_semi) < 3e-3


def test_dipole_geometric_term_vanishes():
    # regression: limb darkening evaluated with unperturbed normals gave ~0.8 radamp
    C = _lc_amplitude(1, 0, 0, radamp=1e-3, teffamp=0.0)
    assert abs(C) < 1e-6


def test_radial_velocity():
    t = np.linspace(0, 1/FREQ, 16, endpoint=False)
    b = _star(90, [0.0])
    b.add_feature('pulsation', component='starA', feature='puls01', l=0, m=0, freq=FREQ, radamp=1e-3)
    b.add_dataset('rv', compute_times=t, dataset='rv01', passband='Johnson:V')
    b.set_value_all('atm', 'blackbody')
    b.set_value_all('ld_mode', 'manual')
    b.set_value_all('ld_func', 'linear')
    b.set_value_all('ld_coeffs', [U])
    b.run_compute(irrad_method='none')
    C = _complex_amplitude(t, b.get_value('rvs', dataset='rv01', context='model'))
    h = lambda mu: 1 - U*(1 - mu)
    projection = quad(lambda mu: h(mu)*mu*mu, 0, 1)[0]/quad(lambda mu: h(mu)*mu, 0, 1)[0]
    expected = 2*np.pi*FREQ*1e-3*1.0*projection*6.957e5/86400.0   # km/s, R = 1 Rsun
    assert abs(abs(C) - expected)/expected < 2e-3
    # receding (positive RV) while the radius decreases: RV = A sin(Phi)
    assert abs(np.degrees(np.angle(C)) + 90) < 1.0


def test_no_state_carried_between_times():
    # regression: velocities accumulated through a ComputedColumn shared with the standard mesh
    t = np.array([0.0, 0.25, 0.5])/FREQ          # Phi = 0, pi/2, pi
    b = _star(90, [0.0])
    b.add_feature('pulsation', component='starA', feature='puls01', l=0, m=0, freq=FREQ, radamp=1e-3, teffamp=5e-3)
    b.add_dataset('mesh', compute_times=t, dataset='mesh01', columns=['vws', 'teffs'])
    b.run_compute(irrad_method='none')
    vw = [b.get_value('vws', time=ti, dataset='mesh01', context='model') for ti in t]
    teffs = [b.get_value('teffs', time=ti, dataset='mesh01', context='model') for ti in t]
    # pulsational velocity zero at Phi = 0, pi (0.005 km/s floor: rotation with P = 1e4 d),
    # omega radamp R = 0.253 km/s at Phi = pi/2
    assert np.abs(vw[0]).max() < 0.01 and np.abs(vw[2]).max() < 0.01
    assert np.abs(vw[1]).max() > 0.24
    # Teff follows 6000 (1 + 5e-3 cos Phi), not renormalised by the displaced areas
    assert np.allclose([np.mean(te) for te in teffs], [6030.0, 6000.0, 5970.0], atol=0.5)


def test_time_dependent():
    # pulsations are not periodic in the orbital phase: PHOEBE must not fold/interpolate in phase
    b = phoebe.default_binary()
    assert not b.hierarchy.is_time_dependent()
    b.add_feature('pulsation', component='primary', feature='puls01', l=0, m=0, freq=FREQ, teffamp=1e-3)
    assert b.hierarchy.is_time_dependent()


@pytest.mark.parametrize('distortion_method', ['roche', 'rotstar'])
def test_binary_teff_map_updates(distortion_method):
    # regression: a synchronous rotstar binary component kept a frozen T map
    t = np.array([0.0, 0.5])/FREQ                 # Phi = 0, pi: dTeff/Teff = +-teffamp
    b = phoebe.default_binary()
    b.set_value('distortion_method', component='primary', value=distortion_method)
    b.add_feature('pulsation', component='primary', feature='puls01', l=0, m=0, freq=FREQ, teffamp=1e-2)
    b.add_dataset('mesh', compute_times=t, dataset='mesh01', columns=['teffs'])
    b.run_compute(irrad_method='none')
    te = [np.mean(b.get_value('teffs', time=ti, component='primary', dataset='mesh01', context='model')) for ti in t]
    assert abs(te[0]/te[1] - 1.01/0.99) < 2e-3


def test_binary_no_accumulated_state():
    # the displaced mesh is rebuilt from the unperturbed standard mesh at every
    # time: one orbit (P = 1 d = 5 pulsation cycles at FREQ) later the binary looks the same
    t = np.array([0.1, 1.1])
    b = phoebe.default_binary()
    b.add_feature('pulsation', component='primary', feature='puls01', l=2, m=1, freq=FREQ, radamp=2e-3, teffamp=5e-3)
    b.add_dataset('lc', compute_times=t, dataset='lc01')
    b.add_dataset('rv', compute_times=t, dataset='rv01')
    b.run_compute(irrad_method='none')
    f = b.get_value('fluxes', dataset='lc01', context='model')
    rv = b.get_value('rvs', component='primary', dataset='rv01', context='model')
    assert abs(f[1]/f[0] - 1) < 1e-6
    assert abs(rv[1] - rv[0]) < 1e-4


def test_checks():
    b = phoebe.default_star()
    b.add_feature('pulsation', component='starA', feature='puls01', l=1, m=2, freq=FREQ, radamp=1e-3)
    assert not b.run_checks_compute().passed
    b = phoebe.default_star()
    b.add_feature('pulsation', component='starA', feature='puls01', l=1, m=1, freq=0.0, radamp=1e-3)
    assert not b.run_checks_compute().passed
    b = phoebe.default_binary(contact_binary=True)
    b.add_feature('pulsation', component='primary', feature='puls01', l=0, m=0, freq=FREQ, radamp=1e-3)
    assert not b.run_checks_compute().passed


def _eccentric_binary(per0, times, ntriangles=1500):
    # heartbeat-like: P = 8 d, e = 0.5, i = 45 deg (no eclipses); t0@system = t0_perpass = 0
    b = phoebe.default_binary()
    b.set_value('period', component='binary', value=8.0)
    b.set_value('sma', component='binary', value=25.0)
    b.set_value('q', component='binary', value=0.65)
    b.set_value('ecc', component='binary', value=0.5)
    b.set_value('per0', component='binary', value=per0)
    b.set_value('incl', component='binary', value=45.0)
    b.flip_constraint('t0_perpass', solve_for='t0_supconj')
    b.set_value('t0_perpass', component='binary', value=0.0)
    b.set_value('requiv', component='primary', value=2.2)
    b.set_value('requiv', component='secondary', value=1.3)
    b.set_value('ntriangles', component='primary', value=ntriangles)
    b.set_value('ntriangles', component='secondary', value=500)
    b.add_dataset('lc', compute_times=times, dataset='lc01', passband='Johnson:V')
    b.set_value_all('atm', 'blackbody')
    for suffix in ('', '_bol'):
        b.set_value_all('ld_mode' + suffix, 'manual')
        b.set_value_all('ld_func' + suffix, 'linear')
        b.set_value_all('ld_coeffs' + suffix, [U])
    return b


@pytest.mark.parametrize('m', [0, -2])
def test_eccentric_teo_phase_locked(m):
    # a tidally excited oscillation (freq = N f_orb) in an eccentric orbit must be a
    # pure sinusoid at N f_orb with phase m (90 deg - per0) at periastron: the mesh
    # frame follows the line of centres, which does not rotate uniformly
    P, N, per0 = 8.0, 20, 30.0
    t = np.linspace(0, P, 161)[:-1]
    fluxes = []
    for add in (False, True):
        b = _eccentric_binary(per0, t)
        if add:
            b.add_feature('pulsation', component='primary', feature='teo', l=2, m=m, freq=N/P, radamp=1e-4, teffamp=1e-3)
        b.run_compute(irrad_method='none')
        fluxes.append(b.get_value('fluxes', dataset='lc01', context='model'))
    y = fluxes[1]/fluxes[0] - 1
    cols = []
    for k in (-1, 0, 1):
        cols += [np.cos(2*np.pi*(N + k)/P*t), np.sin(2*np.pi*(N + k)/P*t)]
    c, *_ = np.linalg.lstsq(np.vstack(cols + [np.ones_like(t)]).T, y, rcond=None)
    C = [complex(c[2*i], -c[2*i+1]) for i in range(3)]
    assert max(abs(C[0]), abs(C[2])) < 1e-2*abs(C[1])
    dphase = (np.degrees(np.angle(C[1])) - m*(90.0 - per0) + 180) % 360 - 180
    assert abs(dphase) < 1.0


def test_teo_harmonic_check():
    b = _eccentric_binary(30.0, np.linspace(0, 80, 50), ntriangles=500)
    b.add_feature('pulsation', component='primary', feature='teo', l=2, m=-2, freq=20/8.0, radamp=1e-4, teffamp=1e-3)
    is_teo = lambda: any('tidally excited' in str(item.message) for item in b.run_checks_compute().items)
    assert is_teo()
    b.set_value('freq', feature='teo', value=20.37/8.0)
    assert not is_teo()


def test_gravity_linearity_check():
    # a 60 c/d p mode on the Sun-like default star has f_g ~ 50: radamp = 0.05 gives dg/g > 1
    b = phoebe.default_star()
    b.add_feature('pulsation', component='starA', feature='puls01', l=0, m=0, freq=60.0, radamp=0.05)
    assert not b.run_checks_compute().passed


def test_checks_use_local_amplitudes():
    # (2,0) peaks at sqrt(5) = 2.24 times its rms: radamp*f_g = 0.6 is fine globally but
    # makes log g undefined near the poles
    b = phoebe.default_star()
    b.add_feature('pulsation', component='starA', feature='puls01', l=2, m=0, freq=60.0, radamp=0.012)
    report = b.run_checks_compute()
    assert not report.passed
    assert any('max|Y|' in str(item.message) for item in report.items)


def test_prograde_teo_with_negative_corotating_frequency():
    # a low-N prograde TEO in a pseudo-synchronous star has omega_c < 0: physical,
    # so a warning (not an error), and the model must be finite
    b = _eccentric_binary(30.0, np.linspace(0, 8.0, 9), ntriangles=500)
    b.set_value('syncpar', component='primary', value=2.8)
    b.add_feature('pulsation', component='primary', feature='teo', l=2, m=-2, freq=5/8.0, radamp=1e-5, teffamp=1e-3)
    report = b.run_checks_compute()
    assert report.passed
    messages = [str(item.message) for item in report.items]
    assert any('omega_c' in msg for msg in messages)
    assert any('spin parameter' in msg for msg in messages)
    b.run_compute(irrad_method='none')
    assert np.all(np.isfinite(b.get_value('fluxes', dataset='lc01', context='model')))


def test_subclass_is_time_dependent():
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from _subclass_for_tests import NoGravityPulsation
    b = phoebe.default_binary()
    assert not b.hierarchy.is_time_dependent()
    b.add_feature(NoGravityPulsation, component='primary', feature='puls01', l=0, m=0, freq=FREQ, radamp=1e-3)
    assert b.hierarchy.is_time_dependent()


def test_save_load_round_trip(tmp_path):
    # the feature's freq shares its qualifier with freq@star: constraints must still
    # resolve to the star's own parameter when a saved bundle is loaded
    b = phoebe.default_binary()
    b.add_feature('pulsation', component='primary', feature='puls01', l=1, m=1, freq=FREQ, radamp=1e-4)
    fn = str(tmp_path/'puls.bundle')
    b.save(fn)
    b2 = phoebe.load(fn)
    assert b2.get_value('freq', feature='puls01', context='feature') == FREQ
    b2.set_value('syncpar', component='primary', value=2.0)
    b2.run_delayed_constraints()      # other tests may leave constraints in delayed mode
    assert abs(b2.get_value('freq', component='primary', context='component') - 2*b.get_value('freq', component='primary', context='component')) < 1e-10


if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')
    test_harmonics_pole_regular()
    test_sectoral_dYdtheta()
    test_radial_geometric_term()
    for args in [(0, 0, 90, -40.0), (1, 0, 0, 60.0), (2, 0, 60, -120.0)]:
        test_temperature_and_geometric_terms(*args)
    test_dipole_geometric_term_vanishes()
    test_radial_velocity()
    test_no_state_carried_between_times()
    test_checks()
