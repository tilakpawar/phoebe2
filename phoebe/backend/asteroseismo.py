"""
This module exists because the definition of spherical harmonics used
in scipy and the ones commonly used in asteroseismology are slightly
different. Here we implement spherical harmonics after Arfken (1970),
summarized by Townsend (2003), MNRAS 343, 125.
"""

import numpy as np
from scipy.special import legendre
try:
    from scipy.misc import factorial
except:
    from scipy.special import factorial

def as_legendre(l, m, x):
    """
    @l: non-radial degree (the number of longitudinal and latitudinal node lines)
    @m: longitudinal order (the number of longitudinal node lines)
    @x: argument, typically cos \theta

    Computes associated Legendre polynomials P_l^m in x.

    This function uses scipy's version of Legendre polynomials while
    explicitly imposing the following relation for negative values of m:

      P_l^{-m}(x) = (-1)^m (l-m)!/(l+m)! P_l^m (x)
    """

    m_ = abs(m)
    legendre_poly = legendre(l)
    deriv_legpoly_ = legendre_poly.deriv(m=m_)
    deriv_legpoly = np.polyval(deriv_legpoly_,x)
    P_l_m = (-1)**m_ * (1-x**2)**(m_/2.) * deriv_legpoly
    if m < 0:
        P_l_m = (-1)**m_ * factorial(l-m_)/factorial(l+m_) * P_l_m
    return P_l_m

def as_Y(l, m, theta, phi):
    """
    @l: non-radial degree (the number of longitudinal and latitudinal node lines)
    @m: longitudinal order (the number of longitudinal node lines)
    @theta: colatitude, 0 at the +z pole, pi at the -z pole
    @phi: azimuth, 0 in the x direction, goes from 0 to 2pi

    Spherical harmonic according to Townsend (2003), MNRAS 343, 125:

      Y_l^m (\theta, \phi) = (-1)^m \sqrt{ (2l+1)/4pi (l-m)!/(l+m)! } P_l^m(cos\theta) exp(im\phi)
    """

    factor = (-1)**m * np.sqrt( (2*l+1)/(4*np.pi) * factorial(l-m)/factorial(l+m))
    Plm = as_legendre(l, m, np.cos(theta))
    return factor*Plm*np.exp(1j*m*phi)

def _inv_sin(theta, eps=1e-12):
    """
    1/sin(theta), with the value at the poles set to 0 instead of inf/nan.
    The pole is a coordinate singularity of the spherical-harmonic basis, not
    a physical one; without this a mesh vertex landing on a pole yields nan,
    which then propagates into the coordinates.
    """
    s = np.sin(theta)
    return np.where(np.abs(s) < eps, 0.0, 1.0/np.where(np.abs(s) < eps, 1.0, s))

def as_norm_J(l, m):
    """
    @l: non-radial degree (the number of longitudinal and latitudinal node lines)
    @m: longitudinal order (the number of longitudinal node lines)

    Normalization factor for the derivatives.
    """

    return np.sqrt(float((l**2-m**2))/(4*l**2-1)) if abs(m) < l else 0

def as_norm_atlp1(l, m, Omega, k):
    """
    @l: non-radial degree (the number of longitudinal and latitudinal node lines)
    @m: longitudinal order (the number of longitudinal node lines)
    @Omega: spin parameter (Omega_rot/omega_freq)
    @k:

    Normalization factor for the l+1 term.
    """

    if Omega < 1e-6:
        return 0.0
    return Omega * (l-abs(m)+1)/(l+1) * 2./(2*l+1) * (1-l*k)

def as_norm_atlm1(l, m, Omega, k):
    """
    @l: non-radial degree (the number of longitudinal and latitudinal node lines)
    @m: longitudinal order (the number of longitudinal node lines)
    @Omega: spin parameter (Omega_rot/omega_freq)
    @k:

    Normalization factor for the l-1 term.
    """

    if Omega < 1e-6:
        return 0.0
    return Omega * (l+abs(m))/l * 2./(2*l+1) * (1 + (l+1)*k)

def as_dYdtheta(l, m, theta, phi):
    """
    @l: non-radial degree (the number of longitudinal and latitudinal node lines)
    @m: longitudinal order (the number of longitudinal node lines)
    @theta: colatitude, 0 at the +z pole, pi at the -z pole
    @phi: azimuth, 0 in the x direction, goes from 0 to 2pi

    The derivative of a spherical harmonic wrt colatitude.

      sin(theta)*dY/dtheta = (l*J_{l+1}^m * Y_{l+1}^m - (l+1)*J_l^m * Y_{l-1,m})
    """

    if abs(m) > l:
        Y = 0.
    else:
        factor = _inv_sin(theta)
        term1 = l     * as_norm_J(l+1, m) * as_Y(l+1, m, theta, phi)
        # Y_{l-1}^m does not exist for |m| = l (its weight J_l^m is 0, but
        # as_Y would return inf*0 = nan for m = -l)
        term2 = (l+1) * as_norm_J(l,   m) * as_Y(l-1, m, theta, phi) if abs(m) < l else 0.
        Y = factor * (term1 - term2)
    return Y

def as_dYdphi(l, m, theta, phi):
    """
    @l: non-radial degree (the number of longitudinal and latitudinal node lines)
    @m: longitudinal order (the number of longitudinal node lines)
    @theta: colatitude, 0 at the +z pole, pi at the -z pole
    @phi: azimuth, 0 in the x direction, goes from 0 to 2pi

    The derivative of a spherical harmonic wrt longitude.

      dY/dphi = i*m*Y
    """

    return 1j*m*as_Y(l, m, theta, phi)

def as_xi_r(l, m, theta, phi, wt):
    """
    @l: non-radial degree (the number of longitudinal and latitudinal node lines)
    @m: longitudinal order (the number of longitudinal node lines)
    @theta: colatitude, 0 at the +z pole, pi at the -z pole
    @phi: azimuth, 0 in the x direction, goes from 0 to 2pi
    @wt: omega*t, where omega=2 \pi \nu is the angular frequency, and t is time

    Unscaled radial displacement, see
        Aerts et al. (2010), the Asteroseismology book and
        Zima W, A&A 455, 227–234 (2006), Appendix A
    """

    return as_Y(l, m, theta, phi) * np.exp(1j*wt)

def as_xi_theta(l, m, theta, phi, wt, Omega=0, k=0):
    """
    @l: non-radial degree (the number of longitudinal and latitudinal node lines)
    @m: longitudinal order (the number of longitudinal node lines)
    @theta: colatitude, 0 at the +z pole, pi at the -z pole
    @phi: azimuth, 0 in the x direction, goes from 0 to 2pi
    @wt: omega*t, where omega=2 \pi \nu is the angular frequency, and t is time
    @Omega: spin parameter
    @k:

    Unscaled co-latitudinal displacement, see
        Aerts et al. (2010), the Asteroseismology book and
        Zima W, A&A 455, 227–234 (2006), Appendix A
    """

    term1 = k * as_dYdtheta(l, m, theta, phi) * np.exp(1j*wt)
    term2 = as_norm_atlp1(l, m, Omega, k) * _inv_sin(theta) * as_dYdphi(l+1, m, theta, phi) * np.exp(1j*(wt + np.pi/2))
    term3 = as_norm_atlm1(l, m, Omega, k) * _inv_sin(theta) * as_dYdphi(l-1, m, theta, phi) * np.exp(1j*(wt - np.pi/2))
    return term1 + term2 + term3

def as_xi_phi(l, m, theta, phi, wt, Omega=0, k=0):
    """
    @l: non-radial degree (the number of longitudinal and latitudinal node lines)
    @m: longitudinal order (the number of longitudinal node lines)
    @theta: colatitude, 0 at the +z pole, pi at the -z pole
    @phi: azimuth, 0 in the x direction, goes from 0 to 2pi
    @wt: omega*t, where omega=2 \pi \nu is the angular frequency, and t is time
    @Omega: spin parameter
    @k:

    Unscaled longitudinal displacement, see
        Aerts et al. (2010), the Asteroseismology book and
        Zima W, A&A 455, 227–234 (2006), Appendix A
    """

    term1 = k * _inv_sin(theta) * as_dYdphi(l, m, theta, phi) * np.exp(1j*wt)
    term2 = -as_norm_atlp1(l, m, Omega, k) * as_dYdtheta(l+1, m, theta, phi)*np.exp(1j*(wt+np.pi/2))
    term3 = -as_norm_atlm1(l, m, Omega, k) * as_dYdtheta(l-1, m, theta, phi)*np.exp(1j*(wt-np.pi/2))
    return term1 + term2 + term3


# ---------------------------------------------------------------------------
# Pole-regular spherical harmonics on Cartesian unit vectors (used by the
# Pulsation feature).  Convention: orthonormal Y_l^m with the Condon-Shortley
# phase, identical to scipy.special.sph_harm_y and to Townsend (2003), MNRAS
# 343, 125, Eq. 1.  Unlike as_Y above, (-1)^m is applied only once.
# ---------------------------------------------------------------------------

def _unit(x):
    x = np.asarray(x, dtype=float)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

def cart_Y(l, m, xhat):
    """
    Complex orthonormal Y_l^m at the directions xhat (..., 3), which need not
    be normalised.  Returns 0 for l < 0 or |m| > l (convenient for the ladder
    relations in cart_grad_Y).
    """
    xhat = _unit(xhat)
    if l < 0 or abs(m) > l:
        return np.zeros(xhat.shape[:-1], dtype=complex)
    # arctan2 instead of arccos keeps full precision next to the poles
    theta = np.arctan2(np.hypot(xhat[..., 0], xhat[..., 1]), xhat[..., 2])
    phi = np.arctan2(xhat[..., 1], xhat[..., 0])
    try:
        from scipy.special import sph_harm_y
        return sph_harm_y(l, m, theta, phi)
    except ImportError:
        # scipy < 1.15: sph_harm(m, l, azimuth, colatitude)
        from scipy.special import sph_harm
        return sph_harm(m, l, phi, theta)

def cart_grad_Y(l, m, xhat):
    """
    Surface gradient grad_S Y_l^m on the unit sphere, in Cartesian components
    (..., 3), complex.  Regular everywhere, including the poles: it is the
    Cartesian gradient of the solid harmonic r^l Y_l^m (from the ladder
    relations) minus its radial part, grad_S Y = grad(r^l Y) - l Y xhat.

    grad_S = theta_hat d/dtheta + phi_hat (1/sin theta) d/dphi, so the
    horizontal displacement of a spheroidal mode is xi_h = K R grad_S Y.
    """
    xhat = _unit(xhat)
    if l == 0 or abs(m) > l:
        return np.zeros(xhat.shape, dtype=complex)
    c = np.sqrt((2.0*l + 1.0)/(2.0*l - 1.0))
    dz = c*np.sqrt(max((l-m)*(l+m), 0)) * cart_Y(l-1, m, xhat)
    dplus = c*np.sqrt(max((l-m)*(l-m-1), 0)) * cart_Y(l-1, m+1, xhat)     # (d_x + i d_y)
    dminus = -c*np.sqrt(max((l+m)*(l+m-1), 0)) * cart_Y(l-1, m-1, xhat)   # (d_x - i d_y)
    grad = np.stack([0.5*(dplus + dminus), (dplus - dminus)/2j, dz], axis=-1)
    return grad - l*cart_Y(l, m, xhat)[..., None]*xhat

def mode_frame(axis, ref=(1.0, 0.0, 0.0)):
    """
    Rotation matrix whose ROWS are the unit vectors (x', y', z') of the mode
    frame: z' = axis and x' is the projection of `ref` onto the plane normal
    to the axis (y is used instead if `ref` is parallel to the axis).
    x_mode = x @ M.T and v = v_mode @ M.  A different choice of x' is
    equivalent to shifting the pulsation phase by m times the rotation angle.
    """
    z = _unit(np.asarray(axis, dtype=float))
    ref = np.asarray(ref, dtype=float)
    if abs(np.dot(_unit(ref), z)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    x = ref - np.dot(ref, z)*z
    x /= np.linalg.norm(x)
    return np.vstack([x, np.cross(z, x), z])

def orbital_harmonics(freqs, forb, tspan=None, max_frac=0.01):
    """
    Flag frequencies that are integer multiples N*forb of the orbital
    frequency, as tidally excited oscillations (TEOs) are.

    A frequency counts as a harmonic if |f - N forb| <= tol, with tol the
    Rayleigh resolution 1/tspan of the data when tspan is given, but never more
    than max_frac*forb (so at most a fraction 2*max_frac of random frequencies
    is flagged by chance).

    Returns (N, f - N forb, tol, is_harmonic) with N, the offsets and the
    flags as arrays.
    """
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    N = np.rint(freqs/forb).astype(int)
    df = freqs - N*forb
    tol = max_frac*forb if not tspan else min(1.0/tspan, max_frac*forb)
    return N, df, tol, (N >= 1) & (np.abs(df) <= tol)


if __name__=='__main__':
    print('Checking whether negative m values are computed correctly:'),
    ls, x = [0, 1, 2, 3, 4, 5], np.cos(np.linspace(0, np.pi, 100))
    check = 0
    for l in ls:
        for m in range(-l, l+1, 1):
            Ppos = as_legendre(l,  m, x)
            Pneg = as_legendre(l, -m, x)
            mycheck = Pneg, (-1)**m * factorial(l-m)/factorial(l+m) * Ppos
            check += sum(abs(mycheck[0]-mycheck[1])>1e-10)
    if check == 0:
        print('yes')
    else:
        print('no -- exiting')
        exit()
