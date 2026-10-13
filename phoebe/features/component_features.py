import numpy as np
import astropy.units as u

from phoebe.parameters import FloatParameter, IntParameter, BoolParameter, ChoiceParameter, ParameterSet
from phoebe.features.common import BaseFeature


__all__ = ['ComponentFeature', 'Spot', 'GeometricPulsation', 'Pulsation']

_skip_filter_checks = {'check_default': False, 'check_visible': False}


class ComponentFeature(BaseFeature):
    """
    Note that for all features, each of the methods below will be called.  So
    changing the coordinates WILL affect the original/intrinsic loggs which
    will then be used as input for that method call.

    In other words, its probably safest if each feature only overrides a
    SINGLE one of the methods.  Overriding multiple methods should be done
    with great care.

    Each feature may or may not require recomputing a mesh, depending on the
    kind of change it exacts to the mesh. For example, pulsations will require
    recomputing a mesh while spots will not. By default, the mesh will be
    recomputed (set in this superclass' `__init__()` method) but inherited
    classes should overload `self.requires_remeshing`.
    """
    allowed_component_kinds = ['star', 'envelope']
    allowed_dataset_kinds = [None]

    def __repr__(self):
        return f"<ComponentFeature: {self.__class__.__name__}>"

    def requires_remeshing(self):
        """
        Whether this feature requires remeshing of the component mesh.
        """
        return False

    def requires_recompute_instantaneous(self):
        """
        Whether local quantities (loggs, teffs, intensities) must be recomputed
        at every time step even if the mesh itself is reused.
        """
        return False

    def update_frame(self, time, euler, s):
        """
        Called by the Body at every time step, before the other hooks, with the
        Euler angles (etheta, elongan, eincl) that place the mesh frame in the
        system and the spin axis `s` in the mesh frame.  The mesh frame follows
        the line of centres, so in eccentric orbits it does not rotate
        uniformly: features whose pattern is fixed in inertial space or in the
        rotating star can use this to follow the frame exactly.
        """
        return

    def cartesian_to_spherical(self, roche_coords):
        """
        Transform Cartesian Roche coordinates to spherical coordinates.
        
        Parameters
        ----------
        roche_coords : array_like
            Array of Cartesian coordinates with shape (N, 3) where columns
            are [x, y, z]
            
        Returns
        -------
        r : ndarray
            Radial distance from origin
        theta : ndarray
            Colatitude (polar angle) in radians, measured from positive
            z-axis [0, π]
        phi : ndarray
            Longitude (azimuthal angle) in radians, measured from positive
            x-axis [-π, π]
        """
        import numpy as np
        
        x, y, z = roche_coords[:, 0], roche_coords[:, 1], roche_coords[:, 2]
        r = np.sqrt((roche_coords**2).sum(axis=1))
        theta = np.arccos(z/r)  # colatitude [0, π]
        phi = np.arctan2(y, x)  # longitude [-π, π]
        
        return r, theta, phi

    def modify_coords_for_computations(self, coords_for_computations, s, t):
        """
        Method for a feature to modify the coordinates.  Coordinates are
        modified AFTER scaling but BEFORE being placed in orbit.

        NOTE: coords_for_computations affect physical properties only and
        not geometric properties (areas, eclipse detection, etc).  If you
        want to override geometric properties, use the hook for
        modify_coords_for_observations as well.

        Features that affect coordinates_for_computations should override
        this method
        """
        return coords_for_computations

    def modify_coords_for_observations(self, coords_for_computations, coords_for_observations, s, t):
        """
        Method for a feature to modify the coordinates.  Coordinates are
        modified AFTER scaling but BEFORE being placed in orbit.

        NOTE: coords_for_observations affect the geometry only (areas of each
        element and eclipse detection) but WILL NOT affect any physical
        parameters (loggs, teffs, intensities).  If you want to override
        physical parameters, use the hook for modify_coords_for_computations
        as well.

        Features that affect coordinates_for_observations should override this method.
        """
        return coords_for_observations

    def modify_velocities(self, velocities, coords_for_observations, s=[0., 0., 1.], t=None):
        """
        Method for a feature to add surface velocities in the star's own
        (co-rotating) frame, BEFORE rotation and orbital motion are added.

        Units are those of coords_for_observations per day (solRad/d).

        Features that move the surface (e.g. pulsations) should override this
        method and return a NEW array (do not modify `velocities` in place: the
        array may be shared with the standard mesh).
        """
        return velocities

    def modify_rvs(self, rvs, orbit_vel, roche_coords, s=[0., 0., 1.], t=None):
        """
        Method for a feature to modify the radial velocities.

        Features that affect radial velocities (RV+LP datasets) should override this method

        NOTE: orbit_vel[2] is in the OPPOSITE direction of the radial velocity
        """
        return rvs

    def modify_loggs(self, loggs, roche_coords, s=[0., 0., 1.], t=None):
        """
        Method for a feature to modify the loggs.

        Features that affect loggs should override this method
        """
        return loggs

    def modify_teffs(self, teffs, roche_coords, s=[0., 0., 1.], t=None):
        """
        Method for a feature to modify the teffs.

        Features that affect teffs should override this method
        """
        return teffs

    def modify_intensities(self, abs_normal_intensities, abs_intensities,
                           mus, pblum_scale, extinct_factors, boost_factors,
                           roche_coords, s=[0., 0., 1.], t=None):
        """
        Method for a feature to modify the intensities.
        Features that affect intensities should override this method

        Arguments
        ----------
        * `abs_normal_intensities` (ndarray): Absolute normal intensities, already multiplied
            by `extinct_factors`.
        * `abs_intensities` (ndarray): Absolute projected intensities, already multiplied by
            `extinct_factors` and `boost_factors`.
        * `mus` (ndarray): Cosine of the angle between the normal vector and the line of sight
        * `pblum_scale` (ndarray): Scale factor for the pblum, that will be applied to the abs_intensities
            AFTER modify_intensities to result in the scaled intensities
        * `extinct_factors` (ndarray): Extinction factors for the intensities, already applied
        * `boost_factors` (ndarray): Boost factors for the intensities, already applied
        * `roche_coords` (ndarray): Roche coordinates for the computations
        * `s` (array-like): Spin vector in Roche coordinates
        * `t` (float): Current time
        """
        return abs_normal_intensities, abs_intensities


class Spot(ComponentFeature):
    @classmethod
    def create_feature_parameters(cls, feature, **kwargs):
        """
        Create a <phoebe.parameters.ParameterSet> for a spot feature.

        Generally, this will be used as an input to the kind argument in
        <phoebe.frontend.bundle.Bundle.add_feature>.  If attaching through
        <phoebe.frontend.bundle.Bundle.add_feature>, all `**kwargs` will be
        passed on to set the values as described in the arguments below.  Alternatively,
        see <phoebe.parameters.ParameterSet.set_value> to set/change the values
        after creating the Parameters.

        Allowed to attach to:
        * components with kind: star
        * datasets: not allowed

        Arguments
        ----------
        * `colat` (float/quantity, optional): colatitude of the center of the spot
            wrt spin axis.
        * `long` (float/quantity, optional): longitude of the center of the spot wrt
            spin axis.
        * `radius` (float/quantity, optional): angular radius of the spot.
        * `relteff` (float/quantity, optional): temperature of the spot relative
            to the intrinsic temperature.

        Returns
        --------
        * (<phoebe.parameters.ParameterSet>, list): ParameterSet of all newly created
            <phoebe.parameters.Parameter> objects and a list of all necessary
            constraints.
        """
        params = []
        params += [FloatParameter(qualifier="colat", value=kwargs.get('colat', 0.0), default_unit=u.deg, description='Colatitude of the center of the spot wrt spin axis')]
        params += [FloatParameter(qualifier="long", value=kwargs.get('long', 0.0), default_unit=u.deg, description='Longitude of the center of the spot wrt spin axis')]
        params += [FloatParameter(qualifier='radius', value=kwargs.get('radius', 1.0), limits=(0, 180), default_unit=u.deg, description='Angular radius of the spot')]
        params += [FloatParameter(qualifier='relteff', value=kwargs.get('relteff', 1.0), limits=(0, None), default_unit=u.dimensionless_unscaled, description='Temperature of the spot relative to the intrinsic temperature')]

        return ParameterSet(params), []

    @classmethod
    def parse_bundle(cls, b, feature_ps):
        """
        Initialize a Spot feature from the bundle.
        """
        import numpy as np
        colat = feature_ps.get_value(qualifier='colat', unit=u.rad, **_skip_filter_checks)
        longitude = feature_ps.get_value(qualifier='long', unit=u.rad, **_skip_filter_checks)

        if len(b.hierarchy.get_stars())>=2:
            star_ps = b.get_component(component=feature_ps.component, **_skip_filter_checks)
            orbit_ps = b.get_component(component=b.hierarchy.get_parent_of(feature_ps.component), **_skip_filter_checks)
            # TODO: how should this handle dpdt?

            # we won't use syncpar directly because that is defined wrt sidereal period and we want to make sure
            # this translated to roche longitude correctly.  In the non-apsidal motion case
            # syncpar = period_anom_orb / period_star
            period_anom_orb = orbit_ps.get_value(qualifier='period_anom', unit=u.d, **_skip_filter_checks)
            period_star = star_ps.get_value(qualifier='period', unit=u.d, **_skip_filter_checks)
            rot_dlongdt = 2 * np.pi * (period_anom_orb/period_star - 1) / period_anom_orb
        else:
            star_ps = b.get_component(component=feature_ps.component, **_skip_filter_checks)
            rot_dlongdt = star_ps.get_value(qualifier='freq', unit=u.rad/u.d, **_skip_filter_checks)
            longitude += np.pi/2

        radius = feature_ps.get_value(qualifier='radius', unit=u.rad, **_skip_filter_checks)
        relteff = feature_ps.get_value(qualifier='relteff', unit=u.dimensionless_unscaled, **_skip_filter_checks)

        t0 = b.get_value(qualifier='t0', context='system', unit=u.d, **_skip_filter_checks)

        return dict(colat=colat, longitude=longitude, rot_dlongdt=rot_dlongdt, radius=radius, relteff=relteff, t0=t0)

    @classmethod
    def run_checks_compute(cls, b, feature_ps, compute_ps):
        items = []
        relteff_param = feature_ps.get_parameter(qualifier='relteff', **_skip_filter_checks)
        radius_param = feature_ps.get_parameter(qualifier='radius', **_skip_filter_checks)
        if relteff_param.get_value(**_skip_filter_checks) == 1:
            items += [{'msg': 'relteff of spot is 1.0 which will have no affect',
                       'params': [relteff_param],
                       'is_error': False}]
        if radius_param.get_value(**_skip_filter_checks) == 0:
            items += [{'msg': 'radius of spot is 0.0 which will have no affect',
                       'params': [radius_param],
                       'is_error': False}]
        return items

    def requires_remeshing(self):
        return True

    def instantaneous_position(self, s, time):
        """
        s is the spin vector in roche coordinates
        time is the current time
        """
        t = time - self.kwargs['t0']
        longitude = self.kwargs['longitude'] + self.kwargs['rot_dlongdt'] * t
        colat = self.kwargs['colat']
        return longitude, colat

    def pointing_vector(self, s, time):
        """
        s is the spin vector in roche coordinates
        time is the current time
        """
        import numpy as np
        longitude, colat = self.instantaneous_position(s, time)

        """
        define the basis vectors in the spin (primed) coordinates in terms of
        the Roche coordinates.
        - the z' direction is explicitly given by the spin vector
        ez' = s
        - the x' direction should point in the longitudinal direction that the 
        companion is in the rotating frame, which means that the y' direction should 
        always be orthogonal to x. Thus, to get this, we'll first calculate the y' 
        direction by taking the cross product of the z' and x directions 
        ey' = ez' x ex
        if the pole aligns with the x-axis, then the y' direction is the z direction
        - the x' direction is then defined by the cross product of the other two
        ex' = ey' x ez'
        """
        ex = np.array([1., 0., 0.])
        ezp = s / np.linalg.norm(s)
        if (s == ex).all():
            eyp = np.array([0., 0., 1.])
        else:
            eyp = np.cross(ezp, ex)
        exp = np.cross(eyp, ezp)

        # now we can express the pointing vector in terms of the primed basis
        pv = (np.sin(colat)*np.cos(longitude)*exp +
              np.sin(colat)*np.sin(longitude)*eyp +
              np.cos(colat)*ezp)

        # renormalize and return pointing vector
        return pv / np.linalg.norm(pv)

    def modify_teffs(self, teffs, roche_coords, s=[0., 0., 1.], t=None):
        """
        Change the local effective temperatures for any values within the
        "cone" defined by the spot.  Any teff within the spot will have its
        current value multiplied by the "relteff" factor

        :parameter array teffs: array of teffs for computations
        :parameter array coords: array of coords for computations
        :t float: current time
        """
        import numpy as np
        s = np.asarray(s)
        if t is None:
            # then assume at t0
            t = self.kwargs['t0']

        pointing_vector = self.pointing_vector(s, t)
        self.logger.debug("spot.modify_teffs at t={} with pointing_vector={} and radius={}".format(t, pointing_vector, self.kwargs['radius']))

        cos_alpha_coords = np.dot(roche_coords, pointing_vector) / np.linalg.norm(roche_coords, axis=1)
        cos_alpha_spot = np.cos(self.kwargs['radius'])

        filter_ = cos_alpha_coords > cos_alpha_spot
        teffs[filter_] = teffs[filter_] * self.kwargs['relteff']

        return teffs


class GeometricPulsation(ComponentFeature):
    def requires_remeshing(self):
        return True

    def requires_recompute_instantaneous(self):
        # pulsations change the surface at every time step (also marks the
        # system as time-dependent: no phase folding)
        return True
        
    @classmethod
    def create_feature_parameters(cls, feature, **kwargs):
        params = []
        params += [FloatParameter(qualifier='freq', value=kwargs.get('freq', 1.0), default_unit=u.d**-1, limits=(0, None), description='Pulsation frequency')]
        params += [FloatParameter(qualifier='radamp', value=kwargs.get('radamp', 0.0), default_unit=u.dimensionless_unscaled, description='Radial amplitude of the pulsation')]
        params += [FloatParameter(qualifier='phase', value=kwargs.get('phase', 0.0), default_unit=u.dimensionless_unscaled, description='Phase of pulsations at time t0@system')]
        params += [IntParameter(qualifier='l', value=kwargs.get('l', 0), limits=(0, None), description='Spherical harmonic degree l')]
        params += [IntParameter(qualifier='m', value=kwargs.get('m', 0), description='Spherical harmonic order m')]
        return ParameterSet(params), []

    @classmethod
    def parse_bundle(cls, b, feature_ps):
        """
        Initialize a Geometric Pulsation feature from the bundle.
        """
        from phoebe import c
        freq = feature_ps.get_value(qualifier='freq', unit=u.d**-1, **_skip_filter_checks)
        radamp = feature_ps.get_value(qualifier='radamp', unit=u.dimensionless_unscaled, **_skip_filter_checks)
        phase = feature_ps.get_value(qualifier='phase', unit=u.dimensionless_unscaled, **_skip_filter_checks)
        l = feature_ps.get_value(qualifier='l', unit=u.dimensionless_unscaled, **_skip_filter_checks)
        m = feature_ps.get_value(qualifier='m', unit=u.dimensionless_unscaled, **_skip_filter_checks)

        t0 = b.get_value(qualifier='t0', context='system', unit=u.d, **_skip_filter_checks)

        GM = c.G.to('solRad3 / (solMass d2)').value*b.get_value(qualifier='mass', component=feature_ps.component, context='component', unit=u.solMass, **_skip_filter_checks)
        R = b.get_value(qualifier='requiv', component=feature_ps.component, context='component', unit=u.solRad, **_skip_filter_checks)

        # Cowling assumption
        tanamp = GM/R**3/freq**2

        return dict(radamp=radamp, freq=freq, phase=phase, t0=t0, l=l, m=m, tanamp=tanamp)

    def modify_coords_for_computations(self, coords_for_computations, s, t):
        """
        """
        from phoebe.backend import asteroseismo

        x, y, z = coords_for_computations[:, 0], coords_for_computations[:, 1], coords_for_computations[:, 2]
        r = np.sqrt((coords_for_computations**2).sum(axis=1))
        theta = np.arccos(z/r)
        phi = np.arctan2(y, x)
        phase = 2*np.pi*self.kwargs['freq']*(t - self.kwargs['t0']) + self.kwargs['phase']

        l, m = self.kwargs['l'], self.kwargs['m']
        xi_r = self.kwargs['radamp'] * np.sqrt(4.*np.pi) * asteroseismo.as_xi_r(l, m, theta, phi, phase)
        if l > 0:
            xi_t = self.kwargs['tanamp'] * np.sqrt(4.*np.pi) * asteroseismo.as_xi_theta(l, m, theta, phi, phase)
            xi_p = self.kwargs['tanamp'] * np.sqrt(4.*np.pi) * asteroseismo.as_xi_phi(l, m, theta, phi, phase)
        else:
            xi_t = np.zeros_like(theta)
            xi_p = np.zeros_like(phi)

        new_r = r + xi_r.real
        new_theta = theta + xi_t.real
        new_phi = phi + xi_p.real

        new_coords = np.zeros(coords_for_computations.shape)
        new_coords[:, 0] = new_r * np.sin(new_theta) * np.cos(new_phi)
        new_coords[:, 1] = new_r * np.sin(new_theta) * np.sin(new_phi)
        new_coords[:, 2] = new_r * np.cos(new_theta)

        return new_coords

    def modify_coords_for_observations(self, coords_for_computations, coords_for_observations, s, t):
        """
        Displacement equations:

          xi_r(r, theta, phi)     = a(r) Y_lm (theta, phi) exp(-i*2*pi*f*t)
          xi_theta(r, theta, phi) = b(r) dY_lm/dtheta (theta, phi) exp(-i*2*pi*f*t)
          xi_phi(r, theta, phi)   = b(r)/sin(theta) dY_lm/dphi (theta, phi) exp(-i*2*pi*f*t)

        where:

          b(r) = a(r) GM/(R^3*f^2)
        """
        from phoebe.backend import asteroseismo

        x, y, z = coords_for_observations[:, 0], coords_for_observations[:, 1], coords_for_observations[:, 2]
        r = np.sqrt((coords_for_observations**2).sum(axis=1))
        theta = np.arccos(z/r)
        phi = np.arctan2(y, x)
        phase = 2*np.pi*self.kwargs['freq']*(t - self.kwargs['t0']) + self.kwargs['phase']

        l, m = self.kwargs['l'], self.kwargs['m']
        xi_r = self.kwargs['radamp'] * np.sqrt(4.*np.pi) * asteroseismo.as_xi_r(l, m, theta, phi, phase)
        if l > 0:
            xi_t = self.kwargs['tanamp'] * np.sqrt(4.*np.pi) * asteroseismo.as_xi_theta(l, m, theta, phi, phase)
            xi_p = self.kwargs['tanamp'] * np.sqrt(4.*np.pi) * asteroseismo.as_xi_phi(l, m, theta, phi, phase)
        else:
            xi_t = np.zeros_like(theta)
            xi_p = np.zeros_like(phi)

        new_r = r + xi_r.real
        new_theta = theta + xi_t.real
        new_phi = phi + xi_p.real

        new_coords = np.zeros(coords_for_observations.shape)
        new_coords[:, 0] = new_r * np.sin(new_theta) * np.cos(new_phi)
        new_coords[:, 1] = new_r * np.sin(new_theta) * np.sin(new_phi)
        new_coords[:, 2] = new_r * np.cos(new_theta)

        return new_coords

def _peak_factors(l, m, n=181):
    """max |sqrt(4 pi) Y_l^m| and max |sqrt(4 pi) grad_S Y_l^m| over the sphere: the local
    amplitudes are the global ones times these factors (sqrt(2l+1) for m = 0)."""
    from phoebe.backend import asteroseismo
    th, ph = np.meshgrid(np.linspace(0, np.pi, n), np.linspace(0, 2*np.pi, 2*n, endpoint=False), indexing='ij')
    x = np.stack([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)], axis=-1).reshape(-1, 3)
    Y = np.sqrt(4*np.pi)*np.abs(asteroseismo.cart_Y(l, m, x))
    gY = np.sqrt(4*np.pi)*np.linalg.norm(np.abs(asteroseismo.cart_grad_Y(l, m, x)), axis=-1)
    return float(Y.max()), float(gY.max())


class Pulsation(ComponentFeature):
    r"""
    A single linear spheroidal pulsation mode (l, m) with a physical surface
    map: displacement (radial and horizontal), Lagrangian effective-temperature
    perturbation, effective-gravity perturbation and pulsational velocity.

    Every perturbation is the real part of a complex amplitude times
    Ys = sqrt(4 pi) Y_l^m(x') exp(i Phi(t)), with Y_l^m orthonormal including
    the Condon-Shortley phase (scipy, Townsend 2003) and x' the direction in
    the mode frame.  With this normalisation the amplitudes of a radial mode
    are fractional amplitudes.  Following Cugier et al. (1994), Dupret et al.
    (2003) and Townsend (2003), at a surface point at radius r:

        xi / r      = radamp Re{[Ys r_hat + K grad_S Ys] e^{i Phi}}
        dTeff/Teff  = teffamp Re{e^{i teffphase} Ys e^{i Phi}}
        dg_eff/g    = -radamp f_g Re{Ys e^{i Phi}}
        v           = d xi/dt in the star's co-rotating frame

    so teffamp = radamp f_T and teffphase = psi_T in the notation of Dupret
    et al. (2003): f_T and psi_T are the non-adiabatic Teff amplitude and
    phase relative to the radial displacement, here treated as free (fittable)
    parameters instead of being taken from a pulsation code.  K and f_g follow
    from the frequency in the star's co-rotating frame, omega_c, in the
    Cowling approximation with a zero Lagrangian pressure perturbation at the
    surface:  K = G M / (omega_c^2 R^3),  f_g = 2 + 1/K.

    For axis='spin' the pattern is Re{Ys(theta, psi) exp(i(2 pi freq (t - t0)
    + phase))}, with psi the azimuth about the spin axis in inertial space,
    measured from the direction of the companion at t0@system, so freq is the
    frequency seen by the observer.  The mesh frame follows the line of
    centres, so in the mesh frame Phi(t) = 2 pi freq (t - t0) + m beta(t) +
    phase, where beta(t) is the angle the mesh frame has turned about the
    spin axis since t0.  beta is taken from the orbit at every time step, so
    it is exact in eccentric orbits and with apsidal motion, and it reduces
    to 2 pi (t - t0) / period_anom for circular orbits (0 for single stars).
    The frequency in the star's co-rotating frame is
    omega_c = 2 pi freq + m Omega_star.  With exp(+i(m phi + omega t)), modes
    with m > 0 are retrograde.  For axis='tidal' the mode axis is the line of
    centres, the pattern is fixed in the co-rotating frame and freq is the
    frequency in that frame (the centre of the observed multiplet).

    Tidally excited oscillations (TEOs) are modes forced at harmonics of the
    orbit, freq = N / period_anom (run_checks flags such frequencies).  With
    t0@system = t0_perpass the azimuth origin is the periastron direction and
    `phase` is the phase of the radial displacement relative to the tidal
    forcing: 0 or 180 deg for an adiabatic response.  Aligned orbits force
    l = 2 modes with m = 0 and m = -2 (prograde) in this convention.
    """
    allowed_component_kinds = ['star']

    @classmethod
    def create_feature_parameters(cls, feature, **kwargs):
        """
        Create a <phoebe.parameters.ParameterSet> for a pulsation feature.

        Allowed to attach to:
        * components with kind: star (not contact envelopes)
        * datasets: not allowed

        Arguments
        ----------
        * `l` (int, optional): spherical-harmonic degree.
        * `m` (int, optional): azimuthal order (m > 0: retrograde).
        * `freq` (float/quantity, optional): pulsation frequency as seen by
            the observer (axis='spin') or in the co-rotating frame (axis='tidal').
        * `phase` (float/quantity, optional): pulsation phase at t0@system.
        * `radamp` (float, optional): radial displacement amplitude xi_r/R.
        * `teffamp` (float, optional): Teff amplitude dTeff/Teff (= radamp * f_T).
        * `teffphase` (float/quantity, optional): phase of the Teff variation
            relative to the radial displacement (psi_T).
        * `axis` (string, optional): pulsation axis: 'spin' or 'tidal'.

        Returns
        --------
        * (<phoebe.parameters.ParameterSet>, list): ParameterSet of all newly created
            <phoebe.parameters.Parameter> objects and a list of all necessary
            constraints.
        """
        params = []
        params += [IntParameter(qualifier='l', value=kwargs.get('l', 0), limits=(0, None), description='Spherical-harmonic degree l')]
        params += [IntParameter(qualifier='m', value=kwargs.get('m', 0), limits=(None, None), description='Azimuthal order m (|m| <= l; m > 0 is retrograde)')]
        params += [FloatParameter(qualifier='freq', value=kwargs.get('freq', 1.0), default_unit=u.d**-1, limits=(0, None), description='Pulsation frequency in the observer\'s frame (axis=spin) or in the co-rotating frame (axis=tidal)')]
        params += [FloatParameter(qualifier='phase', value=kwargs.get('phase', 0.0), default_unit=u.deg, description='Pulsation phase at t0@system')]
        params += [FloatParameter(qualifier='radamp', value=kwargs.get('radamp', 0.0), limits=(0, None), default_unit=u.dimensionless_unscaled, description='Radial displacement amplitude xi_r/R (sqrt(4pi) Y_lm normalisation)')]
        params += [FloatParameter(qualifier='teffamp', value=kwargs.get('teffamp', 0.0), limits=(0, None), default_unit=u.dimensionless_unscaled, description='Effective-temperature amplitude dTeff/Teff (= radamp * f_T)')]
        params += [FloatParameter(qualifier='teffphase', value=kwargs.get('teffphase', 0.0), default_unit=u.deg, description='Phase of the Teff variation relative to the radial displacement (psi_T)')]
        params += [ChoiceParameter(qualifier='axis', value=kwargs.get('axis', 'spin'), choices=['spin', 'tidal'], description='Pulsation axis: stellar spin axis, or line of centres (tidally tilted modes)')]
        return ParameterSet(params), []

    @classmethod
    def parse_bundle(cls, b, feature_ps):
        """
        Initialize a Pulsation feature from the bundle.
        """
        from phoebe import c
        get = lambda qualifier, **kw: feature_ps.get_value(qualifier=qualifier, **kw, **_skip_filter_checks)
        component = feature_ps.component
        star_ps = b.get_component(component=component, **_skip_filter_checks)

        l, m = int(get('l')), int(get('m'))
        freq = get('freq', unit=u.d**-1)
        axis = get('axis')

        t0 = b.get_value(qualifier='t0', context='system', unit=u.d, **_skip_filter_checks)
        stars = b.hierarchy.get_stars()
        ref_inertial = None
        if len(stars) >= 2:
            from phoebe.dynamics import keplerian
            from phoebe.backend.mesh import euler_trans_matrix, spin_in_system
            orbit_ps = b.get_component(component=b.hierarchy.get_parent_of(component), **_skip_filter_checks)
            # mean rotation of the mesh frame: only a fallback, the actual frame
            # orientation is passed to update_frame at every time step
            Omega_frame = 2*np.pi/orbit_ps.get_value(qualifier='period_anom', unit=u.d, **_skip_filter_checks)
            # azimuth origin: the direction of the companion (x-axis of the mesh
            # frame) at t0, as a fixed vector in the system frame
            dyn = keplerian.dynamics_from_bundle(b, [t0], compute=None, return_euler=True, ltte=False)
            istar = stars.index(component)
            M0 = euler_trans_matrix(*(float(np.atleast_1d(e[istar])[0]) for e in dyn[7:10]))
            spin = spin_in_system(star_ps.get_value(qualifier='incl', unit=u.rad, **_skip_filter_checks),
                                  star_ps.get_value(qualifier='long_an', unit=u.rad, **_skip_filter_checks))
            ref_inertial = M0 @ np.array([1.0, 0.0, 0.0])
            if abs(np.dot(ref_inertial, spin)) > 0.9:
                # same fallback as asteroseismo.mode_frame
                ref_inertial = M0 @ np.array([0.0, 1.0, 0.0])
        else:
            Omega_frame = 0.0
        Omega_star = 2*np.pi/star_ps.get_value(qualifier='period', unit=u.d, **_skip_filter_checks)

        if axis == 'tidal':
            omega_c = 2*np.pi*freq
        else:
            omega_c = 2*np.pi*freq + m*Omega_star

        GM = c.G.to('solRad3 / (solMass d2)').value*star_ps.get_value(qualifier='mass', unit=u.solMass, **_skip_filter_checks)
        R = star_ps.get_value(qualifier='requiv', unit=u.solRad, **_skip_filter_checks)
        K = GM/(omega_c**2*R**3) if omega_c != 0 else np.inf

        return dict(l=l, m=m, freq=freq, axis=axis,
                    phase=get('phase', unit=u.rad),
                    radamp=get('radamp', unit=u.dimensionless_unscaled),
                    teffamp=get('teffamp', unit=u.dimensionless_unscaled),
                    teffphase=get('teffphase', unit=u.rad),
                    t0=t0, Omega_frame=Omega_frame, ref_inertial=ref_inertial,
                    omega_c=omega_c, Omega_star=Omega_star, K=K, f_g=2.0 + 1.0/K)

    @classmethod
    def run_checks_compute(cls, b, feature_ps, compute_ps):
        items = []
        kw = cls.parse_bundle(b, feature_ps)
        get_param = lambda q: feature_ps.get_parameter(qualifier=q, **_skip_filter_checks)
        if abs(kw['m']) > kw['l']:
            items += [{'msg': '|m| must not exceed l', 'params': [get_param('l'), get_param('m')], 'is_error': True}]
        if kw['freq'] <= 0:
            items += [{'msg': 'pulsation frequency must be positive', 'params': [get_param('freq')], 'is_error': True}]
        if b.hierarchy.is_contact_binary(feature_ps.component):
            items += [{'msg': 'pulsations are not supported for components of contact binaries (common envelope)',
                       'params': [get_param('l')], 'is_error': True}]
        Ymax, gYmax = _peak_factors(kw['l'], kw['m']) if abs(kw['m']) <= kw['l'] else (1.0, 1.0)
        if kw['omega_c'] == 0:
            items += [{'msg': 'pulsation frequency in the co-rotating frame is 0 (2 pi freq + m Omega_rot = 0)',
                       'params': [get_param('freq'), get_param('m')], 'is_error': True}]
        else:
            if kw['omega_c'] < 0:
                items += [{'msg': 'omega_c = 2 pi freq + m Omega_rot < 0: the mode is retrograde in the rotating star but prograde for the observer '
                                  '(physical, e.g. low-N prograde TEOs; for a free mode check the sign of m)',
                           'params': [get_param('freq'), get_param('m')], 'is_error': False}]
            if kw['l'] > 0 and kw['axis'] == 'spin' and 2*kw['Omega_star']/abs(kw['omega_c']) > 0.5:
                items += [{'msg': 'spin parameter 2 Omega_rot/omega_c = {:.1f} > 0.5: the Coriolis force reshapes the mode (not modelled), so the Y_lm maps are approximate'.format(2*kw['Omega_star']/abs(kw['omega_c'])),
                           'params': [get_param('freq'), get_param('m')], 'is_error': False}]
            # local amplitudes = global amplitudes x the peak factors of the pattern
            if kw['radamp']*kw['K']*gYmax > 0.1:
                items += [{'msg': 'horizontal displacement radamp*K*max|grad Y|={:.2f} is not small: linear theory (and the mesh) will be inaccurate'.format(kw['radamp']*kw['K']*gYmax),
                           'params': [get_param('radamp'), get_param('freq')], 'is_error': False}]
            if kw['radamp']*kw['f_g']*Ymax >= 1:
                items += [{'msg': 'gravity perturbation radamp*f_g*max|Y|={:.2f} >= 1 makes log g undefined on part of the surface: reduce radamp'.format(kw['radamp']*kw['f_g']*Ymax),
                           'params': [get_param('radamp'), get_param('freq')], 'is_error': True}]
            elif kw['radamp']*kw['f_g']*Ymax > 0.1:
                items += [{'msg': 'gravity perturbation radamp*f_g*max|Y|={:.2f} is not small: outside the linear regime'.format(kw['radamp']*kw['f_g']*Ymax),
                           'params': [get_param('radamp'), get_param('freq')], 'is_error': False}]
        if len(b.hierarchy.get_stars()) >= 2:
            from phoebe.backend import asteroseismo
            orbit_ps = b.get_component(component=b.hierarchy.get_parent_of(feature_ps.component), **_skip_filter_checks)
            star_ps = b.get_component(component=feature_ps.component, **_skip_filter_checks)
            if kw['axis'] == 'tidal' and (orbit_ps.get_value(qualifier='ecc', **_skip_filter_checks) > 0 or
                                          star_ps.get_value(qualifier='syncpar', **_skip_filter_checks) != 1):
                items += [{'msg': "axis='tidal' assumes a circular, synchronous orbit",
                           'params': [get_param('axis')], 'is_error': False}]
            # tidally excited oscillations sit at harmonics of the orbital frequency
            times = [np.atleast_1d(p.get_value(unit=u.d)) for p in b.filter(qualifier=['times', 'compute_times'], context='dataset', **_skip_filter_checks).to_list()]
            times = np.concatenate(times) if len(times) else np.array([])
            tspan = np.ptp(times) if len(times) > 1 else None
            dperdt = orbit_ps.get_value(qualifier='dperdt', unit=u.rad/u.d, **_skip_filter_checks)
            candidates = [(q, kw['freq']) for q in ('period_anom', 'period')] + [('period_anom', kw['freq'] + kw['m']*dperdt/(2*np.pi))]*bool(dperdt and kw['m'])
            for qualifier, freq in candidates:
                N, _, _, is_harmonic = asteroseismo.orbital_harmonics(freq, 1.0/orbit_ps.get_value(qualifier=qualifier, unit=u.d, **_skip_filter_checks), tspan)
                if is_harmonic[0]:
                    items += [{'msg': 'freq ~= {} x f_orb: possibly a tidally excited oscillation (TEO)'.format(N[0]),
                               'params': [get_param('freq')], 'is_error': False}]
                    break
        if (kw['teffamp']*Ymax > 0.1 or kw['radamp']*Ymax > 0.1) if kw['omega_c'] != 0 else (kw['teffamp'] > 0.1 or kw['radamp'] > 0.1):
            items += [{'msg': 'local pulsation amplitudes (amplitude x max|Y|) above 0.1 are outside the linear regime',
                       'params': [get_param('radamp'), get_param('teffamp')], 'is_error': False}]
        if kw['radamp'] == 0 and kw['teffamp'] == 0:
            items += [{'msg': 'radamp and teffamp are both 0: the pulsation will have no effect',
                       'params': [get_param('radamp'), get_param('teffamp')], 'is_error': False}]
        if kw['l'] > 0:
            ntri = b.get_value(qualifier='ntriangles', component=feature_ps.component, compute=compute_ps.compute, check_default=False, check_visible=False) if 'ntriangles' in compute_ps.qualifiers else None
            if ntri is not None and ntri < 300*kw['l']*(kw['l'] + 1):
                items += [{'msg': 'ntriangles={} may be too coarse to resolve an l={} pattern (suggest >= {})'.format(ntri, kw['l'], 300*kw['l']*(kw['l'] + 1)),
                           'params': [get_param('l')], 'is_error': False}]
        return items

    def requires_remeshing(self):
        # the displacement is applied to a fresh copy of the unperturbed
        # (standard) mesh at every time, so remeshing is not needed ...
        return False

    def requires_recompute_instantaneous(self):
        # ... but local quantities change at every time step
        return True

    def _mode_frame(self, s):
        from phoebe.backend import asteroseismo
        if self.kwargs['axis'] == 'tidal':
            return asteroseismo.mode_frame(axis=(1.0, 0.0, 0.0), ref=(0.0, 0.0, 1.0))
        return asteroseismo.mode_frame(axis=s, ref=(1.0, 0.0, 0.0))

    def update_frame(self, time, euler, s):
        # beta(t): how far the mesh frame has turned about the spin axis since
        # t0, i.e. minus the azimuth of the fixed (inertial) reference direction
        # in the mode frame.  Exact for eccentric orbits and misaligned spins.
        beta = None
        if self.kwargs['axis'] != 'tidal' and self.kwargs.get('ref_inertial') is not None:
            from phoebe.backend.mesh import euler_trans_matrix
            ref = euler_trans_matrix(*euler).T @ self.kwargs['ref_inertial']
            M = self._mode_frame(s)
            beta = -np.arctan2(ref @ M[1], ref @ M[0])
        self._frame = (time, beta)

    def _frame_rotation(self, t):
        if self.kwargs['axis'] == 'tidal':
            # the pattern is fixed in the mesh frame
            return 0.0
        frame = getattr(self, '_frame', None)
        if frame is not None and frame[0] == t and frame[1] is not None:
            return frame[1]
        # mean motion of the frame: exact for circular orbits only
        return self.kwargs['Omega_frame']*(t - self.kwargs['t0'])

    def _complex_phase(self, t):
        if t is None:
            t = self.kwargs['t0']
        return np.exp(1j*(2*np.pi*self.kwargs['freq']*(t - self.kwargs['t0']) +
                          self.kwargs['m']*self._frame_rotation(t) + self.kwargs['phase']))

    def _harmonics(self, coords, s, gradient=False):
        """
        sqrt(4 pi) Y_l^m and (optionally) its surface gradient at the
        directions of `coords`, with the gradient rotated back to the mesh frame.
        """
        from phoebe.backend import asteroseismo
        M = self._mode_frame(s)
        xhat = coords/np.linalg.norm(coords, axis=1)[:, None]
        xm = np.einsum('ij,kj->ik', xhat, M)
        l, m = self.kwargs['l'], self.kwargs['m']
        Y = np.sqrt(4*np.pi)*asteroseismo.cart_Y(l, m, xm)
        if not gradient:
            return xhat, Y, None
        gY = np.einsum('ij,jk->ik', np.sqrt(4*np.pi)*asteroseismo.cart_grad_Y(l, m, xm), M)
        return xhat, Y, gY

    def _displacement(self, coords, s, t, derivative=False):
        """
        Displacement (or its time derivative in the co-rotating frame) at
        `coords`, in the units of `coords`, scaled by the local radius so the
        same amplitude applies to scaled and unscaled meshes.
        """
        if self.kwargs['radamp'] == 0:
            return np.zeros_like(coords)
        xhat, Y, gY = self._harmonics(coords, s, gradient=self.kwargs['l'] > 0)
        shape = Y[:, None]*xhat
        if gY is not None:
            shape = shape + self.kwargs['K']*gY
        amp = self.kwargs['radamp']*self._complex_phase(t)
        if derivative:
            amp = amp*1j*self.kwargs['omega_c']
        r = np.linalg.norm(coords, axis=1)
        return r[:, None]*np.real(amp*shape)

    def modify_coords_for_computations(self, coords_for_computations, s, t):
        return coords_for_computations + self._displacement(coords_for_computations, s, t)

    def modify_coords_for_observations(self, coords_for_computations, coords_for_observations, s, t):
        # displacement evaluated on the (unperturbed) observation coordinates
        # themselves: coords_for_computations have already been displaced
        return coords_for_observations + self._displacement(coords_for_observations, s, t)

    def modify_velocities(self, velocities, coords_for_observations, s=[0., 0., 1.], t=None):
        # pulsational velocity in the co-rotating frame, plus Omega x xi: PHOEBE
        # adds the rotational velocity Omega x x0 of the UNdisplaced position only
        xi = self._displacement(coords_for_observations, s, t)
        Omega = self.kwargs['Omega_star']*np.asarray(s, dtype=float)/np.linalg.norm(s)
        return velocities + self._displacement(coords_for_observations, s, t, derivative=True) + np.cross(Omega, xi)

    def modify_loggs(self, loggs, roche_coords, s=[0., 0., 1.], t=None):
        if self.kwargs['radamp'] == 0:
            return loggs
        _, Y, _ = self._harmonics(roche_coords, s)
        dlng = np.real(-self.kwargs['radamp']*self.kwargs['f_g']*Y*self._complex_phase(t))
        return loggs + np.log10(1.0 + dlng)

    def modify_teffs(self, teffs, roche_coords, s=[0., 0., 1.], t=None):
        if self.kwargs['teffamp'] == 0:
            return teffs
        _, Y, _ = self._harmonics(roche_coords, s)
        dlnT = np.real(self.kwargs['teffamp']*np.exp(1j*self.kwargs['teffphase'])*Y*self._complex_phase(t))
        return teffs*(1.0 + dlnT)
