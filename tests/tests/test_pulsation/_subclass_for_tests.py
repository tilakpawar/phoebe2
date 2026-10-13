"""A user-defined subclass of the pulsation feature (PHOEBE stores its source code)."""
from phoebe.features.component_features import Pulsation


class NoGravityPulsation(Pulsation):
    def modify_loggs(self, loggs, roche_coords, s=[0., 0., 1.], t=None):
        return loggs


from phoebe.features.common import BaseFeature
from phoebe.parameters import FloatParameter, ParameterSet


class OldHooksFeature(BaseFeature):
    """A feature implementing only the ORIGINAL hook set (as in bundles saved by earlier
    versions): no requires_recompute_instantaneous, modify_velocities or update_frame."""
    allowed_component_kinds = ['star']
    allowed_dataset_kinds = [None]

    def __repr__(self):
        return '<Feature OldHooksFeature>'

    @classmethod
    def create_feature_parameters(cls, feature, **kwargs):
        return ParameterSet([FloatParameter(qualifier='factor', value=kwargs.get('factor', 1.0), description='Teff scale')]), []

    @classmethod
    def parse_bundle(cls, b, feature_ps):
        return dict(factor=feature_ps.get_value(qualifier='factor', check_default=False, check_visible=False))

    def requires_remeshing(self):
        return False

    def modify_coords_for_computations(self, coords_for_computations, s, t):
        return coords_for_computations

    def modify_coords_for_observations(self, coords_for_computations, coords_for_observations, s, t):
        return coords_for_observations

    def modify_rvs(self, rvs, orbit_vel, roche_coords, s=[0., 0., 1.], t=None):
        return rvs

    def modify_loggs(self, loggs, roche_coords, s=[0., 0., 1.], t=None):
        return loggs

    def modify_teffs(self, teffs, roche_coords, s=[0., 0., 1.], t=None):
        return teffs*self.kwargs['factor']

    def modify_intensities(self, abs_normal_intensities, abs_intensities, mus, pblum_scale, extinct_factors, boost_factors, roche_coords, s=[0., 0., 1.], t=None):
        return abs_normal_intensities, abs_intensities
