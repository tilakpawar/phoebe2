"""A user-defined subclass of the pulsation feature (PHOEBE stores its source code)."""
from phoebe.features.component_features import Pulsation


class NoGravityPulsation(Pulsation):
    def modify_loggs(self, loggs, roche_coords, s=[0., 0., 1.], t=None):
        return loggs
