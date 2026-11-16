import numpy as np

class AmyloidSimulation:
    def __init__(self, model, valid_mask, stability_factor=0.4, target_time=3600.0):

    """
    Parameters:
    -----------
    model : numpy.ndarray : model of brain, with each cell categorized into gray matter, white matter, or CSF
    valid_mask : numpy.ndarray : Mask indicating valid areas for simulation 
    stability_factor : float : Factor to reduce timestep (0-1) for stability control 
    target_time : float : Target simulation time (seconds)
    """

        self.model = model.astype(np.int8)
