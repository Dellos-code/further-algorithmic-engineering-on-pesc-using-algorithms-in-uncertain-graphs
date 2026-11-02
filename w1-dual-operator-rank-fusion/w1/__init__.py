"""W1: dual-operator polishing with rank fusion for reachability ranking in uncertain graphs."""
from .algorithm import W1, W3, W1Params, w1_from_arrays, w1_scores
from .instances import Instance, load_instance

__all__ = ["W1", "W3", "W1Params", "w1_scores", "w1_from_arrays", "Instance", "load_instance"]
__version__ = "1.0.0"
