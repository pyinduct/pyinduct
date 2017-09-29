from pyinduct.registry import deregister_base


__all__ = ["tear_down"]

def tear_down(labels, plots=None):
    """
    Derigister labels and delet plots.

    Args:
        labels (array-like): All labels to deregister.
        plots (array-like): All plots to delete.
    """

    for label in labels:
        deregister_base(label)

    del(plots)
