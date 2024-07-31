def adjust_for_vtrust(weights, consensus, vtrust_min: float = 0.5):
    """
    Interpolate between the current weight and the normalized consensus weights so that the
    vtrust does not fall below vturst_min, assuming the consensus does not change.
    """
    vtrust_loss_desired = 1 - vtrust_min

    # If the predicted vtrust is already above vtrust_min, then just return the current weights.
    orig_vtrust_loss = np.maximum(0.0, weights - consensus).sum()
    if orig_vtrust_loss <= vtrust_loss_desired:
        bt.logging.info(
            "Weights already satisfy vtrust_min. {} >= {}.".format(
                1 - orig_vtrust_loss, vtrust_min
            )
        )
        return weights

    # If maximum vtrust allowable by the current consensus is less that vtrust_min, then choose the smallest lambda
    # that still maximizes the predicted vtrust. Otherwise, find lambda that achieves vtrust_min.
    vtrust_loss_min = 1 - np.sum(consensus)
    if vtrust_loss_min > vtrust_loss_desired:
        bt.logging.info(
            "Maximum possible vtrust with current consensus is less than vtrust_min. {} < {}.".format(
                1 - vtrust_loss_min, vtrust_min
            )
        )
        vtrust_loss_desired = 1.05 * vtrust_loss_min

    # We could solve this with a LP, but just do rootfinding with scipy.
    consensus_normalized = consensus / np.sum(consensus)

    def fn(lam: float):
        new_weights = (1 - lam) * weights + lam * consensus_normalized
        vtrust_loss = np.maximum(0.0, new_weights - consensus).sum()
        return vtrust_loss - vtrust_loss_desired

    sol = optimize.root_scalar(fn, bracket=[0, 1], method="brentq")
    lam_opt = sol.root

    new_weights = (1 - lam_opt) * weights + lam_opt * consensus_normalized
    vtrust_pred = np.minimum(weights, consensus).sum()
    bt.logging.info(
        "Interpolated weights to satisfy vtrust_min. {} -> {}.".format(
            1 - orig_vtrust_loss, vtrust_pred
        )
    )
    return new_weights
