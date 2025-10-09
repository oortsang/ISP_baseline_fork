# More metrics: rel_l2 and PSNR


from collections.abc import Sequence

import jax
import jax.numpy as jnp


def l2_error(
    pred: jax.Array,
    true: jax.Array,
    *,
    l2_axes: Sequence[int] = (),
    # mean_axes: Sequence[int] | None = None,
    relative: bool = True,
    squared: bool = False,
) -> jax.Array:
    """Computes the relative l2 error in a way that is hopefully compatible
    with the swirl-dynamics code.

    Args:
        pred: The array representing the predictions.
        true: The array representing the ground truths.
        l2_axes: The axes over which the l2 errors will be calculated.
        # mean_axes: The axes over which the average will be taken. If `None`, average
        #     is taken over all axes. If some elements are common between `sum_axes` and
        #     `mean_axes`, the the former takes priority.
        #     The mean is taken after the l2 errors are computed, so the behavior is
        #     different when setting squared=True compared with squaring the output
        #     when setting squared=False.
        relative: indicate whether to divide by the reference l2 norms
        squared:  indicate whether to square the l2 norms before taking any means
    Returns:
        An array of l2 errors
    """
    if pred.shape != true.shape:
        raise ValueError(
            f"`pred` {pred.shape} and `true` {true.shape} must have the same shape."
        )

    l2_diffs = jnp.linalg.norm(pred-true, axis=l2_axes)
    l2_refs  = jnp.linalg.norm(true, axis=l2_axes)

    l2_errs = (l2_diffs / l2_refs) if relative else l2_diffs
    l2_errs = l2_errs**2           if squared  else l2_errs

    out_l2_errs = l2_errs
    # out_l2_err = jnp.mean(l2_errs, axis=mean_axes)
    return out_l2_errs

def psnr(
    pred: jax.Array,
    true: jax.Array,
    *,
    psnr_axes: Sequence[int] = (),
):
    """Figure out if this even works first..."""
    pass
