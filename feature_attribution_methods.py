import numpy as np
from tqdm.auto import tqdm


class AttributionValues:
    """For storing and plotting Shapley values."""

    def __init__(self, values, std):
        self.values = values
        self.std = std


def ShapleySampling(
    surrogate,
    num_players,
    total_samples=512,
    detect_convergence=True,
    thresh=0.01,
    antithetical=False,
    return_all=False,
):
    # Calculate null coalition value.

    null = surrogate([np.zeros((1, num_players), dtype=int)])[0][0]
    # each loop generate batch_size * num_players samples
    # Setup.
    if isinstance(null, np.ndarray):
        values = np.zeros((num_players, len(null)))
        sum_squares = np.zeros((num_players, len(null)))
        deltas = np.zeros((total_samples, num_players, len(null)))
    else:
        values = np.zeros((num_players))
        sum_squares = np.zeros((num_players))
        deltas = np.zeros((total_samples, num_players))
    permutations = np.tile(np.arange(num_players), (total_samples, 1))
    arange = np.arange(total_samples)
    n = 0

    # For tracking progress.
    if return_all:
        N_list = []
        std_list = []
        val_list = []

    for i in range(total_samples):
        if antithetical and i % 2 == 1:
            permutations[i] = permutations[i - 1][::-1]
        else:
            np.random.shuffle(permutations[i])

    # S = np.zeros((total_samples, num_players), dtype=int)
    # S_list = []
    # for i in range(num_players):
    #     S[arange, permutations[:, i]] = 1
    #     S_list += [np.expand_dims(S_, axis=0) for S_ in S]
    # surrogate_output = surrogate(S_list)[:, 0, :]
    # surrogate_output = surrogate_output.reshape(num_players, total_samples, -1)

    S = np.zeros((total_samples, num_players), dtype=int)
    S_list = []
    for i in range(num_players):
        S[arange, permutations[:, i]] = 1
        S_list += [np.expand_dims(S_.copy(), axis=0) for S_ in S]

    S_list = np.array(S_list)[:, 0, :]
    # S_list = [S_list[4 * j : 4 * (j + 1)] for j in range(int(np.ceil(len(S_list) / 4)))]
    S_list = [S_list[4 * j : 4 * (j + 1)] for j in range(int(np.ceil(len(S_list) / 4)))]
    surrogate_output = surrogate(S_list)

    # surrogate_output = surrogate(S_list)
    surrogate_output = surrogate_output.reshape(
        num_players, total_samples, surrogate_output.shape[-1]
    )

    prev_value = null
    for i in range(num_players):
        next_value = surrogate_output[i, :, :]
        deltas[arange, permutations[:, i]] = next_value - prev_value
        prev_value = next_value

    for batch_idx in range(total_samples):
        # Welford's algorithm.
        n += 1
        diff = deltas[batch_idx : batch_idx + 1] - values
        values += np.sum(diff, axis=0) / n
        diff2 = deltas[batch_idx : batch_idx + 1] - values
        sum_squares += np.sum(diff * diff2, axis=0)

        # Calculate progress.
        var = sum_squares / (n**2)
        std = np.sqrt(var)
        ratio = np.max(np.max(std, axis=0) / (values.max(axis=0) - values.min(axis=0)))

        if detect_convergence:
            N_est = (batch_idx + 1) * (ratio / thresh) ** 2

        if return_all:
            val_list.append(np.copy(values))
            std_list.append(np.copy(std))
            if detect_convergence:
                N_list.append(N_est)

    if return_all:
        # Dictionary for progress tracking.
        iters = (np.arange(total_samples) + 1) * num_players
        tracking_dict = {"values": val_list, "std": std_list, "iters": iters}
        if detect_convergence:
            tracking_dict["N_est"] = N_list

        return AttributionValues(values, std), tracking_dict, ratio
    else:
        return AttributionValues(values, std), ratio


import operator as op
from collections import OrderedDict
from functools import reduce


def ncr(n, r):
    """
    Combinatorial computation: number of subsets of size r among n elements
    Efficient algorithm
    """
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer / denom


def projection_step(phi, total):
    return phi - (np.sum(phi, axis=0) - total) / len(phi)


def ShapleySGD(
    surrogate,
    # game,
    d,
    num_subsets=100,
    mbsize=32,
    step=0.001,
    step_type="constant",
    sampling="importance",
    averaging="uniform",
    return_interval=128,
    C=1,
    phi_0=False,
):
    """
    Estimate the Shapley values using projected stochastic gradient descent.
    """
    # Get general information
    assert sampling in ("default", "paired", "importance")
    assert step_type in ("constant", "sqrt", "inverse")
    assert averaging in ("none", "uniform", "tail")

    # Setup for importance sampling
    dict_w_k = OrderedDict()  # weights per size k
    dict_L_k = OrderedDict()  # L-smooth constant per size k
    D = C * np.sqrt(d)
    for k in range(1, d):
        w_k = (d - 1) / (ncr(d, k) * k * (d - k))
        L_k = w_k * np.sqrt(k) * (np.sqrt(k) * D + C)
        dict_w_k.update({k: w_k})
        dict_L_k.update({k: L_k})

    # Summation of all L per coalition (closed formula)
    sum_L = np.sum(
        [(d - 1) / (np.sqrt(k) * (d - k)) * (np.sqrt(k) * D + C) for k in range(1, d)]
    )

    # Subset distributions

    # 1. Importance sampling
    p = [ncr(d, k) for k in range(1, d)]
    p /= np.sum(p)
    p_importance = np.array(list(dict_L_k.values())) * np.array(p)
    p_importance /= np.sum(p_importance)

    # 2. Default distribution or paired sampling
    p_default = 1 / (np.arange(1, d) * (d - np.arange(1, d)))
    p_default /= p_default.sum()

    # Get null/grand and output dimension
    grand = surrogate([np.ones((1, d), dtype=int)])[0][
        0
    ]  # game(np.ones((1, d), dtype=bool))[0]
    null = surrogate([np.zeros((1, d), dtype=int)])[0][
        0
    ]  # game(np.zeros((1, d), dtype=bool))[0]
    assert isinstance(grand, np.ndarray)
    out_dim = len(grand)
    total = grand - null

    # Initialize Shapley value estimates
    if phi_0:
        phi = phi_0.copy()
    else:
        phi = np.zeros((d, out_dim))

    # Projection step
    phi = projection_step(phi, total)

    # Store for iterate averaging
    if out_dim is None:
        phi_iterates = np.zeros((int(np.ceil(num_subsets / mbsize)), d))
    else:
        phi_iterates = np.zeros((int(np.ceil(num_subsets / mbsize)), d, out_dim))

    ## masks generation begin
    masks = np.zeros((num_subsets, d), dtype=int)
    # Sample subset cardinality
    if sampling == "importance":
        k_list = np.random.choice(list(range(1, d)), size=num_subsets, p=p_importance)
    else:
        k_list = np.random.choice(list(range(1, d)), size=num_subsets, p=p_default)

    # Apply permutations
    indices = [np.random.permutation(d)[:k] for k in k_list]
    for i in range(num_subsets):
        if (i % 2 == 1) and (sampling == "paired"):
            masks[i] = 1 - masks[i - 1]
        else:
            masks[i, indices[i]] = 1

    masks_list = [np.expand_dims(masks_.copy(), axis=0) for masks_ in masks]
    masks_list = np.array(masks_list)[:, 0, :]
    masks_list = [
        masks_list[4 * j : 4 * (j + 1)]
        for j in range(int(np.ceil(len(masks_list) / 4)))
    ]

    surrogate_output = surrogate(masks_list)
    surrogate_output = surrogate_output.reshape(num_subsets, surrogate_output.shape[-1])

    ## masks generation end

    for t in range(len(phi_iterates)):
        x = masks[t * mbsize : (t + 1) * mbsize]
        assert x.shape[0] == mbsize, "num_subsets must be a multiple of mbsize"

        y = surrogate_output[t * mbsize : (t + 1) * mbsize] - null

        # Calculate gradient
        residual = x.dot(phi) - y
        grad = x[:, :, None] * residual[:, None, :]
        if sampling == "importance":
            # Get weights w, p for importance sampling
            w = np.array([dict_w_k[k] for k in x.sum(axis=1)])
            p = np.array([dict_L_k[k] / sum_L for k in x.sum(axis=1)])

            # Apply importance sampling weights
            grad *= np.expand_dims(w / p, (1, 2))

        # Average gradient
        grad = np.mean(grad, axis=0)

        # Update phi
        if step_type == "constant":
            phi = phi - step * grad
        elif step_type == "sqrt":
            phi = phi - (step / np.sqrt(t + 1)) * grad
        elif step_type == "inverse":
            phi = phi - (step / (t + 1)) * grad

        # Projection step
        phi = projection_step(phi, total)

        # Update iterate history
        phi_iterates[t] = phi

    # Calculate iterate averages
    if averaging == "none":
        phi_iterates_averaged = phi_iterates
    elif averaging == "uniform":
        phi_iterates_averaged = np.cumsum(phi_iterates, axis=0) / np.expand_dims(
            np.arange(len(phi_iterates)) + 1, (1, 2)
        )
    elif averaging == "tail":
        t = np.expand_dims(np.arange(len(phi_iterates)) + 1, (1, 2))
        phi_iterates_averaged = np.cumsum(2 * phi_iterates * t, axis=0) / (t * (t + 1))

    tracking_dict = {
        "values": [value for value in phi_iterates_averaged],
        "iters": (np.arange(len(phi_iterates)) + 1) * mbsize,
    }
    assert len(tracking_dict["values"]) == len(tracking_dict["iters"])
    if return_interval is not None:
        tracking_dict["values"] = tracking_dict["values"][::return_interval]
        tracking_dict["iters"] = tracking_dict["iters"][::return_interval]

    return tracking_dict


def default_min_variance_samples():
    """Determine min_variance_samples."""
    return 5


def default_variance_batches(num_players, batch_size):
    """
    Determine variance_batches.

    This value tries to ensure that enough samples are included to make A
    approximation non-singular.
    """
    return int(np.ceil(10 * num_players / batch_size))


def calculate_result_shapley(A, b, total):
    """Calculate the regression coefficients."""
    num_players = A.shape[1]
    try:
        if len(b.shape) == 2:
            A_inv_one = np.linalg.solve(A, np.ones((num_players, 1)))
        else:
            A_inv_one = np.linalg.solve(A, np.ones(num_players))
        A_inv_vec = np.linalg.solve(A, b)
        values = A_inv_vec - A_inv_one * (
            np.sum(A_inv_vec, axis=0, keepdims=True) - total
        ) / np.sum(A_inv_one)
    except np.linalg.LinAlgError:
        raise ValueError(
            "singular matrix inversion. Consider using larger " "variance_batches"
        )

    return values


def ShapleyRegression(
    surrogate,
    num_players,
    num_subsets,
    batch_size=512,
    detect_convergence=True,
    thresh=0.01,
    paired_sampling=True,
    return_all=False,
    min_variance_samples=None,
    variance_batches=None,
    verbose=False,
):
    if min_variance_samples is None:
        min_variance_samples = default_min_variance_samples()
    else:
        assert isinstance(min_variance_samples, int)
        assert min_variance_samples > 1

    if variance_batches is None:
        variance_batches = default_variance_batches(num_players, batch_size)
    else:
        assert isinstance(variance_batches, int)
        assert variance_batches >= 1

    # Weighting kernel (probability of each subset size).
    weights = np.arange(1, num_players)
    weights = 1 / (weights * (num_players - weights))
    weights = weights / np.sum(weights)

    # Calculate null and grand coalitions for constraints.
    null = surrogate([np.zeros((1, num_players), dtype=int)])[0][0]
    grand = surrogate([np.ones((1, num_players), dtype=int)])[0][0]

    # Calculate difference between grand and null coalitions.
    total = grand - null

    # Setup.
    n = 0
    b = 0
    A = 0
    estimate_list = []

    # For variance estimation.
    A_sample_list = []
    b_sample_list = []

    # For tracking progress.
    var = np.nan * np.ones(num_players)
    if return_all:
        N_list = []
        std_list = []
        val_list = []
        iters = []

    masks = np.zeros((num_subsets, num_players), dtype=int)
    num_included = np.random.choice(num_players - 1, size=num_subsets, p=weights) + 1
    for masks_idx, (row, num) in enumerate(zip(masks, num_included)):
        if paired_sampling and masks_idx % 2 == 1:
            row = 1 - masks[masks_idx - 1]
        else:
            inds = np.random.choice(num_players, size=num, replace=False)
            row[inds] = 1
    masks_list = [np.expand_dims(masks_.copy(), axis=0) for masks_ in masks]

    masks_list = np.array(masks_list)[:, 0, :]
    masks_list = [
        masks_list[4 * j : 4 * (j + 1)]
        for j in range(int(np.ceil(len(masks_list) / 4)))
    ]
    surrogate_output = surrogate(masks_list)
    # surrogate_output = surrogate(masks_list)
    surrogate_output = surrogate_output.reshape(num_subsets, surrogate_output.shape[-1])

    # Begin sampling.
    for it in range(int(np.ceil(num_subsets / batch_size))):
        # Sample subsets.
        S = masks[batch_size * it : batch_size * (it + 1)]
        game_S = surrogate_output[batch_size * it : batch_size * (it + 1)]

        # Single sample.
        A_sample = np.matmul(
            S[:, :, np.newaxis].astype(float), S[:, np.newaxis, :].astype(float)
        )
        b_sample = (S.astype(float).T * (game_S - null)[:, np.newaxis].T).T

        # Welford's algorithm.
        n += len(S)
        iters.append(n)
        b += np.sum(b_sample - b, axis=0) / n
        A += np.sum(A_sample - A, axis=0) / n

        # Calculate progress.
        values = calculate_result_shapley(A, b, total)
        A_sample_list.append(A_sample)
        b_sample_list.append(b_sample)
        if len(A_sample_list) == variance_batches:
            # Aggregate samples for intermediate estimate.
            A_sample = np.concatenate(A_sample_list, axis=0).mean(axis=0)
            b_sample = np.concatenate(b_sample_list, axis=0).mean(axis=0)
            A_sample_list = []
            b_sample_list = []

            # Add new estimate.
            estimate_list.append(calculate_result_shapley(A_sample, b_sample, total))

            # Estimate current var.
            if len(estimate_list) >= min_variance_samples:
                var = np.array(estimate_list).var(axis=0)

        # Convergence ratio.
        std = np.sqrt(var * variance_batches / (it + 1))
        ratio = np.max(np.max(std, axis=0) / (values.max(axis=0) - values.min(axis=0)))

        # Print progress message.
        if verbose:
            if detect_convergence:
                print(f"StdDev Ratio = {ratio:.4f} (Converge at {thresh:.4f})")
            else:
                print(f"StdDev Ratio = {ratio:.4f}")

        # Forecast number of iterations required.
        if detect_convergence:
            N_est = (it + 1) * (ratio / thresh) ** 2

        # Save intermediate quantities.
        if return_all:
            val_list.append(values)
            std_list.append(std)
            if detect_convergence:
                N_list.append(N_est)

    # Return results.
    if return_all:
        # Dictionary for progress tracking.
        assert len(iters) == len(val_list)
        tracking_dict = {"values": val_list, "std": std_list, "iters": iters}
        if detect_convergence:
            tracking_dict["N_est"] = N_list

        return AttributionValues(values, std), tracking_dict, ratio
    else:
        return AttributionValues(values, std), ratio


def calculate_result_banzhaf(A, b, total):
    """Calculate the regression coefficients."""
    num_players = A.shape[1]
    try:
        if len(b.shape) == 2:
            A_inv_one = np.linalg.solve(A, np.ones((num_players, 1)))
        else:
            A_inv_one = np.linalg.solve(A, np.ones(num_players))
        A_inv_vec = np.linalg.solve(A, b)
        values = A_inv_vec - A_inv_one * (
            np.sum(A_inv_vec, axis=0, keepdims=True) - total
        ) / np.sum(A_inv_one)
    except np.linalg.LinAlgError:
        raise ValueError(
            "singular matrix inversion. Consider using larger " "variance_batches"
        )

    return values


def BanzhafRegression(
    surrogate,
    num_players,
    num_subsets,
    batch_size=512,
    detect_convergence=True,
    thresh=0.01,
    paired_sampling=True,
    return_all=False,
    min_variance_samples=None,
    variance_batches=None,
    verbose=False,
):
    if min_variance_samples is None:
        min_variance_samples = default_min_variance_samples()
    else:
        assert isinstance(min_variance_samples, int)
        assert min_variance_samples > 1

    if variance_batches is None:
        variance_batches = default_variance_batches(num_players, batch_size)
    else:
        assert isinstance(variance_batches, int)
        assert variance_batches >= 1

    import ipdb

    ipdb.set_trace()
    # Weighting kernel (probability of each subset size).
    weights = np.arange(1, num_players)
    weights = 1 / (weights * (num_players - weights))
    weights = weights / np.sum(weights)

    # Calculate null and grand coalitions for constraints.
    null = surrogate([np.zeros((1, num_players), dtype=int)])[0][0]
    grand = surrogate([np.ones((1, num_players), dtype=int)])[0][0]

    # Calculate difference between grand and null coalitions.
    total = grand - null

    # Setup.
    n = 0
    b = 0
    A = 0
    estimate_list = []

    # For variance estimation.
    A_sample_list = []
    b_sample_list = []

    # For tracking progress.
    var = np.nan * np.ones(num_players)
    if return_all:
        N_list = []
        std_list = []
        val_list = []
        iters = []

    masks = np.zeros((num_subsets, num_players), dtype=int)
    num_included = np.random.choice(num_players - 1, size=num_subsets, p=weights) + 1
    for masks_idx, (row, num) in enumerate(zip(masks, num_included)):
        if paired_sampling and masks_idx % 2 == 1:
            row = 1 - masks[masks_idx - 1]
        else:
            inds = np.random.choice(num_players, size=num, replace=False)
            row[inds] = 1
    masks_list = [np.expand_dims(masks_.copy(), axis=0) for masks_ in masks]

    masks_list = np.array(masks_list)[:, 0, :]
    masks_list = [
        masks_list[4 * j : 4 * (j + 1)]
        for j in range(int(np.ceil(len(masks_list) / 4)))
    ]
    surrogate_output = surrogate(masks_list)
    # surrogate_output = surrogate(masks_list)
    surrogate_output = surrogate_output.reshape(num_subsets, surrogate_output.shape[-1])

    # Begin sampling.
    for it in range(int(np.ceil(num_subsets / batch_size))):
        # Sample subsets.
        S = masks[batch_size * it : batch_size * (it + 1)]
        game_S = surrogate_output[batch_size * it : batch_size * (it + 1)]

        # Single sample.
        A_sample = np.matmul(
            S[:, :, np.newaxis].astype(float), S[:, np.newaxis, :].astype(float)
        )
        b_sample = (S.astype(float).T * (game_S - null)[:, np.newaxis].T).T

        # Welford's algorithm.
        n += len(S)
        iters.append(n)
        b += np.sum(b_sample - b, axis=0) / n
        A += np.sum(A_sample - A, axis=0) / n

        # Calculate progress.
        values = calculate_result_shapley(A, b, total)
        A_sample_list.append(A_sample)
        b_sample_list.append(b_sample)
        if len(A_sample_list) == variance_batches:
            # Aggregate samples for intermediate estimate.
            A_sample = np.concatenate(A_sample_list, axis=0).mean(axis=0)
            b_sample = np.concatenate(b_sample_list, axis=0).mean(axis=0)
            A_sample_list = []
            b_sample_list = []

            # Add new estimate.
            estimate_list.append(calculate_result_shapley(A_sample, b_sample, total))

            # Estimate current var.
            if len(estimate_list) >= min_variance_samples:
                var = np.array(estimate_list).var(axis=0)

        # Convergence ratio.
        std = np.sqrt(var * variance_batches / (it + 1))
        ratio = np.max(np.max(std, axis=0) / (values.max(axis=0) - values.min(axis=0)))

        # Print progress message.
        if verbose:
            if detect_convergence:
                print(f"StdDev Ratio = {ratio:.4f} (Converge at {thresh:.4f})")
            else:
                print(f"StdDev Ratio = {ratio:.4f}")

        # Forecast number of iterations required.
        if detect_convergence:
            N_est = (it + 1) * (ratio / thresh) ** 2

        # Save intermediate quantities.
        if return_all:
            val_list.append(values)
            std_list.append(std)
            if detect_convergence:
                N_list.append(N_est)

    # Return results.
    if return_all:
        # Dictionary for progress tracking.
        assert len(iters) == len(val_list)
        tracking_dict = {"values": val_list, "std": std_list, "iters": iters}
        if detect_convergence:
            tracking_dict["N_est"] = N_list

        return AttributionValues(values, std), tracking_dict, ratio
    else:
        return AttributionValues(values, std), ratio