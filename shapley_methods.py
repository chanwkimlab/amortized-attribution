import numpy as np
from tqdm.auto import tqdm


class ShapleyValues:
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
    S = np.zeros((total_samples, num_players), dtype=int)

    prev_value = null
    for i in range(num_players):
        S[arange, permutations[:, i]] = 1
        next_value = surrogate([np.expand_dims(S_, axis=0) for S_ in S])[:, 0, :]
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

        return ShapleyValues(values, std), tracking_dict, ratio
    else:
        return ShapleyValues(values, std), ratio
