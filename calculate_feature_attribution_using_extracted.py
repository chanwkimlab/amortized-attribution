import argparse
import glob
from functools import partial
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from tqdm.auto import tqdm

from utils import read_eval_results


class AttributionValues:
    """For storing and plotting Shapley values."""

    def __init__(self, values, std):
        self.values = values
        self.std = std


def default_min_variance_samples(game):
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
    print(A.shape)
    print(b.shape)
    import time

    start_time = time.time()

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
    print(time.time() - start_time)

    return values


def ShapleyRegressionPrecomputed(
    grand_value,
    null_value,
    model_outputs,
    masks,
    num_players,
    batch_size=512,
    detect_convergence=True,
    thresh=0.01,
    n_samples=None,
    return_all=False,
    min_variance_samples=None,
    variance_batches=None,
    bar=True,
    verbose=False,
):
    # Verify arguments.
    from tqdm.auto import tqdm

    if min_variance_samples is None:
        min_variance_samples = 5
    else:
        assert isinstance(min_variance_samples, int)
        assert min_variance_samples > 1

    if variance_batches is None:
        variance_batches = default_variance_batches(num_players, batch_size)
    else:
        assert isinstance(variance_batches, int)
        assert variance_batches >= 1

    # Possibly force convergence detection.
    if n_samples is None:
        n_samples = 1e20
        if not detect_convergence:
            detect_convergence = True
            if verbose:
                print("Turning convergence detection on")

    if detect_convergence:
        assert 0 < thresh < 1

    # Weighting kernel (probability of each subset size).
    weights = np.arange(1, num_players)
    weights = 1 / (weights * (num_players - weights))
    weights = weights / np.sum(weights)

    # Calculate null and grand coalitions for constraints.
    null = null_value
    grand = grand_value

    # Calculate difference between grand and null coalitions.
    total = grand - null

    # Set up bar.
    n_loops = int(np.ceil(n_samples / batch_size))
    if bar:
        if detect_convergence:
            bar = tqdm(total=1)
        else:
            bar = tqdm(total=n_loops * batch_size)

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

    # Begin sampling.
    for it in range(n_loops):
        # Sample subsets.
        # print(subsets.shape)
        S = masks[batch_size * it : batch_size * (it + 1)]  # (batch_size, num_players)
        game_S = model_outputs[
            batch_size * it : batch_size * (it + 1)
        ]  # (batch_size, num_classes)
        #         print("S", S, S.sum(axis=1))
        #         print("game(s)", game_S)
        #         print("game(s)-null", game_S-null)

        A_sample = np.matmul(
            S[:, :, np.newaxis].astype(float), S[:, np.newaxis, :].astype(float)
        )  # (batch_size, num_players, 1) * (batch_size, 1, num_players) = (batch_size, num_players, num_players)

        b_sample = (
            S.astype(float).T * (game_S - null)[:, np.newaxis].T
        ).T  # (num_players, batch_size) * (1, num_classes, batch_size) = (num_players, num_classes)

        #         print("b", b_sample)
        #         print("variance_batches", variance_batches)

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
            # print(len(estimate_list), min_variance_samples)
            if len(estimate_list) >= min_variance_samples:
                var = np.array(estimate_list).var(axis=0)

        # Convergence ratio.
        std = np.sqrt(var * variance_batches / (it + 1))
        ratio = np.max(np.max(std, axis=0) / (values.max(axis=0) - values.min(axis=0)))
        # print("std", var)
        # Print progress message.
        if verbose:
            if detect_convergence:
                print(f"StdDev Ratio = {ratio:.4f} (Converge at {thresh:.4f})")
            else:
                print(f"StdDev Ratio = {ratio:.4f}")

        # Check for convergence.
        if detect_convergence:
            if ratio < thresh:
                if verbose:
                    print("Detected convergence")

                # Skip bar ahead.
                if bar:
                    bar.n = bar.total
                    bar.refresh()
                break

        # Forecast number of iterations required.
        if detect_convergence:
            N_est = (it + 1) * (ratio / thresh) ** 2
            if bar and not np.isnan(N_est):
                bar.n = np.around((it + 1) / N_est, 4)
                bar.refresh()
        elif bar:
            bar.update(batch_size)

        # Save intermediate quantities.
        if return_all:
            val_list.append(values)
            std_list.append(std)
            if detect_convergence:
                N_list.append(N_est)

        # print("size", batch_size*it, len(masks))
        if batch_size * (it + 1) >= len(masks):
            break
    # print(ratio)
    # Return results.
    if return_all:
        # Dictionary for progress tracking.
        # iters = (np.arange(it + 1) + 1) * batch_size * (1)
        tracking_dict = {"values": val_list, "std": std_list, "iters": iters}
        if detect_convergence:
            tracking_dict["N_est"] = N_list

        return AttributionValues(values, std), tracking_dict, ratio
    else:
        return AttributionValues(values, std), ratio


def get_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description="Process some inputs.")

    # Argument for batch size without default
    parser.add_argument(
        "--batch_size", type=int, required=True, help="The batch size for processing."
    )

    # Argument for input path without default
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input directory."
    )

    # Argument for normalization function
    parser.add_argument(
        "--normalize_function",
        type=str,
        choices=["softmax"],
        required=True,
        help="The normalization function to be used. Options: softmax.",
    )

    parser.add_argument(
        "--num_players", type=int, required=True, help="The number of players"
    )

    parser.add_argument(
        "--target_subset_size",
        type=int,
        required=False,
        default=None,
        help="The target subset size",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Accessing the arguments
    batch_size = args.batch_size
    input_path = args.input_path
    if args.normalize_function == "softmax":
        normalize_function = softmax
    else:
        raise ValueError("Unsupported normalization function")
    num_players = args.num_players
    target_subset_size = args.target_subset_size

    sample_list = glob.glob(str(Path(input_path) / "[0-9]*"))

    pbar = tqdm(sample_list)
    subsets_output_prev = None
    for sample_path in pbar:
        eval_results = read_eval_results(path=sample_path)

        grand_value = eval_results["grand"]["logits"]
        if len(grand_value.shape) == 1:
            grand_value = partial(normalize_function, axis=0)(grand_value)
        else:
            raise RuntimeError(f"Not supported grand shape {grand_value.shape}")

        null_value = eval_results["null"]["logits"]
        if len(null_value.shape) == 1:
            null_value = partial(normalize_function, axis=0)(null_value)
        else:
            raise RuntimeError(f"Not supported null shape {null_value.shape}")

        subsets_output = eval_results["subsets"]["logits"]
        if len(subsets_output.shape) == 2:
            subsets_output = partial(normalize_function, axis=1)(subsets_output)
        else:
            raise RuntimeError(
                f"Not supported subset outputs shape {subsets_output.shape}"
            )
        if subsets_output_prev is not None and len(subsets_output) != len(
            subsets_output_prev
        ):
            print(
                "subsets_output_prev",
                subsets_output_prev.shape,
                "subsets_output",
                subsets_output.shape,
                "sample_path",
                sample_path,
            )

        subsets = eval_results["subsets"]["masks"]

        assert (
            subsets_output.shape[1] == grand_value.shape[0] == null_value.shape[0]
        ), f"Num of classes mismatch {subsets_output.shape[1]} , {grand_value.shape[0]} , {null_value.shape[0]}"
        assert (
            subsets.shape[1] == num_players
        ), f"Num of players mismatch {subsets.shape[1]} != {num_players}"

        if target_subset_size is None:
            _, tracking_dict, ratio = ShapleyRegressionPrecomputed(
                grand_value=grand_value,
                null_value=null_value,
                model_outputs=subsets_output,
                masks=subsets,
                batch_size=batch_size,
                num_players=num_players,
                variance_batches=2,
                min_variance_samples=2,
                return_all=True,
                bar=False,
            )

            torch.save(
                obj=tracking_dict, f=str(Path(sample_path) / "shapley_output.pt")
            )
        else:
            for subset_group_idx in range(len(subsets_output) // target_subset_size):
                _, tracking_dict, ratio = ShapleyRegressionPrecomputed(
                    grand_value=grand_value,
                    null_value=null_value,
                    model_outputs=subsets_output[
                        target_subset_size
                        * subset_group_idx : target_subset_size
                        * (subset_group_idx + 1)
                    ],
                    masks=subsets[
                        target_subset_size
                        * subset_group_idx : target_subset_size
                        * (subset_group_idx + 1)
                    ],
                    batch_size=batch_size,
                    num_players=num_players,
                    variance_batches=2,
                    min_variance_samples=2,
                    return_all=True,
                    bar=False,
                )

                torch.save(
                    obj=tracking_dict,
                    f=str(
                        Path(sample_path)
                        / f"shapley_output_{target_subset_size}_{subset_group_idx}.pt"
                    ),
                )

        pbar.set_postfix(
            ratio=ratio,
            num_masks=len(subsets),
            refresh=True,
        )
