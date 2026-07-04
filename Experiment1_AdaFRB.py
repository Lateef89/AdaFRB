# ===============================================================================
# Example 1: (Sparse logistic regression on LIBSVM benchmarks)
# objective function: min_{x} \sum_{i=1}^{N} \log(1 + exp(-b_{i}a_{i}^\top x)) + \lambda \|x\|_{1}
# ================================================================================

import os, re, bz2, ssl, time, urllib.request
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_svmlight_file
from scipy import sparse
from scipy.sparse.linalg import svds

# ============================================================
# Output folder
# ============================================================

OUT = r"C:\Users\jolla\OneDrive\Latex paper\2026\AdaFRB Main\figures\Example1"
os.makedirs(OUT, exist_ok=True)

print("Figures will be saved to:")
print(OUT)


# ============================================================
# Experiment settings
# ============================================================

N_RUNS = 20
N_ITER = 3000
LAM_REG = 0.01
TOL = 1e-6
METRIC_EVERY = 10

DATASETS = ["a1a", "w1a"]
DATASET_DIR = "datasets"

_BASE_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"

_REGISTRY = {
    "a1a": (123, "a1a"),
    "w1a": (300, "w1a"),
}


# ============================================================
# Method settings
# ============================================================

METHODS = ["MR-FRB", "FISTA", "FRB", "PGD"]

METHOD_PARAMS = {
    "MR-FRB": {"step_factor": 0.99, "theta": 2.0, "restart": False},
    "FISTA": {"step_factor": 0.30},
    "FRB": {"step_factor": 0.30},
    "PGD": {"step_factor": 0.30},
}


# ============================================================
# Plot settings
# ============================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "legend.fontsize": 10,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "axes.linewidth": 1.2,
})

STYLES = {
    "MR-FRB": dict(
        color="red",
        linestyle="-",
        linewidth=3.0,
        marker="*",
        markevery=25,
        markersize=10,
    ),
    "FISTA": dict(
        color="blue",
        linestyle="--",
        linewidth=2.4,
        marker="o",
        markevery=25,
        markersize=6,
    ),
    "FRB": dict(
        color="green",
        linestyle="-.",
        linewidth=2.4,
        marker="s",
        markevery=25,
        markersize=6,
    ),
    "PGD": dict(
        color="purple",
        linestyle=":",
        linewidth=2.6,
        marker="h",
        markevery=25,
        markersize=7,
    ),
}


# ============================================================
# Dataset loading
# ============================================================

def _ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _download(url, dest, timeout=180, retries=3, chunk=1024 * 1024):
    ctx = _ssl_context()
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"}
            )

            temp_file = dest + ".tmp"

            with urllib.request.urlopen(
                request,
                context=ctx,
                timeout=timeout
            ) as response:
                with open(temp_file, "wb") as fout:
                    while True:
                        data = response.read(chunk)
                        if not data:
                            break
                        fout.write(data)

            os.replace(temp_file, dest)
            return

        except Exception as exc:
            last_error = exc

            if os.path.exists(dest + ".tmp"):
                os.remove(dest + ".tmp")

            if attempt < retries:
                time.sleep(2 ** attempt)

    raise RuntimeError(f"Download failed: {url}\nLast error: {last_error}")


def _decompress_bz2(src, dst):
    with bz2.open(src, "rb") as fin:
        with open(dst, "wb") as fout:
            fout.write(fin.read())

    os.remove(src)


def _validate(path, min_valid=3):
    pattern = re.compile(r"^[+-]?\d+(\s+\d+:[^\s]+)*\s*$")

    with open(path, "r", errors="replace") as f:
        lines = [f.readline() for _ in range(10)]

    valid = sum(
        1 for line in lines
        if line.strip() and pattern.match(line.strip())
    )

    if valid < min_valid:
        raise ValueError(
            f"{path} is not a valid LIBSVM file. Delete it and rerun."
        )


def _download_if_needed(name):
    os.makedirs(DATASET_DIR, exist_ok=True)

    _, server_file = _REGISTRY[name]

    local_file = os.path.join(DATASET_DIR, name)

    if os.path.isfile(local_file):
        print(f"     Using cached file: {local_file}")
        return local_file

    url_plain = f"{_BASE_URL}/{server_file}"
    url_bz2 = f"{_BASE_URL}/{server_file}.bz2"

    local_bz2 = local_file + ".bz2"

    print(f"     Downloading {name} from {url_plain}")

    try:
        _download(url_plain, local_file)
        _validate(local_file)
        return local_file

    except Exception:
        if os.path.isfile(local_file):
            os.remove(local_file)

    print("     Plain download failed. Trying compressed .bz2 version.")

    _download(url_bz2, local_bz2)
    _decompress_bz2(local_bz2, local_file)
    _validate(local_file)

    return local_file


def load_dataset(name):
    n_features, _ = _REGISTRY[name]

    local_file = _download_if_needed(name)

    X, y_raw = load_svmlight_file(
        local_file,
        n_features=n_features
    )

    X = X.tocsr().astype(np.float64)

    y = np.where(y_raw > 0, 1.0, -1.0).astype(np.float64)

    col_norms = np.sqrt(X.power(2).sum(axis=0)).A1
    col_norms[col_norms == 0.0] = 1.0

    X = X @ sparse.diags(1.0 / col_norms)

    n_pos = int(np.sum(y == 1.0))

    print(
        f"     Loaded: N={X.shape[0]}, n={X.shape[1]}, "
        f"positive={n_pos}/{len(y)} ({100*n_pos/len(y):.1f}%)"
    )

    return X.tocsr(), y


# ============================================================
# Objective, gradient, prox, metrics
# ============================================================

def logistic_loss_grad(x, A, b):
    margin = b * (A @ x)

    loss = np.sum(np.logaddexp(0.0, -margin))

    sigmoid_negative_margin = 1.0 / (
        1.0 + np.exp(np.clip(margin, -500, 500))
    )

    grad = A.T @ (-b * sigmoid_negative_margin)

    return loss, np.asarray(grad).ravel()


def lipschitz(A):
    smax = svds(A, k=1, return_singular_vectors=False)[0]
    return float((smax ** 2) / 4.0)


def prox_l1(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def objective(x, A, b):
    loss, _ = logistic_loss_grad(x, A, b)
    return loss + LAM_REG * np.linalg.norm(x, 1)


def accuracy(x, A, b):
    pred = np.where(A @ x >= 0, 1.0, -1.0)
    return np.mean(pred == b)


def grad_mapping_norm(x, A, b, lam):
    _, grad = logistic_loss_grad(x, A, b)
    prox_point = prox_l1(x - lam * grad, lam * LAM_REG)
    return np.linalg.norm((x - prox_point) / lam)


def sparsity(x):
    return np.count_nonzero(np.abs(x) > 1e-8)


def empty_hist():
    return {
        "iter": [],
        "cpu": [],
        "objective": [],
        "accuracy": [],
        "gradmap": [],
        "sparsity": [],
    }


def record_metric(hist, x, A, b, lam, start_time, k):
    hist["iter"].append(k)
    hist["cpu"].append(time.perf_counter() - start_time)
    hist["objective"].append(objective(x, A, b))
    hist["accuracy"].append(accuracy(x, A, b))
    hist["gradmap"].append(grad_mapping_norm(x, A, b, lam))
    hist["sparsity"].append(sparsity(x))


# ============================================================
# Algorithms
# ============================================================

def run_pgd(A, b, x0, lam, n_iter):
    x = x0.copy()

    hist = empty_hist()

    start = time.perf_counter()

    record_metric(hist, x, A, b, lam, start, 0)

    for k in range(1, n_iter + 1):
        _, gx = logistic_loss_grad(x, A, b)

        x = prox_l1(
            x - lam * gx,
            lam * LAM_REG
        )

        if k % METRIC_EVERY == 0 or k == n_iter:
            record_metric(hist, x, A, b, lam, start, k)

    return x, {key: np.array(value) for key, value in hist.items()}


def run_fista(A, b, x0, lam, n_iter):
    x = x0.copy()
    x_old = x0.copy()
    t = 1.0

    hist = empty_hist()

    start = time.perf_counter()

    record_metric(hist, x, A, b, lam, start, 0)

    for k in range(1, n_iter + 1):
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0

        beta = (t - 1.0) / t_new

        y = x + beta * (x - x_old)

        _, gy = logistic_loss_grad(y, A, b)

        x_new = prox_l1(
            y - lam * gy,
            lam * LAM_REG
        )

        x_old = x
        x = x_new
        t = t_new

        if k % METRIC_EVERY == 0 or k == n_iter:
            record_metric(hist, x, A, b, lam, start, k)

    return x, {key: np.array(value) for key, value in hist.items()}


def run_frb(A, b, x0, lam, n_iter):
    L = lipschitz(A)

    lam = min(
        lam,
        0.99 / (3.0 * L)
    )

    x = x0.copy()
    x_old = x0.copy()

    _, g_old = logistic_loss_grad(x_old, A, b)
    _, g_cur = logistic_loss_grad(x, A, b)

    hist = empty_hist()

    start = time.perf_counter()

    record_metric(hist, x, A, b, lam, start, 0)

    for k in range(1, n_iter + 1):
        y = x + lam * (g_old - g_cur)

        x_new = prox_l1(
            y - lam * g_cur,
            lam * LAM_REG
        )

        x_old = x
        g_old = g_cur
        x = x_new

        _, g_cur = logistic_loss_grad(x, A, b)

        if k % METRIC_EVERY == 0 or k == n_iter:
            record_metric(hist, x, A, b, lam, start, k)

    return x, {key: np.array(value) for key, value in hist.items()}


def run_mr_frb(A, b, x0, lam, n_iter, theta=2.0, restart=False):
    L = lipschitz(A)

    lam = min(
        lam,
        0.99 / (2.0 * L)
    )

    x = x0.copy()
    x_old = x0.copy()
    t = 1.0

    _, g_old = logistic_loss_grad(x_old, A, b)
    _, g_cur = logistic_loss_grad(x, A, b)

    hist = empty_hist()

    start = time.perf_counter()

    record_metric(hist, x, A, b, lam, start, 0)

    for k in range(1, n_iter + 1):
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0

        beta = (t - 1.0) / t_new

        y = x + beta * (x - x_old)

        if restart and np.dot(y - x, x - x_old) > 0:
            t_new = 1.0
            y = x.copy()

        _, g_y = logistic_loss_grad(y, A, b)

        z = y + theta * (lam / t_new ** 2) * (g_old - g_cur)

        x_new = prox_l1(
            z - lam * g_y,
            lam * LAM_REG
        )

        x_old = x
        g_old = g_cur
        x = x_new
        t = t_new

        _, g_cur = logistic_loss_grad(x, A, b)

        if k % METRIC_EVERY == 0 or k == n_iter:
            record_metric(hist, x, A, b, lam, start, k)

    return x, {key: np.array(value) for key, value in hist.items()}


# ============================================================
# Method wrapper
# ============================================================

def run_method(method, A, b, x0, base_lam, n_iter):
    params = METHOD_PARAMS[method]

    lam = params["step_factor"] * base_lam

    if method == "MR-FRB":
        return run_mr_frb(
            A,
            b,
            x0,
            lam,
            n_iter,
            theta=params["theta"],
            restart=params["restart"],
        )

    if method == "FISTA":
        return run_fista(A, b, x0, lam, n_iter)

    if method == "FRB":
        return run_frb(A, b, x0, lam, n_iter)

    if method == "PGD":
        return run_pgd(A, b, x0, lam, n_iter)

    raise ValueError(method)


# ============================================================
# Plotting
# ============================================================

def save_figure(fig, filename):
    png_file = os.path.join(OUT, filename + ".png")
    eps_file = os.path.join(OUT, filename + ".eps")

    fig.savefig(
        png_file,
        dpi=600,
        bbox_inches="tight"
    )

    fig.savefig(
        eps_file,
        format="eps",
        bbox_inches="tight"
    )

    print(f"Saved: {png_file}")
    print(f"Saved: {eps_file}")


def style_axis(ax):
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.20)

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=5
    )

    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=3
    )

    ax.legend(
        frameon=True,
        loc="best",
        fontsize=10
    )


def plot_combined_metrics_1x4(results, dataset, f_star, filename, N, n):
    """
    One-row four-column figure:
        (a) Objective gap
        (b) Gradient mapping norm
        (c) Training accuracy
        (d) Sparsity

    Each subplot has its own legend.
    """

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(24, 5.5),
        constrained_layout=True,
        sharex=True
    )

    ax_gap = axes[0]
    ax_grad = axes[1]
    ax_acc = axes[2]
    ax_sparse = axes[3]

    for method in METHODS:
        runs = results[method]
        k = runs[0]["iter"]
        style = STYLES[method]

        # ----------------------------------------------------
        # (a) Objective gap
        # ----------------------------------------------------
        gap = np.array([
            np.maximum(run["objective"] - f_star, 1e-15)
            for run in runs
        ])

        gap_mean = gap.mean(axis=0)
        gap_std = gap.std(axis=0)

        ax_gap.semilogy(
            k,
            gap_mean,
            label=method,
            **style
        )

        ax_gap.fill_between(
            k,
            np.maximum(gap_mean - gap_std, 1e-15),
            gap_mean + gap_std,
            color=style["color"],
            alpha=0.12
        )

        # ----------------------------------------------------
        # (b) Gradient mapping norm
        # ----------------------------------------------------
        grad = np.array([
            run["gradmap"]
            for run in runs
        ])

        grad_mean = grad.mean(axis=0)
        grad_std = grad.std(axis=0)

        ax_grad.semilogy(
            k,
            np.maximum(grad_mean, 1e-15),
            label=method,
            **style
        )

        ax_grad.fill_between(
            k,
            np.maximum(grad_mean - grad_std, 1e-15),
            grad_mean + grad_std,
            color=style["color"],
            alpha=0.12
        )

        # ----------------------------------------------------
        # (c) Training accuracy
        # ----------------------------------------------------
        acc = np.array([
            run["accuracy"]
            for run in runs
        ])

        acc_mean = 100.0 * acc.mean(axis=0)
        acc_std = 100.0 * acc.std(axis=0)

        ax_acc.plot(
            k,
            acc_mean,
            label=method,
            **style
        )

        ax_acc.fill_between(
            k,
            acc_mean - acc_std,
            acc_mean + acc_std,
            color=style["color"],
            alpha=0.12
        )

        # ----------------------------------------------------
        # (d) Sparsity
        # ----------------------------------------------------
        sparse = np.array([
            run["sparsity"]
            for run in runs
        ])

        sparse_mean = sparse.mean(axis=0)
        sparse_std = sparse.std(axis=0)

        ax_sparse.plot(
            k,
            sparse_mean,
            label=method,
            **style
        )

        ax_sparse.fill_between(
            k,
            sparse_mean - sparse_std,
            sparse_mean + sparse_std,
            color=style["color"],
            alpha=0.12
        )

    # --------------------------------------------------------
    # Axis formatting
    # --------------------------------------------------------

    ax_gap.set_title("(a) Objective gap")
    ax_gap.set_xlabel("Iteration $k$")
    ax_gap.set_ylabel(r"$F(x_k)-F^\star$")
    style_axis(ax_gap)

    ax_grad.set_title("(b) Gradient mapping")
    ax_grad.set_xlabel("Iteration $k$")
    ax_grad.set_ylabel(r"$\|\mathcal{G}_\lambda(x_k)\|$")
    style_axis(ax_grad)

    ax_acc.set_title("(c) Training accuracy")
    ax_acc.set_xlabel("Iteration $k$")
    ax_acc.set_ylabel("Accuracy (%)")
    style_axis(ax_acc)

    ax_sparse.set_title("(d) Sparsity")
    ax_sparse.set_xlabel("Iteration $k$")
    ax_sparse.set_ylabel(r"$\|x_k\|_0$")
    style_axis(ax_sparse)

    fig.suptitle(
        rf"{dataset}: ${n}\times {N}$",
        fontsize=18,
        y=1.08
    )

    save_figure(fig, filename)

    plt.show()


def plot_gap_cpu(results, dataset, f_star, filename):
    fig, ax = plt.subplots(
        figsize=(7.5, 5.5),
        constrained_layout=True
    )

    for method in METHODS:
        run = results[method][0]

        gap = np.maximum(
            run["objective"] - f_star,
            1e-15
        )

        ax.semilogy(
            run["cpu"],
            gap,
            label=method,
            **STYLES[method]
        )

    ax.set_xlabel("CPU time (seconds)")
    ax.set_ylabel(r"$F(x_k)-F^\star$")
    ax.set_title(f"Objective gap vs CPU time — {dataset}")

    style_axis(ax)

    save_figure(fig, filename)

    plt.show()


# ============================================================
# Experiment runner
# ============================================================

def iters_to_tol(obj_hist, iter_hist, f_star):
    gap = obj_hist - f_star

    idx = np.where(gap <= TOL)[0]

    if len(idx) > 0:
        return int(iter_hist[idx[0]])

    return int(iter_hist[-1])


def run_one_dataset(dataset):
    print("\n" + "=" * 75)
    print(f"Dataset: {dataset}")
    print("=" * 75)

    A, b = load_dataset(dataset)

    N, n = A.shape

    L = lipschitz(A)

    base_lam = 1.0 / L

    print(f"     N={N}, n={n}")
    print(f"     L={L:.8f}")
    print(f"     base step 1/L={base_lam:.8f}")

    results = {method: [] for method in METHODS}

    for method in METHODS:
        print(f"     Running {method}")

        for r in range(N_RUNS):
            rng = np.random.default_rng(42 + r)
            x0 = rng.standard_normal(n) * 0.01

            _, hist = run_method(
                method,
                A,
                b,
                x0,
                base_lam,
                N_ITER
            )

            results[method].append(hist)

    f_star = min(
        np.min(run["objective"])
        for method in METHODS
        for run in results[method]
    )

    print(f"     Estimated F* = {f_star:.10f}")

    rows = []

    for method in METHODS:
        times = []
        iters = []
        final_acc = []
        final_grad = []
        final_sparse = []

        for run in results[method]:
            times.append(run["cpu"][-1])

            iters.append(
                iters_to_tol(
                    run["objective"],
                    run["iter"],
                    f_star
                )
            )

            final_acc.append(run["accuracy"][-1])
            final_grad.append(run["gradmap"][-1])
            final_sparse.append(run["sparsity"][-1])

        rows.append({
            "method": method,
            "time_mean": np.mean(times),
            "time_std": np.std(times),
            "iters_mean": np.mean(iters),
            "iters_std": np.std(iters),
            "acc_mean": np.mean(final_acc),
            "grad_mean": np.mean(final_grad),
            "sparse_mean": np.mean(final_sparse),
        })

        print(
            f"     {method:8s}: "
            f"time={np.mean(times):.3f}±{np.std(times):.3f}, "
            f"iters={np.mean(iters):.0f}±{np.std(iters):.0f}, "
            f"acc={100*np.mean(final_acc):.2f}%, "
            f"gradmap={np.mean(final_grad):.3e}, "
            f"sparsity={np.mean(final_sparse):.1f}"
        )

    prefix = f"improved_{dataset}"

    plot_combined_metrics_1x4(
        results,
        dataset,
        f_star,
        filename=prefix + "_combined_metrics_1x4", N=N, n=n
    )

    plot_gap_cpu(
        results,
        dataset,
        f_star,
        filename=prefix + "_gap_cpu"
    )

    return results, rows, f_star


# ============================================================
# Main
# ============================================================

all_results = {}
all_rows = {}

print("=" * 75)
print("Improved L1 Sparse Logistic Regression Experiment")
print(f"Datasets: {', '.join(DATASETS)}")
print(f"Methods : {', '.join(METHODS)}")
print("=" * 75)

for dataset in DATASETS:
    results, rows, f_star = run_one_dataset(dataset)

    all_results[dataset] = results
    all_rows[dataset] = rows


print("\n" + "=" * 100)
print("Summary table")
print("=" * 100)

print(
    f"{'Dataset':12s} {'Method':8s} {'Time':>12s} "
    f"{'Iters':>15s} {'Acc(%)':>10s} "
    f"{'GradMap':>12s} {'Sparsity':>10s}"
)

print("-" * 100)

for dataset, rows in all_rows.items():
    for row in rows:
        print(
            f"{dataset:12s} "
            f"{row['method']:8s} "
            f"{row['time_mean']:7.3f}±{row['time_std']:.3f} "
            f"{row['iters_mean']:7.0f}±{row['iters_std']:.0f} "
            f"{100*row['acc_mean']:9.2f} "
            f"{row['grad_mean']:12.3e} "
            f"{row['sparse_mean']:10.1f}"
        )

    print()

print("=" * 100)

print("\nAll figures saved in:")
print(OUT)