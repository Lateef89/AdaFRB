"""
Experiment 2 — Nonconvex Sparse Logistic Regression with MCP Penalty
====================================================================

Datasets : LIBSVM a1a and w1a
Methods  : MR-FRB, FRB, PGD, PALM, iPiano

Figures saved as PNG and EPS to:
    C:\\Users\\jolla\\OneDrive\\Latex paper\\2026\\AdaFRB\\figures\\exp2_nonconvex_mcp
"""

# ============================================================
# Imports
# ============================================================

import os
import re
import bz2
import ssl
import time
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_svmlight_file
from scipy import sparse
from scipy.sparse.linalg import svds


# ============================================================
# Output folder
# ============================================================

OUT_BASE = r"C:\Users\jolla\OneDrive\Latex paper\2026\AdaFRB\figures\Example2"
OUT = os.path.join(OUT_BASE, "exp2_nonconvex_mcp")

os.makedirs(OUT, exist_ok=True)

print("Figures will be saved to:")
print(OUT)


# ============================================================
# Experiment settings
# ============================================================

GLOBAL_SEED = 42
RNG = np.random.default_rng(GLOBAL_SEED)

N_RUNS = 20
N_ITER = 3000
METRIC_EVERY = 10

LAM_REG = 0.01
GAMMA_MCP = 2.7
TOL = 1e-6

DATASETS = ["a1a", "w1a"]

DATASET_DIR = "datasets"

BASE_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"

REGISTRY = {
    "a1a": (123, "a1a"),
    "w1a": (300, "w1a"),
}


# ============================================================
# Method parameters  (FISTA removed)
# ============================================================

METHOD_PARAMS = {
    "MR-FRB": {
        "step_factor": 0.95,
        "theta": 0.45,
    },
    "FRB": {
        "step_factor": 0.199,
        "use_frb_cap": True,
    },
    "PGD": {
        "step_factor": 0.40,
    },
    "PALM": {
        "step_factor": 0.35,
    },
    "iPiano": {
        "step_factor": 0.240,
        "beta_coeff": 0.30,
    },
}

METHODS = ["MR-FRB", "FRB", "PGD", "PALM", "iPiano"]


# ============================================================
# Publication-style plotting
# ============================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "axes.linewidth": 1.2,
})

STYLES = {
    "MR-FRB": dict(
        color="#D62728",
        linestyle="-",
        linewidth=3.5,
        marker="*",
        markevery=15,
        markersize=12,
    ),
    "FRB": dict(
        color="#2CA02C",
        linestyle="-.",
        linewidth=2.6,
        marker="s",
        markevery=15,
        markersize=6,
    ),
    "PGD": dict(
        color="#9467BD",
        linestyle=":",
        linewidth=2.8,
        marker="h",
        markevery=15,
        markersize=7,
    ),
    "PALM": dict(
        color="#8C564B",
        linestyle=(0, (5, 2)),
        linewidth=2.5,
        marker="D",
        markevery=15,
        markersize=6,
    ),
    "iPiano": dict(
        color="#FF7F0E",
        linestyle=(0, (3, 1, 1, 1)),
        linewidth=2.5,
        marker="^",
        markevery=15,
        markersize=6,
    ),
}


# ============================================================
# Dataset download and loading
# ============================================================

def ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def download_file(url, dest, timeout=180, retries=3, chunk_size=1 << 20):
    ctx = ssl_context()
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "*/*",
                    "Referer": BASE_URL + "/",
                },
            )

            tmp = dest + ".tmp"

            with urllib.request.urlopen(req, context=ctx, timeout=timeout) as response:
                with open(tmp, "wb") as f:
                    while True:
                        data = response.read(chunk_size)
                        if not data:
                            break
                        f.write(data)

            os.replace(tmp, dest)
            return

        except Exception as exc:
            last_error = exc

            if os.path.isfile(dest + ".tmp"):
                os.remove(dest + ".tmp")

            if attempt < retries:
                wait_time = 2 ** attempt
                print(f"Download failed. Retrying in {wait_time} seconds.")
                time.sleep(wait_time)

    raise RuntimeError(f"Download failed: {url}\nLast error: {last_error}")


def validate_libsvm_file(path):
    pattern = re.compile(r"^[+-]?\d+(\s+\d+:[^\s]+)*\s*$")

    with open(path, "r", errors="replace") as f:
        lines = [f.readline() for _ in range(10)]

    valid = sum(
        1 for line in lines
        if line.strip() and pattern.match(line.strip())
    )

    if valid < 3:
        raise ValueError(
            f"{path} does not look like a valid LIBSVM file. "
            "Delete the cached file and rerun."
        )


def ensure_dataset(name):
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'.")

    os.makedirs(DATASET_DIR, exist_ok=True)

    _, server_file = REGISTRY[name]

    path = os.path.join(DATASET_DIR, name)

    if os.path.isfile(path):
        print(f"     Using cached: {path}")
        return path

    plain_url = f"{BASE_URL}/{server_file}"
    bz2_url = f"{BASE_URL}/{server_file}.bz2"
    bz2_path = path + ".bz2"

    print(f"     Downloading {name} from:")
    print(f"     {plain_url}")

    try:
        download_file(plain_url, path)
        validate_libsvm_file(path)
        print(f"     Saved: {path}")
        return path

    except Exception:
        if os.path.isfile(path):
            os.remove(path)

    print("     Plain download failed. Trying compressed .bz2 version.")

    download_file(bz2_url, bz2_path)

    with bz2.open(bz2_path, "rb") as fin, open(path, "wb") as fout:
        fout.write(fin.read())

    os.remove(bz2_path)

    validate_libsvm_file(path)

    print(f"     Saved: {path}")

    return path


def load_dataset_sparse(name):
    n_features, _ = REGISTRY[name]

    path = ensure_dataset(name)

    X_raw, y_raw = load_svmlight_file(path, n_features=n_features)

    X = X_raw.tocsr().astype(np.float64)

    y = np.where(y_raw > 0, 1.0, -1.0).astype(np.float64)

    col_norms = np.sqrt(X.power(2).sum(axis=0)).A1
    col_norms[col_norms == 0.0] = 1.0

    X = (X @ sparse.diags(1.0 / col_norms)).tocsr()

    n_pos = int(np.sum(y == 1.0))

    print(
        f"     Loaded: N={X.shape[0]}, n={X.shape[1]}, "
        f"positive={n_pos}/{len(y)} ({100*n_pos/len(y):.1f}%)"
    )

    return X, y


# ============================================================
# Smooth logistic loss
# ============================================================

def logistic_loss_grad(x, A, b):
    margin = b * (A @ x)

    loss = float(np.sum(np.logaddexp(0.0, -margin)))

    sig = 1.0 / (1.0 + np.exp(np.clip(margin, -500, 500)))

    grad = A.T @ (-b * sig)

    return loss, np.asarray(grad).ravel()


def lipschitz(A):
    smax = svds(A, k=1, return_singular_vectors=False)[0]
    return float((smax ** 2) / 4.0)


# ============================================================
# MCP penalty and proximal operator
# ============================================================

def mcp_penalty(x, lam, gamma=GAMMA_MCP):
    a = np.abs(x)
    threshold = gamma * lam

    values = np.where(
        a <= threshold,
        lam * a - (a ** 2) / (2.0 * gamma),
        0.5 * gamma * lam ** 2,
    )

    return float(np.sum(values))


def prox_mcp(x, step, gamma=GAMMA_MCP):
    tau = step * LAM_REG
    threshold = gamma * tau

    out = x.copy()

    a = np.abs(x)

    mask = a <= threshold

    soft = np.sign(x) * np.maximum(a - tau, 0.0)

    out[mask] = np.clip(
        soft[mask] / (1.0 - 1.0 / gamma),
        -threshold,
        threshold,
    )

    out[~mask] = x[~mask]

    return out


# ============================================================
# Objective and metrics
# ============================================================

def objective(x, A, b):
    loss, _ = logistic_loss_grad(x, A, b)
    return loss + mcp_penalty(x, LAM_REG, GAMMA_MCP)


def grad_norm(x, A, b):
    _, g = logistic_loss_grad(x, A, b)
    return float(np.linalg.norm(g))


def grad_mapping_norm(x, A, b, step):
    _, g = logistic_loss_grad(x, A, b)
    prox_point = prox_mcp(x - step * g, step)
    return float(np.linalg.norm(x - prox_point) / step)


def step_norm(x_new, x_old):
    return float(np.linalg.norm(x_new - x_old))


def accuracy(x, A, b):
    pred = np.where(A @ x >= 0, 1.0, -1.0)
    return float(np.mean(pred == b))


def sparsity(x):
    return int(np.sum(np.abs(x) > 1e-8))


def empty_history():
    return {
        "iter": [],
        "cpu": [],
        "objective": [],
        "accuracy": [],
        "gradmap": [],
        "gradnorm": [],
        "stepnorm": [],
        "sparsity": [],
    }


def record_metric(hist, x, x_prev, A, b, step, start_time, k):
    hist["iter"].append(k)
    hist["cpu"].append(time.perf_counter() - start_time)
    hist["objective"].append(objective(x, A, b))
    hist["accuracy"].append(accuracy(x, A, b))
    hist["gradmap"].append(grad_mapping_norm(x, A, b, step))
    hist["gradnorm"].append(grad_norm(x, A, b))
    hist["stepnorm"].append(step_norm(x, x_prev))
    hist["sparsity"].append(sparsity(x))


def finalise_history(hist):
    return {key: np.array(value) for key, value in hist.items()}


# ============================================================
# Effective step sizes
# ============================================================

def method_step(method, L):
    params = METHOD_PARAMS[method]
    sf = params["step_factor"]

    if method == "MR-FRB":
        return min(sf / L, 0.99 / (2.0 * L))

    if method == "FRB" and params.get("use_frb_cap", False):
        return min(sf / L, 0.99 / (3.0 * L))

    return sf / L


# ============================================================
# Algorithms  (run_fista removed)
# ============================================================

def run_mr_frb(A, b, x0, L, n_iter):
    step = method_step("MR-FRB", L)
    theta = METHOD_PARAMS["MR-FRB"]["theta"]

    x = x0.copy()
    x_old = x0.copy()
    t = 1.0

    _, g_old = logistic_loss_grad(x_old, A, b)
    _, g_cur = logistic_loss_grad(x, A, b)

    hist = empty_history()
    start_time = time.perf_counter()

    record_metric(hist, x, x_old, A, b, step, start_time, 0)

    for k in range(1, n_iter + 1):
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
        beta = (t - 1.0) / t_new

        y = x + beta * (x - x_old)

        _, g_y = logistic_loss_grad(y, A, b)

        z = y + theta * (step / t_new ** 2) * (g_old - g_cur)

        x_new = prox_mcp(z - step * g_y, step)

        x_prev = x.copy()

        x_old, g_old, x, t = x, g_cur, x_new, t_new

        _, g_cur = logistic_loss_grad(x, A, b)

        if k % METRIC_EVERY == 0 or k == n_iter:
            record_metric(hist, x, x_prev, A, b, step, start_time, k)

    return x, finalise_history(hist)


def run_frb(A, b, x0, L, n_iter):
    step = method_step("FRB", L)

    x = x0.copy()
    x_old = x0.copy()

    _, g_old = logistic_loss_grad(x_old, A, b)
    _, g_cur = logistic_loss_grad(x, A, b)

    hist = empty_history()
    start_time = time.perf_counter()

    record_metric(hist, x, x_old, A, b, step, start_time, 0)

    for k in range(1, n_iter + 1):
        y = x + step * (g_old - g_cur)

        x_new = prox_mcp(y - step * g_cur, step)

        x_prev = x.copy()

        x_old, g_old, x = x, g_cur, x_new

        _, g_cur = logistic_loss_grad(x, A, b)

        if k % METRIC_EVERY == 0 or k == n_iter:
            record_metric(hist, x, x_prev, A, b, step, start_time, k)

    return x, finalise_history(hist)


def run_pgd(A, b, x0, L, n_iter):
    step = method_step("PGD", L)

    x = x0.copy()

    hist = empty_history()
    start_time = time.perf_counter()

    record_metric(hist, x, x, A, b, step, start_time, 0)

    for k in range(1, n_iter + 1):
        _, g = logistic_loss_grad(x, A, b)

        x_prev = x.copy()

        x = prox_mcp(x - step * g, step)

        if k % METRIC_EVERY == 0 or k == n_iter:
            record_metric(hist, x, x_prev, A, b, step, start_time, k)

    return x, finalise_history(hist)


def run_palm(A, b, x0, L, n_iter):
    step = method_step("PALM", L)

    x = x0.copy()

    hist = empty_history()
    start_time = time.perf_counter()

    record_metric(hist, x, x, A, b, step, start_time, 0)

    for k in range(1, n_iter + 1):
        _, g = logistic_loss_grad(x, A, b)

        x_prev = x.copy()

        x = prox_mcp(x - step * g, step)

        if k % METRIC_EVERY == 0 or k == n_iter:
            record_metric(hist, x, x_prev, A, b, step, start_time, k)

    return x, finalise_history(hist)


def run_ipiano(A, b, x0, L, n_iter):
    step = method_step("iPiano", L)
    beta = METHOD_PARAMS["iPiano"]["beta_coeff"] * step

    x = x0.copy()
    x_old = x0.copy()

    hist = empty_history()
    start_time = time.perf_counter()

    record_metric(hist, x, x_old, A, b, step, start_time, 0)

    for k in range(1, n_iter + 1):
        _, g = logistic_loss_grad(x, A, b)

        x_new = prox_mcp(
            x - step * g + beta * (x - x_old),
            step
        )

        x_prev = x.copy()

        x_old, x = x, x_new

        if k % METRIC_EVERY == 0 or k == n_iter:
            record_metric(hist, x, x_prev, A, b, step, start_time, k)

    return x, finalise_history(hist)


def run_method(method, A, b, x0, L, n_iter):
    if method == "MR-FRB":
        return run_mr_frb(A, b, x0, L, n_iter)

    if method == "FRB":
        return run_frb(A, b, x0, L, n_iter)

    if method == "PGD":
        return run_pgd(A, b, x0, L, n_iter)

    if method == "PALM":
        return run_palm(A, b, x0, L, n_iter)

    if method == "iPiano":
        return run_ipiano(A, b, x0, L, n_iter)

    raise ValueError(f"Unknown method: {method}")


# ============================================================
# Reference value F*
# ============================================================

def refine_reference_value(x_start, A, b, L, n_refine=2000):
    step = 0.20 / L

    x = x_start.copy()

    best_value = objective(x, A, b)

    for _ in range(n_refine):
        _, g = logistic_loss_grad(x, A, b)
        x = prox_mcp(x - step * g, step)
        value = objective(x, A, b)
        best_value = min(best_value, value)

    return best_value


# ============================================================
# Plotting helpers
# ============================================================

def save_figure(fig, filename):
    png_file = os.path.join(OUT, filename + ".png")
    eps_file = os.path.join(OUT, filename + ".eps")

    fig.savefig(png_file, dpi=600, bbox_inches="tight")
    fig.savefig(eps_file, format="eps", bbox_inches="tight")

    print(f"Saved: {png_file}")
    print(f"Saved: {eps_file}")


def ax_style(ax):
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.20)
    ax.tick_params(axis="both", which="major", direction="in", length=5)
    ax.tick_params(axis="both", which="minor", direction="in", length=3)
    ax.legend(frameon=True, loc="best")


def mean_std_curves(runs, key):
    arr = np.array([run[key] for run in runs])
    return np.mean(arr, axis=0), np.std(arr, axis=0)


def gap_mean_std(runs, f_star):
    arr = np.maximum(
        np.array([run["objective"] for run in runs]) - f_star,
        1e-15
    )
    return np.mean(arr, axis=0), np.std(arr, axis=0)


def relgap_mean_std(runs, f_star):
    denom = max(1.0, abs(f_star))
    arr = np.maximum(
        (np.array([run["objective"] for run in runs]) - f_star) / denom,
        1e-15
    )
    return np.mean(arr, axis=0), np.std(arr, axis=0)


def plot_metric(results, dataset, f_star, key, ylabel, title, filename, logy=True):
    fig, ax = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)

    for method in METHODS:
        runs = results[method]
        iters = runs[0]["iter"]

        if key == "gap":
            mean, std = gap_mean_std(runs, f_star)
        elif key == "relgap":
            mean, std = relgap_mean_std(runs, f_star)
        else:
            mean, std = mean_std_curves(runs, key)

        style = STYLES[method]

        if logy:
            ax.semilogy(iters, mean, label=method, **style)
            lower = np.maximum(mean - std, 1e-15)
        else:
            ax.plot(iters, mean, label=method, **style)
            lower = mean - std

        ax.fill_between(
            iters,
            lower,
            mean + std,
            color=style["color"],
            alpha=0.16,
        )

    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax_style(ax)

    save_figure(fig, filename)

    plt.show()


def plot_metric_cpu(results, dataset, f_star, key, ylabel, title, filename, logy=True):
    fig, ax = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)

    for method in METHODS:
        runs = results[method]

        final_values = np.array([run["objective"][-1] for run in runs])
        median_idx = int(np.argmin(np.abs(final_values - np.median(final_values))))

        run = runs[median_idx]

        if key == "gap":
            yvals = np.maximum(run["objective"] - f_star, 1e-15)
        elif key == "relgap":
            yvals = np.maximum(
                (run["objective"] - f_star) / max(1.0, abs(f_star)),
                1e-15
            )
        else:
            yvals = run[key]

        style = STYLES[method]

        if logy:
            ax.semilogy(run["cpu"], yvals, label=method, **style)
        else:
            ax.plot(run["cpu"], yvals, label=method, **style)

    ax.set_xlabel("CPU time (seconds)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax_style(ax)

    save_figure(fig, filename)

    plt.show()


def plot_convergence_4panel(results, dataset, f_star):
    """
    2x2 convergence figure:
      Top-left     Objective gap  F(x_k) - F*            (log scale)
      Top-right    Gradient mapping norm ||G_lam(x_k)||  (log scale)
      Bottom-left  Training accuracy                      (linear)
      Bottom-right Sparsity ||x_k||_0                    (linear)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    panels = [
        # (axis,          key,        y-axis label,                                         log?)
        (axes[0, 0], "gap",      r"$F(x_k)-F^*$",                                        True),
        (axes[0, 1], "gradmap",  r"$\|\mathcal{G}_{\lambda}(x_k)\|$",                True),
        (axes[1, 0], "accuracy", "Training accuracy",                                     False),
        (axes[1, 1], "sparsity", r"Sparsity $\|x_k\|_0$",                              False),
    ]

    panel_titles = [
        r"Objective gap $F(x_k)-F^*$",
        r"Gradient mapping norm $\|\mathcal{G}_{\lambda}(x_k)\|$",
        "Training accuracy",
        r"Sparsity $\|x_k\|_0$",
    ]

    for (ax, key, ylabel, logy), title in zip(panels, panel_titles):
        for method in METHODS:
            runs  = results[method]
            iters = runs[0]["iter"]

            if key == "gap":
                mean, std = gap_mean_std(runs, f_star)
            else:
                mean, std = mean_std_curves(runs, key)

            style = STYLES[method]

            if logy:
                ax.semilogy(iters, mean, label=method, **style)
                lower = np.maximum(mean - std, 1e-15)
            else:
                ax.plot(iters, mean, label=method, **style)
                lower = mean - std

            ax.fill_between(
                iters,
                lower,
                mean + std,
                color=style["color"],
                alpha=0.16,
            )

        ax.set_xlabel("Iteration $k$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — {dataset}")
        ax_style(ax)

    save_figure(fig, f"{dataset}_convergence_4panel")
    plt.show()


def performance_profile(metric_values, title, filename):
    methods = list(metric_values.keys())

    matrix = np.array([metric_values[m] for m in methods], dtype=float).T

    best = np.min(matrix, axis=1)

    ratios = matrix / best[:, None]

    tau_max = max(1.1, np.nanmax(ratios))

    tau_grid = np.linspace(1.0, tau_max + 0.1, 250)

    fig, ax = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)

    for j, method in enumerate(methods):
        profile = [np.mean(ratios[:, j] <= tau) for tau in tau_grid]
        ax.plot(tau_grid, profile, label=method, **STYLES[method])

    ax.set_xlabel(r"Performance ratio $\tau$")
    ax.set_ylabel("Fraction of test problems")
    ax.set_title(title)
    ax.set_ylim([0.0, 1.05])

    ax_style(ax)

    save_figure(fig, filename)

    plt.show()


def cpu_boxplot(cpu_values, filename):
    fig, ax = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)

    data = [cpu_values[m] for m in METHODS]

    bp = ax.boxplot(
        data,
        labels=METHODS,
        showmeans=True,
        patch_artist=True,
    )

    colors = [STYLES[m]["color"] for m in METHODS]

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)

    ax.set_ylabel("CPU time (seconds)")
    ax.set_title("CPU time distribution over independent runs")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)

    save_figure(fig, filename)

    plt.show()


# ============================================================
# Summary helpers
# ============================================================

def iters_to_tol(obj_hist, iter_hist, f_star):
    gap = obj_hist - f_star
    idx = np.where(gap <= TOL)[0]
    return int(iter_hist[idx[0]]) if len(idx) else int(iter_hist[-1])


def compute_summary(dataset, results, f_star, L):
    rows = []

    for method in METHODS:
        runs = results[method]

        iters = [
            iters_to_tol(run["objective"], run["iter"], f_star)
            for run in runs
        ]

        cpu  = [run["cpu"][-1]                                          for run in runs]
        gap  = [max(run["objective"][-1] - f_star, 0.0)                 for run in runs]
        relgap = [
            max((run["objective"][-1] - f_star) / max(1.0, abs(f_star)), 0.0)
            for run in runs
        ]
        gm   = [run["gradmap"][-1]                                      for run in runs]
        gn   = [run["gradnorm"][-1]                                     for run in runs]
        sn   = [run["stepnorm"][-1]                                     for run in runs]
        acc  = [run["accuracy"][-1]                                     for run in runs]
        nnz  = [run["sparsity"][-1]                                     for run in runs]

        rows.append({
            "Dataset":        dataset,
            "Method":         method,
            "Step":           method_step(method, L),
            "Iter_mean":      np.mean(iters),
            "Iter_std":       np.std(iters),
            "CPU_mean":       np.mean(cpu),
            "CPU_std":        np.std(cpu),
            "Gap_mean":       np.mean(gap),
            "RelGap_mean":    np.mean(relgap),
            "GradMap_mean":   np.mean(gm),
            "GradNorm_mean":  np.mean(gn),
            "StepNorm_mean":  np.mean(sn),
            "Acc_mean":       np.mean(acc),
            "Sparsity_mean":  np.mean(nnz),
        })

    return rows


def save_histories(dataset, results):
    for method in METHODS:
        path = os.path.join(OUT, f"history_{dataset}_{method}.npz")

        np.savez_compressed(
            path,
            **{
                f"run_{i}_{key}": value
                for i, run in enumerate(results[method])
                for key, value in run.items()
            }
        )

        print(f"Saved history: {path}")


# ============================================================
# Dataset runner
# ============================================================

def run_one_dataset(dataset):
    print("\n" + "=" * 80)
    print(f"Dataset: {dataset}")
    print("=" * 80)

    A, b = load_dataset_sparse(dataset)

    N, n = A.shape

    L = lipschitz(A)

    print(f"     N = {N}, n = {n}")
    print(f"     L = {L:.8f}")
    print(f"     1/L = {1.0/L:.8f}")

    print("\n     Effective step sizes:")

    for method in METHODS:
        step = method_step(method, L)
        print(f"     {method:8s}: step = {step:.6e}, step*L = {step*L:.3f}")

    results = {method: [] for method in METHODS}
    final_points = []

    for method in METHODS:
        print(f"\n     Running {method}")

        for r in range(N_RUNS):
            rng = np.random.default_rng(GLOBAL_SEED + r)

            x0 = rng.standard_normal(n) * 0.01

            x_final, hist = run_method(method, A, b, x0, L, N_ITER)

            results[method].append(hist)
            final_points.append(x_final)

    print("\n     Estimating reference value F*")

    raw_fstar = min(
        np.min(run["objective"])
        for method in METHODS
        for run in results[method]
    )

    best_x = None
    best_value = np.inf

    for x in final_points:
        value = objective(x, A, b)

        if value < best_value:
            best_value = value
            best_x = x.copy()

    refined_fstar = refine_reference_value(best_x, A, b, L, n_refine=2000)

    f_star = min(raw_fstar, refined_fstar)

    print(f"     Raw F* estimate     = {raw_fstar:.10f}")
    print(f"     Refined F* estimate = {refined_fstar:.10f}")
    print(f"     Used F*             = {f_star:.10f}")

    save_histories(dataset, results)

    rows = compute_summary(dataset, results, f_star, L)

    print("\n     Summary for dataset:")
    print(
        f"{'Method':8s} {'Step':>12s} {'Iters':>15s} "
        f"{'CPU':>15s} {'Gap':>12s} {'GradMap':>12s} "
        f"{'Acc(%)':>10s} {'nnz':>8s}"
    )
    print("-" * 105)

    for row in rows:
        print(
            f"{row['Method']:8s} "
            f"{row['Step']:12.4e} "
            f"{row['Iter_mean']:7.0f}±{row['Iter_std']:<6.0f} "
            f"{row['CPU_mean']:7.2f}±{row['CPU_std']:<6.2f} "
            f"{row['Gap_mean']:12.3e} "
            f"{row['GradMap_mean']:12.3e} "
            f"{100*row['Acc_mean']:9.2f} "
            f"{row['Sparsity_mean']:8.1f}"
        )

    print("\n     Generating plots")

    plot_metric(results, dataset, f_star, key="gap",
                ylabel=r"$F(x_k)-F^*$",
                title=f"$F(x_k)-F^*$ — {dataset}",
                filename=f"{dataset}_gap_iter", logy=True)

    plot_metric(results, dataset, f_star, key="relgap",
                ylabel="Relative objective error",
                title=f"Relative Objective Error — {dataset}",
                filename=f"{dataset}_relative_gap_iter", logy=True)

    plot_metric(results, dataset, f_star, key="gradmap",
                ylabel=r"$\|\mathcal{G}_{\lambda}(x_k)\|$",
                title=f"Gradient Mapping Norm — {dataset}",
                filename=f"{dataset}_gradmap_iter", logy=True)

    plot_metric(results, dataset, f_star, key="gradnorm",
                ylabel=r"$\|\nabla g(x_k)\|$",
                title=f"Smooth Gradient Norm — {dataset}",
                filename=f"{dataset}_gradnorm_iter", logy=True)

    plot_metric(results, dataset, f_star, key="stepnorm",
                ylabel=r"$\|x_{k+1}-x_k\|$",
                title=f"Step Difference — {dataset}",
                filename=f"{dataset}_stepnorm_iter", logy=True)

    plot_metric(results, dataset, f_star, key="sparsity",
                ylabel=r"Sparsity $\|x_k\|_0$",
                title=f"Sparsity Evolution — {dataset}",
                filename=f"{dataset}_sparsity_iter", logy=False)

    plot_metric(results, dataset, f_star, key="accuracy",
                ylabel="Training accuracy",
                title=f"Training Accuracy — {dataset}",
                filename=f"{dataset}_accuracy_iter", logy=False)

    plot_metric_cpu(results, dataset, f_star, key="gap",
                    ylabel=r"$F(x_k)-F^*$",
                    title=f"Objective Gap vs CPU Time — {dataset}",
                    filename=f"{dataset}_gap_cpu", logy=True)

    plot_metric_cpu(results, dataset, f_star, key="gradmap",
                    ylabel=r"$\|\mathcal{G}_{\lambda}(x_k)\|$",
                    title=f"Gradient Mapping Norm vs CPU Time — {dataset}",
                    filename=f"{dataset}_gradmap_cpu", logy=True)

    plot_convergence_4panel(results, dataset, f_star)

    return results, rows, f_star


# ============================================================
# Main
# ============================================================

print("=" * 80)
print("Experiment 2 — Nonconvex MCP Sparse Logistic Regression")
print(f"Datasets : {', '.join(DATASETS)}")
print(f"Methods  : {', '.join(METHODS)}")
print(f"Runs     : {N_RUNS}")
print(f"Iters    : {N_ITER}")
print(f"lambda   : {LAM_REG}")
print(f"gamma    : {GAMMA_MCP}")
print("=" * 80)

all_results = {}
all_rows    = {}
all_fstars  = {}

for dataset in DATASETS:
    results, rows, f_star = run_one_dataset(dataset)
    all_results[dataset] = results
    all_rows[dataset]    = rows
    all_fstars[dataset]  = f_star


# ============================================================
# Combined summary table
# ============================================================

summary_rows = []

for dataset in DATASETS:
    summary_rows.extend(all_rows[dataset])

summary_df = pd.DataFrame(summary_rows)

csv_path = os.path.join(OUT, "summary_experiment2.csv")
tex_path = os.path.join(OUT, "summary_experiment2.tex")

summary_df.to_csv(csv_path, index=False)

summary_df.to_latex(
    tex_path,
    index=False,
    float_format="%.4e",
    caption="Summary of Experiment 2 for nonconvex MCP sparse logistic regression.",
    label="tab:exp2_mcp_summary",
)

print("\n" + "=" * 110)
print("Combined Summary Table")
print("=" * 110)

print(
    f"{'Dataset':8s} {'Method':8s} {'Step':>12s} {'Iters':>15s} "
    f"{'CPU':>15s} {'Gap':>12s} {'GradMap':>12s} "
    f"{'Acc(%)':>10s} {'nnz':>8s}"
)
print("-" * 110)

for _, row in summary_df.iterrows():
    print(
        f"{row['Dataset']:8s} "
        f"{row['Method']:8s} "
        f"{row['Step']:12.4e} "
        f"{row['Iter_mean']:7.0f}±{row['Iter_std']:<6.0f} "
        f"{row['CPU_mean']:7.2f}±{row['CPU_std']:<6.2f} "
        f"{row['Gap_mean']:12.3e} "
        f"{row['GradMap_mean']:12.3e} "
        f"{100*row['Acc_mean']:9.2f} "
        f"{row['Sparsity_mean']:8.1f}"
    )

print("=" * 110)

print(f"\nSaved summary CSV : {csv_path}")
print(f"Saved summary LaTeX: {tex_path}")


# ============================================================
# Performance profiles across datasets and runs
# ============================================================

metric_iters = {method: [] for method in METHODS}
metric_cpu   = {method: [] for method in METHODS}

for dataset in DATASETS:
    f_star = all_fstars[dataset]

    for method in METHODS:
        for run in all_results[dataset][method]:
            metric_iters[method].append(
                iters_to_tol(run["objective"], run["iter"], f_star)
            )
            metric_cpu[method].append(run["cpu"][-1])

performance_profile(
    metric_iters,
    title="Performance Profile Based on Iterations",
    filename="performance_profile_iterations",
)

performance_profile(
    metric_cpu,
    title="Performance Profile Based on CPU Time",
    filename="performance_profile_cpu",
)

cpu_boxplot(metric_cpu, filename="cpu_boxplot")

print("\nAll figures, histories, and tables saved in:")
print(OUT)