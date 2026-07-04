
# ============================================================
# Improved Experiment 3 — Distributed FRB on Ring Network
# ============================================================

import os, ssl, bz2, time, urllib.request
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import load_svmlight_file
from scipy import sparse
from scipy.sparse.linalg import svds


try:
    import __main__ as _main
    _IS_NOTEBOOK = not hasattr(_main, "__file__")
except Exception:
    _IS_NOTEBOOK = False

if not _IS_NOTEBOOK:
    matplotlib.use("Agg")


OUT_BASE = r"C:\Users\jolla\OneDrive\Latex paper\2026\AdaFRB\figures"
OUT = os.path.join(OUT_BASE, "exp4_distributed_frb_improved")
os.makedirs(OUT, exist_ok=True)

print("Figures will be saved to:")
print(OUT)


GLOBAL_SEED = 42
M_AGENTS = 20
N_RUNS = 20
N_ITER = 500
LAM_REG = 0.01
DATASET_DIR = "datasets"

STEP_DFRB = 0.79
STEP_DGDGT = 0.080
STEP_DPGD = 0.080
STEP_DGD = 0.060

METHODS = ["D-FRB", "DGD+GT", "D-PGD", "DGD"]

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.dpi": 130,
    "savefig.dpi": 600,
    "axes.linewidth": 1.2,
})

STYLES = {
    "D-FRB": dict(color="#D62728", linestyle="-", linewidth=3.2,
                  marker="*", markevery=50, markersize=12),
    "DGD+GT": dict(color="#1F77B4", linestyle="--", linewidth=2.6,
                   marker="o", markevery=50, markersize=7),
    "D-PGD": dict(color="#2CA02C", linestyle="-.", linewidth=2.6,
                  marker="s", markevery=50, markersize=7),
    "DGD": dict(color="#9467BD", linestyle=":", linewidth=2.8,
                marker="h", markevery=50, markersize=8),
}


BASE_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"


def ssl_ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def download_file(url, dest, retries=3):
    ctx = ssl_ctx()

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

            with urllib.request.urlopen(req, context=ctx, timeout=180) as r:
                with open(tmp, "wb") as f:
                    while True:
                        data = r.read(1 << 20)
                        if not data:
                            break
                        f.write(data)

            os.replace(tmp, dest)
            return

        except Exception as exc:
            if os.path.isfile(dest + ".tmp"):
                os.remove(dest + ".tmp")

            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Download failed: {url}\n{exc}")


def ensure_a1a():
    os.makedirs(DATASET_DIR, exist_ok=True)

    path = os.path.join(DATASET_DIR, "a1a")

    if os.path.isfile(path):
        print(f"Using cached dataset: {path}")
        return path

    try:
        url = BASE_URL + "/a1a"
        print(f"Downloading: {url}")
        download_file(url, path)
        return path

    except Exception:
        if os.path.isfile(path):
            os.remove(path)

    url = BASE_URL + "/a1a.bz2"
    bz2_path = path + ".bz2"

    print(f"Trying compressed file: {url}")
    download_file(url, bz2_path)

    with bz2.open(bz2_path, "rb") as fin, open(path, "wb") as fout:
        fout.write(fin.read())

    os.remove(bz2_path)

    return path


def load_dataset_sparse():
    path = ensure_a1a()

    X_raw, y_raw = load_svmlight_file(path, n_features=123)

    X = X_raw.tocsr().astype(np.float64)
    y = np.where(y_raw > 0, 1.0, -1.0).astype(np.float64)

    col_norms = np.sqrt(X.power(2).sum(axis=0)).A1
    col_norms[col_norms == 0.0] = 1.0

    X = (X @ sparse.diags(1.0 / col_norms)).tocsr()

    print(
        f"Loaded a1a: N={X.shape[0]}, n={X.shape[1]}, "
        f"positive={(y == 1).sum()}/{len(y)}"
    )

    return X, y


def logistic_loss_grad(x, A, b):
    margin = b * (A @ x)

    loss = float(np.sum(np.logaddexp(0.0, -margin)))

    sig = 1.0 / (1.0 + np.exp(np.clip(margin, -500, 500)))

    grad = A.T @ (-b * sig)

    return loss, np.asarray(grad).ravel()


def global_lipschitz(X):
    smax = svds(X, k=1, return_singular_vectors=False)[0]
    return float((smax ** 2) / 4.0)


def prox_l1(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def full_objective(x, X_full, y_full):
    loss, _ = logistic_loss_grad(x, X_full, y_full)
    return loss + LAM_REG * np.linalg.norm(x, 1)


def gradient_mapping_norm(x, X_full, y_full, lam):
    _, grad = logistic_loss_grad(x, X_full, y_full)
    prox_point = prox_l1(x - lam * grad, lam * LAM_REG)
    return float(np.linalg.norm(x - prox_point) / lam)


def ring_mixing_matrix(m):
    W = np.zeros((m, m), dtype=np.float64)

    for i in range(m):
        W[i, i] = 1.0 / 3.0
        W[i, (i - 1) % m] = 1.0 / 3.0
        W[i, (i + 1) % m] = 1.0 / 3.0

    return W


def spectral_gap(W):
    eigvals = np.linalg.eigvalsh(W)
    eigvals = np.sort(np.abs(eigvals))[::-1]
    return float(1.0 - eigvals[1])


def method_steps(L):
    return {
        "D-FRB": STEP_DFRB / (3.0 * L),
        "DGD+GT": STEP_DGDGT / (3.0 * L),
        "D-PGD": STEP_DPGD / (3.0 * L),
        "DGD": STEP_DGD / L,
    }


def compute_f_star(X_full, y_full, L):
    lam = 0.40 / L

    n = X_full.shape[1]
    x = np.zeros(n)

    _, g_prev = logistic_loss_grad(x, X_full, y_full)
    _, g_cur = logistic_loss_grad(x, X_full, y_full)

    best = full_objective(x, X_full, y_full)

    for _ in range(3000):
        y = x + lam * (g_prev - g_cur)
        x_new = prox_l1(y - lam * g_cur, lam * LAM_REG)

        g_prev = g_cur
        x = x_new

        _, g_cur = logistic_loss_grad(x, X_full, y_full)

        best = min(best, full_objective(x, X_full, y_full))

    return best


def local_grad(i, x, local_data):
    Xi, yi = local_data[i]
    _, gi = logistic_loss_grad(x, Xi, yi)
    return gi


def run_distributed_method(method, local_data, X_full, y_full, W, L, x0_agents):
    lam = method_steps(L)[method]

    m = len(local_data)
    n = local_data[0][0].shape[1]

    x_agents = x0_agents.copy()

    if method in ["D-FRB", "DGD+GT", "D-PGD"]:
        s_agents = np.array([
            local_grad(i, x_agents[i], local_data)
            for i in range(m)
        ])
        s_prev = s_agents.copy()
    else:
        s_agents = None
        s_prev = None

    cons_hist = np.empty(N_ITER)
    obj_hist = np.empty(N_ITER)
    gradmap_hist = np.empty(N_ITER)
    nnz_hist = np.empty(N_ITER)
    cpu_hist = np.empty(N_ITER)

    start = time.perf_counter()

    for k in range(N_ITER):

        if method == "D-FRB":
            x_hat = np.empty_like(x_agents)

            for i in range(m):
                y_i = x_agents[i] + lam * (s_prev[i] - s_agents[i])
                x_hat[i] = prox_l1(y_i - lam * s_agents[i], lam * LAM_REG)

            x_new = W @ x_hat

            grad_old = np.array([
                local_grad(i, x_agents[i], local_data)
                for i in range(m)
            ])

            grad_new = np.array([
                local_grad(i, x_new[i], local_data)
                for i in range(m)
            ])

            s_new = W @ s_agents + grad_new - grad_old

            s_prev = s_agents.copy()
            s_agents = s_new
            x_agents = x_new

        elif method in ["DGD+GT", "D-PGD"]:
            x_hat = np.empty_like(x_agents)

            for i in range(m):
                x_hat[i] = prox_l1(
                    x_agents[i] - lam * s_agents[i],
                    lam * LAM_REG
                )

            x_new = W @ x_hat

            grad_old = np.array([
                local_grad(i, x_agents[i], local_data)
                for i in range(m)
            ])

            grad_new = np.array([
                local_grad(i, x_new[i], local_data)
                for i in range(m)
            ])

            s_new = W @ s_agents + grad_new - grad_old

            s_prev = s_agents.copy()
            s_agents = s_new
            x_agents = x_new

        elif method == "DGD":
            grad_now = np.array([
                local_grad(i, x_agents[i], local_data)
                for i in range(m)
            ])

            x_hat = np.empty_like(x_agents)

            for i in range(m):
                x_hat[i] = prox_l1(
                    x_agents[i] - lam * grad_now[i],
                    lam * LAM_REG
                )

            x_agents = W @ x_hat

        x_bar = x_agents.mean(axis=0)

        cons_hist[k] = float(
            np.max(np.linalg.norm(x_agents - x_bar, axis=1))
        )
        obj_hist[k] = full_objective(x_bar, X_full, y_full)
        gradmap_hist[k] = gradient_mapping_norm(x_bar, X_full, y_full, lam)
        nnz_hist[k] = float(np.sum(np.abs(x_bar) > 1e-8))
        cpu_hist[k] = time.perf_counter() - start

    return {
        "cons": cons_hist,
        "obj": obj_hist,
        "gradmap": gradmap_hist,
        "nnz": nnz_hist,
        "cpu": cpu_hist,
    }


def save_fig(fig, stem):
    png = os.path.join(OUT, stem + ".png")
    eps = os.path.join(OUT, stem + ".eps")

    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(eps, format="eps", bbox_inches="tight")

    print("Saved:", png)
    print("Saved:", eps)

    plt.show()
    plt.close(fig)


def ax_style(ax, loc="best"):
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.20)
    ax.tick_params(axis="both", which="major", direction="in", length=5)
    ax.tick_params(axis="both", which="minor", direction="in", length=3)
    ax.legend(frameon=True, loc=loc)


def mean_std(results, method, key):
    arr = np.array([run[key] for run in results[method]])
    return arr.mean(axis=0), arr.std(axis=0)


def plot_metric(results, key, ylabel, title, filename, logy=True, f_star=None):
    iters = np.arange(1, N_ITER + 1)

    fig, ax = plt.subplots(figsize=(8.5, 6), constrained_layout=True)

    for method in METHODS:
        mean, std = mean_std(results, method, key)

        if key == "obj" and f_star is not None:
            mean = np.maximum(mean - f_star, 1e-14)
            lower = np.maximum(mean - std, 1e-14)
            upper = mean + std
        else:
            mean = np.maximum(mean, 1e-16) if logy else mean
            lower = np.maximum(mean - std, 1e-16) if logy else mean - std
            upper = mean + std

        style = STYLES[method]

        if logy:
            ax.semilogy(iters, mean, label=method, **style)
        else:
            ax.plot(iters, mean, label=method, **style)

        ax.fill_between(
            iters,
            lower,
            upper,
            color=style["color"],
            alpha=0.12,
        )

    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax_style(ax, "upper right")

    save_fig(fig, filename)


def plot_gap_cpu(results, f_star):
    fig, ax = plt.subplots(figsize=(8.5, 6), constrained_layout=True)

    for method in METHODS:
        final_objs = np.array([r["obj"][-1] for r in results[method]])
        median_index = int(np.argmin(np.abs(final_objs - np.median(final_objs))))

        run = results[method][median_index]

        gap = np.maximum(run["obj"] - f_star, 1e-14)

        ax.semilogy(
            run["cpu"],
            gap,
            label=method,
            **STYLES[method]
        )

    ax.set_xlabel("CPU time (seconds)")
    ax.set_ylabel(r"$F(\bar{x}^k)-F^*$")
    ax.set_title("Objective Gap vs CPU Time")

    ax_style(ax, "upper right")

    save_fig(fig, "obj_gap_cpu")


def plot_convergence_4panel(results, f_star):
    iters = np.arange(1, N_ITER + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    panels = [
        (axes[0, 0], "cons", r"$\max_i\|x_i^k-\bar{x}^k\|$", "Consensus Error", True),
        (axes[0, 1], "obj", r"$F(\bar{x}^k)-F^*$", "Objective Gap", True),
        (axes[1, 0], "gradmap", r"$\|\mathcal{G}_{\lambda}(\bar{x}^k)\|$", "Gradient Mapping Norm", True),
        (axes[1, 1], "nnz", r"$\|\bar{x}^k\|_0$", "Sparsity of Averaged Iterate", False),
    ]

    for ax, key, ylabel, title, logy in panels:
        for method in METHODS:
            mean, std = mean_std(results, method, key)

            if key == "obj":
                mean = np.maximum(mean - f_star, 1e-14)
                lower = np.maximum(mean - std, 1e-14)
                upper = mean + std
            else:
                mean = np.maximum(mean, 1e-16) if logy else mean
                lower = np.maximum(mean - std, 1e-16) if logy else mean - std
                upper = mean + std

            style = STYLES[method]

            if logy:
                ax.semilogy(iters, mean, label=method, **style)
            else:
                ax.plot(iters, mean, label=method, **style)

            ax.fill_between(
                iters,
                lower,
                upper,
                color=style["color"],
                alpha=0.12,
            )

        ax.set_xlabel("Iteration $k$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        ax_style(ax, "upper right")

    save_fig(fig, "convergence_4panel")


def main():
    print("=" * 80)
    print("Experiment 4 — Distributed FRB on Ring Network")
    print(f"Agents: {M_AGENTS}")
    print(f"Runs: {N_RUNS}")
    print(f"Iterations: {N_ITER}")
    print("=" * 80)

    X_full, y_full = load_dataset_sparse()

    N, n = X_full.shape
    L = global_lipschitz(X_full)

    steps = method_steps(L)

    print(f"\nGlobal Lipschitz constant L = {L:.8f}")

    for method in METHODS:
        print(
            f"{method:8s}: lambda = {steps[method]:.6e}, "
            f"lambda*L = {steps[method] * L:.4f}"
        )

    W = ring_mixing_matrix(M_AGENTS)

    print(f"\nRing spectral gap = {spectral_gap(W):.6f}")
    print(f"Row stochastic error = {np.max(np.abs(W.sum(axis=1)-1)):.2e}")
    print(f"Column stochastic error = {np.max(np.abs(W.sum(axis=0)-1)):.2e}")

    print("\nComputing F* using centralised FRB...")
    f_star = compute_f_star(X_full, y_full, L)
    print(f"Estimated F* = {f_star:.8f}")

    results = {method: [] for method in METHODS}

    for run_id in range(N_RUNS):
        print(f"\nRun {run_id + 1}/{N_RUNS}")

        rng = np.random.default_rng(GLOBAL_SEED + run_id)

        perm = rng.permutation(N)
        splits = np.array_split(perm, M_AGENTS)

        local_data = [
            (X_full[idx], y_full[idx])
            for idx in splits
        ]

        x0_agents = rng.standard_normal((M_AGENTS, n)) * 0.01

        for method in METHODS:
            print(f"  Running {method}...", end=" ")

            start = time.perf_counter()

            out = run_distributed_method(
                method,
                local_data,
                X_full,
                y_full,
                W,
                L,
                x0_agents
            )

            elapsed = time.perf_counter() - start

            results[method].append(out)

            print(
                f"done in {elapsed:.2f}s | "
                f"gap={out['obj'][-1]-f_star:.3e}, "
                f"cons={out['cons'][-1]:.3e}"
            )

    print("\nGenerating figures...")

    plot_metric(
        results,
        key="cons",
        ylabel=r"$\max_i\|x_i^k-\bar{x}^k\|$",
        title="Consensus Error on Ring Network",
        filename="consensus_iter",
        logy=True,
    )

    plot_metric(
        results,
        key="obj",
        ylabel=r"$F(\bar{x}^k)-F^*$",
        title="Objective Gap on Ring Network",
        filename="obj_gap_iter",
        logy=True,
        f_star=f_star,
    )

    plot_metric(
        results,
        key="gradmap",
        ylabel=r"$\|\mathcal{G}_{\lambda}(\bar{x}^k)\|$",
        title="Gradient Mapping Norm",
        filename="gradmap_iter",
        logy=True,
    )

    plot_metric(
        results,
        key="nnz",
        ylabel=r"$\|\bar{x}^k\|_0$",
        title="Sparsity of Averaged Iterate",
        filename="sparsity_iter",
        logy=False,
    )

    plot_gap_cpu(results, f_star)

    plot_convergence_4panel(results, f_star)

    summary_rows = []

    for method in METHODS:
        final_gap = np.array([run["obj"][-1] - f_star for run in results[method]])
        final_cons = np.array([run["cons"][-1] for run in results[method]])
        final_gradmap = np.array([run["gradmap"][-1] for run in results[method]])
        final_nnz = np.array([run["nnz"][-1] for run in results[method]])
        final_cpu = np.array([run["cpu"][-1] for run in results[method]])

        summary_rows.append({
            "Method": method,
            "lambda": steps[method],
            "lambda_times_L": steps[method] * L,
            "Final_gap_mean": final_gap.mean(),
            "Final_gap_std": final_gap.std(),
            "Consensus_mean": final_cons.mean(),
            "Consensus_std": final_cons.std(),
            "GradMap_mean": final_gradmap.mean(),
            "GradMap_std": final_gradmap.std(),
            "Sparsity_mean": final_nnz.mean(),
            "CPU_mean": final_cpu.mean(),
            "CPU_std": final_cpu.std(),
        })

    summary = pd.DataFrame(summary_rows)

    csv_path = os.path.join(OUT, "summary_experiment4.csv")
    npz_path = os.path.join(OUT, "history_experiment4.npz")

    summary.to_csv(csv_path, index=False)

    np.savez_compressed(
        npz_path,
        f_star=f_star,
        **{
            f"{method}_run{r}_{key}": value
            for method in METHODS
            for r, run in enumerate(results[method])
            for key, value in run.items()
        }
    )

    print("\nSummary Table")
    print("=" * 120)
    print(summary)
    print("=" * 120)

    print(f"\nSaved summary CSV: {csv_path}")
    print(f"Saved history NPZ: {npz_path}")
    print(f"All figures saved in: {OUT}")


main()