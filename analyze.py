"""
Compute comparison metrics and plots from results/*/log.csv.

Usage: python analyze.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS = Path("results")

# Moving-average window for CARLA's noisy acceleration signal [s]
SMOOTH_WINDOW = 0.25

# Same colors as the cars in the video
COLORS = {
    "pid": "#1a9e50",
    "mpc": "#1f6fd1",
    "lead": "#8a8a8a",
}

STYLES = {
    "pid": "-",
    "mpc": "--",
}

INK = "#222222"
MUTED = "#666666"
GRID = "#e4e4e4"

plt.rcParams.update({
    "font.size": 12,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
})


def smooth(values, dt):
    window = max(1, round(SMOOTH_WINDOW / dt))
    return (
        pd.Series(values)
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def derived_signals(df, prefix):
    t = df["t_s"].to_numpy()
    dt = float(np.median(np.diff(t)))

    accel = smooth(df[f"{prefix}_accel"].to_numpy(), dt)
    jerk = np.gradient(accel, dt)

    return accel, jerk


def compute_metrics(df, prefix):
    accel, jerk = derived_signals(df, prefix)

    gap = df[f"{prefix}_gap_m"]
    desired = df[f"{prefix}_desired_gap_m"]

    return {
        "min_gap_m": gap.min(),
        "min_time_gap_s": df[f"{prefix}_time_gap_s"].min(),
        "max_jerk_ms3": np.max(np.abs(jerk)),
        "rms_accel_ms2": np.sqrt(np.mean(accel**2)),
        "rms_gap_error_m": np.sqrt(np.mean((gap - desired) ** 2)),
        "collision": bool(df[f"{prefix}_collision"].max()),
    }


def plot_run(df, title, path):
    t = df["t_s"]

    fig, axes = plt.subplots(
        4, 1,
        figsize=(11, 11),
        sharex=True
    )

    ax_speed, ax_gap, ax_accel, ax_jerk = axes

    ax_speed.plot(
        t, df["lead_reference_kmh"],
        color=COLORS["lead"], linestyle=":", label="Lead"
    )

    for prefix in ("pid", "mpc"):
        name = prefix.upper()
        color = COLORS[prefix]
        style = STYLES[prefix]

        accel, jerk = derived_signals(df, prefix)

        ax_speed.plot(
            t, df[f"{prefix}_ego_kmh"],
            color=color, linestyle=style, label=name
        )

        ax_gap.plot(
            t, df[f"{prefix}_gap_m"],
            color=color, linestyle=style, label=f"{name} gap"
        )

        ax_gap.plot(
            t, df[f"{prefix}_desired_gap_m"],
            color=color, linestyle=style, linewidth=1.0, alpha=0.45,
            label=f"{name} desired"
        )

        ax_accel.plot(t, accel, color=color, linestyle=style, label=name)
        ax_jerk.plot(t, jerk, color=color, linestyle=style, label=name)

        hits = df.index[df[f"{prefix}_collision"] > 0]
        if len(hits):
            t_hit = t[hits[0]]
            for ax in axes:
                ax.axvline(t_hit, color=color, linewidth=1.0, alpha=0.6)
            ax_gap.annotate(
                f"{name} collision",
                (t_hit, 0),
                xytext=(5, 10),
                textcoords="offset points",
                color=INK
            )

    ax_gap.axhline(
        0, color=MUTED, linewidth=1.0
    )

    ax_speed.set_ylabel("Speed [km/h]")
    ax_gap.set_ylabel("Gap [m]")
    ax_accel.set_ylabel("Acceleration [m/s²]")
    ax_jerk.set_ylabel("Jerk [m/s³]")
    ax_jerk.set_xlabel("Time [s]")

    ax_speed.legend(loc="upper right", ncol=3)
    ax_gap.legend(loc="upper right", ncol=2, fontsize=10)

    fig.suptitle(title, color=INK, fontsize=15, x=0.06, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_summary(metrics, path):
    good = metrics[
        (metrics["pid_preset"] == "good")
        & (metrics["mpc_preset"] == "good")
    ]

    if good.empty:
        return

    scenarios = list(good["scenario"].unique())

    panels = [
        ("min_gap_m", "Min distance [m]", "higher = safer"),
        ("min_time_gap_s", "Min time gap [s]", "higher = safer"),
        ("max_jerk_ms3", "Max jerk [m/s³]", "lower = more comfortable"),
        ("rms_accel_ms2", "RMS acceleration [m/s²]", "lower = more comfortable"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    x = np.arange(len(scenarios))
    width = 0.38

    for ax, (column, label, hint) in zip(axes, panels):
        for i, controller in enumerate(("PID", "MPC")):
            values = [
                good[
                    (good["scenario"] == s)
                    & (good["controller"] == controller)
                ][column].iloc[0]
                for s in scenarios
            ]

            bars = ax.bar(
                x + (i - 0.5) * (width + 0.02),
                values,
                width,
                color=COLORS[controller.lower()],
                hatch="" if controller == "PID" else "//",
                edgecolor="white",
                linewidth=0,
                label=controller
            )

            ax.bar_label(bars, fmt="%.1f", fontsize=9, color=INK, padding=2)

        ax.set_title(label, color=INK, fontsize=12, loc="left")
        ax.set_xlabel(hint, color=MUTED, fontsize=10)
        ax.set_xticks(x, [s.replace("_", " ") for s in scenarios])
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.15)

    axes[0].legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    logs = sorted(RESULTS.glob("*/log.csv"))

    if not logs:
        print("No results found in results/*/log.csv")
        return

    records = []

    for log in logs:
        run = log.parent.name
        df = pd.read_csv(log)

        scenario, pid_part, mpc_part = run.rsplit("_", 2)
        pid_preset = pid_part.removeprefix("pid-")
        mpc_preset = mpc_part.removeprefix("mpc-")

        for prefix in ("pid", "mpc"):
            records.append({
                "scenario": scenario,
                "pid_preset": pid_preset,
                "mpc_preset": mpc_preset,
                "controller": prefix.upper(),
                "tuning": pid_preset if prefix == "pid" else mpc_preset,
                **compute_metrics(df, prefix),
            })

        title = (
            f"{scenario.replace('_', ' ').capitalize()} - "
            f"PID ({pid_preset}) vs MPC ({mpc_preset})"
        )

        plot_run(df, title, log.parent / "plots.png")
        print(f"Plotted {log.parent / 'plots.png'}")

    metrics = pd.DataFrame(records)

    metrics.to_csv(RESULTS / "metrics.csv", index=False)

    table = metrics.drop(columns=["pid_preset", "mpc_preset"]).copy()
    table["collision"] = table["collision"].map({True: "YES", False: "no"})

    (RESULTS / "metrics.md").write_text(
        table.to_markdown(index=False, floatfmt=".2f"),
        encoding="utf-8"
    )

    plot_summary(metrics, RESULTS / "summary.png")

    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nSaved {RESULTS / 'metrics.csv'}, {RESULTS / 'metrics.md'}")


if __name__ == "__main__":
    main()
