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
SMOOTH_WINDOW = 0.5

# Skip the spawn transient (cars get their initial speed at t=0,
# which gives a huge acceleration spike) [s]
SETTLE_TIME = 1.0

# Metrics are computed from here on. Before this, CARLA's gearbox
# makes the cars lose speed after spawn and they catch up hard,
# which is a simulator artifact and not controller behaviour.
# Every scenario's first event is at t >= 10 s. [s]
EVAL_START = 8.0

SCENARIO_ORDER = ["gentle", "emergency", "stop_and_go"]
TUNING_ORDER = ["good", "pd", "bad"]

TUNING_LABEL = {"good": "good", "pd": "PD (no I-term)", "bad": "bad"}

# PID green and MPC blue, like the labels in the videos
COLORS = {
    "pid": "#008300",
    "mpc": "#2a78d6",
    "lead": "#8a8a8a",
}

# In the good-vs-bad plots the color shows the tuning; "good" keeps
# the controller's own color
TUNING_COLORS = {
    "pd": "#6250d6",
    "bad": "#eb6834",
}

LINE_WIDTH = 2.6

INK = "#222222"
MUTED = "#666666"
GRID = "#e4e4e4"

plt.rcParams.update({
    "font.size": 15,
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
    "lines.linewidth": LINE_WIDTH,
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


# Safety and comfort references drawn in the plots
MIN_DISTANCE = 8.0      # same as main.py [m]
COMFORT_ACCEL = 2.0     # typical ACC comfort limit [m/s^2]

DANGER = "#d64545"
DANGER_FILL = "#fbe3e3"
COMFORT_FILL = "#eef3ea"

NAMES = {"pid": "PID", "mpc": "MPC"}


def brake_start(df):
    """Time when the lead starts slowing down, or None."""
    lead = df["lead_reference_kmh"].to_numpy()
    slowing = np.nonzero(np.diff(lead) < -1e-6)[0]
    return float(df["t_s"].iloc[slowing[0]]) if len(slowing) else None


def mark_brake_start(axes, df):
    t_brake = brake_start(df)
    if t_brake is None:
        return

    for ax in axes:
        ax.axvline(t_brake, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))

    axes[0].annotate(
        "Lead brakes",
        (t_brake, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(6, -4),
        textcoords="offset points",
        va="top",
        color=MUTED,
        fontsize=13
    )


def mark_startup(axes, df):
    t0 = df["t_s"].iloc[0]

    for ax in axes:
        ax.axvspan(t0, EVAL_START, color="#f2f2f2", zorder=0)

    # Short label inside the band, at the top so it never covers
    # the axis tick labels
    axes[0].annotate(
        "start-up",
        ((t0 + EVAL_START) / 2, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(0, -6),
        textcoords="offset points",
        ha="center",
        va="top",
        color=MUTED,
        fontsize=12,
        fontstyle="italic"
    )


def style_gap_axis(ax):
    # Everything below the minimum distance is unsafe
    ax.axhspan(-100, MIN_DISTANCE, color=DANGER_FILL, zorder=0)
    ax.axhline(MIN_DISTANCE, color=DANGER, linewidth=1.0, zorder=1)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Gap [m]")


def style_accel_axis(ax):
    ax.axhspan(-COMFORT_ACCEL, COMFORT_ACCEL, color=COMFORT_FILL, zorder=0)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_ylabel("Acceleration [m/s²]")


def mark_min_gap(ax, df, prefix, color):
    i = int(df[f"{prefix}_gap_m"].idxmin())
    t_min = df["t_s"].iloc[i]
    gap_min = df[f"{prefix}_gap_m"].iloc[i]

    ax.plot(t_min, gap_min, "o", color=color, markersize=8,
            markeredgecolor="white", markeredgewidth=1.5, zorder=5)

    # Keep the label inside the plot when the minimum is at the end
    t = df["t_s"]
    near_end = t_min > t.iloc[0] + 0.8 * (t.iloc[-1] - t.iloc[0])

    ax.annotate(
        f"{NAMES[prefix]} min {gap_min:.1f} m",
        (t_min, gap_min),
        xytext=(-10 if near_end else 10, 30 if prefix == "pid" else 14),
        textcoords="offset points",
        ha="right" if near_end else "left",
        color=COLORS[prefix],
        fontsize=14,
        fontweight="bold"
    )


def mark_collision(ax, df, prefix):
    hits = df.index[df[f"{prefix}_collision"] > 0]
    if not len(hits):
        return

    i = hits[0]
    ax.plot(df["t_s"].iloc[i], df[f"{prefix}_gap_m"].iloc[i], "X",
            color=DANGER, markersize=13, markeredgecolor="white", zorder=6)
    ax.annotate(
        f"{NAMES[prefix]} collision",
        (df["t_s"].iloc[i], df[f"{prefix}_gap_m"].iloc[i]),
        xytext=(10, 6 if prefix == "pid" else 22),
        textcoords="offset points",
        color=DANGER,
        fontsize=11,
        fontweight="bold"
    )


def plot_controllers(ax, t, series, accel=False):
    """
    Draw PID and MPC on one axis. series: {prefix: values}.
    In acceleration plots MPC oscillates, so it is drawn thinner and
    underneath so the PID line stays readable.
    """

    for prefix in ("mpc", "pid"):
        ax.plot(
            t, series[prefix],
            color=COLORS[prefix],
            linewidth=1.6 if (accel and prefix == "mpc") else LINE_WIDTH,
            alpha=0.85 if (accel and prefix == "mpc") else 1.0,
            label=NAMES[prefix],
            zorder=3 if prefix == "pid" else 2
        )


def plot_overview(df, title, path):
    """Speed, gap and acceleration over time for one run (16:9)."""
    t = df["t_s"]

    fig, (ax_speed, ax_gap, ax_accel) = plt.subplots(
        3, 1,
        figsize=(13.33, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.3, 1]}
    )

    mark_startup((ax_speed, ax_gap, ax_accel), df)

    ax_speed.plot(t, df["lead_reference_kmh"], color=COLORS["lead"],
                  linestyle=":", linewidth=LINE_WIDTH, label="Lead car")

    plot_controllers(ax_speed, t, {
        p: df[f"{p}_ego_kmh"] for p in ("pid", "mpc")
    })

    for prefix in ("pid", "mpc"):
        ax_gap.plot(t, df[f"{prefix}_desired_gap_m"], color=COLORS[prefix],
                    linestyle=":", linewidth=1.4, alpha=0.6)

    plot_controllers(ax_gap, t, {
        p: df[f"{p}_gap_m"] for p in ("pid", "mpc")
    })

    plot_controllers(ax_accel, t, {
        p: derived_signals(df, p)[0] for p in ("pid", "mpc")
    }, accel=True)

    for prefix in ("pid", "mpc"):
        mark_min_gap(ax_gap, df, prefix, COLORS[prefix])
        mark_collision(ax_gap, df, prefix)

    ax_speed.set_ylabel("Speed [km/h]")
    ax_speed.set_ylim(bottom=0)
    style_gap_axis(ax_gap)
    style_accel_axis(ax_accel)
    ax_accel.set_xlabel("Time [s]")

    mark_brake_start((ax_speed, ax_gap, ax_accel), df)

    # Order: Lead, PID, MPC
    handles, labels = ax_speed.get_legend_handles_labels()
    order = [labels.index(name) for name in ("Lead car", "PID", "MPC")]
    ax_speed.legend([handles[i] for i in order], [labels[i] for i in order],
                    loc="lower left", ncol=3)

    ax_gap.text(0.995, 0.96, "dotted = desired gap",
                transform=ax_gap.transAxes, ha="right", va="top",
                color=MUTED, fontsize=12)
    ax_gap.text(0.995, 0.02, f"unsafe (< {MIN_DISTANCE:.0f} m)",
                transform=ax_gap.transAxes, ha="right", va="bottom",
                color=DANGER, fontsize=12)
    ax_accel.text(0.995, 0.96, f"shaded = comfortable (±{COMFORT_ACCEL:.0f} m/s²)",
                  transform=ax_accel.transAxes, ha="right", va="top",
                  color=MUTED, fontsize=12)

    ax_accel.set_xlim(t.iloc[0], t.iloc[-1])

    fig.suptitle(title, color=INK, fontsize=18, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_good_vs_good(runs, path):
    """Gap and acceleration for all scenarios, good vs good, side by side."""
    scenarios = [
        s for s in SCENARIO_ORDER
        if (s, "good", "good") in runs
    ]

    if not scenarios:
        return

    fig, axes = plt.subplots(
        2, len(scenarios),
        figsize=(13.33, 7.5),
        sharex="col",
        squeeze=False
    )

    for col, scenario in enumerate(scenarios):
        df = runs[(scenario, "good", "good")]
        t = df["t_s"]
        ax_gap, ax_accel = axes[0, col], axes[1, col]

        mark_startup((ax_gap, ax_accel), df)

        plot_controllers(ax_gap, t, {
            p: df[f"{p}_gap_m"] for p in ("pid", "mpc")
        })
        plot_controllers(ax_accel, t, {
            p: derived_signals(df, p)[0] for p in ("pid", "mpc")
        }, accel=True)

        for prefix in ("pid", "mpc"):
            mark_collision(ax_gap, df, prefix)

        style_gap_axis(ax_gap)
        style_accel_axis(ax_accel)
        mark_brake_start((ax_gap, ax_accel), df)

        ax_gap.set_title(scenario.replace("_", " ").capitalize(),
                         color=INK, fontsize=16, loc="left",
                         fontweight="bold")
        ax_accel.set_xlabel("Time [s]")
        ax_accel.set_xlim(t.iloc[0], t.iloc[-1])

        if col > 0:
            ax_gap.set_ylabel("")
            ax_accel.set_ylabel("")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [labels.index("PID"), labels.index("MPC")]
    fig.legend([handles[i] for i in order], ["PID", "MPC"],
               loc="upper right", ncol=2, fontsize=15)

    fig.suptitle("Good tuning: PID vs MPC in all scenarios",
                 color=INK, fontsize=18, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tuning(runs, scenario, path):
    """
    Good vs bad tuning in one scenario: PID variants on the left,
    MPC variants on the right, one color per tuning.
    """
    # (prefix, preset) -> run that contains it, preferring the good/good run
    lines = {}

    for (s, pid_preset, mpc_preset), df in runs.items():
        if s != scenario:
            continue
        for prefix, preset in (("pid", pid_preset), ("mpc", mpc_preset)):
            key = (prefix, preset)
            if key not in lines or (pid_preset, mpc_preset) == ("good", "good"):
                lines[key] = df

    if not any(preset == "bad" for _, preset in lines):
        return

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13.33, 7.5),
        sharex=True,
        sharey="row"
    )

    for col, prefix in enumerate(("pid", "mpc")):
        ax_gap, ax_accel = axes[0, col], axes[1, col]

        presets = [
            preset for preset in TUNING_ORDER
            if (prefix, preset) in lines
        ]

        for preset in presets:
            df = lines[(prefix, preset)]
            t = df["t_s"]
            color = COLORS[prefix] if preset == "good" else TUNING_COLORS[preset]
            label = TUNING_LABEL.get(preset, preset)

            ax_gap.plot(t, df[f"{prefix}_gap_m"], color=color, label=label,
                        zorder=3 if preset == "good" else 2)
            ax_accel.plot(t, derived_signals(df, prefix)[0], color=color,
                          linewidth=1.8, alpha=0.9,
                          zorder=3 if preset == "good" else 2)

            mark_collision(ax_gap, df, prefix)

        mark_startup((ax_gap, ax_accel), df)
        style_gap_axis(ax_gap)
        style_accel_axis(ax_accel)
        mark_brake_start((ax_gap, ax_accel), df)

        ax_gap.set_title(NAMES[prefix], color=INK, fontsize=16,
                         loc="left", fontweight="bold")
        ax_gap.legend(loc="upper right", frameon=True, facecolor="white",
                      edgecolor="none", framealpha=0.9)
        ax_accel.set_xlabel("Time [s]")
        ax_accel.set_xlim(t.iloc[0], t.iloc[-1])

        if col > 0:
            ax_gap.set_ylabel("")
            ax_accel.set_ylabel("")

    fig.suptitle(
        f"{scenario.replace('_', ' ').capitalize()}: good vs bad tuning",
        color=INK, fontsize=18, x=0.01, ha="left"
    )
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
    runs = {}

    for log in logs:
        run = log.parent.name
        df = pd.read_csv(log)
        df = df[df["t_s"] >= SETTLE_TIME].reset_index(drop=True)

        scenario, pid_part, mpc_part = run.rsplit("_", 2)
        pid_preset = pid_part.removeprefix("pid-")
        mpc_preset = mpc_part.removeprefix("mpc-")

        evaluated = df[df["t_s"] >= EVAL_START].reset_index(drop=True)

        for prefix in ("pid", "mpc"):
            records.append({
                "scenario": scenario,
                "pid_preset": pid_preset,
                "mpc_preset": mpc_preset,
                "controller": prefix.upper(),
                "tuning": pid_preset if prefix == "pid" else mpc_preset,
                **compute_metrics(evaluated, prefix),
            })

        runs[(scenario, pid_preset, mpc_preset)] = df

        title = (
            f"{scenario.replace('_', ' ').capitalize()}: "
            f"PID ({pid_preset}) vs MPC ({mpc_preset})"
        )

        # Replaced by overview.png
        (log.parent / "plots.png").unlink(missing_ok=True)

        plot_overview(df, title, log.parent / "overview.png")
        print(f"Plotted {log.parent / 'overview.png'}")

    plot_good_vs_good(runs, RESULTS / "good_vs_good.png")

    for scenario in ("gentle", "emergency", "stop_and_go"):
        plot_tuning(runs, scenario, RESULTS / f"tuning_{scenario}.png")

    metrics = pd.DataFrame(records)

    # The same controller + tuning appears in several runs (e.g. PID
    # good in pid-good_mpc-good and pid-good_mpc-bad): keep one row,
    # preferring the good/good run
    metrics["from_good_good"] = (
        (metrics["pid_preset"] == "good")
        & (metrics["mpc_preset"] == "good")
    )

    metrics["scenario_order"] = metrics["scenario"].map(
        {s: i for i, s in enumerate(SCENARIO_ORDER)}
    )
    metrics["tuning_order"] = metrics["tuning"].map(
        {s: i for i, s in enumerate(TUNING_ORDER)}
    )

    metrics = (
        metrics
        .sort_values("from_good_good", ascending=False)
        .drop_duplicates(["scenario", "controller", "tuning"])
        .sort_values(
            ["scenario_order", "controller", "tuning_order"],
            ascending=[True, False, True]
        )
        .drop(columns=["from_good_good", "scenario_order", "tuning_order"])
        .reset_index(drop=True)
    )

    metrics.to_csv(RESULTS / "metrics.csv", index=False)

    table = metrics.drop(columns=["pid_preset", "mpc_preset"]).copy()
    table["collision"] = table["collision"].map({True: "YES", False: "no"})
    table = table.rename(columns={"max_jerk_ms3": "max_jerk_ms3 (noisy)"})

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
