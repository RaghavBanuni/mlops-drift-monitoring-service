"""Command line entry points: fit a baseline, scan a window, or reproduce the evaluation.

``python -m driftwatch.cli evaluate`` is the one to run first.  It prints, per scenario,
how many windows each detector took to notice a real change and how often it fired when nothing
had happened - which is the only comparison that distinguishes a monitor from a noise generator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .alerts import AlertPolicy
from .baseline import Baseline, fit_baseline
from .detect import DriftPolicy, decision_table, psi_noise_floor, scan_window
from .monitor import Monitor, MonitorConfig
from .simulate import SCENARIOS, ScenarioConfig, build_scenario, evaluate_all, evaluate_detectors


def _read(path: str) -> pd.DataFrame:
    file = Path(path)
    if not file.exists():
        raise SystemExit(f"no such file: {file}")
    if file.suffix in (".parquet", ".pq"):
        return pd.read_parquet(file)
    return pd.read_csv(file)


def _show(frame: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if frame.empty:
        print("(nothing to show)")
        return
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(frame.to_string(index=False))


def command_baseline(args: argparse.Namespace) -> int:
    frame = _read(args.data)
    baseline = fit_baseline(
        frame,
        features=args.features,
        prediction_column=args.prediction_column,
        target_column=args.target_column,
        n_bins=args.bins,
        model_version=args.model_version,
    )
    destination = baseline.save(args.out)
    print(f"baseline written to {destination}")
    print(
        json.dumps(
            {
                "rows": baseline.n_rows,
                "numeric": list(baseline.numeric),
                "categorical": list(baseline.categorical),
                "baseline_auc": baseline.baseline_auc,
                "model_version": baseline.model_version,
            },
            indent=2,
        )
    )
    floor = psi_noise_floor(baseline.n_rows, args.window_rows, args.bins)
    print(
        f"\nwith {args.window_rows}-row windows and {args.bins} bins, PSI sits around "
        f"{floor:.4f} with no drift at all; the effective warn threshold will be "
        f"max(0.10, 3 x {floor:.4f}) = {max(0.10, 3 * floor):.4f}"
    )
    return 0


def command_scan(args: argparse.Namespace) -> int:
    baseline = Baseline.load(args.baseline)
    monitor = Monitor(
        baseline,
        policy=DriftPolicy(psi_warn=args.psi_warn, psi_alert=args.psi_alert, alpha=args.alpha),
        config=MonitorConfig(prediction_column=args.prediction_column, id_column=args.id_column),
    )
    outcome = monitor.ingest(_read(args.data), window=args.window)
    report = monitor.reports[-1]

    _show(report.schema.frame(), "schema")
    _show(report.frame(), "drift")
    print(f"\nverdict: {outcome['verdict']}")
    for alert in outcome["alerts"]:
        print(f"  [{alert['severity']}] {alert['scope']}: {alert['reason']}")
    if not outcome["alerts"]:
        print("  (nothing paged: a first window can flag but not confirm)")
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"\nfull report written to {args.json}")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    config = ScenarioConfig(
        n_reference=args.reference_rows,
        n_window=args.window_rows,
        n_windows=args.windows,
        change_at=args.change_at,
        seed=args.seed,
    )
    if args.scenario == "all":
        frame = evaluate_all(config)
        for name, group in frame.groupby("scenario", sort=False):
            _show(group.drop(columns="scenario"), f"{name}")
        _show(
            frame.groupby("detector", sort=False)
            .agg(
                false_alarm_windows=("false_alarm_windows", "sum"),
                mean_delay=("delay_windows", "mean"),
                missed=("detected", lambda column: int((column == False).sum())),
            )
            .reset_index(),
            "across every scenario (fewer false alarms and lower delay both matter)",
        )
    else:
        _show(evaluate_detectors(args.scenario, config), args.scenario)
    print(
        "\nread the columns together: a detector that fires instantly and also fires when "
        "nothing happened is not better, it is louder"
    )
    return 0


def command_replay(args: argparse.Namespace) -> int:
    """Run a scenario through the full Monitor, including delayed labels."""
    config = ScenarioConfig(
        n_reference=args.reference_rows,
        n_window=args.window_rows,
        n_windows=args.windows,
        change_at=args.change_at,
        seed=args.seed,
    )
    scenario = build_scenario(args.scenario, config)
    baseline = fit_baseline(
        scenario.reference,
        features=[
            column
            for column in scenario.reference.columns
            if column not in ("row_id", "prediction", "label")
        ],
        prediction_column="prediction",
        target_column="label",
    )
    monitor = Monitor(
        baseline,
        alert_policy=AlertPolicy(confirm_windows=args.confirm, of_windows=args.of_windows),
        config=MonitorConfig(prediction_column="prediction", id_column="row_id"),
    )
    for index, window in enumerate(scenario.windows):
        monitor.ingest(window, index)
        matured = index - args.label_lag  # outcomes arrive `label_lag` windows late
        if matured >= 0:
            source = scenario.windows[matured]
            monitor.add_labels(source["row_id"], source["label"], index)
        if monitor.labels.matched:
            monitor.quality(window=index)

    _show(monitor.history(), f"{scenario.name}: {scenario.description}")
    _show(monitor.alerts.frame(), "alerts raised")
    print("\nalerting:", json.dumps(monitor.alerts.stats()))
    print("coverage:", json.dumps(monitor.coverage()))
    print("reading:", json.dumps(monitor.interpret()["case"]))
    print("action:", monitor.interpret()["action"])
    if scenario.change_at is not None:
        print(f"(ground truth: the change was injected at window {scenario.change_at})")
    return 0


def command_decisions(_: argparse.Namespace) -> int:
    _show(decision_table(), "what a drift signal means once quality is known")
    return 0


def command_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as error:  # pragma: no cover - depends on the install extras
        raise SystemExit("uvicorn is not installed; pip install 'driftwatch[service]'") from error
    from .service import create_app

    monitor = None
    if args.baseline:
        monitor = Monitor(
            Baseline.load(args.baseline),
            config=MonitorConfig(prediction_column=args.prediction_column, id_column=args.id_column),
        )
    uvicorn.run(create_app(monitor), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="driftwatch",
        description="Drift and quality monitoring for models already in production.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("baseline", help="fit and save a pinned reference profile")
    fit.add_argument("--data", required=True, help="CSV or Parquet of the reference period")
    fit.add_argument("--out", default="baseline.json")
    fit.add_argument("--features", nargs="*")
    fit.add_argument("--prediction-column", default=None)
    fit.add_argument("--target-column", default=None)
    fit.add_argument("--bins", type=int, default=10)
    fit.add_argument("--window-rows", type=int, default=1000, help="expected window size")
    fit.add_argument("--model-version", default="unversioned")
    fit.set_defaults(handler=command_baseline)

    scan = subparsers.add_parser("scan", help="score one window against a saved baseline")
    scan.add_argument("--baseline", required=True)
    scan.add_argument("--data", required=True)
    scan.add_argument("--window", type=int, default=0)
    scan.add_argument("--prediction-column", default=None)
    scan.add_argument("--id-column", default=None)
    scan.add_argument("--psi-warn", type=float, default=0.10)
    scan.add_argument("--psi-alert", type=float, default=0.25)
    scan.add_argument("--alpha", type=float, default=0.05)
    scan.add_argument("--json", default=None, help="also write the full report here")
    scan.set_defaults(handler=command_scan)

    evaluate = subparsers.add_parser(
        "evaluate", help="detection delay against false alarms, on scenarios with known truth"
    )
    evaluate.add_argument("--scenario", default="all", choices=("all", *SCENARIOS))
    evaluate.add_argument("--reference-rows", type=int, default=4000)
    evaluate.add_argument("--window-rows", type=int, default=800)
    evaluate.add_argument("--windows", type=int, default=20)
    evaluate.add_argument("--change-at", type=int, default=10)
    evaluate.add_argument("--seed", type=int, default=7)
    evaluate.set_defaults(handler=command_evaluate)

    replay = subparsers.add_parser(
        "replay", help="run a scenario through the full monitor, with delayed labels"
    )
    replay.add_argument("--scenario", default="sudden_covariate", choices=SCENARIOS)
    replay.add_argument("--reference-rows", type=int, default=4000)
    replay.add_argument("--window-rows", type=int, default=800)
    replay.add_argument("--windows", type=int, default=20)
    replay.add_argument("--change-at", type=int, default=10)
    replay.add_argument("--seed", type=int, default=7)
    replay.add_argument("--confirm", type=int, default=2)
    replay.add_argument("--of-windows", type=int, default=3)
    replay.add_argument("--label-lag", type=int, default=3, help="windows before outcomes arrive")
    replay.set_defaults(handler=command_replay)

    subparsers.add_parser(
        "decisions", help="print the drift-versus-quality decision table"
    ).set_defaults(handler=command_decisions)

    serve = subparsers.add_parser("serve", help="run the HTTP monitoring sidecar")
    serve.add_argument("--baseline", default=None)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--prediction-column", default="prediction")
    serve.add_argument("--id-column", default=None)
    serve.set_defaults(handler=command_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
