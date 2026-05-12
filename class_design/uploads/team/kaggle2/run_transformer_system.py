from __future__ import annotations

import argparse
import shutil
import shlex
import subprocess
import sys
from pathlib import Path

from pseudo_common import get_preset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command two-round runner for strict assets, KNN, calibration, soft pseudo, and student re-ensemble."
    )
    parser.add_argument("--asset-dir", default="models/strict_assets")
    parser.add_argument("--routes", nargs="+", default=["swin", "beit"], help="Base Transformer routes used to build assets.")
    parser.add_argument("--prob-models", nargs="+", default=["swin", "beit"])
    parser.add_argument("--feature-models", nargs="+", default=["swin", "beit", "dinov2_crop", "dinov2_full"])
    parser.add_argument("--train-routes", nargs="+", default=["transformer", "beit"], help="Routes to pseudo fine-tune.")

    parser.add_argument("--teacher-prefix", default="graph_knn")
    parser.add_argument("--calibrated-prefix", default="graph_knn_calibrated")
    parser.add_argument("--conservative-prefix", default="graph_knn_conservative")
    parser.add_argument("--conservative-calibrated-prefix", default="graph_knn_conservative_calibrated")
    parser.add_argument("--round1-teacher-blend-prefix", default="round1_teacher_blend")
    parser.add_argument("--round2-prefix", default="final_graph_knn")
    parser.add_argument("--round2-calibrated-prefix", default="final_graph_knn_calibrated")
    parser.add_argument("--round2-conservative-prefix", default="final_graph_knn_conservative")
    parser.add_argument("--round2-conservative-calibrated-prefix", default="final_graph_knn_conservative_calibrated")
    parser.add_argument("--round2-prob-models", nargs="+", default=None)
    parser.add_argument("--round2-feature-models", nargs="+", default=None)
    parser.add_argument("--pseudo-output", default="pseudo_soft_labels.csv")
    parser.add_argument("--final-submission", default="submission_transformer_system.csv")

    parser.add_argument("--skip-assets", action="store_true")
    parser.add_argument("--skip-dinov2", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--skip-conservative-graph", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-pseudo", action="store_true")
    parser.add_argument("--skip-round1-teacher-blend", action="store_true")
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--skip-second-round", action="store_true")
    parser.add_argument("--skip-final-blender", action="store_true")
    parser.add_argument("--skip-asset-manifest", action="store_true")
    parser.add_argument("--second-round-from-existing", action="store_true")
    parser.add_argument("--skip-final-copy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--dinov2-model", default="facebook/dinov2-base")
    parser.add_argument(
        "--dinov2-local-files-only",
        action="store_true",
        default=True,
        help="Load DINOv2 from local files/cache only. This is the default; no download is attempted.",
    )
    parser.add_argument(
        "--allow-dinov2-download",
        action="store_false",
        dest="dinov2_local_files_only",
        help="Explicitly allow transformers to download DINOv2 if it is not available locally.",
    )
    parser.add_argument("--dinov2-assets", nargs="+", default=["crop", "full"], choices=["crop", "full"])
    parser.add_argument("--no-normalize-features", action="store_true")
    parser.add_argument(
        "--prob-tta-modes",
        nargs="+",
        default=None,
        choices=["base", "zoom", "hflip"],
        help="Regenerate route OOF/test probabilities from fold weights using these TTA modes.",
    )
    parser.add_argument(
        "--infer-base-probs-from-weights",
        action="store_true",
        default=True,
        help="For base assets, regenerate OOF/test probabilities from supervised fold weights.",
    )
    parser.add_argument(
        "--use-existing-base-probs",
        action="store_false",
        dest="infer_base_probs_from_weights",
        help="Use existing OOF NPY and test CSV files for base assets instead of re-inferring from weights.",
    )

    parser.add_argument("--k-grid", default="5,10,20,40")
    parser.add_argument("--alpha-grid", default="0,5,15,30")
    parser.add_argument("--clip-grid", default="1e-7,1e-6,1e-5")
    parser.add_argument("--weight-grid", default="0,0.25,0.5,0.75,1.0,1.5")
    parser.add_argument("--prob-weight-grid", default="0,0.5,1.0,1.5,2.0")
    parser.add_argument("--no-prob-weight-search", action="store_true")
    parser.add_argument("--no-feature-preset-search", action="store_true")
    parser.add_argument("--feature-dirichlet-count", type=int, default=5)
    parser.add_argument("--per-class-calibration", action="store_true", default=True)
    parser.add_argument("--no-per-class-calibration", action="store_false", dest="per_class_calibration")

    parser.add_argument("--soft-threshold", type=float, default=0.90)
    parser.add_argument("--hard-threshold", type=float, default=0.98)
    parser.add_argument("--min-margin", type=float, default=0.10)
    parser.add_argument("--hard-min-margin", type=float, default=0.25)
    parser.add_argument("--per-class-limit", type=int, default=4000)
    parser.add_argument("--max-pseudo", type=int, default=40000)
    parser.add_argument("--pseudo-require-teacher-agreement", action="store_true", default=True)
    parser.add_argument("--no-pseudo-require-teacher-agreement", action="store_false", dest="pseudo_require_teacher_agreement")

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--pseudo-weight", type=float, default=0.30)
    parser.add_argument("--pseudo-sample-ratio", type=float, default=1.0)
    parser.add_argument("--soft-kl-weight", type=float, default=0.7)
    parser.add_argument("--hard-pseudo-weight", type=float, default=0.3)
    parser.add_argument("--kl-temperature", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)

    parser.add_argument("--device", default=None)
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def script_path(name: str) -> str:
    return str(Path(__file__).resolve().parent / name)


def add_optional(command: list[str], flag: str, value) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def run_command(command: list[str], dry_run: bool) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"\n[RUN] {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def run_asset_manifest(args: argparse.Namespace) -> None:
    if args.skip_asset_manifest:
        return
    command = python_cmd("validate_strict_assets.py") + [
        "--asset-dir",
        args.asset_dir,
        "--fail-on-issues",
    ]
    run_command(command, args.dry_run)


def copy_file(source: Path, destination: Path, dry_run: bool) -> None:
    print(f"\n[COPY] {source} -> {destination}", flush=True)
    if dry_run:
        return
    if not source.exists():
        raise FileNotFoundError(f"Final prediction CSV not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def python_cmd(script_name: str) -> list[str]:
    return [sys.executable, script_path(script_name)]


def route_asset_name(route: str) -> str:
    return get_preset(route).name


def pseudo_asset_name(route: str) -> str:
    return f"pseudo_{route_asset_name(route)}"


def pseudo_oof_path(route: str) -> Path:
    preset = get_preset(route)
    return Path(preset.save_dir) / preset.oof_name


def pseudo_weight_pattern(route: str) -> str:
    return get_preset(route).pseudo_weight_pattern


def build_strict_assets(
    args: argparse.Namespace,
    route: str,
    asset_name: str | None = None,
    weights: str | None = None,
    oof_preds: str | Path | None = None,
    test_preds_csv: str | Path | None = None,
    infer_probs_from_weights: bool | None = None,
) -> None:
    command = python_cmd("build_strict_oof_assets.py") + [
        "--route",
        route,
        "--output-dir",
        args.asset_dir,
    ]
    if asset_name:
        command.extend(["--asset-name", asset_name])
    add_optional(command, "--weights", weights)
    add_optional(command, "--oof-preds", oof_preds)
    add_optional(command, "--test-preds-csv", test_preds_csv)
    if not args.no_normalize_features:
        command.append("--normalize-features")
    add_optional(command, "--device", args.device)
    add_optional(command, "--batch-size", args.batch_size)
    add_optional(command, "--num-workers", args.num_workers)
    should_infer_probs = bool(args.prob_tta_modes) or bool(infer_probs_from_weights)
    if should_infer_probs:
        command.append("--infer-probs-from-weights")
        command.extend(["--prob-tta-modes", *(args.prob_tta_modes or ["base"])])
    if args.compile_model:
        command.append("--compile")
    if args.no_amp:
        command.append("--no-amp")
    run_command(command, args.dry_run)


def run_graph(
    args: argparse.Namespace,
    output_prefix: str,
    prob_models: list[str],
    feature_models: list[str],
    oof_neighbor_mode: str,
) -> None:
    command = python_cmd("graph_smoothing.py") + [
        "--asset-dir",
        args.asset_dir,
        "--output-prefix",
        output_prefix,
        "--k-grid",
        args.k_grid,
        "--alpha-grid",
        args.alpha_grid,
        "--clip-grid",
        args.clip_grid,
        "--weight-grid",
        args.weight_grid,
        "--prob-weight-grid",
        args.prob_weight_grid,
        "--feature-dirichlet-count",
        str(args.feature_dirichlet_count),
        "--oof-neighbor-mode",
        oof_neighbor_mode,
        "--prob-models",
        *prob_models,
        "--feature-models",
        *feature_models,
    ]
    if args.no_prob_weight_search:
        command.append("--no-prob-weight-search")
    if args.no_feature_preset_search:
        command.append("--no-feature-preset-search")
    run_command(command, args.dry_run)


def run_calibration(args: argparse.Namespace, input_prefix: str, output_prefix: str) -> None:
    command = python_cmd("calibrate_predictions.py") + [
        "--asset-dir",
        args.asset_dir,
        "--oof-preds",
        f"{input_prefix}_oof_preds.npy",
        "--test-preds",
        f"{input_prefix}_test_preds.npy",
        "--output-prefix",
        output_prefix,
    ]
    if args.per_class_calibration:
        command.append("--per-class")
    run_command(command, args.dry_run)


def run_final_blender(args: argparse.Namespace) -> Path:
    command = python_cmd("final_candidate_blender.py") + [
        "--asset-dir",
        args.asset_dir,
        "--submission",
        args.final_submission,
    ]
    run_command(command, args.dry_run)
    return Path(args.final_submission)


def run_candidate_blender(
    args: argparse.Namespace,
    candidate_prefixes: list[str],
    output_prefix: str,
    submission: Path,
) -> Path:
    command = python_cmd("final_candidate_blender.py") + [
        "--asset-dir",
        args.asset_dir,
        "--output-prefix",
        output_prefix,
        "--submission",
        str(submission),
        "--candidate-prefixes",
        *candidate_prefixes,
    ]
    run_command(command, args.dry_run)
    return submission


def generate_soft_pseudo(args: argparse.Namespace, teacher_csv: Path, agreement_csvs: list[Path] | None = None) -> None:
    command = python_cmd("make_soft_pseudo.py") + [
        "--teacher-csv",
        str(teacher_csv),
        "--soft-threshold",
        str(args.soft_threshold),
        "--hard-threshold",
        str(args.hard_threshold),
        "--min-margin",
        str(args.min_margin),
        "--hard-min-margin",
        str(args.hard_min_margin),
        "--per-class-limit",
        str(args.per_class_limit),
        "--max-pseudo",
        str(args.max_pseudo),
        "--output",
        args.pseudo_output,
    ]
    if agreement_csvs:
        command.extend(["--agreement-csvs", *[str(path) for path in agreement_csvs]])
    if args.pseudo_require_teacher_agreement and agreement_csvs:
        command.append("--require-agreement")
    run_command(command, args.dry_run)


def finetune_students(args: argparse.Namespace) -> None:
    for route in args.train_routes:
        command = python_cmd("train_with_pseudo.py") + [
            "--route",
            route,
            "--pseudo-csv",
            args.pseudo_output,
            "--soft-pseudo",
            "--soft-kl-weight",
            str(args.soft_kl_weight),
            "--hard-pseudo-weight",
            str(args.hard_pseudo_weight),
            "--kl-temperature",
            str(args.kl_temperature),
            "--pseudo-weight",
            str(args.pseudo_weight),
            "--pseudo-sample-ratio",
            str(args.pseudo_sample_ratio),
            "--epochs",
            str(args.epochs),
        ]
        add_optional(command, "--device", args.device)
        add_optional(command, "--batch-size", args.batch_size)
        add_optional(command, "--num-workers", args.num_workers)
        if args.compile_model:
            command.append("--compile")
        if args.no_amp:
            command.append("--no-amp")
        run_command(command, args.dry_run)


def generate_student_test_probs(args: argparse.Namespace, route: str, asset_name: str) -> Path:
    asset_dir = Path(args.asset_dir)
    prob_csv = asset_dir / f"test_preds_{asset_name}.csv"
    hard_pseudo_csv = asset_dir / f"hard_pseudo_from_{asset_name}.csv"
    command = python_cmd("pseudo_labeling.py") + [
        "--route",
        route,
        "--weights",
        pseudo_weight_pattern(route),
        "--prob-output",
        str(prob_csv),
        "--output",
        str(hard_pseudo_csv),
    ]
    add_optional(command, "--device", args.device)
    add_optional(command, "--batch-size", args.batch_size)
    add_optional(command, "--num-workers", args.num_workers)
    if args.compile_model:
        command.append("--compile")
    if args.no_amp:
        command.append("--no-amp")
    run_command(command, args.dry_run)
    return prob_csv


def build_second_round_student_assets(args: argparse.Namespace) -> list[str]:
    student_assets: list[str] = []
    for route in args.train_routes:
        asset_name = pseudo_asset_name(route)
        test_prob_csv = None if args.prob_tta_modes else generate_student_test_probs(args, route, asset_name)
        build_strict_assets(
            args,
            route=route,
            asset_name=asset_name,
            weights=pseudo_weight_pattern(route),
            oof_preds=None if args.prob_tta_modes else pseudo_oof_path(route),
            test_preds_csv=test_prob_csv,
        )
        student_assets.append(asset_name)
    return student_assets


def main() -> None:
    args = parse_args()
    asset_dir = Path(args.asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_assets:
        for route in args.routes:
            build_strict_assets(args, route, infer_probs_from_weights=args.infer_base_probs_from_weights)

    if not args.skip_dinov2:
        for dinov2_asset in args.dinov2_assets:
            command = python_cmd("extract_dinov2_features.py") + [
                "--model-name",
                args.dinov2_model,
                "--output-dir",
                args.asset_dir,
                "--asset-name",
                f"dinov2_{dinov2_asset}",
            ]
            if dinov2_asset == "crop":
                command.extend(
                    [
                        "--train-dir",
                        "dataset/imgs/train_cropped_v2",
                        "--train-fallback-dir",
                        "dataset/imgs/train",
                        "--test-dir",
                        "dataset/imgs/test_cropped_v2",
                        "--test-fallback-dir",
                        "dataset/imgs/test",
                    ]
                )
            else:
                command.extend(
                    [
                        "--train-dir",
                        "dataset/imgs/train",
                        "--train-fallback-dir",
                        "dataset/imgs/train",
                        "--test-dir",
                        "dataset/imgs/test",
                        "--test-fallback-dir",
                        "dataset/imgs/test",
                    ]
                )
            if not args.no_normalize_features:
                command.append("--normalize-features")
            if args.dinov2_local_files_only:
                command.append("--local-files-only")
            add_optional(command, "--device", args.device)
            add_optional(command, "--batch-size", args.batch_size)
            add_optional(command, "--num-workers", args.num_workers)
            run_command(command, args.dry_run)

    run_asset_manifest(args)

    final_csv = asset_dir / f"{args.calibrated_prefix}_test_preds.csv"

    if not args.skip_graph:
        run_graph(args, args.teacher_prefix, args.prob_models, args.feature_models, "transductive")
        if not args.skip_calibration:
            run_calibration(args, args.teacher_prefix, args.calibrated_prefix)
            final_csv = asset_dir / f"{args.calibrated_prefix}_test_preds.csv"

        if not args.skip_conservative_graph:
            run_graph(args, args.conservative_prefix, args.prob_models, args.feature_models, "train_only")
            if not args.skip_calibration:
                run_calibration(args, args.conservative_prefix, args.conservative_calibrated_prefix)

    teacher_csv = asset_dir / f"{args.calibrated_prefix}_test_preds.csv"
    agreement_csvs: list[Path] = []
    if args.skip_calibration:
        teacher_csv = asset_dir / f"{args.teacher_prefix}_test_preds.csv"
    elif not args.skip_graph and not args.skip_conservative_graph:
        transductive_teacher = asset_dir / f"{args.calibrated_prefix}_test_preds.csv"
        conservative_teacher = asset_dir / f"{args.conservative_calibrated_prefix}_test_preds.csv"
        if not args.skip_round1_teacher_blend:
            teacher_csv = run_candidate_blender(
                args,
                candidate_prefixes=[args.calibrated_prefix, args.conservative_calibrated_prefix],
                output_prefix=args.round1_teacher_blend_prefix,
                submission=asset_dir / f"{args.round1_teacher_blend_prefix}_test_preds.csv",
            )
        agreement_csvs = [transductive_teacher, conservative_teacher]

    if not args.skip_pseudo:
        generate_soft_pseudo(args, teacher_csv, agreement_csvs)

    if not args.skip_finetune:
        finetune_students(args)

    run_second_round = not args.skip_second_round and (not args.skip_finetune or args.second_round_from_existing)
    if run_second_round:
        student_assets = build_second_round_student_assets(args)
        run_asset_manifest(args)
        round2_prob_models = args.round2_prob_models or [*args.prob_models, *student_assets]
        round2_feature_models = args.round2_feature_models or [*args.feature_models, *student_assets]

        if not args.skip_graph:
            run_graph(args, args.round2_prefix, round2_prob_models, round2_feature_models, "transductive")
            if not args.skip_calibration:
                run_calibration(args, args.round2_prefix, args.round2_calibrated_prefix)
                final_csv = asset_dir / f"{args.round2_calibrated_prefix}_test_preds.csv"
            else:
                final_csv = asset_dir / f"{args.round2_prefix}_test_preds.csv"

            if not args.skip_conservative_graph:
                run_graph(args, args.round2_conservative_prefix, round2_prob_models, round2_feature_models, "train_only")
                if not args.skip_calibration:
                    run_calibration(args, args.round2_conservative_prefix, args.round2_conservative_calibrated_prefix)

    if not args.skip_final_blender and not args.skip_graph and not args.skip_calibration:
        final_csv = run_final_blender(args)
    elif not args.skip_final_copy and args.final_submission:
        copy_file(final_csv, Path(args.final_submission), args.dry_run)

    print("\n[INFO] Transformer system pipeline finished.", flush=True)


if __name__ == "__main__":
    main()
