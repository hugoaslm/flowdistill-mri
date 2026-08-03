from __future__ import annotations

import argparse
import json
from pathlib import Path

from flowdistill_mri.config import load_config
from flowdistill_mri.evaluation.generation import evaluate_generation
from flowdistill_mri.training.builders import build_teacher
from flowdistill_mri.training.distill import distill_freeflow
from flowdistill_mri.training.runtime import create_run_directory
from flowdistill_mri.training.teacher import train_teacher


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)


def _train_teacher(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    output = args.output or create_run_directory(cfg, "teacher") / "checkpoint"
    print(train_teacher(cfg, output, resume=args.resume))


def _distill(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    output = args.output or create_run_directory(cfg, "freeflow") / "checkpoint"
    print(distill_freeflow(cfg, args.teacher_checkpoint, output, resume=args.resume))


def _evaluate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    output = args.output or create_run_directory(cfg, "evaluation")
    results = evaluate_generation(cfg, args.teacher_checkpoint, args.student_checkpoint, output)
    print(json.dumps(results, indent=2))


def _inspect(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    parameters = sum(value.numel() for value in build_teacher(cfg).parameters())
    print(json.dumps({"config": cfg.to_dict(), "parameters": parameters}, indent=2))


def _smoke(args: argparse.Namespace) -> None:
    cfg = load_config(Path("configs") / f"smoke_{args.tier}.yaml")
    run = create_run_directory(cfg, f"smoke-{args.tier}")
    print(f"[smoke] tier={args.tier} run={run}", flush=True)
    print("[smoke] stage 1/3: teacher", flush=True)
    teacher_checkpoint = train_teacher(cfg, run / "teacher")
    print("[smoke] stage 2/3: FreeFlow distillation", flush=True)
    student_checkpoint = distill_freeflow(cfg, teacher_checkpoint, run / "freeflow")
    print("[smoke] stage 3/3: evaluation", flush=True)
    results = evaluate_generation(cfg, teacher_checkpoint, student_checkpoint, run / "evaluation")
    print("[smoke] complete", flush=True)
    print(json.dumps({"run": str(run), "results": results}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlowDistill MRI research CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect", help="validate config and report model size")
    _config_argument(inspect)
    inspect.set_defaults(handler=_inspect)
    teacher = subparsers.add_parser("train-teacher", help="train a rectified-flow teacher")
    _config_argument(teacher)
    teacher.add_argument("--output", type=Path)
    teacher.add_argument("--resume", action="store_true")
    teacher.set_defaults(handler=_train_teacher)
    distill = subparsers.add_parser("distill-freeflow", help="distill Track A")
    _config_argument(distill)
    distill.add_argument("--teacher-checkpoint", required=True, type=Path)
    distill.add_argument("--output", type=Path)
    distill.add_argument("--resume", action="store_true")
    distill.set_defaults(handler=_distill)
    evaluate = subparsers.add_parser("evaluate-generation", help="compare teacher and student")
    _config_argument(evaluate)
    evaluate.add_argument("--teacher-checkpoint", required=True, type=Path)
    evaluate.add_argument("--student-checkpoint", required=True, type=Path)
    evaluate.add_argument("--output", type=Path)
    evaluate.set_defaults(handler=_evaluate)
    smoke = subparsers.add_parser("smoke", help="run an opt-in synthetic pipeline")
    smoke.add_argument("--tier", choices=("ci", "local"), default="ci")
    smoke.set_defaults(handler=_smoke)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
