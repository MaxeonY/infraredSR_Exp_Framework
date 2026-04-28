import argparse
import os
import runpy
import sys
from pathlib import Path
from typing import Dict, List


SCRIPT_MAP: Dict[str, str] = {
    "preprocess": "datasets/preprocess.py",
    "train": "train.py",
    "test": "test.py",
    "compare_results": "compare_results.py",
    "infer": "infer.py",
}


def _strip_leading_double_dash(args: List[str]) -> List[str]:
    if len(args) > 0 and args[0] == "--":
        return args[1:]
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="InfraredSR unified entrypoint. Dispatch to preprocess/train/test/infer scripts."
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Project root directory. Defaults to this file's directory.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print the resolved script and forwarded args without executing.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    for cmd in SCRIPT_MAP:
        sp = subparsers.add_parser(cmd, help=f"Run {SCRIPT_MAP[cmd]}")
        sp.add_argument(
            "script_args",
            nargs=argparse.REMAINDER,
            help="Arguments forwarded to target script. You can prefix with '--' for clarity.",
        )

    return parser


def run_target_script(project_root: Path, command: str, script_args: List[str], dry_run: bool = False) -> int:
    if command not in SCRIPT_MAP:
        raise ValueError(f"Unknown command: {command}")

    script_rel = SCRIPT_MAP[command]
    script_path = (project_root / script_rel).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Target script does not exist: {script_path}")

    forwarded = _strip_leading_double_dash(script_args)

    print("=" * 70)
    print(f"[main] command    : {command}")
    print(f"[main] script     : {script_path}")
    print(f"[main] project    : {project_root.resolve()}")
    print(f"[main] args       : {forwarded if forwarded else '[]'}")
    print("=" * 70)

    if dry_run:
        return 0

    old_argv = sys.argv
    old_cwd = Path.cwd()
    exit_code = 0

    try:
        os.chdir(project_root)
        sys.argv = [str(script_path)] + forwarded
        runpy.run_path(str(script_path), run_name="__main__")
    except SystemExit as e:
        if isinstance(e.code, int):
            exit_code = e.code
        elif e.code is None:
            exit_code = 0
        else:
            exit_code = 1
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    return exit_code


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        raise FileNotFoundError(f"project_root does not exist: {project_root}")
    if not project_root.is_dir():
        raise NotADirectoryError(f"project_root is not a directory: {project_root}")

    code = run_target_script(
        project_root=project_root,
        command=args.command,
        script_args=args.script_args,
        dry_run=args.dry_run,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
