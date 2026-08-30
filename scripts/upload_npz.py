import argparse
import os


def ensure_wandb_runtime() -> None:
    """Fail early with a helpful message when the active Python env is missing required CFFI crypto dependencies."""
    try:
        import _cffi_backend  # noqa: F401
        import cryptography  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment-specific failure path
        raise RuntimeError(
            "W&B artifact upload failed because the active Python environment is missing the native CFFI/cryptography dependency.\n"
            "Fix with one of:\n"
            "  source /root/gpufree-data/envs/isaaclab/bin/activate\n"
            "  python -m pip install --upgrade cffi cryptography\n"
            "or\n"
            "  conda install -n isaaclab cffi cryptography"
        ) from exc


ensure_wandb_runtime()
import wandb


def build_registry_target(entity: str | None, registry_name: str, collection_name: str) -> str:
    """Normalize a registry path to the form used by replay_npz.py.

    Examples:
        - "wandb-registry-motions/Motions"
        - "1760298213-beihang-university-org/wandb-registry-motions/Motions"
        - "1760298213-beihang-university-org/wandb-registry-motions/Motions:latest"
    """
    target = registry_name.strip().strip("/")

    if "wandb-registry-" not in target:
        target = f"wandb-registry-{target}"

    if target.startswith("wandb-registry-") and entity:
        if "/" not in target:
            target = f"{entity}/{target}"

    if not target.endswith(f"/{collection_name}") and not target.endswith(f"/{collection_name}:latest"):
        target = f"{target}/{collection_name}"

    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a .npz motion file to the W&B registry.")
    parser.add_argument("--artifact_path", type=str, default="./motion.npz", help="Path to the .npz file to upload.")
    parser.add_argument("--artifact_name", type=str, default=None, help="Artifact name; defaults to the file stem.")
    parser.add_argument("--project", type=str, default="csv_to_npz", help="W&B project name.")
    parser.add_argument(
        "--run_entity",
        type=str,
        default=os.environ.get("WANDB_TEAM_ENTITY") or os.environ.get("WANDB_ENTITY"),
        help="W&B entity allowed for run logging. Usually your team/user entity, not the org.",
    )
    parser.add_argument(
        "--registry_entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY") or os.environ.get("WANDB_TEAM_ENTITY"),
        help="Organization/entity that owns the registry. Usually the org name.",
    )
    parser.add_argument(
        "--registry_name",
        type=str,
        default="wandb-registry-motions",
        help="Registry path, e.g. wandb-registry-motions or 1760298213-beihang-university-org/wandb-registry-motions",
    )
    parser.add_argument("--collection_name", type=str, default="Motions", help="Artifact collection name in the registry.")
    args = parser.parse_args()

    if not os.path.exists(args.artifact_path):
        raise FileNotFoundError(f"Artifact file not found: {args.artifact_path}")

    if not args.run_entity:
        raise ValueError("Need a valid run entity: pass --run_entity or set WANDB_TEAM_ENTITY/WANDB_ENTITY.")

    artifact_name = args.artifact_name or os.path.splitext(os.path.basename(args.artifact_path))[0]
    registry_target = build_registry_target(args.registry_entity, args.registry_name, args.collection_name)

    run = wandb.init(project=args.project, name=artifact_name, entity=args.run_entity)
    print(f"[INFO]: Uploading '{args.artifact_path}' to W&B project '{args.project}' via run entity '{args.run_entity}' ...")

    logged_artifact = run.log_artifact(artifact_or_path=args.artifact_path, name=artifact_name, type="motions")
    run.link_artifact(artifact=logged_artifact, target_path=registry_target)

    print(f"[INFO]: Registry target: {registry_target}")
    print(f"[INFO]: Replay command: python scripts/replay_npz.py --registry_name={registry_target}:latest")


if __name__ == "__main__":
    main()
