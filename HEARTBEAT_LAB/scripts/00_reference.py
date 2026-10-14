"""Copy an existing pinned author checkout; preserve its code and GPL license."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

PIN = "3ccaa4f8ed11769b904af885e17018c39c89a2ac"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "experiments/temporal_garmin/cache/vendor/multirocket")
    args = parser.parse_args()
    head = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(args.source), "diff", "HEAD", "--", "multirocket", "LICENSE"], text=True)
    if head != PIN or dirty:
        raise SystemExit("Expected a clean MultiRocket author checkout at " + PIN)
    destination = ROOT / "vendor/MultiRocket"
    if destination.exists():
        raise SystemExit("Reference already exists; inspect it before replacing it.")
    destination.mkdir(parents=True)
    shutil.copytree(args.source / "multirocket", destination / "multirocket",
                    ignore=shutil.ignore_patterns("__pycache__", "*.nbc", "*.nbi"))
    for filename in ("LICENSE", "README.md"):
        shutil.copy2(args.source / filename, destination / filename)
    hashes = {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((destination / "multirocket").glob("*.py"))}
    (destination / "provenance.json").write_text(json.dumps({"commit": PIN, "sha256": hashes}, indent=2) + "\n")
    print("Prepared pinned MultiRocket reference in", destination)


if __name__ == "__main__":
    main()
