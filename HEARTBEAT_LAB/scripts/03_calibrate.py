"""Add validation-fitted sigmoid calibration to existing model bundles."""
import argparse
from pathlib import Path
from heartbeat_lab.calibration import calibrate_bundle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundles", nargs="+", type=Path)
    for path in parser.parse_args().bundles:
        result = calibrate_bundle(path)
        print(path.name, "validation n =", result["n"], "test =", result.get("test"))
