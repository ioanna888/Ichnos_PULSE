"""Verify the configured oxidative source against its manifest."""

import hashlib
import math
from pathlib import Path
import re
import runpy
import sys
import xml.etree.ElementTree as ET


def main():
    root = Path(__file__).resolve().parent.parent
    manifest = root / "docs/oxidative_model_manifest.md"
    config = root / "python/ichnos_config.py"

    try:
        text = manifest.read_text(encoding="utf-8")
        path_match = re.search(
            r"Canonical source SBML:\s*`([^`]+)`", text
        )
        hash_match = re.search(
            r"SHA256(?: του SBML)?:\s*`([a-fA-F0-9]{64})`", text
        )
        if not path_match or not hash_match:
            raise ValueError(
                "Missing Canonical source SBML or SHA256 in manifest."
            )

        declared = (root / path_match.group(1)).resolve()
        configured = Path(
            runpy.run_path(str(config))["VARIANTS"]["ox"]["sensing_file"]
        ).resolve()

        if configured != declared:
            raise ValueError(
                f"Source path mismatch: {configured} != {declared}"
            )

        payload = configured.read_bytes()
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != hash_match.group(1).lower():
            raise ValueError(f"SHA256 mismatch: {actual_hash}")

        ns = {"s": "http://www.sbml.org/sbml/level3/version2/core"}
        model = ET.fromstring(payload)
        params = {
            p.get("id"): p
            for p in model.findall(
                "s:model/s:listOfParameters/s:parameter", ns
            )
        }

        count = 0
        for line in text.splitlines():
            row = re.match(
                r"^\|\s*`([^`]+)`\s*\|\s*([0-9]+(?:\.[0-9]+)?)",
                line,
            )
            if not row:
                continue
            name, value = row.groups()
            if name not in params:
                raise ValueError(f"Missing parameter: {name}")
            actual = float(params[name].get("value"))
            if not math.isclose(
                actual, float(value), rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(
                    f"{name}: SBML={actual}, manifest={value}"
                )
            count += 1

        if count < 11:
            raise ValueError(
                f"Only {count} readable parameter rows; expected 11."
            )

        if params["S_ox"].get("constant") != "true":
            raise ValueError("Source S_ox must be constant.")
        if "k_clear" in params:
            raise ValueError("Unexpected k_clear in frozen source.")
        rules = model.findall(
            "s:model/s:listOfRules/s:rateRule", ns
        )
        if any(rule.get("variable") == "S_ox" for rule in rules):
            raise ValueError("Unexpected source rate rule for S_ox.")

        print(f"PASS: source = {path_match.group(1)}")
        print(f"PASS: SHA256 = {actual_hash}")
        print(f"PASS: {count} manifest parameters match.")
        print("PASS: source S_ox is constant, without clearance.")
        return 0

    except (OSError, ValueError, KeyError, ET.ParseError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
    