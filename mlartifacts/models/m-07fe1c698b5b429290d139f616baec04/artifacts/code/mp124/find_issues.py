"""Stage 1 entry point: hunt the seven data problems and fill in the
data_issues part of reports/findings.yaml.

    PYTHONPATH=src python -m mp124.find_issues          # print + write findings
    PYTHONPATH=src python -m mp124.find_issues --dry-run # print only

The findings file is edited line by line (not re-dumped with yaml.safe_dump)
so that every comment and every key from the original form is kept.
"""

from __future__ import annotations

import argparse
import re

import yaml

from mp124.checks import run_all_checks
from mp124.config import file_sha256, load_csv, load_params, resolve


def to_yaml_scalar(value) -> str:
    """Write one value the way the findings form expects."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return f'"{value}"'


def update_findings(text: str, issues: dict) -> str:
    """Replace `key: value` lines under data_issues -> <issue> -> <field>,
    keeping any trailing comment on the line."""
    out, section, issue = [], None, None
    for line in text.splitlines():
        top = re.match(r"^(\w+):", line)
        if top:
            section, issue = top.group(1), None
        sub = re.match(r"^  (\w+):\s*(#.*)?$", line)
        if section == "data_issues" and sub:
            issue = sub.group(1)
        field = re.match(r"^(    )(\w+):(\s*)([^#\n]*?)(\s*#.*)?$", line)
        if section == "data_issues" and issue in issues and field:
            indent, key, _, _, comment = field.groups()
            if key in issues[issue]:
                new = f"{indent}{key}: {to_yaml_scalar(issues[issue][key])}"
                if comment:  # keep the form's explanation comment
                    new = f"{new.ljust(26)} {comment.strip()}"
                line = new
        out.append(line)
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print only, do not write")
    args = parser.parse_args()

    params = load_params()
    train, later, holdout = (load_csv(n, params) for n in ("train", "later", "holdout"))

    print(f"train.csv sha256 = {file_sha256(resolve(params['data']['train']))}")
    print(f"rows: train={len(train)} later={len(later)} holdout={len(holdout)}\n")

    results = run_all_checks(train, later, holdout, params)
    for name, res in results.items():
        details = {k: v for k, v in res.items() if k != "evidence"}
        print(f"[{name}] {details}\n    -> {res['evidence']}")

    if args.dry_run:
        return
    path = resolve(params["reports"]["findings_out"])
    new_text = update_findings(path.read_text(encoding="utf-8"), results)
    yaml.safe_load(new_text)  # fail loudly if we ever produce invalid YAML
    path.write_text(new_text, encoding="utf-8")
    print(f"\nwrote data_issues to {path}")


if __name__ == "__main__":
    main()
