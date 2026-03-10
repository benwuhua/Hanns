#!/usr/bin/env python3
"""
Validate feature-list.json structure and integrity.
"""

from __future__ import annotations

import json
import sys


REQUIRED_FIELDS = {
    "id",
    "category",
    "title",
    "description",
    "priority",
    "status",
    "verification_steps",
}
VALID_STATUSES = {"failing", "passing"}
VALID_PRIORITIES = {"high", "medium", "low"}


def validate(path: str) -> list[str]:
    errors: list[str] = []

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, FileNotFoundError) as exc:
        return [f"Cannot read feature-list.json: {exc}"]

    if "features" not in data:
        return ['"features" key missing from root object']

    features = data["features"]
    if not isinstance(features, list):
        return ['"features" must be an array']

    ids_seen: set[object] = set()

    for index, feature in enumerate(features):
        prefix = f"Feature [{index}]"

        if not isinstance(feature, dict):
            errors.append(f"{prefix}: must be an object")
            continue

        missing = REQUIRED_FIELDS - set(feature.keys())
        if missing:
            errors.append(f"{prefix}: missing fields: {sorted(missing)}")

        feature_id = feature.get("id")
        if feature_id is not None:
            if feature_id in ids_seen:
                errors.append(f"{prefix}: duplicate id={feature_id}")
            ids_seen.add(feature_id)

        status = feature.get("status")
        if status and status not in VALID_STATUSES:
            errors.append(
                f"{prefix} (id={feature_id}): invalid status '{status}', "
                f"must be one of {sorted(VALID_STATUSES)}"
            )

        priority = feature.get("priority")
        if priority and priority not in VALID_PRIORITIES:
            errors.append(
                f"{prefix} (id={feature_id}): invalid priority '{priority}', "
                f"must be one of {sorted(VALID_PRIORITIES)}"
            )

        steps = feature.get("verification_steps")
        if steps is not None and (not isinstance(steps, list) or not steps):
            errors.append(
                f"{prefix} (id={feature_id}): verification_steps must be a non-empty array"
            )

    all_ids = {feature.get("id") for feature in features if isinstance(feature, dict)}
    for feature in features:
        if not isinstance(feature, dict):
            continue
        feature_id = feature.get("id")
        dependencies = feature.get("dependencies", [])
        if not isinstance(dependencies, list):
            errors.append(f"Feature id={feature_id}: dependencies must be an array")
            continue
        for dependency in dependencies:
            if dependency not in all_ids:
                errors.append(
                    f"Feature id={feature_id}: dependency id={dependency} does not exist"
                )

    return errors


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: validate_features.py <path/to/feature-list.json>")
        raise SystemExit(1)

    errors = validate(sys.argv[1])
    if errors:
        print(f"VALIDATION FAILED - {len(errors)} error(s):\n")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        data = json.load(handle)
    features = data["features"]
    passing = sum(1 for feature in features if feature.get("status") == "passing")
    failing = sum(1 for feature in features if feature.get("status") == "failing")
    print(f"VALID - {len(features)} features ({passing} passing, {failing} failing)")


if __name__ == "__main__":
    main()
