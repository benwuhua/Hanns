#!/usr/bin/env python3
"""
Validate feature-list.json structure and integrity.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path


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
CURRENT_FOCUS_PATTERN = re.compile(r"^- Current focus: `([^`]+)`(?: .*)?$", re.MULTILINE)
NEXT_FEATURE_PATTERN = re.compile(r"^- Next feature: `([^`]+)`(?: .*)?$", re.MULTILINE)
PROGRESS_PATTERN = re.compile(
    r"^- Progress: (\d+)/(\d+) features passing \((\d+)%\)$",
    re.MULTILINE,
)
SESSION_LOG_PATTERN = re.compile(r"^## Session Log$", re.MULTILINE)
SESSION_HEADING_PATTERN = re.compile(r"^### Session .+$", re.MULTILINE)
TEMP_ARTIFACT_SUFFIXES = (".new", ".old", ".tmp")
SKIP_SCAN_DIRECTORIES = {".git", "target", "__pycache__"}
REQUIRED_RELEASE_NOTES_MARKERS = ("# Release Notes", "## [Unreleased]")


def read_json(path: Path) -> dict | list | str:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_required_match(
    pattern: re.Pattern[str], text: str, error_message: str
) -> re.Match[str] | None:
    match = pattern.search(text)
    if match is None:
        return None
    return match


def extract_latest_session(task_progress: str) -> tuple[str | None, str | None]:
    session_log_match = SESSION_LOG_PATTERN.search(task_progress)
    if session_log_match is None:
        return None, "task-progress.md missing `## Session Log` section"

    session_matches = list(SESSION_HEADING_PATTERN.finditer(task_progress, session_log_match.end()))
    if not session_matches:
        return None, "task-progress.md missing a latest `### Session ...` block"

    start = session_matches[0].start()
    end = session_matches[1].start() if len(session_matches) > 1 else len(task_progress)
    return task_progress[start:end], None


def find_latest_session_placeholder(latest_session: str) -> str | None:
    for line in latest_session.splitlines():
        stripped = line.strip()
        if "pending rerun after durable-state update" in stripped:
            return "pending rerun after durable-state update"
        if "-> pending" in stripped:
            return stripped
    return None


def find_bad_verification_step(step: str) -> str | None:
    try:
        tokens = shlex.split(step)
    except ValueError:
        return None

    if len(tokens) < 2 or tokens[0] != "cargo" or tokens[1] != "test":
        return None

    cargo_test_tokens: list[str] = []
    for token in tokens[2:]:
        if token == "--":
            break
        cargo_test_tokens.append(token)

    if "--tests" not in cargo_test_tokens:
        return None

    tests_index = cargo_test_tokens.index("--tests")
    for token in cargo_test_tokens[tests_index + 1 :]:
        if not token.startswith("-"):
            return (
                f"verification step '{step}' uses `--tests` with a bare filter; "
                "use `--test <binary>` for integration test binaries"
            )
    return None


def find_temp_artifacts(repo_root: Path) -> list[Path]:
    temp_artifacts: list[Path] = []

    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_SCAN_DIRECTORIES for part in path.parts):
            continue
        if path.suffix in TEMP_ARTIFACT_SUFFIXES:
            temp_artifacts.append(path)

    return sorted(temp_artifacts)


def validate(path: str) -> list[str]:
    errors: list[str] = []
    feature_list_path = Path(path).resolve()
    repo_root = feature_list_path.parent

    try:
        data = read_json(feature_list_path)
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
        elif isinstance(steps, list):
            for step in steps:
                if not isinstance(step, str):
                    errors.append(
                        f"{prefix} (id={feature_id}): verification_steps entries must be strings"
                    )
                    continue
                bad_pattern = find_bad_verification_step(step)
                if bad_pattern is not None:
                    errors.append(f"Feature id={feature_id}: {bad_pattern}")

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

    task_progress_path = repo_root / "task-progress.md"
    release_notes_path = repo_root / "RELEASE_NOTES.md"
    try:
        task_progress = read_text(task_progress_path)
    except FileNotFoundError:
        errors.append(f"Missing durable workflow file: {task_progress_path}")
        task_progress = ""

    try:
        release_notes = read_text(release_notes_path)
    except FileNotFoundError:
        errors.append(f"Missing durable workflow file: {release_notes_path}")
        release_notes = ""

    if release_notes:
        for marker in REQUIRED_RELEASE_NOTES_MARKERS:
            if marker not in release_notes:
                errors.append(f"RELEASE_NOTES.md missing required marker: {marker}")

    if task_progress:
        current_focus_match = parse_required_match(
            CURRENT_FOCUS_PATTERN,
            task_progress,
            "task-progress.md missing `- Current focus: ` line",
        )
        failing = sum(1 for feature in features if feature.get("status") == "failing")

        if current_focus_match is None:
            errors.append("task-progress.md missing `- Current focus: ` line")
        else:
            current_focus = current_focus_match.group(1)
            if current_focus == "none":
                if failing != 0:
                    errors.append(
                        "task-progress.md current focus may be `none` only when all features are passing"
                    )
            elif current_focus not in all_ids:
                errors.append(
                    f"task-progress.md current focus `{current_focus}` does not exist in feature-list.json"
                )

        next_feature_match = parse_required_match(
            NEXT_FEATURE_PATTERN,
            task_progress,
            "task-progress.md missing `- Next feature: ` line",
        )
        if next_feature_match is None:
            errors.append("task-progress.md missing `- Next feature: ` line")
        else:
            next_feature = next_feature_match.group(1)
            if next_feature == "none":
                if failing != 0:
                    errors.append(
                        "task-progress.md next feature may be `none` only when all features are passing"
                    )
            elif next_feature not in all_ids:
                errors.append(
                    f"task-progress.md next feature `{next_feature}` does not exist in feature-list.json"
                )

        progress_match = parse_required_match(
            PROGRESS_PATTERN,
            task_progress,
            "task-progress.md missing `- Progress: ` summary line",
        )
        passing = sum(1 for feature in features if feature.get("status") == "passing")
        total = len(features)
        if progress_match is None:
            errors.append("task-progress.md missing `- Progress: ` summary line")
        else:
            logged_passing = int(progress_match.group(1))
            logged_total = int(progress_match.group(2))
            logged_percent = int(progress_match.group(3))
            expected_percent = round((passing / total) * 100) if total else 0
            if logged_passing != passing or logged_total != total:
                errors.append(
                    f"task-progress.md progress summary says {logged_passing}/{logged_total} "
                    f"features passing but feature-list.json has {passing}/{total}"
                )
            if logged_percent != expected_percent:
                errors.append(
                    f"task-progress.md progress summary says {logged_percent}% but "
                    f"feature-list.json implies {expected_percent}%"
                )

        latest_session, session_error = extract_latest_session(task_progress)
        if session_error is not None:
            errors.append(session_error)
        elif latest_session is not None:
            placeholder = find_latest_session_placeholder(latest_session)
            if placeholder is not None:
                errors.append(
                    f"Latest session still contains unresolved placeholder: {placeholder}"
                )

    for artifact in find_temp_artifacts(repo_root):
        errors.append(
            f"Repository contains temporary/orphan artifact: {artifact.relative_to(repo_root)}"
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
    print(
        f"VALID - {len(features)} features ({passing} passing, {failing} failing); "
        "workflow/doc checks passed"
    )


if __name__ == "__main__":
    main()
