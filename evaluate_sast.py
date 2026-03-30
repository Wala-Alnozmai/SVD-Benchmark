"""
evaluate_sast.py — Semgrep and SonarQube baseline evaluation.

Implements Section 3.2 (Static Analysis Baselines) of the paper:
  "A CWE-Aware Benchmark and Comparative Evaluation of LLMs for Java
   Software Vulnerability Detection"

Baselines used (Section 3.2):
  - Semgrep   : high-velocity intra-procedural analysis.
                TP only if the tool flags the SPECIFIC VULNERABLE LINES
                defined in the expert-validated ground truth.
  - SonarQube : inter-procedural taint analysis.
                Run with the full build environment reconstructed for
                the parent commit of each vulnerability.

CodeQL and Snyk are EXCLUDED from evaluation (they were used to
construct the ground truth labels, so including them would introduce
circularity bias — Section 3.2).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import DATASET_PATH, RESULTS_DIR
from data_utils import load_dataset, get_eval_set


# ---------------------------------------------------------------------------
# Semgrep
# ---------------------------------------------------------------------------
def run_semgrep(java_file: Path) -> list[int]:
    """
    Run Semgrep on a single Java file and return the list of flagged line
    numbers.

    Uses the Java security rule pack (p/java) which covers the CWE
    categories present in the benchmark.
    """
    result = subprocess.run(
        [
            "semgrep", "--config", "p/java",
            "--json",
            str(java_file),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):   # semgrep exits 1 when findings exist
        return []

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    flagged_lines = []
    for finding in data.get("results", []):
        start = finding.get("start", {}).get("line")
        end   = finding.get("end",   {}).get("line")
        if start:
            flagged_lines.extend(range(start, (end or start) + 1))
    return flagged_lines


def semgrep_is_tp(
    flagged_lines:   list[int],
    vulnerable_line: str | int,
    line_offset:     int = 0,
) -> bool:
    """
    A Semgrep detection is a True Positive only if it flags the SPECIFIC
    vulnerable line(s) defined in the expert-validated ground truth
    (Section 3.2).

    line_offset accounts for any boilerplate lines prepended to the temp
    file before the snippet begins.  When the snippet is wrapped inside
    ``public class Snippet_N {\\n<code>\\n}``, the first line of the
    original snippet appears at line 2 in the temp file, so pass
    ``line_offset=1``.
    """
    if not flagged_lines:
        return False
    try:
        vline = int(str(vulnerable_line).strip().split(",")[0])
        return (vline + line_offset) in flagged_lines
    except (ValueError, TypeError):
        return False


def evaluate_semgrep(df: pd.DataFrame) -> list[dict]:
    """Evaluate Semgrep on the full evaluation set."""
    eval_set = get_eval_set(df)
    results  = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, row in tqdm(eval_set.iterrows(), total=len(eval_set), desc="Semgrep"):
            code          = row["Code Snippet"]
            label         = row["label"]
            vuln_line_num = row.get("Line Number", "")

            # Write snippet to a temporary .java file
            tmp_file = Path(tmpdir) / f"Snippet_{idx}.java"
            tmp_file.write_text(
                f"public class Snippet_{idx} {{\n{code}\n}}"
            )

            flagged   = run_semgrep(tmp_file)
            is_vuln   = label != "BENIGN"

            # Line offset: the snippet is wrapped inside
            #   public class Snippet_N {\n<code>\n}
            # so the class-declaration line shifts every snippet line by +1.
            LINE_OFFSET = 1

            # Prediction logic:
            # - If Semgrep flags ANY line: predicted vulnerable
            # - A True Positive requires the specific vulnerable line to be flagged
            if flagged:
                # For TP accuracy: check line-level match (accounting for class wrapper)
                line_match = semgrep_is_tp(flagged, vuln_line_num, line_offset=LINE_OFFSET) if is_vuln else False
                prediction = label if line_match else "CWE-UNKNOWN"
            else:
                prediction = "BENIGN"

            results.append({
                "index":          idx,
                "ground_truth":   label,
                "prediction":     prediction,
                "flagged_lines":  flagged,
                "line_match":     semgrep_is_tp(flagged, vuln_line_num, line_offset=LINE_OFFSET) if is_vuln else None,
            })

    return results


# ---------------------------------------------------------------------------
# SonarQube
# ---------------------------------------------------------------------------
def run_sonarqube(project_dir: Path, project_key: str, token: str) -> dict:
    """
    Run SonarQube analysis on a project directory.

    Per Section 3.2, the full build environment is reconstructed for the
    parent commit of each vulnerability to enable deep inter-procedural
    taint-flow analysis.

    Returns a dict mapping file paths to lists of flagged line numbers.
    """
    sonar_props = project_dir / "sonar-project.properties"
    sonar_props.write_text(
        f"sonar.projectKey={project_key}\n"
        f"sonar.sources=.\n"
        f"sonar.java.binaries=.\n"
        f"sonar.host.url={os.environ.get('SONAR_HOST_URL', 'http://localhost:9000')}\n"
        f"sonar.token={token}\n"
    )

    subprocess.run(
        ["sonar-scanner"],
        cwd=project_dir,
        check=True,
        capture_output=True,
    )

    # Query issues via SonarQube Web API
    import urllib.request, urllib.parse
    host  = os.environ.get("SONAR_HOST_URL", "http://localhost:9000")
    url   = (
        f"{host}/api/issues/search?"
        f"componentKeys={urllib.parse.quote(project_key)}&resolved=false&ps=500"
    )
    req   = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())

    findings: dict[str, list[int]] = {}
    for issue in data.get("issues", []):
        comp      = issue.get("component", "")
        line      = issue.get("line")
        file_path = comp.split(":")[-1] if ":" in comp else comp
        if line:
            findings.setdefault(file_path, []).append(line)

    return findings


def evaluate_sonarqube(df: pd.DataFrame) -> list[dict]:
    """
    Evaluate SonarQube on the full evaluation set.

    Requires:
      - SONAR_HOST_URL env var (default: http://localhost:9000)
      - SONAR_TOKEN    env var with a valid SonarQube authentication token
      - Project directories cloned and compiled at their parent commits

    Per Section 3.2, the full build environment is reconstructed for each
    vulnerability's parent commit to enable inter-procedural taint analysis.
    """
    token    = os.environ.get("SONAR_TOKEN", "")
    if not token:
        raise EnvironmentError("Set SONAR_TOKEN environment variable to run SonarQube evaluation.")

    eval_set = get_eval_set(df)
    results  = []

    # Group by project to minimise repeated SonarQube scans
    for project_name, group in eval_set.groupby("Project Name"):
        project_dir = Path(os.environ.get("PROJECTS_ROOT", ".")) / project_name
        if not project_dir.exists():
            print(f"  [WARN] Project directory not found: {project_dir}. Skipping.")
            for idx, row in group.iterrows():
                results.append({
                    "index":        idx,
                    "ground_truth": row["label"],
                    "prediction":   "BENIGN",
                    "note":         "project directory not found",
                })
            continue

        project_key = f"svd-benchmark-{project_name.lower().replace(' ', '-')}"
        findings    = run_sonarqube(project_dir, project_key, token)

        for idx, row in group.iterrows():
            vuln_file = row.get("Vulnerable File", "")
            vuln_line = row.get("Line Number", "")
            label     = row["label"]

            # Check if SonarQube flagged the vulnerable file/line
            file_findings = findings.get(vuln_file, [])
            try:
                vline     = int(str(vuln_line).strip().split(",")[0])
                predicted = label if vline in file_findings else "BENIGN"
            except (ValueError, TypeError):
                predicted = "BENIGN"

            results.append({
                "index":        idx,
                "ground_truth": label,
                "prediction":   predicted,
                "flagged_lines": file_findings,
            })

    return results


def save_results(results: list[dict], tool: str) -> Path:
    out_dir  = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tool}_deterministic.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate SAST baselines on SVD-Benchmark.")
    parser.add_argument("--tool", choices=["semgrep", "sonarqube", "all"],
                        default="all")
    args = parser.parse_args()

    df = load_dataset(DATASET_PATH)

    if args.tool in ("semgrep", "all"):
        print("\n=== Semgrep | Deterministic (Intra-procedural) ===")
        results = evaluate_semgrep(df)
        save_results(results, "semgrep")

    if args.tool in ("sonarqube", "all"):
        print("\n=== SonarQube | Deterministic (Inter-procedural) ===")
        results = evaluate_sonarqube(df)
        save_results(results, "sonarqube")


if __name__ == "__main__":
    main()
