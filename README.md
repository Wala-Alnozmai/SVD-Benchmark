# SVD-Benchmark: A CWE-Aware Benchmark and Comparative Evaluation of LLMs for Java Software Vulnerability Detection

[![Dataset](https://img.shields.io/badge/Dataset-5054%20samples-blue)](https://github.com/Wala-Alnozmai/SVD-Benchmark)
[![CWE Coverage](https://img.shields.io/badge/CWE%20Types-37-green)](https://cwe.mitre.org/)
[![Paper](https://img.shields.io/badge/Paper-Software%20Quality%20Journal-red)](https://github.com/Wala-Alnozmai/SVD-Benchmark)

This repository contains the **SVD-Benchmark** dataset and evaluation framework from our paper:

**"A CWE-Aware Benchmark and Comparative Evaluation of LLMs for Java Software Vulnerability Detection"**

*Wala Alnozami, Obieda Ananbeh, Dae-Kyoo Kim*
*School of Engineering and Computer Science, Oakland University*

> Submitted to **Software Quality Journal** — Special Issue: *Software Quality in an AI-Driven World*

---

## Dataset Overview

The SVD-Benchmark provides the first publicly reproducible, CWE-labeled benchmark for Java software vulnerability detection (SVD), grounded in confirmed CVEs from real-world open-source projects.

### Statistics

| Attribute | Value |
|-----------|-------|
| Total samples | 5,054 (balanced) |
| Vulnerable instances | 2,527 |
| Benign instances | 2,527 |
| Java projects | 20 |
| CWE categories | 37 |
| Annotator agreement | Cohen's κ = 0.81 |
| Third-reviewer accuracy | 97% (194/200 holdout) |

### Benchmark Creation Pipeline

1. **Project selection** — 20 Java projects with confirmed CVEs from the GitHub Security Advisory Database
2. **Dual-tool SAST labeling** — each snippet scanned with both CodeQL and Snyk; labeled vulnerable only when **both** flag it (intersection strategy to minimize false positives)
3. **Manual review** — discrepant cases (one tool flags, the other does not) reviewed by two independent domain experts; Cohen's κ = 0.81
4. **AST-based deduplication** — hash-based dedup on normalized AST representations; 1,675 duplicates removed from 6,729 raw → 5,054 unique
5. **Comment stripping** — non-semantic comments and formatting artifacts removed to prevent leakage of superficial signals
6. **Project-level stratified downsampling** — benign class downsampled per project to achieve a 1:1 ratio using `sklearn.utils.resample` (see `src/build_dataset.py`)

### CSV Columns

| Column | Description |
|--------|-------------|
| `CWE ID` | CWE identifier (e.g., `CWE-798`) or `BENIGN` |
| `Project Name` | Source Java project |
| `Vulnerable File` | Relative file path within the project |
| `Programming Language` | Always `Java` |
| `Line Number` | Starting line of the snippet |
| `Code Snippet` | Full method-level code snippet |
| `Exact Vulnerable Line` | The specific line(s) flagged as vulnerable |
| `Description` | Vulnerability description from SAST output |
| `Status` | `vulnerable` or `benign` |
| `test` | `yes` = RAG retrieval pool; `no` = training/evaluation set |

---

## Project Coverage

| Domain | Projects |
|--------|----------|
| Web Frameworks | Apache CXF, Undertow, Vert.x |
| Big Data | Apache Flink, Apache Hadoop, Apache Hive |
| Developer Tools | OpenRefine, FitNesse |
| Identity & Access | Keycloak |
| Databases | HyperSQL, OpenTSDB, Infinispan |
| Communication | OpenMeetings, Wire-AVS |
| Graph | HugeGraph-Toolchain |
| Messaging | Apache Dubbo, Apache James, Apache NiFi |
| Network | Netty |

### Detailed Project Table

| **Project** | **CVE-ID** | **Description** | **Affected Versions** |
|-------------|------------|-----------------|----------------------|
| eclipse-vertx | CVE-2019-17640 | Vert.x core: event-driven, non-blocking HTTP/TCP framework | >= 3.0.0, < 3.9.4 |
| Apache Flink | CVE-2020-17518 | Real-time stream and batch data processing | >= 1.5.1, < 1.11.3 |
| OpenTSDB | CVE-2020-35476 | Distributed time series database on HBase | <= 2.4.0 |
| Apache Hadoop | CVE-2022-25168 | Distributed data processing with MapReduce/HDFS | >= 2.0.0 < 2.10.2; >= 3.0.0-alpha < 3.2.4; >= 3.3.0 < 3.3.3 |
| Netty | CVE-2022-41915 | Asynchronous event-driven network framework | >= 4.1.83.Final, < 4.1.86.Final |
| Undertow | CVE-2018-1067 | Java web server with blocking/non-blocking APIs | <= 7.1.1.GA |
| wire-avs | CVE-2021-41193 | Secure messaging platform (WebRTC AVS) | <= 7.1.1.GA |
| Apache Dubbo | CVE-2021-36161 | High-performance Java RPC framework | < 7.1.12 |
| Apache NiFi | CVE-2018-17195 | Data flow automation platform | < 2.7.13 |
| James | CVE-2022-45935 | Modular Java mail server | >= 1.0.0, <= 1.7.1 |
| Infinispan | CVE-2019-10174 | In-memory key/value data store and cache | <= 8.2.11.Final; >= 9.0.0.Final <= 9.4.16.Final |
| HyperSQL | CVE-2022-41853 | In-memory/disk-based Java RDBMS | < 2.7.1 |
| Apache Hive | CVE-2022-41137 | SQL querying for Hadoop data warehousing | 4.0.0-alpha-1 |
| Openmeetings | CVE-2024-54676 | Video conferencing and collaboration platform | >= 2.1.0, < 8.0.0 |
| Eclipse GlassFish | CVE-2024-9329 | Jakarta EE-compliant application server | >= 2.1.0, < 8.0.0 |
| Keycloak | CVE-2024-8883 | Identity and access management (SSO/OAuth2/SAML) | < 7.0.17 |
| Hugegraph-toolchain | CVE-2024-27347 | HugeGraph utilities for graph data management | <= 22.0.12; >= 23.0.0 <= 24.0.7; >= 25.0.0 <= 25.0.5 |
| Apache CXF | CVE-2024-28752 | SOAP and RESTful web services framework | >= 1.0.0, < 1.3.0 |
| FitNesse | CVE-2024-39610 | Web-based acceptance testing tool | < 3.5.8; >= 3.6.0 < 3.6.3; >= 4.0.0 < 4.0.4 |
| OpenRefine | CVE-2024-47882 | Java tool for cleaning and transforming messy data | < 3.8.0 |

### CWE Distribution

| **CWE-ID** | **Count** | **CWE-ID** | **Count** | **CWE-ID** | **Count** | **CWE-ID** | **Count** |
|------------|-----------|------------|-----------|------------|-----------|------------|-----------|
| CWE-798 | 632 | CWE-400 | 58 | CWE-942 | 18 | CWE-280 | 3 |
| CWE-23 | 280 | CWE-1004 | 52 | CWE-208 | 18 | CWE-330 | 2 |
| CWE-319 | 216 | CWE-547 | 52 | CWE-326 | 14 | CWE-613 | 1 |
| CWE-918 | 201 | CWE-78 | 49 | CWE-502 | 11 | CWE-200 | 1 |
| CWE-79 | 168 | CWE-295 | 48 | CWE-134 | 8 | CWE-297 | 1 |
| CWE-113 | 149 | CWE-614 | 46 | CWE-327 | 8 | CWE-114 | 1 |
| CWE-611 | 121 | CWE-352 | 31 | CWE-90 | 8 | CWE-347 | 1 |
| CWE-470 | 93 | CWE-209 | 29 | CWE-501 | 7 | | |
| CWE-601 | 75 | CWE-89 | 29 | CWE-256 | 3 | | |
| CWE-916 | 68 | CWE-732 | 22 | CWE-074 | 3 | | |

---

## Repository Structure

```
SVD-Benchmark/
├── SVD-Benchmark.csv          # Full benchmark dataset (5,054 instances)
├── requirements.txt           # Python dependencies
└── src/
    ├── config.py              # Seeds, model names, shared constants
    ├── data_utils.py          # Dataset loading and split helpers
    ├── prompts.py             # One-Shot and Few-Shot prompt templates
    ├── rag.py                 # SimCSE-based RAG retrieval
    ├── evaluate_llm.py        # Proprietary LLM inference (ChatGPT/Claude/Gemini)
    ├── evaluate_slm.py        # SLM inference via Ollama + RAG
    ├── evaluate_sast.py       # Semgrep and SonarQube baseline evaluation
    ├── metrics.py             # Precision/Recall/F1/Accuracy/Detection rates
    └── statistical_validation.py  # Bootstrap CI, McNemar's test, Odds ratio
```

---

## Setup

```bash
pip install -r requirements.txt
```

Set API keys in environment variables:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

SLMs are served locally via [Ollama](https://ollama.com). Pull the required models:

```bash
ollama pull codegemma:2b
ollama pull qwen2.5-coder:1.5b
ollama pull llama3.2:1b
```

---

## Usage

### Evaluate Proprietary LLMs

```bash
# One-Shot
python src/evaluate_llm.py --model chatgpt --strategy one_shot

# Few-Shot ICL (k=3, seed=42 fixed)
python src/evaluate_llm.py --model claude --strategy few_shot

# All models, all strategies
python src/evaluate_llm.py --all
```

### Evaluate Open-Source SLMs with RAG

```bash
python src/evaluate_slm.py --model codegemma --strategy rag
python src/evaluate_slm.py --all
```

### Evaluate SAST Baselines

```bash
python src/evaluate_sast.py --tool semgrep
python src/evaluate_sast.py --tool sonarqube
```

### Compute Statistical Validation

```bash
python src/statistical_validation.py \
    --predictions_a results/chatgpt_few_shot.json \
    --predictions_b results/codegemma_rag.json \
    --ground_truth SVD-Benchmark.csv
```

---

## Evaluation Methodology

### Prompting Strategies

| Strategy | Models | Description |
|----------|--------|-------------|
| One-Shot | All LLMs | Instruction + target code; no examples |
| Few-Shot ICL | All LLMs | k=3 fixed examples (seed=42) + target code |
| RAG | SLMs only | Top-k SimCSE-retrieved exemplars replace random examples |

**Output format**: models must return exactly `BENIGN` or `CWE-{ID}`. Any response not matching `^(BENIGN\|CWE-\d{2,4})$` is collapsed to `BENIGN`.

### Statistical Validation

- **Bootstrap CIs**: B=5,000 resamples with replacement; 2.5th/97.5th percentiles → 95% CI
- **McNemar's test**: continuity-corrected; χ² = (|n₁₀ − n₀₁| − 1)² / (n₁₀ + n₀₁)
- **Odds ratio**: n₁₀ / n₀₁ from McNemar discordant counts

---

## Citation

```bibtex
@article{alnozami2026cweaware,
  title={A {CWE}-Aware Benchmark and Comparative Evaluation of {LLMs} for {Java} Software Vulnerability Detection},
  author={Alnozami, Wala and Ananbeh, Obieda and Kim, Dae-Kyoo},
  journal={},
  year={2026},
  publisher={}
}
```

---

**Note**: This benchmark is intended for research purposes. Vulnerability assessments should be independently verified before application to production systems.
