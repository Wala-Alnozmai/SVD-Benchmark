"""
prompts.py — One-Shot and Few-Shot prompt templates.

Implements exactly the templates shown in Figure 2 of the paper:
  "A CWE-Aware Benchmark and Comparative Evaluation of LLMs for Java
   Software Vulnerability Detection"

Output contract (Section 3.3):
  - Return exactly  BENIGN          if the code is NOT vulnerable.
  - Return exactly  CWE-{ID}        if the code IS vulnerable.
  - No explanations, comments, or extra text.
  - Any response not matching ^(BENIGN|CWE-\\d{2,4})$ is collapsed to BENIGN.
"""

# ---------------------------------------------------------------------------
# Shared instruction block (identical for One-Shot and Few-Shot)
# ---------------------------------------------------------------------------
INSTRUCTION = (
    "Analyze the following Java code snippet and determine if it is vulnerable.\n"
    "If it is NOT vulnerable, return exactly: BENIGN.\n"
    "If it IS vulnerable, return exactly the CWE identifier in the form: CWE-{ID}.\n"
    "Do NOT include any explanations, comments, or extra text."
)

# ---------------------------------------------------------------------------
# Fixed few-shot examples from the paper (Figure 2, Section 3.3)
# These are the two canonical examples shown in the paper.
# ---------------------------------------------------------------------------
PAPER_EXAMPLES = [
    {
        "code": (
            "public String greetUser(String name) {\n"
            "    if (name == null) {\n"
            "        return \"Hello, Guest!\";\n"
            "    }\n"
            "    return \"Hello, \" + name;\n"
            "}"
        ),
        "label": "BENIGN",
    },
    {
        "code": (
            "public ResultSet getUser(Connection conn, String username)\n"
            "        throws SQLException {\n"
            "    Statement stmt = conn.createStatement();\n"
            "    String query = \"SELECT * FROM users WHERE name = '\"\n"
            "            + username + \"'\";\n"
            "    return stmt.executeQuery(query);\n"
            "}"
        ),
        "label": "CWE-89",
    },
]


def build_one_shot_prompt(code_snippet: str) -> str:
    """
    Configuration A — One-Shot (Figure 2).

    Provides the instruction and target code only; no examples.
    Serves as the primary control to measure intrinsic model knowledge.
    """
    return (
        f"{INSTRUCTION}\n\n"
        f"Code:\n{code_snippet}"
    )


def build_few_shot_prompt(code_snippet: str, examples: list[dict]) -> str:
    """
    Configuration B — Few-Shot ICL (Figure 2).

    Prepends k=3 examples before the target snippet.  Examples are
    passed in as a list of dicts with keys 'code' and 'label'.

    The examples argument should always be the same set (sampled once
    with seed=42 from data_utils.sample_few_shot_examples) so that all
    models receive identical context.
    """
    parts = [INSTRUCTION, ""]

    for i, ex in enumerate(examples, start=1):
        parts.append(f"Example {i}:")
        parts.append(f"Input:\n{ex['code']}")
        parts.append(f"Output: {ex['label']}")
        parts.append("")

    parts.append(f"Code:\n{code_snippet}")
    return "\n".join(parts)


def build_rag_prompt(code_snippet: str, retrieved_examples: list[dict]) -> str:
    """
    RAG prompt — same structure as Few-Shot but with dynamically retrieved
    exemplars replacing the random k=3 examples (Section 3.4).

    retrieved_examples: list of dicts with keys 'code' and 'label',
    returned by rag.retrieve().
    """
    return build_few_shot_prompt(code_snippet, retrieved_examples)
