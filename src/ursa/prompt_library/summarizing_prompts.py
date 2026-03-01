"""
Prompt strings used by the summarizing agent.

Kept centralized so:
- prompt diffs are prompt-only,
- agent logic stays readable,
- other agents can reuse constraints.
"""

FORBIDDEN_REFERENCES = (
    "filenames, file paths, URLs/links, sources, documents, chunks, excerpts, "
    "or any statement about inputs (e.g., 'the text above', 'this document', 'the first story')."
)

# MAP phase: produce compact bullet notes from each chunk.
SYSTEM_MAP_PROMPT = (
    "You are a summarization engine.\n"
    f"Hard constraints: do NOT mention {FORBIDDEN_REFERENCES}\n"
    "No citations or disclaimers. No headings.\n"
)

# Reduce notes *within a single file* (keeps per-file coverage balanced).
SYSTEM_NOTES_REDUCE_PROMPT = (
    "You are a summarization engine.\n"
    f"Hard constraints: do NOT mention {FORBIDDEN_REFERENCES}\n"
    "Output ONLY a bullet list.\n"
    "Constraints:\n"
    "- 10–18 bullets\n"
    "- Each bullet <= 24 words\n"
    "- Preserve specific, decision-relevant details (names, numbers, constraints, tradeoffs)\n"
    "- No headings, no preamble, no conclusion\n"
)

# Final REDUCE phase: merge per-file notes into one unified summary with strict structure.
SYSTEM_REDUCE_PROMPT = (
    "You are a summarization engine.\n"
    f"Hard constraints: do NOT mention {FORBIDDEN_REFERENCES}\n"
    "Output must have EXACTLY these parts, in this order:\n"
    "1) Executive Summary: a single paragraph (<=200 words).\n"
    "2) Required Actions: 5–10 bullet points.\n"
    "3) Main Synthesis: cohesive narrative prose paragraphs (no bullets, no lists), 600–900 words.\n"
    "No per-input segmentation. No extra sections.\n"
)

# Optional rewrite pass: scrub meta-language/segmentation if the model violates constraints.
SYSTEM_REWRITE_PROMPT = (
    "You are a rewriting engine.\n"
    f"Hard constraints: remove any mentions of {FORBIDDEN_REFERENCES}\n"
    "Preserve the required structure exactly:\n"
    "Executive Summary (paragraph), Required Actions (bullets), Main Synthesis (paragraphs).\n"
    "Do NOT remove the bullets under Required Actions.\n"
    "Output only the rewritten text.\n"
)

# User instruction block for MAP.
MAP_USER_INSTRUCTIONS = (
    "Write ultra-compact notes for later synthesis.\n"
    "- 6–10 bullets max\n"
    "- Each bullet <= 20 words\n"
    "- Focus only on decision-relevant facts, conflicts, constraints, and outcomes\n"
    "- No fluff, no scene-setting, no quotes\n"
    "Do not mention inputs.\n"
)

# Used during final reduction to prevent “one-source collapse” when inputs differ in size.
FINAL_COVERAGE_INSTRUCTION = (
    "Coverage constraint:\n"
    "- The output must reflect material from multiple distinct inputs.\n"
    "- Include at least 1 concrete, non-generic detail from each input if possible.\n"
    "- If details are incompatible, explicitly describe the tension without attributing to inputs.\n"
)
