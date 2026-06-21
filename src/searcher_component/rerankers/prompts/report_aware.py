"""Prompt templates for the report-aware reranker.

Prompt sets:
1. Outline generation – given the main query and sub-queries, produce a
   structured JSON outline with sections.
   Output format: <think>...</think> <outline>...</outline>
2. Batch-pointwise passage scoring – given a single section and a batch of
   passages, score each passage for relevance to that section.
   Output format: <answer> [{pid, think, score}, ...] </answer>
3. Pointwise passage scoring – given a single section and a *single* passage,
   score it for relevance.
   Output format: <answer> [{"pid": "p1", "score": N}] </answer>
4. Listwise passage ranking – given a single section and a set of passages,
   produce a full ranking.
   Output format: <answer> [p3] > [p1] > [p2] > ... </answer>
5. Writer – ReAct-style report writer that retrieves evidence from a memory
   bank (built from the top-k documents) and writes the report section by
   section with inline <cite> tags.  Ranking is derived from citation order.
"""

import json
import re
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Regex patterns for output parsing
# ---------------------------------------------------------------------------

# Outline generation: <outline>...</outline>
OUTLINE_PATTERN = re.compile(
    r"<outline>(.*?)</outline>", re.DOTALL
)

# Passage assignment: <answer>...</answer>
ASSIGNMENT_ANSWER_PATTERN = re.compile(
    r"<answer>(.*?)</answer>", re.DOTALL
)


def parse_outline_sections(raw: str) -> List[Dict[str, str]]:
    """Parse an LLM outline response into a list of ``{"title": ..., "description": ...}`` dicts.

    Handles:
    - ``<outline>...</outline>`` tagged output
    - Bare JSON (with optional markdown fences)
    - Both flat list format and ``{"sections": [...]}`` dict format

    This is the single source of truth for outline parsing, shared by
    ``ReportOutlineGenerator`` (decomposition pipeline) and
    ``ReportAwareReranker`` (post-fusion reranker).
    """
    text = raw.strip() if isinstance(raw, str) else str(raw).strip()
    outline_match = OUTLINE_PATTERN.search(text)
    if outline_match:
        outline_text = outline_match.group(1).strip()
    else:
        # Fallback: strip markdown fences and try parsing the whole response
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json) and last line (```)
            end = -1 if lines[-1].strip() in ("```", "") else len(lines)
            outline_text = "\n".join(lines[1:end])
        else:
            outline_text = text

    data = json.loads(outline_text)

    # Support both flat list and dict format
    if isinstance(data, list):
        sections_raw = data
    else:
        sections_raw = data.get("sections", [])

    return [
        {
            "title": sec.get("title", ""),
            "description": sec.get("description") or sec.get("plan", ""),
        }
        for sec in sections_raw
    ]


# ---------------------------------------------------------------------------
# 1. Outline generation
# ---------------------------------------------------------------------------

REPORT_OUTLINE_SYSTEM_PROMPT = """\
You are a professional report generation expert, skilled at creating high-quality report outlines. Your task is to analyze the user's query and provide a structured article outline (top-level sections only).

# Rules
- The outline must be comprehensive, logically sound, and aligned with the user's stated preferences and requirements.
- Each section title should be highly specific and targeted — reflecting a concrete sub-aspect or follow-up inquiry about the original query, something a person wouldn't know from general knowledge.
- Consider any current events, real-time developments, or precise facts that can be used to craft better section titles.
- Sections should be complementary and together address the full scope of the query.
- Sub-queries are hints — you may merge, split, or rename them as you see fit. The outline should be organized for a reader, not for retrieval.
- The output language must match the language of the user's query.

# Output Format
<think> Brief reasoning about how to structure the report </think>
<outline>
[
  {"title": "...", "description": "..."},
  {"title": "...", "description": "..."},
  ...
]
</outline>

Output strictly in the specified format."""

REPORT_OUTLINE_USER_PROMPT = """\
# User Query

{query}

# Sub-queries (aspects already identified)

{sub_queries}

# Task

Analyze the query and sub-queries above, then generate a report outline in the <outline> tag. Each section must have a "title" and a "description"."""


# ---------------------------------------------------------------------------
# 2. Batch-pointwise passage scoring (section-centric scoring)
# ---------------------------------------------------------------------------

BATCH_POINTWISE_SYSTEM_PROMPT_WITH_THINKING = """\
You are an expert research analyst. You will be given a specific section of a report outline and a set of passages. Score each passage for its relevance to that section.

# Scoring Scale
5 - The passage is directly relevant. It directly provides key evidence or answers for the section. It matches the topics and concepts described in the section.
4 - The passage is highly relevant. It provides strong evidence closely matching one or more of the specific topics in the section.
2 - The passage is partially relevant. It helps synthesize content for the section. It matches one or more of the topics and concepts described in the section.
1 - The passage is not relevant. It might be on a similar topic, with many similar words, but it does not help this section.
0 - The passage is completely unrelated to this section.

# Output Format
<answer>
[
  {"pid": "p1", "think": "brief reasoning about relevance to the section", "score": 4},
  {"pid": "p2", "think": "brief reasoning about relevance to the section", "score": 0},
  ...
]
</answer>

Output strictly in the specified format. List every passage exactly once."""

BATCH_POINTWISE_SYSTEM_PROMPT_WITHOUT_THINKING = """\
You are an expert research analyst. You will be given a specific section of a report outline and a set of passages. Score each passage for its relevance to that section.

# Scoring Scale
5 - The passage is directly relevant. It directly provides key evidence or answers for the section. It matches the topics and concepts described in the section.
4 - The passage is highly relevant. It provides strong evidence closely matching one or more of the specific topics in the section.
2 - The passage is partially relevant. It helps synthesize content for the section. It matches one or more of the topics and concepts described in the section.
1 - The passage is not relevant. It might be on a similar topic, with many similar words, but it does not help this section.
0 - The passage is completely unrelated to this section.

# Output Format
<answer>
[
  {"pid": "p1", "score": 4},
  {"pid": "p2", "score": 0},
  ...
]
</answer>

Output strictly in the specified format. List every passage exactly once."""


def get_batch_pointwise_system_prompt(thinking: bool = True) -> str:
    """Return the passage assignment system prompt with or without per-score thinking."""
    if thinking:
        return BATCH_POINTWISE_SYSTEM_PROMPT_WITH_THINKING
    return BATCH_POINTWISE_SYSTEM_PROMPT_WITHOUT_THINKING

BATCH_POINTWISE_USER_PROMPT = """\
# Main Query

{query}

{section_context}

# Passages

{passages}

# Task

Score each passage (p1, p2, ...) for relevance to the section above. Return a JSON list in <answer> tags."""


# ---------------------------------------------------------------------------
# 3. Pointwise passage scoring (one passage at a time)
# ---------------------------------------------------------------------------

POINTWISE_SYSTEM_PROMPT_WITH_THINKING = """\
You are an expert research analyst assembling a comprehensive report. You will be given a specific section of a report outline and a single passage. Evaluate how well the passage provides evidence, data, or insights that would strengthen this section.

# Scoring Scale
5 - The passage is directly relevant. It directly provides key evidence or answers for the section. It matches the topics and concepts described in the section.
4 - The passage is highly relevant. It provides strong evidence closely matching one or more of the specific topics in the section.
2 - The passage is partially relevant. It helps synthesize content for the section. It matches one or more of the topics and concepts described in the section.
1 - The passage is not relevant. It might be on a similar topic, with many similar words, but it does not help this section.
0 - The passage is completely unrelated to this section.

# Output Format
<answer>
[
  {"pid": "p1", "think": "brief reasoning about relevance to the section", "score": 4}
]
</answer>

Output strictly in the specified format."""

POINTWISE_SYSTEM_PROMPT_WITHOUT_THINKING = """\
You are an expert research analyst assembling a comprehensive report. You will be given a specific section of a report outline and a single passage. Evaluate how well the passage provides evidence, data, or insights that would strengthen this section.

# Scoring Scale
5 - The passage is directly relevant. It directly provides key evidence or answers for the section. It matches the topics and concepts described in the section.
4 - The passage is highly relevant. It provides strong evidence closely matching one or more of the specific topics in the section.
2 - The passage is partially relevant. It helps synthesize content for the section. It matches one or more of the topics and concepts described in the section.
1 - The passage is not relevant. It might be on a similar topic, with many similar words, but it does not help this section.
0 - The passage is completely unrelated to this section.

# Output Format
<answer>
[
  {"pid": "p1", "score": 4}
]
</answer>

Output strictly in the specified format."""


def get_pointwise_system_prompt(thinking: bool = True) -> str:
    """Return the pointwise system prompt with or without per-score thinking."""
    if thinking:
        return POINTWISE_SYSTEM_PROMPT_WITH_THINKING
    return POINTWISE_SYSTEM_PROMPT_WITHOUT_THINKING


POINTWISE_USER_PROMPT = """\
# Main Query

{query}

{section_context}

# Passage

{passage}

# Task

Score the passage above for relevance to the report section. Return a JSON list in <answer> tags."""


# ---------------------------------------------------------------------------
# 4. Listwise passage ranking (full ordering per section)
# ---------------------------------------------------------------------------

LISTWISE_SYSTEM_PROMPT_WITH_THINKING = """\
You are an expert research analyst assembling a comprehensive report. You will be given a specific section of a report outline and a set of passages, each indicated by a numerical identifier []. Rank the passages based on how well they provide evidence, data, or insights that would strengthen this section of the report.

First, think step-by-step inside <think> tags about which passages are most relevant to the section, then produce your ranking in <answer> tags."""

LISTWISE_SYSTEM_PROMPT_WITHOUT_THINKING = """\
You are an expert research analyst assembling a comprehensive report. You will be given a specific section of a report outline and a set of passages, each indicated by a numerical identifier []. Rank the passages based on how well they provide evidence, data, or insights that would strengthen this section of the report."""

# Keep backward-compatible alias
LISTWISE_SYSTEM_PROMPT = LISTWISE_SYSTEM_PROMPT_WITHOUT_THINKING


def get_listwise_system_prompt(thinking: bool = True) -> str:
    """Return the listwise system prompt with or without thinking."""
    if thinking:
        return LISTWISE_SYSTEM_PROMPT_WITH_THINKING
    return LISTWISE_SYSTEM_PROMPT_WITHOUT_THINKING


LISTWISE_USER_PROMPT_WITH_THINKING = """\
# Main Query

{query}

{section_context}

# Passages

{passages}

# Task

Rank the {num} passages above based on their relevance to the report section. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [p2] > [p1].

<think> brief reasoning about relevance to the section </think>
<answer> ranking here </answer>"""

LISTWISE_USER_PROMPT_WITHOUT_THINKING = """\
# Main Query

{query}

{section_context}

# Passages

{passages}

# Task

Rank the {num} passages above based on their relevance to the report section. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [p2] > [p1]. Output your ranking in <answer> tags. Do not explain.

<answer> ranking here </answer>"""

# Keep backward-compatible alias
LISTWISE_USER_PROMPT = LISTWISE_USER_PROMPT_WITHOUT_THINKING


def get_listwise_prompts(thinking: bool = True) -> tuple:
    """Return (system_prompt, user_prompt_template) for listwise ranking."""
    if thinking:
        return LISTWISE_SYSTEM_PROMPT_WITH_THINKING, LISTWISE_USER_PROMPT_WITH_THINKING
    return LISTWISE_SYSTEM_PROMPT_WITHOUT_THINKING, LISTWISE_USER_PROMPT_WITHOUT_THINKING


# ---------------------------------------------------------------------------
# Section context builder (with / without full outline)
# ---------------------------------------------------------------------------

def build_section_context(
    section_title: str,
    section_description: str,
    pass_whole_outline: bool = False,
    outline_section_titles: Optional[List[str]] = None,
) -> str:
    """Build the section context block inserted into scoring prompts.

    Parameters
    ----------
    section_title : str
        Title of the targeted section.
    section_description : str
        Description of the targeted section.
    pass_whole_outline : bool
        When True, prepend the full report outline (titles only) before
        the targeted section so the LLM sees the broader report structure.
    outline_section_titles : list of str, optional
        Ordered list of *all* section titles in the outline.  Required
        when ``pass_whole_outline`` is True.

    Returns
    -------
    str
        Ready-to-insert prompt block.  Two variants:

        **Without outline** (default)::

            # Report Section

            Title: ...
            Description: ...

        **With outline**::

            # Report Outline

            Section 1: ...
            Section 2: ...
            ...

            # Targeted Section

            Title: ...
            Description: ...
    """
    if pass_whole_outline and outline_section_titles:
        outline_lines = [
            f"Section {i}: {title}"
            for i, title in enumerate(outline_section_titles, 1)
        ]
        outline_text = "\n".join(outline_lines)
        return (
            f"# Report Outline\n\n{outline_text}\n\n"
            f"# Targeted Section\n\n"
            f"Title: {section_title}\n"
            f"Description: {section_description}"
        )
    return (
        f"# Report Section\n\n"
        f"Title: {section_title}\n"
        f"Description: {section_description}"
    )


# Regex for parsing listwise ranking output: extract [pN] identifiers
LISTWISE_RANKING_PATTERN = re.compile(r"\[p(\d+)\]")


# ---------------------------------------------------------------------------
# 5. Writer (ReAct-style report generation with citation-based ranking)
# ---------------------------------------------------------------------------

WRITER_SYSTEM_PROMPT = """\
You are a report writer. You receive a citation-grounded outline and a memory bank of documents. You must write the full report section by section, using only the evidence you retrieve for each part.

## Workflow
You operate in a loop. In each step you must:
1. **Think** (inside <think>...</think> tags): identify the section to write and which document IDs it needs.
2. **Act** with exactly one of:
   - **retrieve**: pull evidence for the current section from the memory bank.
     <tool_call>
     {"name": "retrieve", "arguments": {"retrieve_id": ["id_1", "id_2", ...], "goal": "Why you need these for this section"}}
     </tool_call>
   - **write**: after receiving evidence, write the section content.
     <write>
     ... section prose with inline citations <cite id="id_1">...</cite> ...
     </write>
   - **terminate**: when all sections are done, output <terminate> and nothing else.

## Rules
- Write one section (or a few closely related subsections) per write block.
- Cite sources inline as <cite id="id_X">...</cite> where id_X matches the memory bank.
- Do not invent facts; use only retrieved evidence.
- Preserve narrative flow and coherence with previously written sections.
- You MUST cite relevant documents. The citations determine how documents are ranked."""

WRITER_USER_TEMPLATE = """\
## Research question
{question}

## Outline (with section titles and descriptions)
{outline}

## Memory bank (document IDs and summaries; full text retrieved on demand)
{memory_bank_summary}

## Already written report (so far)
{report_so_far}{last_observation_block}

## Your task
Continue from the outline. Output <think>...</think> then either retrieve (for the next section) or write (after you have evidence), or <terminate> if the report is complete."""

# Regex patterns for writer output parsing
WRITER_CITE_PATTERN = re.compile(r'<cite\s+id="([^"]+)"')
WRITER_WRITE_PATTERN = re.compile(r"<write>\s*(.*?)\s*</write>", re.DOTALL)
WRITER_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)
WRITER_THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
WRITER_TERMINATE_PATTERN = re.compile(r"<terminate>")


# ---------------------------------------------------------------------------
# 6. Iterative writer – expand outline + append content across doc batches
# ---------------------------------------------------------------------------

# -- 6a. Expand outline (decide how to grow the outline given new evidence) --

EXPAND_OUTLINE_SYSTEM_PROMPT = """\
You are a professional report-generation expert skilled at crafting high-quality report outlines.
Given the user's query, the current report outline (with content written so far), and a new batch of retrieved passages, you must determine whether and how to expand the outline.

## Important Notes
1. Select the single most impactful expansion action. You may add a new top-level section OR expand an existing section/subsection with child subsections.
2. If no expansion is needed (the current outline already covers everything the new passages offer), output a "nop" action.
3. Expand only when the new passages contain substantial information that is NOT already covered by existing sections.
4. New sections/subsections must NOT be redundant with existing ones.
5. New sections/subsections must be relevant and coherent with the rest of the outline.
6. The output language must match the language of the user's query.

## Available Actions
- **add-section**: Add a new top-level section to the outline.
  Format: `{"name": "add-section", "section": {"title": "...", "description": "..."}}`
- **extend-plan**: Expand an existing section by adding subsections (e.g., section-1 → section-1.1, section-1.2).
  Format: `{"name": "extend-plan", "position": "section-X" or "section-X.Y", "subsections": [{"title": "...", "description": "..."}, ...]}`
- **nop**: No operation needed.
  Format: `{"name": "nop"}`

## Output Format
<think> Provide detailed reasoning about whether expansion is needed and why </think>
<action> Action (in JSON format) </action>

Output strictly in the specified format."""

EXPAND_OUTLINE_USER_PROMPT = """\
** User Query **
{query}

** Current Report Outline **
{current_outline}

** New Retrieved Passages **
{new_passages}

** Task **
Analyze the new passages and decide whether to expand the outline. Output your decision in the specified format."""

# Regex for parsing expand-outline actions
EXPAND_OUTLINE_ACTION_PATTERN = re.compile(
    r"<action>\s*(.*?)\s*</action>", re.DOTALL
)

# -- 6b. Append to section (write NEW content for a section using new passages) --

ITER_APPEND_SECTION_SYSTEM_PROMPT = """\
You are an expert report writer. You are iteratively building a report by processing batches of retrieved documents.

You will be given:
- The user's query
- The current report (with content already written for some sections)
- An instruction indicating which section to append content to
- A NEW batch of retrieved passages (not seen before)

Your task is to write NEW content for the specified section using ONLY the new passages. Do NOT repeat or paraphrase content already written — only add genuinely new information, evidence, or perspectives from the new passages.

## Citation Rules
1. Every factual statement must end with citation(s) using the corresponding document ID(s).
   Example: "Quantum entanglement is a physical phenomenon[1]."
2. Citation format must strictly be `[id]` (square brackets only).
3. If multiple documents support a single statement, append citations consecutively, e.g. `[1][3][5]`.
4. The content should be in the same language as the user's query.

## Important
- If the new passages contain NO new relevant information for this section, output an empty action: `<action></action>`
- Focus on adding complementary information, not restating what is already written.

PLEASE JUST OUTPUT THE NEW CONTENT, DO NOT OUTPUT THE SECTION TITLE."""

ITER_APPEND_SECTION_USER_PROMPT = """\
** User Query **
{user_query}

** Current Article Summary **
{current_survey}

** Instruction **
{current_instruction}

** Previously Written Content for This Section **
{existing_content}

** New Retrieved Passages **
{new_passages}

## Action Example:
<action> new content with citations like [1][2] </action>

** Output Format **
<thought> Your thought process — what new information do these passages add? </thought>
<action> Your New Content (in Markdown format) (include [id] citations within the content) </action>

Please strictly follow the specified output format."""


# ---------------------------------------------------------------------------
# 6c. Cite-only batch scoring (lightweight alternative to per-section append)
#
# Instead of generating prose for each section × batch, this prompt asks the
# LLM to map passage IDs to sections in a single call per batch.  This
# drastically reduces the number of LLM calls and output tokens for
# iterations 2+, while still populating the citation score matrix.
# ---------------------------------------------------------------------------

CITE_ONLY_BATCH_SYSTEM_PROMPT = """\
You are an expert document analyst. You are given a report outline (with content already written) and a NEW batch of retrieved passages.

Your task is to identify which passages are relevant to which sections of the report. You do NOT need to write any prose — just map passage IDs to sections.

## Rules
1. For each section, list the passage IDs (integers) that contain relevant evidence for that section.
2. A passage can be relevant to multiple sections.
3. If a passage is not relevant to any section, omit it.
4. If no passages are relevant to a section, use an empty list.
5. Output ONLY the JSON mapping — no explanation, no prose.

## Output Format
<answer>
{"section-1": [3, 7, 12], "section-2": [5, 7], "section-3": [], ...}
</answer>

Use the exact section keys from the outline (e.g., "section-1", "section-1.1", "section-2")."""

CITE_ONLY_BATCH_USER_PROMPT = """\
** User Query **
{user_query}

** Current Report Outline **
{current_outline}

** New Retrieved Passages **
{new_passages}

** Task **
For each section in the outline, list the passage IDs that are relevant. Output strictly in the specified JSON format."""
