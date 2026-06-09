"""WebWeaver user prompt templates for planner and writer (paper: WebWeaver ICLR 2026).

System prompts live in the companion .txt files:
  planner_system.txt
  writer_system.txt
"""

PLANNER_USER_TEMPLATE = """\
## Open-ended research question
{question}

## Memory bank (evidence so far)
{memory_bank}

## Previous steps (thought, action, observation)
{history}

What is your next step? Output <think>...</think> then exactly one action: search, write outline, or <terminate>.\
"""

WRITER_USER_TEMPLATE = """\
## Research question
{question}

## Outline (with citation IDs)
{outline}

## Memory bank (IDs and summaries; full evidence retrieved on demand)
{memory_bank_summary}

## Already written report (so far)
{report_so_far}

## Previous writer steps
{writer_history}{last_observation_block}

## Your task
Continue writing the report from where you left off. Look at "Already written report" to see what sections are done. Do NOT re-retrieve or re-write sections that already appear in the report.
Output <think>...</think> then either retrieve (for the next unwritten section), write (after you have evidence), or <terminate> if the report is complete.\
"""
