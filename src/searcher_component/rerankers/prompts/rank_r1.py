"""Prompt templates for Rank-R1 setwise reranking (ielab/llm-rankers)."""

import re

RANK_R1_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the "
    "answer. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think> <answer> answer here </answer>."
)

RANK_R1_USER_PROMPT = (
    'Given the query: "{query}", which of the following documents is most relevant?\n'
    "{docs}\n"
    "After completing the reasoning process, please provide only the label "
    "of the most relevant document to the query, enclosed in square brackets, "
    "within the answer tags. For example, if the third document is the most "
    "relevant, the answer should be: <think> reasoning process here </think> "
    "<answer>[3]</answer>."
)

RANK_R1_ANSWER_PATTERN = re.compile(
    r"<think>.*?</think>\s*<answer>(.*?)</answer>", re.DOTALL
)
