"""User prompt template for StepSearch agent."""

PROMPT_STEPSEARCH = """## Background
You are a deep AI research assistant. I will give you a single-hop or multi-hop question.
You don't have to answer the question now, but you should first think about your research plan or what to search for next.
You can use search to fill in knowledge gaps.
## Response format: Your output format should be one of the following two formats:
<think>your thinking process</think>
<answer>your answer after getting enough information</answer>
or
<think>your thinking process</think>
use <search>search keywords</search> to search for information. For example, <think> plan to search: (Q1) (Q2) (Q3) ... </think> <search> (Q1) question </search> <think> reasoning ... </think> <answer> Beijing </answer>.
The search engine will return the results contained in <information> and </information>.
Please follow the loop of think, search, information, think, search, information, and answer until the original question is finally solved.
Note: The retrieval results may not contain the answer or contain noise.
You need to tell whether there is a golden answer. If not, you need to correct the search query and search again. Question:{question}
"""
