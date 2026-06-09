"""Plan update prompt for ReAct with plan agent."""

REACT_PLAN_UPDATE_PROMPT = """You are updating a retrieval plan based on recent search results.

User Query: {query}

Current Plan:
{current_plan}

Last Search Action: {last_search_query}

Search Results Summary:
{search_summary}

Your task is to update the plan by:
1. Determining which task(s) the search results addressed
2. Marking those tasks as completed with ✓
3. Adding a summary of findings under each completed task (indented with →)
4. Updating the progress bar to reflect completed tasks

Format your response EXACTLY as follows:

✓ Completed task [task number(s)]

Progress: [█░░░] X/N complete

✓ [Task number]. [Task description from original plan]
   → [1-2 sentence summary of what was found in the search results]

□ [Task number]. [Remaining task description - unchanged]

□ [Task number]. [Remaining task description - unchanged]

Guidelines:
- Use ✓ for completed tasks and □ for remaining tasks
- Preserve ALL previously checked (✓) tasks with their summaries - never remove or lose completed items
- Add "→" indented summaries ONLY for newly completed tasks from the current search
- Progress bar: Use █ for completed tasks and ░ for remaining (e.g., [█░░░] means 1 of 4 done)
- Update progress fraction (X/N) where X is completed count and N is total
- Start with "✓ Completed task [number(s)]" line
- Keep task descriptions unchanged unless refinement is necessary
- DO NOT add new tasks unless absolutely critical
- Maintain all blank lines between sections for readability

Updated Plan:"""
