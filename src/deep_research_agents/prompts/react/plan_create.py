"""Plan creation prompt for ReAct with plan agent."""

REACT_PLAN_CREATE_PROMPT = """You are helping to create a retrieval plan for gathering documents to answer a user query.

User Query: {query}

Your task is to create a structured research discovery plan with numbered tasks for the key aspects that need to be researched.

Format your response EXACTLY as follows:

Research Discovery for: [restate the user query]

Progress: [░░░░] 0/N complete

□ 1. [First research task description - be specific about what documents/information to search for]

□ 2. [Second research task description - be specific about what documents/information to search for]

□ 3. [Third research task description - be specific about what documents/information to search for]

□ 4. [Fourth research task description - be specific about what documents/information to search for]

Guidelines:
- Create 3-5 specific research tasks (N tasks total)
- Each task should describe what specific type of documents or information to retrieve
- Tasks should cover different aspects or dimensions of the query
- Use □ (unchecked) for all tasks initially
- Progress bar should show 0/N where N is the total number of tasks
- Progress bar uses ░ characters (all empty initially)
- Keep task descriptions detailed but concise (1-2 sentences each)

Retrieval Plan:"""
