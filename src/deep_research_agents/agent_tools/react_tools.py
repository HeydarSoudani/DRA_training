"""Tools for ReAct retrieval model."""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_retrieval_research.llm_utils.litellm_client import LiteLLMClient
from prompts.react.plan_create import REACT_PLAN_CREATE_PROMPT
from prompts.react.plan_update import REACT_PLAN_UPDATE_PROMPT


class PlanTool:
    """Tool for managing retrieval plans using LLM-generated structured plans."""

    def __init__(self, llm_client: LiteLLMClient):
        """Initialize the PlanTool.

        Args:
            llm_client: LiteLLM client for generating plans
        """
        self.llm_client = llm_client
        self.current_plan = None
        self.plan_history = []
        self.query = None

    def execute(self, mode: str, query: str, last_search_query: Optional[str] = None, search_results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Execute a plan action using LLM to generate structured plans.

        Args:
            mode: Plan mode - 'create' or 'update'
            query: The original user query
            last_search_query: The last search query (for update mode)
            search_results: The search results (for update mode)

        Returns:
            Dictionary containing:
                - success: bool indicating if operation was successful
                - observation: str the generated plan with checkmarks
                - plan: str the current plan
        """
        mode = mode.lower()
        self.query = query

        if mode == 'create':
            return self._create_plan(query)

        elif mode == 'update':
            if self.current_plan is None:
                return {
                    'success': False,
                    'observation': 'Cannot update plan. No plan has been created yet. Use Plan[create] first.',
                    'plan': None
                }
            return self._update_plan(query, last_search_query, search_results)

        else:
            return {
                'success': False,
                'observation': f"Invalid plan mode: '{mode}'. Use 'create' or 'update'.",
                'plan': self.current_plan
            }

    def _create_plan(self, query: str) -> Dict[str, Any]:
        """Create a new plan using LLM.

        Args:
            query: The user query

        Returns:
            Dictionary with success, observation, and plan
        """
        # Format the prompt
        prompt = REACT_PLAN_CREATE_PROMPT.format(query=query)

        # Generate plan using LLM
        messages = [
            {"role": "system", "content": "You are a helpful research planning assistant."},
            {"role": "user", "content": prompt}
        ]

        plan = self.llm_client.complete(messages, temperature=0.7, max_tokens=300)

        # Store the plan
        self.current_plan = plan.strip()
        self.plan_history.append({
            'mode': 'create',
            'plan': self.current_plan,
            'timestamp': len(self.plan_history)
        })

        # Return the plan as observation
        return {
            'success': True,
            'observation': self.current_plan,
            'plan': self.current_plan
        }

    def _update_plan(self, query: str, last_search_query: Optional[str], search_results: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Update the plan based on search results using LLM.

        Args:
            query: The original user query
            last_search_query: The last search query executed
            search_results: The documents retrieved

        Returns:
            Dictionary with success, observation, and plan
        """
        # Create a summary of search results with improved formatting
        if search_results and len(search_results) > 0:
            # Summarize first few results with title and text
            summary_parts = []
            for idx, doc in enumerate(search_results[:3], 1):
                metadata = doc.get('metadata', {})
                title = metadata.get('title', 'Untitled Document')
                text = doc.get('relevant_text', '')[:150]  # First 150 chars
                summary_parts.append(f"[{idx}] {title}\n{text}...")
            search_summary = "\n\n".join(summary_parts)
            search_summary += f"\n\n(Total: {len(search_results)} documents retrieved)"
        else:
            search_summary = "No documents were retrieved."

        # Format the prompt
        prompt = REACT_PLAN_UPDATE_PROMPT.format(
            query=query,
            current_plan=self.current_plan,
            last_search_query=last_search_query or "N/A",
            search_summary=search_summary
        )

        # Generate updated plan using LLM
        messages = [
            {"role": "system", "content": "You are a helpful research planning assistant."},
            {"role": "user", "content": prompt}
        ]

        updated_plan = self.llm_client.complete(messages, temperature=0.7, max_tokens=400)

        # Store the updated plan
        self.current_plan = updated_plan.strip()
        self.plan_history.append({
            'mode': 'update',
            'plan': self.current_plan,
            'last_search_query': last_search_query,
            'timestamp': len(self.plan_history)
        })

        # Return the updated plan as observation
        return {
            'success': True,
            'observation': self.current_plan,
            'plan': self.current_plan
        }

    def get_current_plan(self) -> Optional[str]:
        """Get the current active plan.

        Returns:
            Current plan content or None if no plan exists
        """
        return self.current_plan

    def get_plan_history(self) -> list:
        """Get the full history of plan operations.

        Returns:
            List of plan operations with timestamps
        """
        return self.plan_history.copy()

    def reset(self):
        """Reset the plan tool state."""
        self.current_plan = None
        self.plan_history = []
        self.query = None

