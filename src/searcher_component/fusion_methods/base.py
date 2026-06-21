"""Base class for all fusion methods."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseFusion(ABC):
    """Abstract base class for fusion methods.

    All fusion methods must implement `fuse()`. The shared `_get_doc_id()`
    helper handles the two possible id field names ('doc_id' or 'id').
    """

    @abstractmethod
    def fuse(self, ranked_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Fuse multiple ranked lists into a single deduplicated ranked list.

        Args:
            ranked_lists: List of ranked result lists. Each result dict must
                          contain at least a 'doc_id' or 'id' key.

        Returns:
            Fused and deduplicated list of results.
        """
        ...

    def _get_doc_id(self, result: Dict[str, Any]) -> Optional[str]:
        """Extract the document id from a result dict."""
        return result.get("doc_id") or result.get("id")
