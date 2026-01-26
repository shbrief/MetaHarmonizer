"""Base classes and data structures for matchers."""
from typing import List, Tuple, Callable
from dataclasses import dataclass

@dataclass
class MatchStrategy:
    """Defines a matching strategy."""
    name: str
    match_func: Callable[[str], List[Tuple[str, float, str]]]
    threshold: float = 1.0

class BaseMatcher:
    """Base class for all matchers."""
    
    def __init__(self, engine):
        """
        Args:
            engine: Reference to the main SchemaMapEngine instance
        """
        self.engine = engine
    
    def match(self, col: str) -> List[Tuple[str, float, str]]:
        """
        Perform matching for a column.
        
        Returns:
            List of (field_name, score, source) tuples
        """
        raise NotImplementedError