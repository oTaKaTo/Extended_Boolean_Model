"""
Query Parser for Extended Boolean Model
Parses Boolean queries and converts them to expression trees
"""

import re
from typing import List


class QueryParser:
    """
    Parse Boolean queries with AND, OR, NOT operators
    Supports parentheses for grouping
    """
    
    def __init__(self, valid_terms: List[str]):
        """
        Initialize parser with valid terms
        
        Args:
            valid_terms: List of valid keywords that can be used in queries
        """
        self.valid_terms = set(valid_terms)
    
    def tokenize(self, query: str) -> List[str]:
        """
        Tokenize the query string into terms and operators
        
        Args:
            query: Query string (e.g., "bird AND cat OR dog")
            
        Returns:
            List of tokens
        """
        # Replace various representations of operators
        query = query.upper()
        
        # Split by whitespace and parentheses while keeping them
        pattern = r'(\(|\)|AND|OR|NOT|\w+)'
        tokens = re.findall(pattern, query)
        
        return tokens
    
    def parse(self, query: str) -> dict:
        """
        Parse a Boolean query into an expression tree
        
        Args:
            query: Query string (e.g., "bird AND cat", "(bird OR cat) AND dog")
            
        Returns:
            Dictionary representing the query tree
        """
        tokens = self.tokenize(query)
        
        if not tokens:
            return {'op': 'OR', 'terms': list(self.valid_terms)[:1]}  # Default
        
        # Parse the tokens into a tree
        result, _ = self._parse_or(tokens, 0)
        return result
    
    def _parse_or(self, tokens: List[str], pos: int) -> tuple:
        """Parse OR expressions (lowest precedence)"""
        left, pos = self._parse_and(tokens, pos)
        
        while pos < len(tokens) and tokens[pos] == 'OR':
            pos += 1  # Skip 'OR'
            right, pos = self._parse_and(tokens, pos)
            left = {'op': 'OR', 'terms': [left, right]}
        
        return left, pos
    
    def _parse_and(self, tokens: List[str], pos: int) -> tuple:
        """Parse AND expressions (medium precedence)"""
        left, pos = self._parse_not(tokens, pos)
        
        while pos < len(tokens) and tokens[pos] == 'AND':
            pos += 1  # Skip 'AND'
            right, pos = self._parse_not(tokens, pos)
            left = {'op': 'AND', 'terms': [left, right]}
        
        return left, pos
    
    def _parse_not(self, tokens: List[str], pos: int) -> tuple:
        """Parse NOT expressions (highest precedence)"""
        if pos < len(tokens) and tokens[pos] == 'NOT':
            pos += 1  # Skip 'NOT'
            operand, pos = self._parse_primary(tokens, pos)
            return {'op': 'NOT', 'operand': operand}, pos
        
        return self._parse_primary(tokens, pos)
    
    def _parse_primary(self, tokens: List[str], pos: int) -> tuple:
        """Parse primary expressions (terms or parenthesized expressions)"""
        if pos >= len(tokens):
            # Return a default term if we run out of tokens
            return {'op': 'TERM', 'value': list(self.valid_terms)[0]}, pos
        
        token = tokens[pos]
        
        # Parenthesized expression
        if token == '(':
            pos += 1  # Skip '('
            expr, pos = self._parse_or(tokens, pos)
            if pos < len(tokens) and tokens[pos] == ')':
                pos += 1  # Skip ')'
            return expr, pos
        
        # Term
        term = token.lower()
        if term in self.valid_terms:
            return {'op': 'TERM', 'value': term}, pos + 1
        else:
            # If invalid term, try to find a close match or skip
            pos += 1
            if pos < len(tokens):
                return self._parse_primary(tokens, pos)
            else:
                return {'op': 'TERM', 'value': list(self.valid_terms)[0]}, pos
    
    def parse_simple(self, query: str) -> dict:
        """
        Simple parser for basic queries without complex nesting
        Handles: "term1 AND term2", "term1 OR term2", "term1"
        
        Args:
            query: Simple query string
            
        Returns:
            Query tree dictionary
        """
        query = query.strip().lower()
        
        # Check for AND
        if ' and ' in query:
            terms = [t.strip() for t in re.split(r'\s+and\s+', query, flags=re.IGNORECASE)]
            terms = [t for t in terms if t in self.valid_terms]
            if terms:
                return {'op': 'AND', 'terms': terms}
        
        # Check for OR
        if ' or ' in query:
            terms = [t.strip() for t in re.split(r'\s+or\s+', query, flags=re.IGNORECASE)]
            terms = [t for t in terms if t in self.valid_terms]
            if terms:
                return {'op': 'OR', 'terms': terms}
        
        # Check for NOT
        if query.startswith('not '):
            term = query[4:].strip()
            if term in self.valid_terms:
                return {'op': 'NOT', 'operand': term}
        
        # Single term
        if query in self.valid_terms:
            return {'op': 'TERM', 'value': query}
        
        # Default: OR of all matching terms found in query
        found_terms = [t for t in self.valid_terms if t in query]
        if found_terms:
            return {'op': 'OR', 'terms': found_terms}
        
        # Fallback: first valid term
        return {'op': 'TERM', 'value': list(self.valid_terms)[0]}


def query_tree_to_string(tree: dict) -> str:
    """
    Convert a query tree back to a readable string
    
    Args:
        tree: Query tree dictionary
        
    Returns:
        String representation of the query
    """
    op = tree.get('op', '').upper()
    
    if op == 'TERM':
        return tree.get('value', '')
    
    elif op == 'AND':
        terms = tree.get('terms', [])
        term_strs = []
        for term in terms:
            if isinstance(term, dict):
                term_strs.append(f"({query_tree_to_string(term)})")
            else:
                term_strs.append(str(term))
        return ' AND '.join(term_strs)
    
    elif op == 'OR':
        terms = tree.get('terms', [])
        term_strs = []
        for term in terms:
            if isinstance(term, dict):
                term_strs.append(f"({query_tree_to_string(term)})")
            else:
                term_strs.append(str(term))
        return ' OR '.join(term_strs)
    
    elif op == 'NOT':
        operand = tree.get('operand')
        if isinstance(operand, dict):
            return f"NOT ({query_tree_to_string(operand)})"
        else:
            return f"NOT {operand}"
    
    return ''
