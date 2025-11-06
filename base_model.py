"""
Extended Boolean Model (EBM) Information Retrieval System
Implements DeMorgan AND and OR with p=2
"""

import math
from typing import Dict, List, Tuple
from collections import Counter


class ExtendedBooleanModel:
    """
    Extended Boolean Model for Information Retrieval
    Uses p=2 (Euclidean distance) for DeMorgan formulas
    """
    
    def __init__(self, documents: List[List[str]], keywords: List[str]):
        """
        Initialize the EBM with documents and keywords
        
        Args:
            documents: List of documents, each document is a list of terms
            keywords: List of all possible keywords/terms
        """
        self.documents = documents
        self.keywords = keywords
        self.N = len(documents)  # Total number of documents
        
        # Precompute statistics
        self.tf_matrix = self._compute_tf()
        self.idf_dict = self._compute_idf()
        self.idf_norm_dict = self._compute_idf_norm()
        self.weight_matrix = self._compute_weights()
        
    def _compute_tf(self) -> List[Dict[str, float]]:
        """
        Compute term frequency (tf) for each term in each document
        tf(t, d_j) = freq(t, d_j) / max_u freq(u, d_j)
        
        Returns:
            List of dictionaries, one per document, mapping term -> tf value
        """
        tf_matrix = []
        
        for doc in self.documents:
            term_counts = Counter(doc)
            max_freq = max(term_counts.values()) if term_counts else 1
            
            tf_doc = {}
            for term in self.keywords:
                tf_doc[term] = term_counts.get(term, 0) / max_freq
            
            tf_matrix.append(tf_doc)
        
        return tf_matrix
    
    def _compute_idf(self) -> Dict[str, float]:
        """
        Compute inverse document frequency (idf) for each term
        idf(t) = log10(N / n_t)
        
        Returns:
            Dictionary mapping term -> idf value
        """
        idf_dict = {}
        
        for term in self.keywords:
            n_t = sum(1 for doc in self.documents if term in doc)
            if n_t > 0:
                idf_dict[term] = math.log10(self.N / n_t)
            else:
                idf_dict[term] = 0.0
        
        return idf_dict
    
    def _compute_idf_norm(self) -> Dict[str, float]:
        """
        Compute normalized idf for each term
        idf_norm(t) = idf(t) / max_v idf(v)
        
        Returns:
            Dictionary mapping term -> normalized idf value
        """
        max_idf = max(self.idf_dict.values()) if self.idf_dict else 1.0
        
        idf_norm_dict = {}
        for term in self.keywords:
            idf_norm_dict[term] = self.idf_dict[term] / max_idf if max_idf > 0 else 0.0
        
        return idf_norm_dict
    
    def _compute_weights(self) -> List[Dict[str, float]]:
        """
        Compute term weights for each term in each document
        W_t,j = tf(t, d_j) * idf_norm(t)
        
        Returns:
            List of dictionaries, one per document, mapping term -> weight
        """
        weight_matrix = []
        
        for doc_idx in range(self.N):
            weights_doc = {}
            for term in self.keywords:
                weights_doc[term] = self.tf_matrix[doc_idx][term] * self.idf_norm_dict[term]
            weight_matrix.append(weights_doc)
        
        return weight_matrix
    
    def get_weight(self, term: str, doc_idx: int) -> float:
        """Get the weight of a term in a specific document"""
        return self.weight_matrix[doc_idx].get(term, 0.0)
    
    def sim_and(self, terms: List[str], doc_idx: int) -> float:
        """
        Compute similarity for AND query using DeMorgan form (p=2)
        sim(q_and, d_j) = 1 - sqrt(sum((1 - W_i,j)^2) / n)
        
        Args:
            terms: List of terms in the AND query
            doc_idx: Document index
            
        Returns:
            Similarity score
        """
        if not terms:
            return 0.0
        
        n = len(terms)
        sum_squares = sum((1 - self.get_weight(term, doc_idx)) ** 2 for term in terms)
        
        similarity = 1 - math.sqrt(sum_squares / n)
        return similarity
    
    def sim_or(self, terms: List[str], doc_idx: int) -> float:
        """
        Compute similarity for OR query (p=2)
        sim(q_or, d_j) = sqrt(sum(W_i,j^2) / n)
        
        Args:
            terms: List of terms in the OR query
            doc_idx: Document index
            
        Returns:
            Similarity score
        """
        if not terms:
            return 0.0
        
        n = len(terms)
        sum_squares = sum(self.get_weight(term, doc_idx) ** 2 for term in terms)
        
        similarity = math.sqrt(sum_squares / n)
        return similarity
    
    def sim_not(self, term: str, doc_idx: int) -> float:
        """
        Compute similarity for NOT query
        sim(NOT t, d_j) = 1 - W_t,j
        
        Args:
            term: Term to negate
            doc_idx: Document index
            
        Returns:
            Similarity score
        """
        return 1 - self.get_weight(term, doc_idx)
    
    def evaluate_query(self, query_tree: dict, doc_idx: int) -> float:
        """
        Recursively evaluate a query tree for a specific document
        
        Args:
            query_tree: Dictionary representing the query structure
                       e.g., {'op': 'AND', 'terms': ['bird', 'cat']}
                       or {'op': 'OR', 'terms': [sub_tree1, sub_tree2]}
            doc_idx: Document index
            
        Returns:
            Similarity score for the document
        """
        op = query_tree.get('op', '').upper()
        
        if op == 'TERM':
            # Single term - return its weight
            term = query_tree.get('value')
            return self.get_weight(term, doc_idx)
        
        elif op == 'AND':
            # Get all term weights or sub-query results
            terms = query_tree.get('terms', [])
            weights = []
            for item in terms:
                if isinstance(item, dict):
                    weights.append(self.evaluate_query(item, doc_idx))
                else:
                    weights.append(self.get_weight(item, doc_idx))
            
            # Apply DeMorgan AND formula
            n = len(weights)
            if n == 0:
                return 0.0
            sum_squares = sum((1 - w) ** 2 for w in weights)
            return 1 - math.sqrt(sum_squares / n)
        
        elif op == 'OR':
            # Get all term weights or sub-query results
            terms = query_tree.get('terms', [])
            weights = []
            for item in terms:
                if isinstance(item, dict):
                    weights.append(self.evaluate_query(item, doc_idx))
                else:
                    weights.append(self.get_weight(item, doc_idx))
            
            # Apply OR formula
            n = len(weights)
            if n == 0:
                return 0.0
            sum_squares = sum(w ** 2 for w in weights)
            return math.sqrt(sum_squares / n)
        
        elif op == 'NOT':
            # Get the term or sub-query result
            operand = query_tree.get('operand')
            if isinstance(operand, dict):
                weight = self.evaluate_query(operand, doc_idx)
            else:
                weight = self.get_weight(operand, doc_idx)
            return 1 - weight
        
        else:
            return 0.0
    
    def rank_documents(self, query_tree: dict) -> List[Tuple[int, float]]:
        """
        Rank all documents based on query similarity
        
        Args:
            query_tree: Parsed query tree structure
            
        Returns:
            List of (doc_num, similarity) tuples, sorted by similarity (descending)
            In case of ties, smaller document id comes first
        """
        results = []
        
        for doc_idx in range(self.N):
            similarity = self.evaluate_query(query_tree, doc_idx)
            doc_num = doc_idx + 1  # Document numbers start at 1
            results.append((doc_num, similarity))
        
        # Sort by similarity (descending), then by document number (ascending) for ties
        results.sort(key=lambda x: (-x[1], x[0]))
        
        # Add rank information
        ranked_results = []
        for rank, (doc_num, similarity) in enumerate(results, 1):
            ranked_results.append({
                'document': doc_num,
                'rank': rank,
                'similarity': similarity
            })
        
        return ranked_results
