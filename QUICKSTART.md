# Quick Start Guide

## Running the Application

### Option 1: Streamlit GUI (Recommended)
```powershell
streamlit run streamlit_app.py
```
Then open your browser to: http://localhost:8501

### Option 2: Test Script (CLI)
```powershell
python test_ebm.py
```

### Option 3: Python Interactive
```python
from base_model import ExtendedBooleanModel
from query_parser import QueryParser

# Define your documents and keywords
documents = [
    ['bird', 'cat', 'bird'],
    ['cat', 'tiger', 'dog'],
    # ... more documents
]
keywords = ['bird', 'cat', 'dog', 'tiger']

# Initialize model
ebm = ExtendedBooleanModel(documents, keywords)
parser = QueryParser(keywords)

# Parse and evaluate query
query_tree = parser.parse("bird AND cat")
ranking = ebm.rank_documents(query_tree)

# Display results
for result in ranking[:5]:
    print(f"D{result['document']}: {result['similarity']}")
```

## Example Queries

### Simple Queries
- `bird` - Single term search
- `cat` - Another single term
- `tiger` - Yet another single term

### AND Queries (DeMorgan form)
- `bird AND cat` - Documents with both terms
- `dog AND tiger` - Both dog and tiger
- `bird AND cat AND dog` - All three terms

### OR Queries
- `bird OR cat` - Documents with either term
- `dog OR tiger` - Either dog or tiger
- `bird OR cat OR dog` - Any of the three

### Complex Queries (with parentheses)
- `(bird OR cat) AND dog` - Either bird or cat, plus dog
- `(bird AND cat) OR tiger` - Either both bird+cat, or tiger
- `(bird OR cat) AND (dog OR tiger)` - Complex combination

### NOT Queries
- `NOT tiger` - Documents without tiger
- `bird AND NOT cat` - Bird but not cat
- `(bird OR dog) AND NOT tiger` - Bird or dog, but no tiger

## GUI Features

### Left Sidebar
- **Document Corpus**: View all 15 documents
- **Keywords**: bird, cat, dog, tiger

### Main Panel
1. **Query Input**: Enter Boolean queries
2. **TF Table**: Term frequency for all documents
3. **IDF Table**: Inverse document frequency values
4. **IDF_norm Table**: Normalized IDF values
5. **Weights Table**: W_t,j for all terms and documents
6. **Similarity Scores**: Computed similarity for each document
7. **Final Ranking**: Sorted results (ties broken by doc ID)
8. **Top 5 Details**: Expanded view of top documents
9. **Export**: Download results as CSV or JSON

## Understanding the Results

### Similarity Scores
- **Range**: 0.0 to 1.0
- **1.0**: Perfect match
- **0.5**: Moderate relevance
- **0.0**: No relevance

### Tie Breaking
When two documents have the same similarity score, the one with the smaller document ID ranks higher.

Example: If D3 and D7 both have similarity 0.500, D3 ranks higher.

## Formulas Used

### Term Frequency (TF)
```
tf(t, d) = count(t in d) / max_count(any term in d)
```

### Inverse Document Frequency (IDF)
```
idf(t) = log10(Total Docs / Docs containing t)
```

### Normalized IDF
```
idf_norm(t) = idf(t) / max(idf of all terms)
```

### Term Weight
```
W(t,d) = tf(t,d) × idf_norm(t)
```

### AND Similarity (DeMorgan, p=2)
```
sim(AND) = 1 - sqrt(Σ(1 - W_i)² / n)
```

### OR Similarity (p=2)
```
sim(OR) = sqrt(Σ(W_i²) / n)
```

## Tips

1. **Use parentheses** for complex queries to control operator precedence
2. **AND has higher precedence** than OR (like multiplication before addition)
3. **NOT has highest precedence** and applies to the immediate next term
4. **Check the parsed query** to verify your query was interpreted correctly
5. **Expand sections** in the GUI to see detailed calculations

## Troubleshooting

### App won't start
```powershell
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Import errors
Make sure you're in the correct directory:
```powershell
cd d:\SoftwareDev\IR\EBM
```

### Port already in use
Kill the existing Streamlit process or use a different port:
```powershell
streamlit run streamlit_app.py --server.port 8502
```

## Files

- `base_model.py` - Core EBM implementation
- `query_parser.py` - Boolean query parser
- `streamlit_app.py` - Web GUI application
- `test_ebm.py` - CLI test suite
- `requirements.txt` - Python dependencies
- `README.md` - Full documentation
- `QUICKSTART.md` - This file
