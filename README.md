# Extended Boolean Model Information Retrieval System

A complete implementation of the Extended Boolean Model (EBM) for Information Retrieval with DeMorgan AND and OR operators (p=2).

## Features

- ✅ Extended Boolean Model with p=2 (Euclidean distance)
- ✅ DeMorgan AND formula implementation
- ✅ OR formula implementation
- ✅ TF-IDF weighting with normalization
- ✅ Query parser supporting AND, OR, NOT, and parentheses
- ✅ Interactive Streamlit GUI
- ✅ Step-by-step calculation display
- ✅ Document ranking with tie-breaking

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the Streamlit GUI:
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Example Queries:
- `bird AND cat` - Documents containing both bird and cat
- `dog OR tiger` - Documents containing dog or tiger
- `(bird OR cat) AND dog` - Complex query with grouping
- `NOT tiger` - Documents without tiger

## Formulas

### Term Frequency (TF):
```
tf(t, d_j) = freq(t, d_j) / max_u freq(u, d_j)
```

### Inverse Document Frequency (IDF):
```
idf(t) = log10(N / n_t)
```

### Normalized IDF:
```
idf_norm(t) = idf(t) / max_v idf(v)
```

### Term Weight:
```
W_t,j = tf(t, d_j) × idf_norm(t)
```

### Similarity (AND - DeMorgan form):
```
sim(q_AND, d_j) = 1 - sqrt(Σ(1 - W_i,j)² / n)
```

### Similarity (OR):
```
sim(q_OR, d_j) = sqrt(Σ(W_i,j²) / n)
```

## Document Corpus

The system includes 15 documents with 4 keywords: bird, cat, dog, tiger

- D1: [bird, cat, bird, cat, dog, dog, bird]
- D2: [cat, tiger, cat, dog]
- D3: [dog, bird, bird]
- ... (see streamlit_app.py for full corpus)

## Files

- `base_model.py` - Extended Boolean Model implementation
- `query_parser.py` - Boolean query parser
- `streamlit_app.py` - Interactive GUI application
- `requirements.txt` - Python dependencies
- `app.py` - Original Gemini API test (legacy)

## Project Structure

```
EBM/
├── base_model.py          # Core EBM logic
├── query_parser.py        # Query parsing & tree construction
├── streamlit_app.py       # GUI application
├── requirements.txt       # Dependencies
├── README.md             # This file
└── app.py                # Legacy API test
```

## Features in GUI

1. **Document Corpus Display** - View all 15 documents in sidebar
2. **Query Input** - Enter Boolean queries with natural syntax
3. **TF Calculation** - See term frequency for each document
4. **IDF Calculation** - View inverse document frequency values
5. **IDF Normalization** - See normalized IDF values
6. **Weight Matrix** - Complete W_t,j values for all terms/documents
7. **Similarity Scores** - Computed similarity for each document
8. **Final Ranking** - Documents ranked by similarity (ties broken by document ID)
9. **Export Results** - Download rankings as CSV or JSON

## Author

Information Retrieval System - Extended Boolean Model Implementation
