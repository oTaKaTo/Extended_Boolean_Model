"""
Streamlit GUI for Extended Boolean Model Information Retrieval System
"""

import streamlit as st
import pandas as pd
import json
import re
import datetime
import os
from base_model import ExtendedBooleanModel
from query_parser import QueryParser, query_tree_to_string

# Import Gemini API
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# Document corpus (from your specification)
DOCUMENTS = [
    ['bird', 'cat', 'bird', 'cat', 'dog', 'dog', 'bird'],  # D1
    ['cat', 'tiger', 'cat', 'dog'],  # D2
    ['dog', 'bird', 'bird'],  # D3
    ['cat', 'tiger'],  # D4
    ['tiger', 'tiger', 'dog', 'tiger', 'cat'],  # D5
    ['cat', 'cat', 'tiger', 'tiger'],  # D6
    ['bird', 'cat', 'dog'],  # D7
    ['dog', 'cat', 'bird'],  # D8
    ['cat', 'dog', 'tiger'],  # D9
    ['tiger', 'cat', 'tiger'],  # D10
    ['dog', 'bird', 'cat', 'tiger', 'dog', 'bird'],  # D11
    ['tiger', 'cat', 'dog', 'cat', 'bird', 'tiger'],  # D12
    ['bird', 'bird', 'tiger', 'cat', 'tiger', 'dog', 'cat'],  # D13
    ['cat', 'tiger', 'dog', 'bird', 'bird', 'cat'],  # D14
    ['tiger', 'dog', 'tiger', 'cat', 'bird', 'cat', 'dog'],  # D15
]

KEYWORDS = ['bird', 'cat', 'dog', 'tiger']


def query_gemini_api(query_text, api_key, model="gemini-2.5-flash"):
    """
    Query Gemini API and return ranking results
    
    Args:
        query_text: The Boolean query string
        api_key: Gemini API key
        model: Model to use
        
    Returns:
        Dictionary with ranking data or None if error
    """
    try:
        client = genai.Client(api_key=api_key)
        
        prompt_template = '''You are an expert Information Retrieval System Analyst.  
Your task is to rank documents using the **Extended Boolean Model (p = 2)**  
based on the following formulas:

AND (DeMorgan form):  
sim(q_and, d_j) = 1 - sqrt(((1 - W_1,j)^2 + (1 - W_2,j)^2) / 2)

OR:  
sim(q_or, d_j) = sqrt((W_1,j^2 + W_2,j^2) / 2)

Term weight computation:
tf(t, d_j) = freq(t, d_j) / max_u freq(u, d_j)  
idf(t) = log(N / n_t)  **Using logarithm base 10 ONLY**
idf_norm(t) = idf(t) / max_v idf(v)  
W_t,j = tf(t, d_j) * idf_norm(t)  

Where:
- N = total number of documents (here N = 15)
- n_t = number of documents containing term t
- Keywords = [bird, cat, dog, tiger]

Documents:
D1: [bird, cat, bird, cat, dog, dog, bird]
D2: [cat, tiger, cat, dog]
D3: [dog, bird, bird]
D4: [cat, tiger]
D5: [tiger, tiger, dog, tiger, cat]
D6: [cat, cat, tiger, tiger]
D7: [bird, cat, dog]
D8: [dog, cat, bird]
D9: [cat, dog, tiger]
D10: [tiger, cat, tiger]
D11: [dog, bird, cat, tiger, dog, bird]
D12: [tiger, cat, dog, cat, bird, tiger]
D13: [bird, bird, tiger, cat, tiger, dog, cat]
D14: [cat, tiger, dog, bird, bird, cat]
D15: [tiger, dog, tiger, cat, bird, cat, dog]

Query: <<USER_QUERY>>

Instructions:
1. Compute idf, idf_norm, and W_t,j for all terms. *Re-count again of term if it have log(1) = 0 in idf , you may count term wrong*
2. Use the formulas above to calculate similarity sim(q, d_j) recursively according to the Boolean structure of the query.
3. Round similarity values to 4 decimals.
4. Rank documents in descending order of similarity (ties → smaller document id first).
5. Return the following JSON format:

{ "ranking": [
  {"document": 1, "rank": 1, "similarity": 0.9811},
  {"document": 2, "rank": 2, "similarity": 0.7822},
  {"document": 6, "rank": 3, "similarity": 0.3843}
]}
'''
        
        prompt = prompt_template.replace('<<USER_QUERY>>', query_text)
        
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
        
        # Save to file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f'gemini_{timestamp}.txt'
        with open(fname, "w", encoding="utf-8") as f:
            f.write(f'model: {model}\n')
            f.write(f'prompt: {prompt}\n')
            f.write(f'response: {response.text}\n')
        
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*"ranking"[\s\S]*\][\s\S]*\}', response.text)
        
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            return data, fname
        else:
            return None, fname
            
    except Exception as e:
        st.error(f"Error querying Gemini API: {str(e)}")
        return None, None


def main():
    st.set_page_config(page_title="Extended Boolean Model IR System", layout="wide")
    
    st.title("🔍 Extended Boolean Model Information Retrieval System")
    st.markdown("### DeMorgan AND & OR with p=2")
    
    # Sidebar: Document corpus
    with st.sidebar:
        st.header("📚 Document Corpus")
        st.markdown(f"**Total Documents:** {len(DOCUMENTS)}")
        st.markdown(f"**Keywords:** {', '.join(KEYWORDS)}")
        
        with st.expander("View All Documents"):
            for i, doc in enumerate(DOCUMENTS, 1):
                st.markdown(f"**D{i}:** {doc}")
    
    # Initialize the model
    if 'ebm' not in st.session_state:
        st.session_state.ebm = ExtendedBooleanModel(DOCUMENTS, KEYWORDS)
        st.session_state.parser = QueryParser(KEYWORDS)
    
    ebm = st.session_state.ebm
    parser = st.session_state.parser
    
    # Execution mode selection
    st.markdown("---")
    st.header("⚙️ Execution Mode")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        run_algorithm = st.checkbox("🔢 Run Local Algorithm", value=True, help="Execute the local EBM implementation")
    
    with col2:
        if GEMINI_AVAILABLE:
            run_gemini = st.checkbox("🤖 Run Gemini API", value=False, help="Query Gemini API for comparison")
        else:
            st.warning("⚠️ Gemini API not available")
            run_gemini = False
    
    with col3:
        if run_algorithm and run_gemini:
            st.info("📊 Both modes - Comparison will be shown")
        elif run_algorithm:
            st.info("🔢 Algorithm only")
        elif run_gemini:
            st.info("🤖 Gemini only")
        else:
            st.warning("⚠️ Select at least one mode")
    
    # Gemini API Key input (if Gemini is selected)
    if run_gemini and GEMINI_AVAILABLE:
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Enter your Gemini API key"
        )
        
        gemini_model = st.selectbox(
            "Gemini Model",
            ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
            index=0
        )
    else:
        gemini_api_key = None
        gemini_model = "gemini-2.5-flash"
    
    # Query input
    st.markdown("---")
    st.header("📝 Enter Your Query")
    
    # Separate input fields based on execution mode
    query = None
    gemini_query = None
    
    if run_algorithm and not run_gemini:
        # Only local algorithm - single English input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Boolean Query (English only)",
                placeholder="e.g., bird AND cat, dog OR tiger, (bird OR cat) AND dog",
                help="Use AND, OR, NOT operators. Parentheses supported. English keywords only."
            )
        
        with col2:
            st.markdown("**Example queries:**")
            if st.button("bird AND cat"):
                query = "bird AND cat"
            if st.button("dog OR tiger"):
                query = "dog OR tiger"
            if st.button("(bird OR cat) AND dog"):
                query = "(bird OR cat) AND dog"
    
    elif run_gemini and not run_algorithm:
        # Only Gemini - single multilingual input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            gemini_query = st.text_input(
                "Boolean Query (Any Language)",
                placeholder="e.g., นก AND แมว, bird AND cat, 鳥 AND 猫",
                help="Use AND, OR, NOT operators. Can use any language for keywords."
            )
            query = gemini_query  # For consistency in later code
        
        with col2:
            st.markdown("**Example:**")
            st.markdown("bird AND cat")
            st.markdown("นก AND แมว (Thai)")
            st.markdown("鳥 AND 猫 (Japanese)")
    
    else:
        # Both modes - two separate inputs
        st.markdown("**Local Algorithm Query (English only):**")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "English Query for Local Algorithm",
                placeholder="e.g., bird AND cat, dog OR tiger",
                help="English keywords only: bird, cat, dog, tiger",
                key="local_query"
            )
        
        with col2:
            st.markdown("**Examples:**")
            if st.button("bird AND cat", key="ex1"):
                query = "bird AND cat"
            if st.button("dog OR tiger", key="ex2"):
                query = "dog OR tiger"
        
        st.markdown("**Gemini API Query (Any Language):**")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            gemini_query = st.text_input(
                "Query for Gemini (Any Language)",
                placeholder="e.g., นก AND แมว, bird AND cat, 鳥 AND 猫",
                help="Can use any language - Gemini will translate to match documents",
                key="gemini_query"
            )
        
        with col2:
            st.markdown("**Examples:**")
            st.markdown("นก AND แมว")
            st.markdown("鳥 AND 猫")
    
    # Process queries
    if query or gemini_query:
        # Validation
        if run_algorithm and not query:
            st.warning("⚠️ Please enter a query for the local algorithm")
            return
        if run_gemini and not gemini_query:
            st.warning("⚠️ Please enter a query for Gemini API")
            return
        
        st.markdown("---")
        
        # Initialize variables
        ranking = None
        gemini_ranking = None
        gemini_file = None
        
        # Run Gemini API if selected
        if run_gemini:
            if not gemini_api_key:
                st.error("❌ Please provide Gemini API Key")
                return
            
            # Use gemini_query if available, otherwise fall back to query
            query_for_gemini = gemini_query if gemini_query else query
            
            with st.spinner("🤖 Querying Gemini API..."):
                gemini_data, gemini_file = query_gemini_api(query_for_gemini, gemini_api_key, gemini_model)
                
                if gemini_data and 'ranking' in gemini_data:
                    gemini_ranking = gemini_data['ranking']
                    st.success(f"✅ Gemini API completed - Results saved to `{gemini_file}`")
                else:
                    st.error(f"❌ Failed to get valid ranking from Gemini API (saved to `{gemini_file}`)")
        
        # Run local algorithm if selected
        if run_algorithm:
            # Parse the query
            try:
                query_tree = parser.parse(query)
                parsed_query_str = query_tree_to_string(query_tree)
                
                st.success(f"✅ Parsed Query: **{parsed_query_str}**")
                
                # Display query tree
                with st.expander("🌳 View Query Tree Structure"):
                    st.json(query_tree)
            except Exception as e:
                st.error(f"❌ Error parsing query: {str(e)}")
                return
        
        # Show results based on execution mode
        try:
            # LOCAL ALGORITHM SECTION
            if run_algorithm:
                # Section 1: TF (Term Frequency) Calculation
                st.markdown("---")
                st.header("1️⃣ Term Frequency (TF) Calculation")
                st.latex(r"tf(t, d_j) = \frac{freq(t, d_j)}{max_u \, freq(u, d_j)}")
                
                tf_data = []
                for doc_idx, doc in enumerate(DOCUMENTS, 1):
                    row = {'Document': f'D{doc_idx}'}
                    for term in KEYWORDS:
                        row[term] = round(ebm.tf_matrix[doc_idx - 1][term], 3)
                    tf_data.append(row)
                
                tf_df = pd.DataFrame(tf_data)
                st.dataframe(tf_df, use_container_width=True)
                
                # Section 2: IDF Calculation
                st.markdown("---")
                st.header("2️⃣ Inverse Document Frequency (IDF) Calculation")
                st.latex(r"idf(t) = log_{10}\left(\frac{N}{n_t}\right)")
                st.markdown(f"**N (Total Documents):** {ebm.N}")
                
                idf_data = []
                for term in KEYWORDS:
                    n_t = sum(1 for doc in DOCUMENTS if term in doc)
                    idf_data.append({
                        'Term': term,
                        'n_t (Doc Count)': n_t,
                        'IDF': round(ebm.idf_dict[term], 3)
                    })
                
                idf_df = pd.DataFrame(idf_data)
                st.dataframe(idf_df, use_container_width=True)
                
                # Section 3: Normalized IDF
                st.markdown("---")
                st.header("3️⃣ Normalized IDF Calculation")
                st.latex(r"idf_{norm}(t) = \frac{idf(t)}{max_v \, idf(v)}")
                
                max_idf = max(ebm.idf_dict.values())
                st.markdown(f"**Max IDF:** {round(max_idf, 3)}")
                
                idf_norm_data = []
                for term in KEYWORDS:
                    idf_norm_data.append({
                        'Term': term,
                        'IDF': round(ebm.idf_dict[term], 3),
                        'IDF_norm': round(ebm.idf_norm_dict[term], 3)
                    })
                
                idf_norm_df = pd.DataFrame(idf_norm_data)
                st.dataframe(idf_norm_df, use_container_width=True)
                
                # Section 4: Term Weights
                st.markdown("---")
                st.header("4️⃣ Term Weights (W_t,j) Calculation")
                st.latex(r"W_{t,j} = tf(t, d_j) \times idf_{norm}(t)")
                
                weight_data = []
                for doc_idx in range(len(DOCUMENTS)):
                    row = {'Document': f'D{doc_idx + 1}'}
                    for term in KEYWORDS:
                        row[term] = round(ebm.weight_matrix[doc_idx][term], 3)
                    weight_data.append(row)
                
                weight_df = pd.DataFrame(weight_data)
                st.dataframe(weight_df, use_container_width=True)
                
                # Section 5: Similarity Formulas
                st.markdown("---")
                st.header("5️⃣ Similarity Formulas (p = 2)")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**AND (DeMorgan form) :**")
                    st.latex(r"sim(q_{AND}, d_j) = 1 - \sqrt{\frac{(1 - W_{1,j})^2 + (1 - W_{2,j})^2}{2}}")

                
                with col2:
                    st.markdown("**OR :**")
                    st.latex(r"sim(q_{OR}, d_j) = \sqrt{\frac{W_{1,j}^2 + W_{2,j}^2}{2}}")

                
                st.markdown("**NOT:**")
                st.latex(r"sim(NOT \, t, d_j) = 1 - W_{t,j}")
                
                # Section 6: Document Similarities
                st.markdown("---")
                st.header("6️⃣ Document Similarity Scores")
                
                similarity_data = []
                for doc_idx in range(len(DOCUMENTS)):
                    sim_score = ebm.evaluate_query(query_tree, doc_idx)
                    similarity_data.append({
                        'Document': f'D{doc_idx + 1}',
                        'Similarity': sim_score
                    })
                
                similarity_df = pd.DataFrame(similarity_data)
                st.dataframe(similarity_df, use_container_width=True)
                
                # Section 7: Final Ranking
                st.markdown("---")
                st.header("🏆 Final Document Ranking")
                st.markdown("**Sorted by:** Similarity (descending), Document ID (ascending for ties)")
                
                ranking = ebm.rank_documents(query_tree)
                ranking_df = pd.DataFrame(ranking)
                
                # Highlight top results
                def highlight_top(row):
                    if row['rank'] <= 0:
                        return ['background-color: #d4edda'] * len(row)
                    return [''] * len(row)
                
                styled_df = ranking_df.style.apply(highlight_top, axis=1)
                st.dataframe(styled_df, use_container_width=True)
                
                # Show top 5 documents
                st.markdown("---")
                st.subheader("📊 Top 5 Documents")
                
                for i, result in enumerate(ranking[:5], 1):
                    doc_num = result['document']
                    doc_content = DOCUMENTS[doc_num - 1]
                    similarity = result['similarity']
                    
                    with st.expander(f"Rank {i}: D{doc_num} (Similarity: {similarity})"):
                        st.markdown(f"**Content:** {doc_content}")
                        
                        # Show term weights for this document
                        st.markdown("**Term Weights:**")
                        weight_info = []
                        for term in KEYWORDS:
                            weight = ebm.weight_matrix[doc_num - 1][term]
                            weight_info.append(f"{term}: {round(weight, 3)}")
                        st.markdown(", ".join(weight_info))
                
                # Export results
                st.markdown("---")
                st.subheader("💾 Export Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = ranking_df.to_csv(index=False)
                    st.download_button(
                        label="Download Ranking (CSV)",
                        data=csv,
                        file_name=f"ebm_ranking_{query.replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    json_str = ranking_df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="Download Ranking (JSON)",
                        data=json_str,
                        file_name=f"ebm_ranking_{query.replace(' ', '_')}.json",
                        mime="application/json"
                    )
            
            # COMPARISON SECTION (if both algorithms ran)
            if run_algorithm and run_gemini and gemini_ranking:
                st.markdown("---")
                st.header("📊 Algorithm Comparison")
                st.markdown(f"**Gemini Results File:** `{gemini_file}`")
                
                # Create comparison dataframe
                st.subheader("📊 Side-by-Side Comparison")
                
                # Convert to dictionaries for easier lookup
                local_dict = {r['document']: {'rank': r['rank'], 'similarity': r['similarity']} 
                             for r in ranking}
                gemini_dict = {r['document']: {'rank': r['rank'], 'similarity': r['similarity']} 
                              for r in gemini_ranking}
                
                # Create comparison table
                comparison_data = []
                all_docs = sorted(set(local_dict.keys()) | set(gemini_dict.keys()))
                
                for doc_id in all_docs:
                    local_info = local_dict.get(doc_id, {'rank': '-', 'similarity': '-'})
                    gemini_info = gemini_dict.get(doc_id, {'rank': '-', 'similarity': '-'})
                    
                    row = {
                        'Document': f'D{doc_id}',
                        'Local Rank': local_info['rank'],
                        'Local Similarity': local_info['similarity'],
                        'Gemini Rank': gemini_info['rank'],
                        'Gemini Similarity': gemini_info['similarity'],
                    }
                    
                    # Calculate differences if both exist
                    if isinstance(local_info['similarity'], (int, float)) and isinstance(gemini_info['similarity'], (int, float)):
                        row['Sim Difference'] = round(abs(local_info['similarity'] - gemini_info['similarity']), 6)
                    else:
                        row['Sim Difference'] = '-'
                    
                    comparison_data.append(row)
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Highlight differences
                def highlight_differences(row):
                    colors = [''] * len(row)
                    if row['Local Rank'] != row['Gemini Rank']:
                        colors[1] = 'background-color: #fff3cd'  # Local Rank
                        colors[3] = 'background-color: #fff3cd'  # Gemini Rank
                    if isinstance(row['Sim Difference'], (int, float)) and row['Sim Difference'] > 0.01:
                        colors[2] = 'background-color: #f8d7da'  # Local Similarity
                        colors[4] = 'background-color: #f8d7da'  # Gemini Similarity
                    return colors
                
                styled_comparison = comparison_df.style.apply(highlight_differences, axis=1)
                st.dataframe(styled_comparison, use_container_width=True)
                
                # Statistics
                st.markdown("---")
                st.subheader("📈 Comparison Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Count matching ranks
                    matching_ranks = sum(1 for row in comparison_data 
                                       if row['Local Rank'] == row['Gemini Rank'] 
                                       and row['Local Rank'] != '-')
                    total_compared = sum(1 for row in comparison_data 
                                       if row['Local Rank'] != '-' and row['Gemini Rank'] != '-')
                    st.metric("Matching Ranks", f"{matching_ranks}/{total_compared}")
                
                with col2:
                    # Average similarity difference
                    valid_diffs = [row['Sim Difference'] for row in comparison_data 
                                  if isinstance(row['Sim Difference'], (int, float))]
                    if valid_diffs:
                        avg_diff = sum(valid_diffs) / len(valid_diffs)
                        st.metric("Avg Similarity Diff", f"{avg_diff:.6f}")
                    else:
                        st.metric("Avg Similarity Diff", "N/A")
                
                with col3:
                    # Max similarity difference
                    if valid_diffs:
                        max_diff = max(valid_diffs)
                        st.metric("Max Similarity Diff", f"{max_diff:.6f}")
                    else:
                        st.metric("Max Similarity Diff", "N/A")
                
                # Top 5 comparison
                st.markdown("---")
                st.subheader("🏆 Top 5 Comparison")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Local System Top 5:**")
                    for i, result in enumerate(ranking[:5], 1):
                        st.markdown(f"{i}. D{result['document']} - Similarity: {result['similarity']}")
                
                with col2:
                    st.markdown("**Gemini API Top 5:**")
                    for i, result in enumerate(gemini_ranking[:5], 1):
                        st.markdown(f"{i}. D{result['document']} - Similarity: {result['similarity']}")
            
            # GEMINI ONLY RESULTS (if only Gemini ran)
            elif run_gemini and gemini_ranking and not run_algorithm:
                st.markdown("---")
                st.header("🤖 Gemini API Results")
                st.markdown(f"**Results File:** `{gemini_file}`")
                
                st.subheader("🏆 Ranking Results")
                
                # Display ranking
                for i, result in enumerate(gemini_ranking, 1):
                    st.markdown(f"{i}. D{result['document']} - Similarity: {result['similarity']}")
                
                # Show as dataframe
                st.markdown("---")
                gemini_df = pd.DataFrame(gemini_ranking)
                st.dataframe(gemini_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            st.exception(e)
    
    else:
        st.info("👆 Enter a Boolean query above to start searching!")
        
        # Show formulas even without query
        st.markdown("---")
        st.header("📐 Extended Boolean Model Formulas")
        
        st.markdown("**Term Frequency:**")
        st.latex(r"tf(t, d_j) = \frac{freq(t, d_j)}{max_u \, freq(u, d_j)}")
        
        st.markdown("**Inverse Document Frequency:**")
        st.latex(r"idf(t) = log_{10}\left(\frac{N}{n_t}\right)")
        
        st.markdown("**Normalized IDF:**")
        st.latex(r"idf_{norm}(t) = \frac{idf(t)}{max_v \, idf(v)}")
        
        st.markdown("**Term Weight:**")
        st.latex(r"W_{t,j} = tf(t, d_j) \times idf_{norm}(t)")
        
        st.markdown("**Similarity (AND) :**")
        st.latex(r"sim(q_{AND}, d_j) = 1 - \sqrt{\frac{(1 - W_{1,j})^2 + (1 - W_{2,j})^2}{2}}")
        
        st.markdown("**Similarity (OR) :**")
        st.latex(r"sim(q_{OR}, d_j) = \sqrt{\frac{W_{1,j}^2 + W_{2,j}^2}{2}}")
        


if __name__ == "__main__":
    main()
