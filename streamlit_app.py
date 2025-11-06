"""
Streamlit GUI for Extended Boolean Model Information Retrieval System
"""

import streamlit as st
import pandas as pd
from base_model import ExtendedBooleanModel
from query_parser import QueryParser, query_tree_to_string


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
    
    # Query input
    st.markdown("---")
    st.header("📝 Enter Your Query")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Boolean Query",
            placeholder="e.g., bird AND cat, dog OR tiger, (bird OR cat) AND dog",
            help="Use AND, OR, NOT operators. Parentheses supported."
        )
    
    with col2:
        st.markdown("**Example queries:**")
        if st.button("bird AND cat"):
            query = "bird AND cat"
        if st.button("dog OR tiger"):
            query = "dog OR tiger"
        if st.button("(bird OR cat) AND dog"):
            query = "(bird OR cat) AND dog"
    
    if query:
        st.markdown("---")
        
        # Parse the query
        try:
            query_tree = parser.parse(query)
            parsed_query_str = query_tree_to_string(query_tree)
            
            st.success(f"✅ Parsed Query: **{parsed_query_str}**")
            
            # Display query tree
            with st.expander("🌳 View Query Tree Structure"):
                st.json(query_tree)
            
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
                st.markdown("**AND (DeMorgan form):**")
                st.latex(r"sim(q_{AND}, d_j) = 1 - \sqrt{\frac{\sum_{i=1}^{n} (1 - W_{i,j})^2}{n}}")
            
            with col2:
                st.markdown("**OR:**")
                st.latex(r"sim(q_{OR}, d_j) = \sqrt{\frac{\sum_{i=1}^{n} W_{i,j}^2}{n}}")
            
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
        
        st.markdown("**Similarity (AND):**")
        st.latex(r"sim(q_{AND}, d_j) = 1 - \sqrt{\frac{\sum_{i=1}^{n} (1 - W_{i,j})^2}{n}}")
        
        st.markdown("**Similarity (OR):**")
        st.latex(r"sim(q_{OR}, d_j) = \sqrt{\frac{\sum_{i=1}^{n} W_{i,j}^2}{n}}")


if __name__ == "__main__":
    main()
