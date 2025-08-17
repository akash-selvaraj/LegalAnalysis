import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from io import BytesIO
import requests
import json

# For ODT support, requires odfpy library
try:
    from odf.opendocument import load as opendocument_load
    from odf.table import Table, TableRow, TableCell
    from odf.text import P
    odt_support = True
except ImportError:
    odt_support = False
    st.warning("odfpy library not installed. ODT files are not supported.")

st.set_page_config(layout="wide")
st.title("Legal Survey Analysis App")

# --- Sidebar for Chatbot ---
st.sidebar.title("Chat with Your Data")
api_key = st.sidebar.text_input("Enter your Google AI Studio API Key", type="password")

with st.sidebar.expander("How to Get an API Key"):
    st.sidebar.markdown("""
    1. Visit [https://aistudio.google.com/](https://aistudio.google.com/).
    2. Log in and click "Get API key".
    3. Create and copy your new API key.
    """)

# Initialize session state
if 'ai_responses' not in st.session_state:
    st.session_state.ai_responses = []

# --- Main App Logic ---
uploaded_file = st.file_uploader("Upload your tabular data file", type=["xlsx", "xls", "csv", "tsv", "odt"])

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_ext = file_name.lower().split('.')[-1]
    file_content = uploaded_file.read()
    
    try:
        if file_ext in ['xlsx', 'xls']:
            tabular_data = pd.read_excel(BytesIO(file_content))
        elif file_ext == 'csv':
            tabular_data = pd.read_csv(BytesIO(file_content), encoding="utf-8")
        elif file_ext == 'tsv':
            tabular_data = pd.read_csv(BytesIO(file_content), sep="\t", encoding="utf-8")
        elif file_ext == 'odt':
            if not odt_support:
                raise ImportError("odfpy not installed. Cannot process ODT files.")
            doc = opendocument_load(BytesIO(file_content))
            data_rows = []
            for elem in doc.getElementsByType(Table):
                for row in elem.getElementsByType(TableRow):
                    row_data = []
                    for cell in row.getElementsByType(TableCell):
                        cell_text = ''.join(p.firstChild.data if p.firstChild else '' for p in cell.getElementsByType(P))
                        row_data.append(cell_text)
                    data_rows.append(row_data)
            tabular_data = pd.DataFrame(data_rows[1:], columns=data_rows[0] if data_rows else [])
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        st.success(f"File '{file_name}' uploaded successfully!")
        st.session_state.data = tabular_data
        
        data = st.session_state.data
        numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = numerical_cols + categorical_cols

        st.subheader("Data Preview")
        st.dataframe(data.head())

        # --- On-Demand, User-Driven Visualization ---
        st.header("Create a Custom Visualization")
        
        analysis_type = st.selectbox("Select Analysis Type", ["Univariate", "Bivariate"])

        if analysis_type == "Univariate":
            selected_col = st.selectbox("Select a Column", all_cols)
            if st.button("Generate Graph"):
                fig, ax = plt.subplots(figsize=(10, 6))
                st.subheader(f"Bar Chart for {selected_col}")

                if selected_col in categorical_cols:
                    vc = data[selected_col].value_counts()
                    bars = ax.bar(vc.index, vc.values, color='#87CEEB')
                    ax.set_ylabel('Count')
                else: # Numerical - bin first, then bar chart
                    # Bin the numerical data into intervals
                    binned_data = pd.cut(data[selected_col], bins=10)
                    vc = binned_data.value_counts().sort_index()
                    # Convert interval index to string for plotting
                    vc.index = vc.index.astype(str)
                    bars = ax.bar(vc.index, vc.values, color='#90EE90')
                    ax.set_ylabel('Count (Frequency)')
                
                # Calculate percentages for labels
                total = vc.sum()
                if total > 0:
                    labels = [f'{(v.get_height() / total) * 100:.1f}%' for v in bars]
                    ax.bar_label(bars, labels=labels, label_type='center', color='black')

                ax.set_xlabel(selected_col)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                fig.tight_layout()
                st.pyplot(fig)

        elif analysis_type == "Bivariate":
            col1 = st.selectbox("Select the First Column", all_cols, key='col1')
            col2 = st.selectbox("Select the Second Column", all_cols, key='col2')
            
            if st.button("Generate Graph"):
                if col1 == col2:
                    st.warning("Please select two different columns.")
                else:
                    fig, ax = plt.subplots(figsize=(12, 7))
                    st.subheader(f"Clustered Bar Chart: {col1} vs {col2}")

                    # Use a copy for plotting to avoid changing original data
                    plot_data = data.copy()

                    # Bin any numerical columns to treat them as categorical
                    if col1 in numerical_cols:
                        plot_data[col1] = pd.cut(plot_data[col1], bins=5, labels=[f"Bin {i+1}" for i in range(5)])
                    if col2 in numerical_cols:
                        plot_data[col2] = pd.cut(plot_data[col2], bins=5, labels=[f"Bin {i+1}" for i in range(5)])

                    # Now, create a crosstab and plot a clustered bar chart for any combination
                    crosstab = pd.crosstab(plot_data[col1], plot_data[col2])
                    crosstab.plot(kind='bar', stacked=False, ax=ax, colormap='Paired')
                    ax.set_ylabel('Count')
                    
                    # Calculate percentages for labels based on the total number of entries
                    total = len(data)
                    if total > 0:
                        for container in ax.containers:
                            labels = [f'{(v.get_height() / total) * 100:.1f}%' for v in container]
                            ax.bar_label(container, labels=labels, label_type='center', color='black', fontsize=9, zorder=10)

                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                    fig.tight_layout()
                    st.pyplot(fig)
        
        with st.expander("View Data Summaries"):
            if numerical_cols:
                st.subheader("Numerical Columns Summary")
                st.table(data[numerical_cols].describe())
            for col in categorical_cols:
                st.subheader(f"Value Counts for {col}")
                vc = data[col].value_counts()
                vc_df = pd.DataFrame({'Count': vc})
                st.table(vc_df)

    except Exception as e:
        st.error(f"Error processing file: {e}")

# --- Handle Chatbot Interaction in Sidebar ---
if 'data' in st.session_state:
    
    st.sidebar.markdown("---")
    ai_mode = st.sidebar.radio("What would you like the AI to do?", 
                               ("Answer a question", "Generate a visualization"), 
                               key="ai_mode")
    
    question = st.sidebar.text_input("Ask a question about your data...", key="chatbot_input")

    if st.sidebar.button("Generate AI Response") and question:
        if not api_key:
            st.sidebar.error("Please enter your Gemini API Key first.")
        else:
            try:
                data = st.session_state.data
                # For large datasets, send summary stats. For smaller ones, send the whole CSV.
                data_desc = "Data summary:\n" + data.describe(include='all').to_csv() if len(data) > 100 else data.to_csv()

                # Select the prompt based on the chosen AI mode
                if ai_mode == "Generate a visualization":
                    prompt = f"""You are an expert data visualization analyst. Your task is to generate Python code for a Matplotlib graph based on the user's question.

Data: {data_desc}
User question: {question}

**Response Instructions:**
1.  Respond in a single valid JSON object with keys 'answer' (string explanation) and 'code' (string with Python code, or null).
2.  The code MUST generate a visually appealing Matplotlib graph assigned to a variable `fig`.
3.  DO NOT include `plt.show()` or import statements. Assume `pd`, `plt`, `np`, and `data` are available.
"""
                else: # Answer a question
                    prompt = f"""You are a helpful data analyst. Your task is to answer the user's question based on the provided data. Provide a concise, text-only answer.

Data: {data_desc}
User question: {question}

**Response Instructions:**
1.  Respond in a single valid JSON object with keys 'answer' (your textual analysis) and 'code' (this MUST be null).
2.  Do NOT generate any Python code. Your entire response should be in the 'answer' field.
"""

                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
                headers = {"Content-Type": "application/json"}
                payload = {"contents": [{"parts": [{"text": prompt}]}],"generationConfig": {"response_mime_type": "application/json","temperature": 0.5,"maxOutputTokens": 2048}}

                with st.spinner("Asking the AI..."):
                    response = requests.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    result = response.json()
                    content_text = result['candidates'][0]['content']['parts'][0]['text']
                    content = json.loads(content_text)
                
                st.session_state.ai_responses.insert(0, content)

            except requests.exceptions.HTTPError as err:
                st.sidebar.error(f"API Error: {err.response.status_code} - {err.response.text}")
            except Exception as e:
                st.sidebar.error(f"An error occurred: {e}")

# --- Display AI-Generated Responses ---
if st.session_state.ai_responses:
    st.header("AI Responses")
    for content in st.session_state.ai_responses:
        st.markdown("---")
        st.write(content.get('answer', 'No textual answer provided.'))
        
        if content.get('code'):
            try:
                # Prepare a namespace for the exec function
                namespace = {'data': st.session_state.data, 'pd': pd, 'plt': plt, 'np': np, 'fig': None, 'ax': None, 'df': None}
                exec(content['code'], namespace)

                # Check for generated plot or dataframe and display them
                if namespace.get('fig') is not None:
                    st.pyplot(namespace['fig'])
                if namespace.get('df') is not None:
                    st.dataframe(namespace['df'])
            except Exception as e:
                st.error(f"Error executing generated code: {e}")
                st.code(content['code'], language='python')
