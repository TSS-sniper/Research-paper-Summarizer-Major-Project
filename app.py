import streamlit as st
import pandas as pd
import os
import pdfplumber
import concurrent.futures
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnableSequence
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from evaluation import evaluate_summary

# Load API Key from .env
load_dotenv()
default_api_key = os.getenv("GROQ_API_KEY")

# Streamlit UI
st.set_page_config(page_title="PDF Summarizer", layout="wide")
st.title("📄 Research Paper Summarizer")

# Initialize session state
st.session_state.api_key = default_api_key

# File uploader
uploaded_files = st.file_uploader(
    "📂 Upload PDF(s)", type=["pdf"], accept_multiple_files=True
)

# **New**: Select summarization style
summary_type = st.selectbox(
    "📝 Choose Summary Type",
    ["📌 Short Summary", "🔍 Key Insights"],
)

# **New**: Different Prompt Templates
PROMPT_TEMPLATES = {
    "📌 Short Summary": """
    Summarize the given research paper in a **concise** manner. Focus on the main idea, methodology, key findings, and conclusions. Keep the summary within **300 words**.
    
    ### **Title:**  
    [Extract the title of the paper]  

    ### ***Authors & Institution:***  
    [List the authors and their affiliations]

    **Summary:**  
    {text}
    """,
    "🔍 Key Insights": """
    Extract and list the **most important insights** from the research paper. Each insight should be brief and impactful.

    **Key Insights from the Paper:**  
    - 🔹 [Insight 1]  
    - 🔹 [Insight 2]  
    - 🔹 [Insight 3]  
    - 🔹 [Insight 4]  

    {text}
    """,
}


# Extract text
def extract_text_tables(pdf_file):
    text = ""
    tables = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            tables.extend(page.extract_tables())
    return text.strip(), tables


# Summarization function
def summarize_pdf(pdf_file, filename, api_key, prompt_template):
    """Extracts text from PDF and generates a summary using the selected prompt."""

    # Initialize LLM
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192")

    # Extract text
    text, tables = extract_text_tables(pdf_file)
    if not text:
        return "⚠️ No readable text extracted."

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_docs = text_splitter.split_text(text)[:3]  # Process first 3 chunks

    # Select the correct prompt template
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = RunnableSequence(prompt | llm)

    try:
        # Step 1: Summarize each chunk separately
        chunk_summaries = [
            chain.invoke({"text": chunk}).content for chunk in split_docs
        ]

        # Step 2: Merge and summarize again
        combined_text = "\n".join(chunk_summaries)
        final_summary = chain.invoke({"text": combined_text}).content

        return final_summary  # Return the final structured summary
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Process PDFs when button is clicked
if st.button("⚡ Summarize PDF(s)"):
    if uploaded_files:
        api_key = st.session_state.api_key  # Get API key from session state
        with st.spinner("🔄 Processing..."):
            file_names = [file.name for file in uploaded_files]
            prompt_template = PROMPT_TEMPLATES[summary_type]  # Get selected prompt

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                results = list(
                    executor.map(
                        lambda args: summarize_pdf(*args, api_key, prompt_template),
                        zip(uploaded_files, file_names),
                    )
                )  # ,

            # Store results in session state
            st.session_state.summaries = {
                name: summary for name, summary in zip(file_names, results)
            }

## Display summaries if available
if "summaries" in st.session_state and st.session_state.summaries:
    combined_summary = ""
    for file, summary in st.session_state.summaries.items():
        st.subheader(f"📄 Summary for: {file}")
        st.write(summary)
        st.write("\n" + "=" * 80 + "\n")  # Separator

        combined_summary += f"📄 **{file}**\n{summary}\n\n{'='*80}\n\n"

    # Download all summaries as one file
    st.download_button(
        label="📥 Download All Summaries",
        data=combined_summary,
        file_name="All_PDF_Summaries.txt",
        mime="text/plain",
    )


# Showing Tables
# Function to extract and show tables from the entire PDF
def show_tables(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        table_found = False
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table_idx, table in enumerate(tables):
                if not table or len(table) < 2:
                    continue  # skip empty or invalid tables

                # Fix headers: replace None or empty strings and ensure uniqueness
                raw_headers = table[0]
                headers = []
                for i, col in enumerate(raw_headers):
                    if not col or col.strip() == "":
                        headers.append(f"Column_{i+1}")
                    else:
                        headers.append(col.strip())

                # Ensure headers are unique
                seen = {}
                unique_headers = []
                for h in headers:
                    count = seen.get(h, 0)
                    if count:
                        unique_headers.append(f"{h}_{count}")
                    else:
                        unique_headers.append(h)
                    seen[h] = count + 1

                # Convert to DataFrame
                df = pd.DataFrame(table[1:], columns=unique_headers)

                st.markdown(
                    f"**📊 Table from Page {page_num + 1}, Table {table_idx + 1}:**"
                )
                st.dataframe(df)
                table_found = True

        if not table_found:
            st.info("ℹ️ No tables found in this PDF.")


# Show tables for each uploaded PDF
for file in uploaded_files:
    st.subheader(f"📄 Tables in: {file.name}")
    show_tables(file)


# Evaluation:

if "summaries" in st.session_state and st.session_state.summaries:
    for file in uploaded_files:
        text, tables = extract_text_tables(file)

    reference_text = text  # Replace with actual ref if available

    # Evaluate summary
    with st.spinner("🔄 Processing..."):
        try:
            scores = evaluate_summary(summary, reference_text)
            st.subheader("**📈📋 Evaluation Scores:**")
            st.json(scores)
        except Exception as e:
            st.error(f"Evaluation failed: {e}")

        with st.expander("ℹ️ What do these evaluation scores mean?"):
            st.markdown(
            """
    - **BERTScore**: Focuses on meaning by comparing contextual word embeddings.
    - **F1 Score**: Balances capturing key info (recall) with avoiding irrelevant content (precision).
    - **Recall**: Measures how much important info from the original is present in the summary.
    - **Accuracy**: General correctness score (less relevant for text tasks).
    - **ROUGE-1**: Matches individual words between summaries.
    - **ROUGE-2**: Matches two-word sequences, reflecting phrase overlap.
    - **ROUGE-L**: Considers the longest shared word sequence, showing structural similarity.
    """
        )
