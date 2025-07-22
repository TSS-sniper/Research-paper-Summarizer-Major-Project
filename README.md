# Research Paper Summarizer with Real-time Evaluation 
## Introduction:
* Built an LLM-powered application that can summarize any research paper into a concise summary, extract tables from the research paper, and give a real-time evaluation score for the generated summary.
* Utilized groq’s inference for Meta’s LLaMA 3–8B to generate summaries, evaluated them using BERTScore (F1) and ROUGE, and extracted tables with pdfplumber and pandas.
* Achieved an average BERTScore (F1) of 0.87 across a benchmark of 10+ scientific papers, reflecting strong semantic alignment with original content.
