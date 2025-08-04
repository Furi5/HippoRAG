import os
from typing import List
import json
import argparse
import logging
import pandas as pd
from src.hipporag import HippoRAG

def main():
    df = pd.read_csv("pmid_article.csv")

    docs = list(df['article'])[:40]

    save_dir = 'output_RAG'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_model_name = 'gemma3:27b'  # Any OpenAI model name
    embedding_model_name = 'nomic-embed-text:latest'  # Embedding model name (NV-Embed, GritLM or Contriever for now)

    # Startup a HippoRAG instance
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        embedding_model_name=embedding_model_name,
                        embedding_base_url="http://localhost:11434",
                        llm_base_url="http://localhost:11434/v1")

    # Run indexing
    hipporag.index(docs=docs)
    
    
    with open('QA.json', 'r') as f:
        qa_data = json.load(f)
    for group, qa_dict in qa_data.items():
        for key, qa in qa_dict.items():

            print(f"question: {qa['question']}, options: {qa['options']}")
            
            queries_solutions, all_response_message, all_metadata = hipporag.rag_qa(
                queries=[f"{qa['question']} {qa['options']}"],
                )
            for query in queries_solutions:
                retrieved_docs = query.docs
                options = query.answer
                answer = all_response_message
                print(f"Retrieved documents: {retrieved_docs}")
                print(f"Options: {options}")
                print(f"Answer: {answer}")
            break
        break


if __name__ == "__main__":
    main()
