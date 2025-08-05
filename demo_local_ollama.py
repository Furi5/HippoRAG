import os
from typing import List
import json
import argparse
import logging
import random
import pandas as pd
from src.hipporag import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig



def main():
     # Set temperature for sampling
    with open('QA.json', 'r') as f:
        qa_data = json.load(f)
    for index in range(5):
        output_dict = {}
        temperature = random.uniform(0, 1)
        
        # save_dir = 'output_RAG'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
        # llm_model_name = '/home/mindrank/fuli/DS-32B'  # Any OpenAI model name
        # embedding_model_name = 'nomic-embed-text:latest'  # Embedding model name (NV-Embed, GritLM or Contriever for now)

        # Startup a HippoRAG instance
        
        BaseConfig.llm_name = "gemma3:27b"
        BaseConfig.llm_base_url = "http://localhost:11434"
        BaseConfig.temperature = temperature
        BaseConfig.embedding_name = "nomic-embed-text:latest"
        BaseConfig.embedding_base_url = "http://localhost:11434"
        
        hipporag = HippoRAG(save_dir='output_RAG',
                            llm_model_name="gemma3:27b",
                            embedding_model_name="nomic-embed-text:latest",
                            embedding_base_url="http://localhost:11434",
                            llm_base_url="http://localhost:11434/v1",
                            global_config=BaseConfig(temperature=temperature)
                            )
        # hipporag = HippoRAG()

        
        # Run indexing
        # df = pd.read_csv("pmid_article.csv")
        # docs = list(df['article'])
        # hipporag.index(docs=docs)
    

        for group, qa_dict in qa_data.items():
            output_dict[group] = {}
            for key, qa in qa_dict.items():
                queries_solutions, all_response_message, all_metadata = hipporag.rag_qa(
                    queries=[f"Question: {qa['question']}\n Option: {qa['options']} please answer the question based on the options provided."],
                    )
                for query in queries_solutions:
                    retrieved_docs = query.docs
                    options = query.answer
                    answer = all_response_message
                    pmid = [f"PMID{doc.split('|t|')[0]}" for doc in retrieved_docs]
                output_dict[group][key] = {
                    "question": qa['question'],
                    "options": qa['options'],
                    "hipporag2_options": options,
                    "answer": answer,
                    "retrieved_docs": pmid[:10],
                }

        with open(f'hipporag2_{index}.json', 'w') as f:
            json.dump(output_dict, f, indent=4)
        



if __name__ == "__main__":
    main()
