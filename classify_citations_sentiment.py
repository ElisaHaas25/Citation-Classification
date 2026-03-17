# classify citations in the references section of a pdf into wether the distance catalogue is used, the methodology is used or the citation is only for background or other reasons

import functions
from functions import *
from transformers import pipeline
import os
import csv 
import torch
import random

random.seed(123) # for reproducibility
# configuration

pdf_directory = "ads_papers" 

# define target references with regex patterns to identify them in the reference section
target_references = {
    "bj_2015": [
        r"Bailer[- ]Jones\s+et\s+al\.\s*,?\s*2015",
        r"Bailer[- ]Jones\s+et\s+al\.\s*\(\s*(2015)\s*\)",
        r"Bailer[- ]Jones\s*\(\s*2015\s*\)",
        r"Bailer[- ]Jones\s*\s*2015\s*"
    ],
    "bj_2018": [
        r"Bailer[- ]Jones\s+et\s+al\.\s*,?\s*2018",
        r"Bailer[- ]Jones\s+et\s+al\.\s*\(\s*(2018)\s*\)",
        r"Bailer[- ]Jones\s*\(\s*2018\s*\)",
        r"Bailer[- ]Jones\s*\s*2018\s*"
    ],
    "bj_2021": [
        r"Bailer[- ]Jones\s+et\s+al\.\s*,?\s*2021",
        r"Bailer[- ]Jones\s+et\s+al\.\s*\(\s*(2021)\s*\)",
        r"Bailer[- ]Jones\s*\(\s*2021\s*\)",
        r"Bailer[- ]Jones\s*\s*2021\s*"
    ],
    "bj_2023": [
        r"Bailer[- ]Jones\s+et\s+al\.\s*,?\s*2023",
        r"Bailer[- ]Jones\s+et\s+al\.\s*\(\s*(2023)\s*\)",
        r"Bailer[- ]Jones\s*\(\s*2023\s*\)",
        r"Bailer[- ]Jones\s*\s*2023\s*"
    ],
    "astraatmadja_2016": [
        r"Astraatmadja\s+and\s+Bailer[- ]Jones\s*\(\s*2016\s*\)",
        r"Astraatmadja\s*&\s*Bailer[- ]Jones\s*\(\s*2016\s*\)",
        r"Astraatmadja\s*&\s*Bailer[- ]Jones\s+2016",
        r"Astraatmadja\s+et\s+al\.\s*,?\s*2016"
    ]
}
# 

sentiment_pipeline = pipeline("sentiment-analysis",
                              #model = "CardiffNLP/twitter-roberta-base-sentiment-latest",
                              model='ProsusAI/finbert')



pdfs_all = [f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")]
pdfs = random.sample(pdfs_all, 100)   #pdfs_all[40:50]   # limit to 100 for testing

with open("output_sentiment_test.csv", mode="w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    
    writer.writerow(["bibcode", "citation_context_no","citation_context","target_reference", "sentiment_result", "score"])

    for i in range(len(pdfs)):  

        filename = pdfs[i]

        print(f"Processing {i+1}/{len(pdfs)}: {filename}")
        print("")
        pdf_path = os.path.join(pdf_directory, filename)
        
        # 1. Extract text
        
        text = extract_text(pdf_path)
        text_cleaned = clean_pdf_text(text)
        
        # 2. Split body and references
        
        body_text, ref_section = split_body_and_references(text_cleaned)
        
        # 3. Build numeric citation map
        
        numeric_map = build_numeric_reference_map(ref_section, target_references)

        # 4. extract each context and cited paper (target_reference) for the pdf

        contexts = []
        ref_keys_found = []

        for ref_key in target_references.keys():

            numeric_numbers = [
                num for num, info in numeric_map.items()
                if info["ref_key"] == ref_key
            ]

            # 5. Extract citation contexts (BODY ONLY)

            context = extract_citation_contexts(
                body_text = body_text,
                author_patterns = target_references[ref_key],
                numeric_numbers = numeric_numbers
            )

            contexts.append(context)

            for i in range(len(context)):
                ref_keys_found.append(ref_key)

        contexts = [item for sublist in contexts if sublist for item in sublist]

        print(f"Found {len(contexts)} citation contexts:\n")
        print(ref_keys_found)

        # 6. Classify each context
        
        for j in range(len(contexts)):

            context = contexts[j]

            print(f"[Context {j+1}]")
            print("")
            print(context)

            result = sentiment_pipeline(context)[0]

            # Extract scores into dictionary

            
            writer.writerow([filename.replace(".pdf", ""), j+1,context, ref_keys_found[j], result['label'], result['score']])

            print(" ")

            print("Prediction:", result)

            print("-" * 60)




