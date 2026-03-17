# classify citations in the references section of a pdf into wether the distance catalogue is used or not

import functions
from functions import *
from transformers import pipeline
import os
import csv 
import torch
import random

random.seed(123) 

#from gliclass import GLiClassModel, ZeroShotClassificationPipeline
#from transformers import AutoTokenizer

# configuration

## define category labels for classification
#classification_categories = [
#    "The authors adopt or use distance values from Bailer-Jones et al. in their analysis, tables or figures.",
#    "The authors cite Bailer-Jones et al. for background, methodology, a passing mention or other non-distance reasons."
#]

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


# Zero-shot classifier using a pre-trained model
# predicts a class that wasn't seen by the model during training, based on the description of the classes and the input text
# useful if amount of labeled data scarce or if we want to classify into very specific categories that are not covered by general models


#classifier = pipeline(
#    "zero-shot-classification",
#    model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
#    #model="microsoft/deberta-large-mnli",
#    device=-1   # CPU
#)
#
#def classify_context(context):
#      
#    result = classifier(
#        context,
#        classification_categories,
#        multi_label=False
#    )
#
#    return result

# for more detailed classification, we can use a text generation model to directly output the label based on the prompt

classifier = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype=torch.float16,
    device_map="auto"
)

def classify_context(context):

    prompt = f"""
You are classifying the function of a scientific citation of Bailer-Jones et al.

Decide whether the citation context below shows that the authors adopt or use numerical distance values from Bailer-Jones et al., or if the citation is used only for background or other reasons.

IMPORTANT:
- Only classify as "uses_distance_catalogue" if the authors clearly adopt or compute using distance values from Bailer-Jones et al. or use distance values in a table or figure.
- If ambiguous, classify as "does_not_use_distance_catalogue".

Citation context:
\"\"\"
{context}
\"\"\"

Respond with exactly one label:
uses_distance_catalogue
or
does_not_use_distance_catalogue
"""

    output = classifier(
        prompt,
        max_new_tokens=10,
        do_sample=False,
        temperature=0.0
    )

    response = output[0]["generated_text"].split(prompt)[-1].strip()

    if "uses_distance_catalogue" in response:
        return "uses_distance_catalogue"
    else:
        return "does_not_use_distance_catalogue"


#model = GLiClassModel.from_pretrained("knowledgator/gliclass-modern-base-v2.0-init")
#tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-modern-base-v2.0-init", add_prefix_space=True)
#pipeline = ZeroShotClassificationPipeline(model, tokenizer, classification_type='multi-label', device='cuda:0')


pdfs_all = [f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")]

pdfs = pdfs_all
#pdfs = random.sample(pdfs_all, 100)     # limit to 100 random samples for testing

with open("output_textgen_2cat.csv", mode="w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    #writer.writerow(["bibcode", "citation_context", "predicted_label","distance_used_score", "distance_not_used_score","used_finetune_model"])
    writer.writerow(["bibcode", "citation_context_no","citation_context","target_reference", "predicted_label"])

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

            predicted_label = classify_context(context)

            writer.writerow([filename.replace(".pdf", ""), j+1, context, ref_keys_found[j], predicted_label])
            
            print(" ")
            print("Prediction:", predicted_label)

            #results = pipeline(context, classification_categories, threshold=0.0)[0]
            #
            #print("Classification results:")
            #for res in results:
            #    print(f"{res['label']}: {res['score']:.4f}")

            print("-" * 60)





