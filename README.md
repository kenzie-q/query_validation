# Query Validation 
By: McKenzie Quinn 

## Overview
The goal is to verify the accuracy of the query product matching labels using a Large Language Model (LLM) to ensure correct classification. 

Our dataset contains labeled query-product pairs, where the label "E" (Exact Match) indicates that a product completely satisfies a query. However, some cases are incorrectly labeled. The task is to:
* Identify misapplied "E" labels for three specific queries.
* Use an LLM to verify product-query matches and reformulate incorrect queries.
* Generate structured output with the corrected labels.


## Getting Started 
The data referenced in this project is from [Amazon esci-data](https://github.com/amazon-science/esci-data/blob/main/README.md). Clone this repo to reference the data accordingly. If errors occur while reading the parquet files, manually download and replace the files. 
1. Clone repository 
2. Create environment `conda env create -f environment.yml`
3. Acitvate environment `conda activate llm`


## Approach 

Begin by using a local LLM (Mistral) to extract structured product attributes from product titles, then product queries. The LLM is instructed to output product attributes in a JSON format. 

Then we ask another LLM to compare the two attribute dictionaries to determine if the product dictionary satifies the query dictionary. Respond with a JSON following this format: `{"satisfied": true or false, "fixed_query": {{ ... }} }`. If satisfied is true, return the original query, otherwise, send the `fixed_query` dictionary to an LLM to rephrase the query.

**Fine-tuning to this system needs to be done through prompt engineering**

LLM: `mistral-7b-instruct-v0.2.Q3_K_M.gguf`

## Repo Layout 
```
│   README.md
│   .gitignore   
|
└───data
│   │   llm_results.csv     - results from LLM execution
│   
└───notebooks
│   │   execution.ipynb     - notebook to execute task         
│
└───src
    │   config.py           - configuration details
    │   utils.py            - script containing utility functions
```
