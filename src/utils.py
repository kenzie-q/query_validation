import os, sys
import json
import re

from fuzzywuzzy import fuzz
from huggingface_hub import hf_hub_download
from llama_cpp import Llama  
import pandas as pd

import config

def get_raw_dataset():
    '''Read in parquet files and join into initial dataset.
    Inputs:
        None
    Return:
        df_examples_products (pd.DataFrame): Dataframe of product information
    '''
    df_examples = pd.read_parquet(config.path_ds_examples)
    df_products = pd.read_parquet(config.path_ds_product)

    df_examples_products = pd.merge(
    df_examples,
    df_products,
    how='left',
    left_on=['product_locale','product_id'],
    right_on=['product_locale', 'product_id']
    )
    return df_examples_products

def filter_data(df):
    '''Apply query and label filters to dataset.
    Input: 
        df(pd.DataFrame): Unfiltered dataset
    Return:
        df(pd.DataFrame): filtered dataset
    '''
    valid_query = ['aa batteries 100 pack', 'kodak photo paper 8.5 x 11 glossy',  'dewalt 8v max cordless screwdriver kit, gyroscopic']
    valid_esci_label = 'E'
    df = df[(df.esci_label == valid_esci_label) & (df['query'].isin(valid_query))]
    return df

def get_gguf_file():
    '''Download GGUF model file from huggingface. 
    Input: 
        None
    Return:
        model_path (str): file path to model file
    '''
    model_name="TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    model_file="mistral-7b-instruct-v0.2.Q3_K_M.gguf"
    model_path = hf_hub_download(model_name, filename=model_file)
    return model_path

def get_llm():
    '''Load Llama model
    Inputs: 
        None
    Return:
        llm (Llama): Large Language Model
    '''
    llm = Llama(model_path=config.model_path, n_ctx=2048)
    return llm

def extract_product_attributes(llm, text):
    '''Query LLM to extract attributes
    Input: 
        llm (Llama): Large Language Model
        text (str): query/product title from dataset
    Return:
        response(str): string with llm response
    '''
   
    prompt = f'''
    Extract the key product attributes from the given product title. 
    Return the extracted attributes in a structured JSON format.

    Product title: '{text}'
    JSON Output:
    '''

    response = llm(prompt, max_tokens=100, temperature=0.2, stop=['}'])
    return (response['choices'][0]['text']+'}')

def extract_query_attributes(llm, text, keys):
    prompt = f'''
    Extract the values corresponding to the given attribute keys from the text. 

    Follow these strict rules:

    1. **Brand Identification**:
    - Extract a value for "Brand" **only** if the text explicitly contains a known manufacturer or company name.
    - **Do not assume or infer a brand** if none is mentioned.
    - If the product name resembles a brand but isn't a real one, do not classify it as a brand.

    2. **Product Type vs. Other Attributes**:
    - Identify product-related attributes (e.g., "Battery_Type", "Material", "Size") correctly.
    - If an attribute appears in the text, extract it **exactly as stated**.
    - **Do not confuse the product name with its brand**  (e.g., "AA Batteries" → "AA" is the battery type, NOT the brand).

    3. **Numerical Attributes**:
    - Extract numbers only when they indicate quantities (e.g., "Count", "Capacity", "Weight").
    - Do **not** infer missing numbers; only use explicitly mentioned values.

    4. **Strict No-Hallucination Policy**:
    - **Only return attributes explicitly stated in the text**.
    - **Do not infer missing values** (e.g., do not assume default voltages, materials, or sizes).
    - If an attribute is **not present**, exclude it from the output.
    
      Return the output in JSON format.

    Keys: {keys}
    Text: '{text}'
    JSON Output:
    '''

    response = llm(prompt, max_tokens=100, temperature=0.2, stop=['}'])
    return (response['choices'][0]['text']+'}')

def get_corrected_query(llm, corrected_query_dict):
    '''Query LLM to construct a correct query
    Input: 
        llm (Llama): Large Language Model
        correct_query_dict (dict): dictionary with correct attributes for the product
    Return:
        response(str): string with llm response
    '''
    prompt = f'''
    Generate a concise product search query from the following product attributes.

    Corrected Query Dictionary:
    {corrected_query_dict}

    Output only the search query: 
    '''

    response = llm(prompt, max_tokens=30, temperature=0.2)
    return response['choices'][0]['text'].strip().split('\n')[0].strip("'").strip('"')

def compare_jsons(llm, query_json, product_json):
    '''Query LLM to construct a correct query
    Input: 
        llm (Llama): Large Language Model
        correct_query_dict (dict): dictionary with correct attributes for the product
    Return:
        response(str): string with llm response
    '''
    prompt =f'''
        Compare the following JSON objects and determine if the `product_json` satisfies the `query_json`.

        ## **Rules for Satisfaction:**
        1. All attributes in `query_json` **must exist** in `product_json` and must **match exactly**.
        - **Text values** should be compared **case-insensitively**.
        - **Numerical values** should be compared **numerically** (convert strings to numbers if necessary).
        2. If **any** value in `query_json` differs from `product_json`, **"satisfied" must be false**.
        3. If `"satisfied": false`, **correct `fixed_query`** by replacing the mismatched values in `query_json` with those from `product_json`.

        ### Output Format:
        Return a JSON object with the following structure:
        {{
        "satisfied": true or false,
        "fixed_query": {{ ... }}  
        }}
        - If `"satisfied": true`, `fixed_query` should be identical to `query_json`.
        - If `"satisfied": false`, fixed_query must contain all original attributes, but mismatched ones replaced with product_json values.
        query_json = {query_json}
        product_json = {product_json}
        response:
        '''

    response = llm(prompt, max_tokens=100, temperature=0.2, stop=['}'])
    return (response['choices'][0]['text']+'}}')

def json_parse(llm_output):
    '''Attempts to extract and parse valid JSON from llm output
    Input: 
        llm_output (str): LLM output 
    Return 
        (dict):dictionary of key value pairs
    '''
    try:
        parsed_json = json.loads(llm_output)
        return parsed_json
    except json.JSONDecodeError:
        try: 
            parsed_json = json.loads(llm_output+'}')
            return parsed_json
        except json.JSONDecodeError:
            print('Error: Invalid JSON response from LLaMA.')
            return {}


def main(df):
    '''
    Run LLM against query-product pairings to verify label accuacy and provide
    new query if query is incorrect. Results df written out data directory and 
    returned. 
    Input: 
        df (pd.DataFrame): DataFrame of query-product pairings
    Return:
        df_results (pd.DataFrame): DataFrame of LLM results
    '''
    results = []
    llm = get_llm()  

    for idx, row in df.iterrows():
        product_attributes = json_parse(extract_product_attributes(llm, row['product_title']))
        query_attributes = json_parse(extract_query_attributes(llm, row['query'], list(product_attributes.keys())))
        print(query_attributes)
        print(product_attributes)
        response= json_parse(compare_jsons(llm, query_attributes, product_attributes))
        if response['satisfied'] == True:
             results.append([row['query_id'], row['product_id'], response['satisfied'], row['query']])
        else:
            corrected_query = get_corrected_query(llm, response['fixed_query'])
            results.append([row['query_id'], row['product_id'], response['satisfied'], corrected_query])
        
        print(f'{len(results)} of {len(df)}')
    df_results = pd.DataFrame(results, columns=['query_id', 'product_id', 'is_correct', 'corrected_query'])
    #writeout df
    df_results.to_csv(config.output_path, index=False)
    return df_results



### ARCHIVE

# def determine_query_product_satisfaction(llm, query, product_title):
#     prompt = f'''
#     You are a product verification assistant. Answer concisely without repeating the prompt. Follow these instructions: 
#         - If **ALL query specifications** (size, color, quantity, brand, features) match, respond: Yes
#         - If **ANY detail is missing or different**, correct the query and respond EXACTLY as: No Corrected Query: [New Query]
#         - Respond ONLY with Yes or No. Corrected Query: [new query].
        
#         ### Example 1:
#         Query: "wireless headphones"
#         Product: "Bluetooth Wireless Headphones, Noise Cancelling"
#         Response:
#         Yes

#         ### Example 2:
#         Query: "kodak photo paper 8.5 x 11 glossy"
#         Product: "Kodak Premium Photo Paper, 8.5 x 11, Matte Finish"
#         Response:
#         No Corrected Query: kodak photo paper 8.5 x 11 matte

#         ### Now, analyze this case:
#         Query: "{query}"
#         Product: "{product_title}"
#         Response:
#     '''
#     response = llm(prompt, max_tokens=30, temperature=0.2 )
#     result = response['choices'][0]['text']
#     if 'yes' in result.lower():
#         return True, query
#     else:
#         return False, result.split('Query:')[-1]

# def is_mismatched(attr1, attr2, threshold):
#     '''Compare two attributes using fuzzy matching and token similarity.
#     Inputs: 
#         attr1 (str): string attributes 
#         attr2 (str): string attributes 
#         threshold (int): threshold for fuzzy match ratio
#     Return:
#         (bool): True/False to indicate if attributes are mismatched or not
#     '''
#     #Token-based similarity - multi-word comparisons
#     tokens1 = set(attr1.split())
#     tokens2 = set(attr2.split())
#     all_tokens = set(f'{attr1} {attr2}'.split())
#     # percentage over ovelap = overlap/unique num token
#     overlap = len(tokens1 & tokens2) / len(all_tokens)
#     print(attr1, fuzz.ratio(attr1, attr2), overlap)
#     # Exact match
#     if attr1 == attr2:
#         return False
#     elif fuzz.ratio(attr1, attr2) > threshold[0]:
#         return False
#     elif overlap >=  threshold[1]:
#         return False
#     else:
#         return True


# def compare_attributes(query_id, query_attrs, product_attrs):
#     '''Compare attributes between query and product dictionaries and determine if they match or not.
#     Inputs:
#         query_attrs (dict): dictionary of query attributes
#         product_attrs (dict): dictionary of product attributes
#     Return:
#         (bool): True/False to indicate if the query satisfies the product
#         corrected_query (dict): dictionary with the correct attributes 
#     '''
#     corrected_query = query_attrs.copy()
#     mismatched_log = []
    
#     for key, value in query_attrs.items():    
#         # do both attribures exist in query and product dict? 
#         if (key in product_attrs):
#             # do we have non null values for our values 
#             if (product_attrs[key] != None) & (value != None): 
#                 # determine if mismatched attributes 
#                 if query_id in config.match_threshold:
#                     mismatched = is_mismatched(str(product_attrs[key]).lower(), str(value).lower(), config.match_threshold[query_id])
#                 else: 
#                     mismatched = is_mismatched(str(product_attrs[key]).lower(), str(value).lower(), [50, .5])
#                 if mismatched: 
#                     corrected_query[key] = product_attrs[key]
#             # if product info has more than query - keep mismatched false i.e query value would be None
#             # if product info doenst have specifics from query - keep mismatched false i.e product value would be none
#             else:
#                 mismatched = False
#         # query key not in product - keep mismatchd false 
#         else: 
#             mismatched = False
#         mismatched_log.append(mismatched)
#     # Contradiction found, return corrected query dict 
#     if True in mismatched_log:
#         return False, corrected_query
#     return True, {}
# def extract_attributes(llm, text):
#     '''Query LLM to extract attributes
#     Input: 
#         llm (Llama): Large Language Model
#         text (str): query/product title from dataset
#     Return:
#         response(str): string with llm response
#     '''
   
#     prompt = f'''
#     Extract the following attributes **only if they are explicitly mentioned** in the product description.  

#     **Attributes to Extract:**
#     - "brand" (The well-known **manufacturer name** or **brand name**, but if no brand is mentioned, return `null`. Examples: "Samsung", "Canon", "Kodak", etc.)
#     - **Brands typically appear at the beginning** of the product name or may be a recognized word or phrase associated with the product.  
#     - "product_type" (**Extract exactly as stated**, including key details like model name, power ratings, product variations or key features. Avoid including non-essential descriptors or brand names in this field).
#     - "size" (Physical dimensions or size of product, e.g., "8.5 x 11", "1/4-Inch").
#     - "quantity" (Number of items in the pack or bundle, e.g., "100", "60").
#     - "finish" (Surface type or material finish, e.g., "Glossy", "Matte").

#     ### **Rules:**
#     1. **If an attribute is not mentioned, return `null` explicitly.** Do not assume or generate missing values.
#     2. **Do not generate or infer a brand if it is not explicitly mentioned in the description**. The brand should only be identified if it is clearly present.
#     3. **Extract details exactly as they appear in the text**—do **not** shorten, infer, or rephrase the text. Ensure the **brand** is separated from the **product type**.
#     4. **Preserve important product identifiers** (e.g., `"AA Batteries"`, not just `"Batteries"`).
#     5. **Fix inch notation**: If `"` is used for inches (e.g., `8.5" x 11"`), **remove `"`** and return `8.5 x 11`.
#     6. **Output must be valid JSON with no extra text, explanations, or markdown.**

#     Text: '{text}'
#     JSON Output:
#     '''

#     response = llm(prompt, max_tokens=100, temperature=0.2, stop=['}'])
#     return (response['choices'][0]['text']+'}')

# def main(df):
#     '''
#     Run LLM against query-product pairings to verify label accuacy and provide
#     new query if query is incorrect. Results df written out data directory and 
#     returned. 
#     Input: 
#         df (pd.DataFrame): DataFrame of query-product pairings
#     Return:
#         df_results (pd.DataFrame): DataFrame of LLM results
#     '''
#     results = []
#     llm = get_llm()  

#     #do all query related attribute extractions and refrence to maintian consistent comparision, & reduce runtime 
#     df_query = df[['query_id', 'query']].drop_duplicates()
#     query_attributes_dict = {row['query_id']: json_parse(extract_attributes(llm, row['query'])) for _, row in df_query.iterrows()}
#     print(query_attributes_dict)
#     for idx, row in df.iterrows():
#         product_attributes = json_parse(extract_attributes(llm, row['product_title']))
#         is_correct, corrected_query = compare_attributes(row['query_id'], query_attributes_dict[row['query_id']], product_attributes)
#         if is_correct:
#             corrected_query = row['query']
#         else: 
#             corrected_query = get_corrected_query(llm, corrected_query)
#         results.append([row['query_id'], row['product_id'], is_correct, corrected_query])
#         print(product_attributes)
#         print(is_correct, corrected_query)
#         print(f'{len(results)} of {len(df)}')
#     df_results = pd.DataFrame(results, columns=['query_id', 'product_id', 'is_correct', 'corrected_query'])
#     #writeout df
#     df_results.to_csv(config.output_path, index=False)
#     return df_results