import os, sys

# Data paths
base_path = '../../esci-data/shopping_queries_dataset'
path_ds_examples = os.path.join(base_path, 'shopping_queries_dataset_examples.parquet')
path_ds_product = os.path.join(base_path, 'shopping_queries_dataset_products.parquet')
output_path = '../data/llm_results.csv'
results_log = '../data/results_log.csv'

# model_path = '/Users/mckenziequinn/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-GGUF/snapshots/b4e04e128f421c93a5f1e34ac4d7ca9b0af47b80/llama-2-7b.Q4_K_M.gguf'
model_path ='/Users/mckenziequinn/.cache/huggingface/hub/models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/snapshots/3a6fbf4a41a1d52e415a4958cde6856d34b2db93/mistral-7b-instruct-v0.2.Q3_K_M.gguf'
match_threshold = {6014: [40, 0.1], 
              32814:[80, 0.43], 
              58953: [24, 0.5]}
