from transformers import AutoTokenizer
import pandas as pd

# 1. Load the tokenizer for your Sentence Transformer model
# Replace 'sentence-transformers/all-MiniLM-L6-v2' with your specific model name
model_name = "cambridgeltl/SapBERT-from-PubmedBERT-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Define your input text
df = pd.read_csv("/Users/sehyunoh/OmicsMLRepo/MetaHarmonizer/EDA/data/cbio_dict_for_embed.csv")

# Initialize an empty list to store token counts
token_counts = []

for i in range(len(df)):
    colname = df["ColName"][i]
    
    # 3. Tokenize the input text
    # The `encode` method returns a list of token IDs
    token_ids = tokenizer.encode(df["Description"][i])
    
    # 4. Get the token count by checking the length of the token_ids list
    token_count = len(token_ids)
    token_counts.append(token_count)
    
    print(f"The input text: '{colname}'")
    print(f"The token count is: {token_count}")

# Add the token counts as a new column
df["token_count"] = token_counts
