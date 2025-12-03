import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# Load preprocessed clinical text
df = pd.read_csv("/Users/sehyunoh/OmicsMLRepo/MetaHarmonizer/EDA/data/cbio_dict_for_embed.csv")

# Load the Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
model = AutoModel.from_pretrained("yikuan8/Clinical-Longformer")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # Set to evaluation mode

texts = df["Description"]

# Process all texts and collect embeddings
all_embeddings = []

for text in texts:
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=4096)
    
    # Move inputs to the same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get embeddings (no gradient computation needed)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        cls_embedding = last_hidden_states[:, 0, :]  # Shape: (1, 768)
    
    # Move to CPU and convert to numpy, then append
    all_embeddings.append(cls_embedding.cpu().numpy())

# Stack all embeddings into a single array (39 x 768)
import numpy as np
embeddings_matrix = np.vstack(all_embeddings)

print(f"Embeddings shape: {embeddings_matrix.shape}")  # Should be (39, 768)

# Optional: Convert to DataFrame for easier handling
embeddings_df = pd.DataFrame(embeddings_matrix)
embeddings_df.to_csv("/Users/sehyunoh/OmicsMLRepo/MetaHarmonizer/EDA/data/target_colname_embeddings_clinical_lf.csv", index=False)

