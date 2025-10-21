##### This script is using cBioPortal curated_treatment_name as an example:

## Import the sentence transformer and model
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")

## Load the bodysite curation map
import pandas as pd
import os

source_dir = "~/OmicsMLRepo/OmicsMLRepoHarmonizer"
map_url = os.path.join(source_dir, "data/cBioPortal_treatment_name_map.csv")
df = pd.read_csv(map_url)
df = df[["original_value", "curated_ontology", "curated_ontology_term_id"]]
orig_cura_map = dict(zip(df["original_value"], df["curated_ontology"]))

## Collect all the original and curated values
orig = []
for x in df["original_value"].unique():
    orig.extend(x.split("<;>"))  # separate rows containing multiple values

cura = []
for x in df["curated_ontology"].unique():
    cura.extend(x.split(";"))

## Embed the curated results (which would, more generally, be the set of ontology terms of interest)
cura_embed = model.encode(cura)
corpus = cura
corpus_embeddings = cura_embed

from sentence_transformers import util
import torch

top_k = 5  # the number of top most similar vectors from the corpus

queries = orig  # query entired original values
all_results = {
    "original_value": [],
    "curated_ontology": [],
    "match_level": [],
    "top1_match": [],
    "top1_score": [],
    "top2_match": [],
    "top2_score": [],
    "top3_match": [],
    "top3_score": [],
    "top4_match": [],
    "top4_score": [],
    "top5_match": [],
    "top5_score": [],
}

for i in range(len(queries)):

    print(i, "out of", len(queries))
    query = queries[i]
    ## Embed each uncurated term
    query_embedding = model.encode(query, convert_to_tensor=True)

    ## Use cosine-similarity and torch.topk to find the high-scored vectors
    ## Seach in the corpus embeddings
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    ## Look up curated value
    curated_value = orig_cura_map[query]

    # print("\n======================\n")
    # print("Query:", query)
    # print("Curated Term:", curated_value)
    # print("Top 5 most similar sentences in corpus:")
    score = top_results[0]
    idx = top_results[1]
    results = zip(score, idx)
    result_labels = []

    for index, (score, idx) in enumerate(results):
        match_rank = index + 1
        matched_term = corpus[idx]
        all_results[f"top{match_rank}_match"].append(matched_term)
        all_results[f"top{match_rank}_score"].append("{:.4f}".format(score))

        # print(f"Index: {index}, match: {matched_term}, score: {score}")

        result = matched_term + " (Score: {:.4f})".format(score)
        result_labels.append(matched_term)

    match_level = 99
    if curated_value in result_labels:
        match_level = result_labels.index(curated_value) + 1
        # print(f"Top {match_level} match!")

    ## Add results back to df via lookup
    all_results["original_value"].append(query)
    all_results["curated_ontology"].append(curated_value)
    all_results["match_level"].append(match_level)

    """
    # Alternatively, we can also use util.semantic_search to perform cosine similarty + topk
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
    hits = hits[0]      # Get the hits for the first query
    for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
    """

## Print `all_results`
results_df = pd.DataFrame.from_dict(all_results)
results_df

## Save the result
data_dir = "~/OmicsMLRepo/OmicsMLRepoHarmonizer/outputs/cbio_treatment_name"
results_df.to_csv(os.path.join(data_dir, "trt_name_match_from_curated.csv"))
