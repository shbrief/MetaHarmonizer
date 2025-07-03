##### This script is using cBioPortal curated_treatment_name as an example:
## Dynamic enum node: "Pharmacologic Substance" (NCIT:C1909)

## Import the sentence transformer and model
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")

## Load the bodysite curation map
import pandas as pd
import os

source_dir = "~/OmicsMLRepo/OmicsMLRepoHarmonizer"
map_url = os.path.join(source_dir, "data/cBioPortal_treatment_name_map.csv")
df = pd.read_csv(map_url)

## Collect all the original values
orig = []
for x in df["original_value"].unique():
    orig.extend(x.split("<;>"))  # separate rows containing multiple values

## Import the comprehensive corpus
outputDir = "~/OmicsMLRepo/OmicsMLRepoHarmonizer/outputs"
fpath = os.path.join(outputDir, "cbio_treatment_name/corpus_from_NCIT:C1909.csv")
corpus = pd.read_csv(fpath, header=None, error_bad_lines=False)
corpus = list(corpus[0])

## Embed the curated results (which would, more generally, be the set of ontology terms of interest)
corpus_embeddings = model.encode(corpus)

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
    max_k = min(len(cos_scores), top_k)
    top_results = torch.topk(cos_scores, k=max_k)

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

## Print `all_results`
results_df = pd.DataFrame(
    dict([(key, pd.Series(value)) for key, value in all_results.items()])
)
results_df

## Save the result
data_dir = "~/OmicsMLRepo/OmicsMLRepoHarmonizer/outputs/cbio_treatment_name"
results_df.to_csv(os.path.join(data_dir, "trt_name_match_from_corpus.csv"))
