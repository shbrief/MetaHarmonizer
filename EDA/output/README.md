# Configuration

## Default parameters
```
OUTPUT_DIR = "~/OmicsMLRepo/MetaHarmonizer/EDA/output"
VALUE_DICT_PATH = os.getenv("FIELD_VALUE_JSON") or "data/schema/field_value_dict.json"
```

### Alias options
```
## `cbio_alias`
ALIAS_DICT_PATH = os.getenv("ALIAS_DICT_PATH") or "data/schema/curated_fields_source_latest_with_flags.csv" 
## `llm_alias`
ALIAS_DICT_PATH = os.getenv("ALIAS_DICT_PATH") or "~/OmicsMLRepo/MetaHarmonizer/EDA/data/heterogeneous_attribute_mapping_ver1.csv"
## `no_alias`
ALIAS_DICT_PATH = ""
```



# Script

## cBioPortal Train Data
```{python eval=FALSE}
from src.models.schema_mapper import SchemaMapEngine
file = "~/OmicsMLRepo/MetaHarmonizer/EDA/data/test.csv"
engine = SchemaMapEngine(file, mode="manual", top_k=5)
results = engine.run_schema_mapping()
```

## New cBioPortal Test Data
```{python eval=FALSE}
from src.models.schema_mapper import SchemaMapEngine
file = "~/OmicsMLRepo/MetaHarmonizer_Test_Data/source/combined_origin_data.csv"
engine = SchemaMapEngine(file, mode="manual", top_k=5)
results = engine.run_schema_mapping()
```

## Validation
```{python eval=FALSE}
from evaluation import schema_mapping_evaluation
metrics = schema_mapping_evaluation.compute_accuracy(
     pred_file="~/OmicsMLRepo/MetaHarmonizer/EDA/output/test_auto.csv",    # output from schema mapping
     truth_file="~/OmicsMLRepo/MetaHarmonizer/data/schema_mapping_eval/truth.csv",
     top_k=5,
     save_eval=True,
     out_dir="~/OmicsMLRepo/MetaHarmonizer/EDA/output"
```


# Files
### `train_*`

Input = `~/OmicsMLRepo/MetaHarmonizer/EDA/data/test.csv`
Gold-standard = `~/OmicsMLRepo/MetaHarmonizer/data/schema_mapping_eval/truth.csv`
CURATED_DICT_PATH = `data/schema/curated_fields.csv`


#### `test_*`

pred_file = `~/OmicsMLRepo/MetaHarmonizer_Test_Data/source/combined_origin_dadta.csv`   # subjected to schema mapper
truth_file = `~/OmicsMLRepo/MetaHarmonizer_Test_Data/source/truth_schema_mapper.csv`    # manually-built gold standard
CURATED_DICT_PATH = `~/OmicsMLRepo/MetaHarmonizer_Test_Data/source/curated_dict.csv`    # data schema (`field_name` and `is_numeric_field`)

