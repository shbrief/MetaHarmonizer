## Analysis-ready formatting ----------------------
library(OmicsMLRepoCuration)
ms_fpath <- "data/cBioPortal_merging_schema.csv"
ms <- readr::read_csv(ms_fpath)
cols <- c("curated_field", "original_field_num", "original_field")
ms_long <- OmicsMLRepoR::getLongMetaTb(ms[cols],
                                       targetCols = c("original_field"), 
                                       delim = ";")
readr::write_csv(x, "outputs/cbio_merging_schema/merging_schema_cols.csv")
