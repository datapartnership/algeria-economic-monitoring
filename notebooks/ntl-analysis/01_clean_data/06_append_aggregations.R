# Append Aggregations

# Appends the individual files created by 05_extract_ntl.R into one dataset per
# unit and time period. Saves as .Rds, .csv, and .dta to:
#   ntl_bm_dir/aggregated/ntl_{unit}_{time_type}.{Rds,csv,dta}

out_dir <- file.path(ntl_bm_dir, "aggregated")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

for(unit in NTL_UNITS){
  for(time_type in NTL_TIME_TYPES){

    files <- file.path(ntl_bm_dir, "aggregated_individual_files", unit, time_type) %>%
      list.files(full.names = TRUE, pattern = "\\.Rds$")

    if(length(files) == 0){
      message("No files for ", unit, " / ", time_type, "; skipping")
      next
    }

    message("Appending: ", unit, " / ", time_type)

    ntl_df <- files %>%
      map_df(readRDS) %>%
      arrange(date)

    #### Cleanup
    # Where the sum is 0 (e.g., no gas flaring locations within the unit), the
    # mean is NaN; set to 0.
    for(mean_var in str_subset(names(ntl_df), "^ntl_.*mean$")){
      sum_var <- str_replace(mean_var, "mean$", "sum")
      if(!(sum_var %in% names(ntl_df))) next

      ntl_df[[mean_var]][ntl_df[[sum_var]] %in% 0] <- 0
    }

    name_root <- paste0("ntl_", unit, "_", time_type)

    saveRDS(ntl_df,   file.path(out_dir, paste0(name_root, ".Rds")))
    write_csv(ntl_df, file.path(out_dir, paste0(name_root, ".csv")))
    write_dta(ntl_df, file.path(out_dir, paste0(name_root, ".dta")))
  }
}
