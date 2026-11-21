# Download GADM Boundaries

# Downloads GADM administrative boundaries (levels 0-2) for the country.
# geodata::gadm() skips the download if the file already exists in `path`.

dir.create(gadm_dir, showWarnings = FALSE, recursive = TRUE)

for(level_i in 0:2){
  gadm(iso3_code, level = level_i, path = gadm_dir)
}
