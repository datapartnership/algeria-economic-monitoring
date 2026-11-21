# Download Black Marble Data

# Downloads VIIRS Black Marble nighttime lights rasters for the country:
#   - Annual  (VNP46A4)
#   - Monthly (VNP46A3), plus the monthly quality flag layer
#   - Daily   (VNP46A2), if DOWNLOAD_NTL_DAILY is TRUE
#
# bm_raster() skips files that already exist in `file_dir`, so this script
# only downloads new dates.

# Setup ------------------------------------------------------------------------
#### Bearer token from NASA
# To get a token, see: https://github.com/worldbank/blackmarbler
nasa_bearer <- read_csv(bearer_path, show_col_types = FALSE) %>%
  pull(token)

#### Region of interest
roi_sf <- load_gadm(0)

#### Directories
raster_dir <- file.path(ntl_bm_dir, "rasters")

for(dir_i in c("annual", "monthly", "monthly_quality", "daily")){
  dir.create(file.path(raster_dir, dir_i), showWarnings = FALSE, recursive = TRUE)
}

# Annual -----------------------------------------------------------------------
bm_raster(roi_sf = roi_sf,
          product_id = "VNP46A4",
          date = ntl_years,
          bearer = nasa_bearer,
          output_location_type = "file",
          file_dir = file.path(raster_dir, "annual"))

# Monthly ----------------------------------------------------------------------
bm_raster(roi_sf = roi_sf,
          product_id = "VNP46A3",
          date = ntl_months,
          bearer = nasa_bearer,
          output_location_type = "file",
          file_dir = file.path(raster_dir, "monthly"))

bm_raster(roi_sf = roi_sf,
          product_id = "VNP46A3",
          date = ntl_months,
          bearer = nasa_bearer,
          variable = "NearNadir_Composite_Snow_Free_Quality",
          output_location_type = "file",
          file_dir = file.path(raster_dir, "monthly_quality"))

# Daily ------------------------------------------------------------------------
if(DOWNLOAD_NTL_DAILY){
  bm_raster(roi_sf = roi_sf,
            product_id = "VNP46A2",
            date = rev(ntl_days),
            bearer = nasa_bearer,
            output_location_type = "file",
            file_dir = file.path(raster_dir, "daily"))
}
