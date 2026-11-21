# Extract Nighttime Lights

# Aggregates Black Marble rasters to each unit of analysis (NTL_UNITS) and time
# period (NTL_TIME_TYPES). Saves one file per unit/time period/date to:
#   ntl_bm_dir/aggregated_individual_files/{unit}/{time_type}/ntl_{date}.Rds
#
# Files that already exist are skipped, so only new dates are processed. To
# re-extract everything (e.g., after changing variables), delete the
# `aggregated_individual_files` folder.
#
# Variables:
#   ntl_sum, ntl_mean          : Sum / mean of NTL
#   ntl_sum_l{x}na             : Sum of NTL, setting pixels below x to 0
#   ntl_gf_{5,10}km_{sum,mean} : NTL within 5/10km of gas flaring locations
#   ntl_nogf_{5,10}km_{sum,mean}: NTL excluding 5/10km of gas flaring locations
#   n_qual_{0,1,2}             : [Monthly only] Number of pixels with each
#                                quality flag (0 = high quality, main algorithm;
#                                1 = high quality, backup algorithm;
#                                2 = poor quality)
#   n_pixels, n_non_na_pixels, prop_non_na_pixels: Pixel counts

raster_dir <- file.path(ntl_bm_dir, "rasters")

# Raster file names by time type
raster_prefix <- c(
  "annual"  = "VNP46A4_NearNadir_Composite_Snow_Free_qflag_t",
  "monthly" = "VNP46A3_NearNadir_Composite_Snow_Free_qflag_t",
  "daily"   = "VNP46A2_Gap_Filled_DNB_BRDF-Corrected_NTL_qflag_t"
)

quality_prefix <- "VNP46A3_NearNadir_Composite_Snow_Free_Quality_qflag_t"

# Load gas flaring -------------------------------------------------------------
gs_5km_sf  <- readRDS(file.path(gas_flare_dir, "finaldata", "gas_flare_locations_5km.Rds"))
gs_10km_sf <- readRDS(file.path(gas_flare_dir, "finaldata", "gas_flare_locations_10km.Rds"))

has_gas_flares <- nrow(gs_5km_sf) > 0

if(has_gas_flares){
  gs_5km_v  <- vect(gs_5km_sf)
  gs_10km_v <- vect(gs_10km_sf)
}

# Function to extract NTL for one raster ---------------------------------------
extract_ntl <- function(roi_sf, ntl_r, qual_r = NULL){

  #### Raster with low values set to 0
  ntl_r_1na   <- ifel(ntl_r < 1,   0, ntl_r)
  ntl_r_0_5na <- ifel(ntl_r < 0.5, 0, ntl_r)
  ntl_r_0_1na <- ifel(ntl_r < 0.1, 0, ntl_r)

  #### Sum / mean
  roi_sf$ntl_sum        <- exact_extract(ntl_r, roi_sf, "sum",  progress = FALSE)
  roi_sf$ntl_mean       <- exact_extract(ntl_r, roi_sf, "mean", progress = FALSE)

  roi_sf$ntl_sum_l1na   <- exact_extract(ntl_r_1na,   roi_sf, "sum", progress = FALSE)
  roi_sf$ntl_sum_l0_5na <- exact_extract(ntl_r_0_5na, roi_sf, "sum", progress = FALSE)
  roi_sf$ntl_sum_l0_1na <- exact_extract(ntl_r_0_1na, roi_sf, "sum", progress = FALSE)

  #### Quality
  if(!is.null(qual_r)){
    roi_sf$n_qual_0 <- exact_extract(qual_r == 0, roi_sf, "sum", progress = FALSE)
    roi_sf$n_qual_1 <- exact_extract(qual_r == 1, roi_sf, "sum", progress = FALSE)
    roi_sf$n_qual_2 <- exact_extract(qual_r == 2, roi_sf, "sum", progress = FALSE)
  }

  #### Gas flaring
  if(has_gas_flares){
    ntl_gf_5km_r    <- mask(ntl_r, gs_5km_v)
    ntl_nogf_5km_r  <- mask(ntl_r, gs_5km_v, inverse = TRUE)
    ntl_gf_10km_r   <- mask(ntl_r, gs_10km_v)
    ntl_nogf_10km_r <- mask(ntl_r, gs_10km_v, inverse = TRUE)

    roi_sf$ntl_gf_5km_sum     <- exact_extract(ntl_gf_5km_r,    roi_sf, "sum",  progress = FALSE)
    roi_sf$ntl_nogf_5km_sum   <- exact_extract(ntl_nogf_5km_r,  roi_sf, "sum",  progress = FALSE)
    roi_sf$ntl_gf_10km_sum    <- exact_extract(ntl_gf_10km_r,   roi_sf, "sum",  progress = FALSE)
    roi_sf$ntl_nogf_10km_sum  <- exact_extract(ntl_nogf_10km_r, roi_sf, "sum",  progress = FALSE)

    roi_sf$ntl_gf_5km_mean    <- exact_extract(ntl_gf_5km_r,    roi_sf, "mean", progress = FALSE)
    roi_sf$ntl_nogf_5km_mean  <- exact_extract(ntl_nogf_5km_r,  roi_sf, "mean", progress = FALSE)
    roi_sf$ntl_gf_10km_mean   <- exact_extract(ntl_gf_10km_r,   roi_sf, "mean", progress = FALSE)
    roi_sf$ntl_nogf_10km_mean <- exact_extract(ntl_nogf_10km_r, roi_sf, "mean", progress = FALSE)
  }

  #### Pixel counts
  roi_sf$n_pixels <- exact_extract(ntl_r, roi_sf,
                                   function(values, coverage_fraction) length(values),
                                   progress = FALSE)

  roi_sf$n_non_na_pixels <- exact_extract(ntl_r, roi_sf,
                                          function(values, coverage_fraction) sum(!is.na(values)),
                                          progress = FALSE)

  roi_sf$prop_non_na_pixels <- roi_sf$n_non_na_pixels / roi_sf$n_pixels

  roi_sf
}

# Extract ----------------------------------------------------------------------
for(unit in NTL_UNITS){

  roi_sf <- load_unit(unit)
  roi_sf$date <- NULL # Avoid conflicts with date variable created below

  for(time_type in NTL_TIME_TYPES){

    out_dir <- file.path(ntl_bm_dir, "aggregated_individual_files", unit, time_type)
    dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

    #### Rasters to process
    raster_files <- file.path(raster_dir, time_type) %>%
      list.files(pattern = paste0("^", raster_prefix[[time_type]], ".*\\.tif$")) %>%
      rev()

    for(file_i in raster_files){

      #### Date
      # Annual: "2012" -> 2012; Monthly: "2012_01" -> "2012-01-01";
      # Daily: "2021_01_01" -> "2021-01-01"
      date_raw_i <- file_i %>%
        str_remove(raster_prefix[[time_type]]) %>%
        str_remove("\\.tif$")

      date_i <- switch(time_type,
                       "annual"  = as.integer(date_raw_i),
                       "monthly" = paste0(str_replace_all(date_raw_i, "_", "-"), "-01"),
                       "daily"   = str_replace_all(date_raw_i, "_", "-"))

      OUT_FILE <- file.path(out_dir, paste0("ntl_", date_i, ".Rds"))
      if(file.exists(OUT_FILE)) next

      #### Load rasters
      ntl_r <- rast(file.path(raster_dir, time_type, file_i))

      qual_r <- NULL
      if(time_type == "monthly"){
        qual_file <- file.path(raster_dir, "monthly_quality",
                               paste0(quality_prefix, date_raw_i, ".tif"))

        if(!file.exists(qual_file)){
          message("Skipping ", unit, " ", date_i, ": quality raster not found")
          next
        }

        qual_r <- rast(qual_file)
      }

      #### Extract and export
      message("Extracting: ", unit, " / ", time_type, " / ", date_i)

      roi_ntl_df <- extract_ntl(roi_sf, ntl_r, qual_r) %>%
        st_drop_geometry()

      roi_ntl_df$date <- date_i

      saveRDS(roi_ntl_df, OUT_FILE)
    }
  }
}
