# Algeria Economic Monitor
# Nighttime Lights Analysis: Main Script
#
# Sets parameters, paths, and packages, then runs the scripts that download,
# extract, and aggregate VIIRS Black Marble nighttime lights data.
#
# Outputs (in `ntl_bm_dir/aggregated`):
#   ntl_{unit}_{time_type}.Rds / .csv / .dta
#   e.g., ntl_adm1_monthly.Rds

# Packages ---------------------------------------------------------------------
library(dplyr)
library(readr)
library(purrr)
library(tidyr)
library(stringr)
library(lubridate)
library(sf)
library(terra)
library(exactextractr)
library(blackmarbler)
library(geodata)
library(readxl)
library(janitor)
library(haven)

# Parameters -------------------------------------------------------------------
iso3_code <- "DZA"
iso2_code <- "DZ"

#### Steps to run
RUN_SCRIPTS        <- TRUE
DOWNLOAD_INPUTS    <- TRUE  # GADM, cities, gas flaring
DOWNLOAD_NTL       <- TRUE  # Black Marble rasters (requires NASA bearer token)
DOWNLOAD_NTL_DAILY <- FALSE # Daily rasters (VNP46A2); large, so off by default
EXTRACT_NTL        <- TRUE
APPEND_NTL         <- TRUE

#### Units of analysis to aggregate nighttime lights to
# Options: "adm0", "adm1", "adm2", "Wilaya", "cities5km", "cities10km",
#          "citiesbuff"
NTL_UNITS <- c("adm0", "adm1", "adm2", "Wilaya")

#### Time periods to aggregate
# Options: "annual", "monthly", "daily"
NTL_TIME_TYPES <- c("annual", "monthly")

#### Dates to download
# Black Marble releases with a lag; dates that aren't yet available are skipped.
ntl_years  <- 2012:2025
ntl_months <- seq.Date(ymd("2012-01-01"),
                       floor_date(Sys.Date(), "month") %m-% months(1),
                       by = "month") %>%
  as.character()
ntl_days   <- seq.Date(ymd("2021-01-01"), Sys.Date() - 1, by = "day") %>%
  as.character()

# Filepaths --------------------------------------------------------------------
#### Root paths
if(Sys.info()[["user"]] == "rmarty"){
  git_dir  <- "~/Github/algeria-economic-monitoring"
  proj_dir <- "/Users/rmarty/Library/CloudStorage/OneDrive-WBG/Development Data Partnership - Algeria Economic Monitor"
  bearer_path <- "~/Dropbox/bearer_bm.csv"
}

#### Data
data_dir      <- file.path(proj_dir, "Data")
ntl_bm_dir    <- file.path(data_dir, "Nighttime Lights BlackMarble")
gadm_dir      <- file.path(data_dir, "GADM")
gas_flare_dir <- file.path(data_dir, "gas-flaring")
cities_dir    <- file.path(data_dir, "cities")
wilaya_dir    <- file.path(data_dir, "Wilaya")

#### Code
code_dir  <- file.path(git_dir, "notebooks", "ntl-analysis")
clean_dir <- file.path(code_dir, "01_clean_data")

# Functions --------------------------------------------------------------------
#### Load GADM boundaries at a given level
load_gadm <- function(level){
  file.path(gadm_dir, "gadm") %>%
    list.files(full.names = TRUE,
               pattern = paste0("_", iso3_code, "_", level, "_pk.rds")) %>%
    map_df(function(file_path){
      readRDS(file_path) %>%
        st_as_sf()
    })
}

#### Load unit of analysis as an sf object
load_unit <- function(unit){
  switch(
    unit,
    "adm0"       = load_gadm(0),
    "adm1"       = load_gadm(1),
    "adm2"       = load_gadm(2),
    "Wilaya"     = read_sf(file.path(wilaya_dir, "RawData", "Wilaya_58.shp")) %>%
      st_transform(4326),
    "cities5km"  = readRDS(file.path(cities_dir, "cities_country_5km_sf.Rds")),
    "cities10km" = readRDS(file.path(cities_dir, "cities_country_10km_sf.Rds")),
    "citiesbuff" = readRDS(file.path(cities_dir, "cities_country_buff_sf.Rds")),
    stop(paste0("Unknown unit: ", unit))
  )
}

# Run scripts ------------------------------------------------------------------
if(RUN_SCRIPTS){

  if(DOWNLOAD_INPUTS){
    source(file.path(clean_dir, "01_download_gadm.R"))
    source(file.path(clean_dir, "02_clean_cities.R"))
    source(file.path(clean_dir, "03_clean_gas_flaring.R"))
  }

  if(DOWNLOAD_NTL) source(file.path(clean_dir, "04_download_black_marble.R"))
  if(EXTRACT_NTL)  source(file.path(clean_dir, "05_extract_ntl.R"))
  if(APPEND_NTL)   source(file.path(clean_dir, "06_append_aggregations.R"))

}
