# Clean Gas Flaring Data

# Creates dataset of gas flaring locations in the country, and 5km and 10km
# buffers around them.
# Raw data from the World Bank Global Gas Flaring Database:
# https://datacatalog.worldbank.org/search/dataset/0037743

raw_dir   <- file.path(gas_flare_dir, "rawdata")
final_dir <- file.path(gas_flare_dir, "finaldata")

dir.create(raw_dir,   showWarnings = FALSE, recursive = TRUE)
dir.create(final_dir, showWarnings = FALSE, recursive = TRUE)

# Download data ----------------------------------------------------------------
gf_urls <- c(
  "2017" = "https://datacatalogfiles.worldbank.org/ddh-published/0037743/DR0045623/viirs_global_flaring_d.7_slope_0.029353_2017_web_v1.xlsx?versionId=2023-01-18T20:03:32.2273754Z",
  "2018" = "https://datacatalogfiles.worldbank.org/ddh-published/0037743/DR0045622/viirs_global_flaring_d.7_slope_0.029353_2018_web.xlsx?versionId=2023-01-18T20:02:43.3965005Z",
  "2019" = "https://datacatalogfiles.worldbank.org/ddh-published/0037743/DR0045621/viirs_global_flaring_d.7_slope_0.029353_2019_web_v20201114-3.xlsx?versionId=2023-01-18T20:03:09.2456111Z",
  "2020" = "https://datacatalogfiles.worldbank.org/ddh-published/0037743/DR0084248/2020%20Global%20Gas%20Flaring%20Volumes.xlsx?versionId=2023-01-18T20:03:53.8309309Z",
  "2021" = "https://datacatalogfiles.worldbank.org/ddh-published/0037743/DR0087112/2021%20Global%20Gas%20Flaring%20Volumes.xlsx?versionId=2023-01-18T20:02:21.4951166Z"
)

for(year_i in names(gf_urls)){
  OUT_FILE <- file.path(raw_dir, paste0("viirs_global_flaring_", year_i, ".xlsx"))

  if(!file.exists(OUT_FILE)){
    download.file(url = gf_urls[[year_i]], destfile = OUT_FILE, mode = "wb")
  }
}

# Append data ------------------------------------------------------------------
# Sheets that contain flaring locations differ across years. Order determines
# the `uid` of each location, so keep this order fixed.
gf_sheets <- list(
  "2021" = 2,
  "2020" = 1:3,
  "2019" = 1,
  "2018" = 4:6,
  "2017" = 1:3
)

gs_df <- map_df(names(gf_sheets), function(year_i){
  map_df(gf_sheets[[year_i]], function(sheet_i){
    read_xlsx(file.path(raw_dir, paste0("viirs_global_flaring_", year_i, ".xlsx")),
              sheet = sheet_i) %>%
      clean_names() %>%
      dplyr::filter(iso_code %in% iso3_code) %>%
      dplyr::transmute(latitude  = as.numeric(latitude),
                       longitude = as.numeric(longitude))
  })
})

gs_df <- gs_df %>%
  distinct() %>%
  dplyr::mutate(uid = row_number())

# Buffer -----------------------------------------------------------------------
gs_sf <- gs_df %>%
  st_as_sf(coords = c("longitude", "latitude"),
           crs = 4326)

gs_5km_sf  <- gs_sf %>% st_buffer(dist = 5000)
gs_10km_sf <- gs_sf %>% st_buffer(dist = 10000)

# Export -----------------------------------------------------------------------
saveRDS(gs_df,      file.path(final_dir, "gas_flare_locations.Rds"))
saveRDS(gs_5km_sf,  file.path(final_dir, "gas_flare_locations_5km.Rds"))
saveRDS(gs_10km_sf, file.path(final_dir, "gas_flare_locations_10km.Rds"))
