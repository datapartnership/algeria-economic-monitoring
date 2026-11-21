# Clean Cities Data

# Creates datasets of cities in the country, including buffers around cities.
# Raw data from Geonames (cities with a population over 1,000):
# https://public.opendatasoft.com/explore/dataset/geonames-all-cities-with-a-population-1000/

dir.create(cities_dir, showWarnings = FALSE, recursive = TRUE)

# Download global cities -------------------------------------------------------
OUT_FILE_ALL <- file.path(cities_dir, "cities_all.Rds")

if(!file.exists(OUT_FILE_ALL)){
  city_df <- read_delim("https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/geonames-all-cities-with-a-population-1000/exports/csv?lang=en&timezone=America%2FNew_York&use_labels=true&delimiter=%3B",
                        delim = ";")

  saveRDS(city_df, OUT_FILE_ALL)
}

# Country cities ---------------------------------------------------------------
OUT_FILE_COUNTRY_DF      <- file.path(cities_dir, "cities_country_df.Rds")
OUT_FILE_COUNTRY_SF      <- file.path(cities_dir, "cities_country_sf.Rds")
OUT_FILE_COUNTRY_5K_SF   <- file.path(cities_dir, "cities_country_5km_sf.Rds")
OUT_FILE_COUNTRY_10K_SF  <- file.path(cities_dir, "cities_country_10km_sf.Rds")
OUT_FILE_COUNTRY_BUFF_SF <- file.path(cities_dir, "cities_country_buff_sf.Rds")

if(!file.exists(OUT_FILE_COUNTRY_SF)){

  city_df <- readRDS(OUT_FILE_ALL)

  #### Cleanup
  city_country_df <- city_df[city_df$`Country Code` %in% iso2_code,] %>%
    clean_names() %>%
    separate(coordinates, into = c("lat", "lon"), sep = ",")

  #### Spatially define
  city_country_sf <- city_country_df %>%
    st_as_sf(coords = c("lon", "lat"),
             crs = 4326)

  #### Fixed buffers
  city_country_5km_sf  <- city_country_sf %>% st_buffer(dist = 5000)
  city_country_10km_sf <- city_country_sf %>% st_buffer(dist = 10000)

  #### Dynamic buffer, based on population (km)
  city_country_sf <- city_country_sf %>%
    dplyr::mutate(buffer = case_when(
      population >= 10000000 ~ 30,
      population >= 5000000  ~ 25,
      population >= 1000000  ~ 20,
      population >= 500000   ~ 15,
      population >= 100000   ~ 10,
      TRUE ~ 5
    ))

  city_buff_sf <- city_country_sf %>%
    split(.$buffer) %>%
    map_df(function(sf_i){
      st_buffer(sf_i, dist = sf_i$buffer[1] * 1000)
    })

  #### Export
  saveRDS(city_country_df,      OUT_FILE_COUNTRY_DF)
  saveRDS(city_country_sf,      OUT_FILE_COUNTRY_SF)
  saveRDS(city_country_5km_sf,  OUT_FILE_COUNTRY_5K_SF)
  saveRDS(city_country_10km_sf, OUT_FILE_COUNTRY_10K_SF)
  saveRDS(city_buff_sf,         OUT_FILE_COUNTRY_BUFF_SF)
}
