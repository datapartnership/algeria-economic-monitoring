# Nighttime Lights: Data Pipeline

Downloads VIIRS Black Marble nighttime lights and aggregates them to administrative units and cities in Algeria.

## Running

Open [`_main.R`](_main.R), set the parameters at the top (units, time periods, which steps to run), and source it. Running it needs a NASA bearer token for Black Marble; see [blackmarbler](https://github.com/worldbank/blackmarbler).

Every step is incremental. Files that already exist are skipped, so a re-run only downloads and processes new dates.

## Scripts (`01_clean_data/`)

| Script | Description |
|---|---|
| `01_download_gadm.R` | Downloads GADM boundaries (ADM 0–2) |
| `02_clean_cities.R` | Geonames cities (pop. > 1,000), with 5km, 10km, and population-based buffers |
| `03_clean_gas_flaring.R` | Gas flaring locations from the [World Bank Global Gas Flaring Database](https://datacatalog.worldbank.org/search/dataset/0037743), with 5km and 10km buffers |
| `04_download_black_marble.R` | Black Marble annual (VNP46A4), monthly (VNP46A3) and monthly quality rasters, plus optional daily (VNP46A2) |
| `05_extract_ntl.R` | Aggregates each raster to each unit; one file per unit, time period, and date |
| `06_append_aggregations.R` | Combines those files into one dataset per unit and time period |

## Outputs

Saved to `Data/Nighttime Lights BlackMarble/aggregated/ntl_{unit}_{time_type}.{Rds,csv,dta}`. For example, `ntl_adm1_monthly.Rds`.

| Variable | Description |
|---|---|
| `ntl_sum`, `ntl_mean` | Sum and mean of nighttime lights |
| `ntl_sum_l1na`, `ntl_sum_l0_5na`, `ntl_sum_l0_1na` | Sum of lights after setting pixels below 1, 0.5, or 0.1 to 0 |
| `ntl_gf_{5,10}km_{sum,mean}` | Lights within 5km or 10km of gas flaring locations |
| `ntl_nogf_{5,10}km_{sum,mean}` | Lights excluding areas within 5km or 10km of gas flaring locations |
| `n_qual_{0,1,2}` | (Monthly only) Number of pixels with each quality flag: 0 = high quality, main algorithm; 1 = high quality, backup algorithm; 2 = poor quality |
| `n_pixels`, `n_non_na_pixels`, `prop_non_na_pixels` | Pixel counts |
| `date` | Year (annual), or `YYYY-MM-01` (monthly) |
