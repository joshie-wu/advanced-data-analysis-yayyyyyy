library(dplyr)
library(lubridate)
library(tidyr)

df <- read.csv('./DOHMH_New_York_City_Restaurant_Inspection_Results_20260501.csv')
mapping <- read.csv('https://raw.githubusercontent.com/nychealth/Food-Safety-Health-Code-Reference/refs/heads/main/Violation-Health-Code-Mapping.csv')


mapping_clean <- mapping %>%
  select(
    Violation_Code,
    starts_with("Condition"),
    Violation_Summary,
    Category_Description
  )

mapping_clean <- mapping_clean %>%
  group_by(Violation_Code) %>%
  summarise(
    across(everything(), ~ first(na.omit(.))),
    .groups = "drop"
  ) 

mapping_clean <- mapping_clean %>%
  select(Violation_Code,Category_Description)

df_mapped <- df %>%
  filter(
    INSPECTION.DATE > as.Date("01/01/1990")
  ) %>%
  left_join(mapping_clean, by = c("VIOLATION.CODE" = "Violation_Code"))%>%
  select(restaurant_id = "CAMIS", score = "SCORE",  borough = "BORO", cuisine="CUISINE.DESCRIPTION", 
         inspection_date = "INSPECTION.DATE", action = "ACTION", violation_code = "VIOLATION.CODE",
         inspection_type = "INSPECTION.TYPE", latitude = "Latitude", longitude = "Longitude") %>%
  drop_na(longitude,latitude) %>%
  filter(longitude != 0 & latitude != 0)


rm(df,mapping,mapping_clean)

#joining nta

library(sf)
nta_map <- read.csv('/Users/joshwu/Downloads/DiTecT_Lab/2020_Neighborhood_Tabulation_Areas_(NTAs)_20260217.csv')

library(tidyr)

df_mapped <- df_mapped %>%
  st_as_sf(coords = c("longitude","latitude"), crs=4326)


nta_map$geometry <- st_as_sfc(nta_map$the_geom, crs=4326)
nta_map <- st_as_sf(nta_map)

df_joined <- st_join(df_mapped, nta_map %>% select(NTA2020,geometry), join = st_within) %>%
  st_drop_geometry()

rm(df_mapped,nta_map)
library(readxl)
nta_demo <- read_excel('/Users/joshwu/Downloads/DiTecT_Lab/5-yr ACS 2023/5-yr ACS 2023/Neighborhood-NTA/Demographic/Dem_1923_NTA.xlsx') %>%
  select(nta = GeoID, Pop_1E)
nta_econ <- read_excel('/Users/joshwu/Downloads/DiTecT_Lab/5-yr ACS 2023/5-yr ACS 2023/Neighborhood-NTA/Economic/Econ_1923_NTA.xlsx') %>%
  select(nta = GeoID, MdEWrkE)


df_full <- df_joined %>% 
  left_join(nta_demo, by = c("NTA2020" = "nta")) %>%
  left_join(nta_econ, by = c("NTA2020" = "nta")) 

write.csv(df_full, "inspection_results_with_ntas.csv")
