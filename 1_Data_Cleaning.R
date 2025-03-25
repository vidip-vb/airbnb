# Data Cleaning #

cat("\014")   
rm(list = ls())  

library(tidyverse)
library(tidytext)
library(lubridate)  # For date filtering

df <- read.csv("London.csv.gz")

## 1. Data Filtering & Cleaning

# 1.1 Filtering for the Last Trimester (Sep–Dec 2024)
df <- df %>%
  filter(ymd(last_scraped) >= ymd("2024-09-01"))  # This keeps the listings scraped from September onwards

# 1.2 Dropping Irrelevant Columns
# - URLs, images and metadeta provide no predictive value for price variable
# - 'license' and 'neighbourhood_group_cleansed' columns is empty in this dataset.
df <- df %>% select(-c("listing_url", "host_url", "host_thumbnail_url", "host_picture_url", "picture_url",
                       "calendar_updated", "scrape_id", "source", "license", "neighbourhood_group_cleansed"))

# - Host name and Host location are not useful for predicting price, host neighbourhood and neighbourhood are unstructured duplicates of neibourhood cleansed variable (which is kept).
# - Historical scrape data is not useful for predicting price, as it only shows when data was last pulled
# - Additionally, description, neighbourhood overview, and host about variables are unstructured free-text, making them less relevant without the use of advanced NLP techniques, which is outside the scope of this essay.

df <- df %>% select(-c("name", "host_name", "host_location", "host_neighbourhood", "neighbourhood", "calendar_last_scraped", "last_scraped", 
                       "description", "neighborhood_overview","host_about", "longitude","latitude")) # Name variable is kept for the sake of data exploration using text analysis

# - We can also remove the IDs as these are just identifiers so they dont provide predictive value in our ML model
df <- df %>% select(-"id", -"host_id")

# 1.3 Manipulating Relevant Character Variables
# - Checking for All Character String Variables
char_vars <- sapply(df, is.character)
print("Character Variables in Dataset:")
print(names(df)[char_vars])

# 1.4 Dates
df$host_since <- as.numeric(difftime(as.Date(df$host_since), as.Date("2000-01-01"), units = "days"))
df$first_review <- as.numeric(difftime(as.Date(df$first_review), as.Date("2000-01-01"), units = "days"))
df$last_review <- as.numeric(difftime(as.Date(df$last_review), as.Date("2000-01-01"), units = "days"))
# - since features like last_review have many different dates, using as a categorical variable would be unfeasible (because of large no. of dummies). Instead they are converted to numeric 'since' features where it uses days as units.

# 1.5 Percentages
df$host_response_rate <- as.numeric(gsub("%", "", df$host_response_rate)) / 100
df$host_acceptance_rate <- as.numeric(gsub("%", "", df$host_acceptance_rate)) / 100
# - These percentages were stored as strings, so we remove the % and convert it to its decimal equivalence.

# 1.6 (Boolean) T/F Variables to Categorical (Factor) Variables 
df$host_is_superhost <- as.factor(df$host_is_superhost)
df$host_response_time <- as.factor(df$host_response_time)
df$host_has_profile_pic <- as.factor(df$host_has_profile_pic)
df$host_identity_verified <- as.factor(df$host_identity_verified)
df$instant_bookable <- as.factor(df$instant_bookable)
df$has_availability <- as.factor(df$has_availability)

# 1.7 Price: 
df$price <- as.numeric(gsub("[$,]", "", df$price))


# 1.8 Special Cases
# 1.8.1 Host verifications act as a trust indicator, broken up into a list of verification methods (e.g., emails, phone, gov't id). Some listings have more verifications which may act as more trustworthiness. So it may be important to keep this variable.
# - To convert the text from format '[...], [...], ...' into one that is more useful for the purpose of models, they are seperated and transformed into binary indicators.
df$email_verified <- grepl("email", df$host_verifications)
df$phone_verified <- grepl("phone", df$host_verifications)
df$work_email_verified <- grepl("work_email", df$host_verifications)
df$photographer_verified <- grepl("photographer", df$host_verifications) # - This should encompass every type of verification method.

# - Therefore, we can drop the original text column
df <- df %>% select(-host_verifications)

# 1.8.2 Neighbourhood Cleansed contains London district names as a categorical variable. ML models handle this well as a factor (ie. The variable is converted into a factor w/ 33 levels "Barking and Dagenham","Barnet", etc.)
df$neighbourhood_cleansed <- as.factor(df$neighbourhood_cleansed)

# 1.8.3 Room Types
# - There are 4 total types of rooms in this variable (entire home/apt, private room, shared room, hotel room), so it can be made into a categorical (factor) variable
df$room_type <- as.factor(df$room_type)

# 1.8.4 Property Types
# - This is a bit more tricky to manage since there are ideas that overlap and its not as limited (~98 total options). Analysing the frequency of each property type, will show which are the most common and rare property types.
table(df$property_type) %>% sort(decreasing = TRUE)

# - The table also portrays that many categories overlap or are related to the ideas in the room type variable.  Since room_type already distinguishes between entire/private/shared spaces, property_type is simplified to focus purely on the type of living space (e.g., House, Apartment, Hotel, etc.).
df$property_type_grouped <- case_when(
  grepl("rental unit|condo|serviced apartment", df$property_type) ~ "Apartment/Condo",
  grepl("home|townhouse|guesthouse|villa", df$property_type) ~ "House/Townhouse",
  grepl("hotel|boutique hotel|aparthotel", df$property_type) ~ "Hotel/Serviced Living",
  grepl("Shared room|hostel", df$property_type) ~ "Shared Living Space",
  grepl("Houseboat|Campsite|Tent|Yurt|Barn|Castle|Treehouse|Dome|Shepherd’s hut|Cycladic home|Lighthouse", df$property_type) ~ "Unique Stay",
  TRUE ~ "Other"
)

df$property_type_grouped <- as.factor(df$property_type_grouped)
df <- df %>% select(-property_type) # Can now remove the original property type column 


# 1.8.5 Bathrooms Text
# - This describes bathroom count or type but it is stored as a string of text. To convert it into a numeric format, the orignial column is dropped and numerical values are assigned to the number of values. 
df$bathrooms_text <- ifelse(df$bathrooms_text == "Half-bath", 0.5, 
                            as.numeric(gsub("[^0-9.]", "", df$bathrooms_text))) #half-bath is converted to 0.5 and gsub removes non-numeric characters for the rest of the assignments

df$bathrooms[is.na(df$bathrooms)] <- df$bathrooms_text[is.na(df$bathrooms)]
# - This replaces the missing values in bathroom with information from bathroom text, allowing us to delete the batroom_text variable
df <- df %>% select(-bathrooms_text)

# 1.8.6 Amenities
# - This free-text column contains comma-separated lists of features available in the listing (eg. Wifi, Kitchen, Heating, etc.) 
# - For a start, the Amenities column is cleaned
df$amenities <- as.character(df$amenities)
amenities_cleaned <- gsub("\\[|\\]|\"", "", df$amenities) # removing brackets and quotes
amenities_cleaned <- strsplit(amenities_cleaned, ", ")  # splitting into individual amenities

amenity_counts <- table(unlist(amenities_cleaned)) # flattening the list and count occurances
amenity_counts_df <- as.data.frame(amenity_counts) %>%
  arrange(desc(Freq)) # converting this into a data frame so we can sort it from most to least occuring amenity

head(amenity_counts_df, 100) #top 100 most common amenities


# The amenity categories are defined as follows:
amenity_categories <- list(
  kitchen = c("Kitchen", "Cooking basics", "Stove", "Oven", "Microwave", "Refrigerator", "Toaster", "Freezer", "Baking sheet"),
  tv = c("TV", "Netflix", "TV with standard cable", "Smart TV"),
  work_friendly = c("Dedicated workspace", "Ethernet connection"),
  bathroom = c("Shampoo", "Conditioner", "Body soap", "Hair dryer", "Bathtub", "Shower gel"),
  laundry = c("Washer", "Dryer", "Free washer – In unit", "Free dryer – In unit", "Laundromat nearby"),
  heating_cooling = c("Heating", "Central heating", "Air conditioning", "Portable fans"),
  safety = c("Smoke alarm", "Carbon monoxide alarm", "Fire extinguisher", "First aid kit", "Safe"),
  sleeping_comfort = c("Bed linens", "Extra pillows and blankets", "Room-darkening shades"),
  child_friendly = c("Crib", "High chair", "Pack ’n play/Travel crib", "Children’s books and toys", "Children’s dinnerware"),
  luxury = c("Hot water kettle", "Wine glasses", "Garden view", "City skyline view"),
  outdoor_space = c("Private patio or balcony", "Backyard", "Outdoor dining area", "Outdoor furniture", "BBQ grill"),
  security = c("Lockbox", "Smart lock", "Keypad", "Exterior security cameras")
)

calculate_score <- function(amenities, keywords) {
  rowSums(sapply(keywords, function(term) grepl(term, amenities, ignore.case = TRUE)))} # Function to calculate scores

# Applying scoring to all categories
for (category in names(amenity_categories)) {
  df[[paste0(category, "_score")]] <- calculate_score(df$amenities, amenity_categories[[category]])
}

# The "amenities" and "amenities_cleaned" columns can now be dropped
df <- df %>% select(-amenities)


# 1.8.7 Host Reponse Time
# - shows how quicly the host reponds in a string format (eg. within a few hours, within a day, etc.)
# - We can convert this into a variabel with orderedd numerical values and since there are only 4 possible values, this is quite reasonable.
df$host_response_time <- factor(df$host_response_time, 
                                levels = c("within an hour", "within a few hours", "within a day", "a few days or more"),
                                labels = c(1, 2, 3, 4),
                                ordered = TRUE)
# - This means that 4 indicates the slowest response and 1 indicates the quickest.


# 1.9 Removing Price Outliers

# 1.9.1 Computing IQR for price
Q1 <- quantile(df$price, 0.25, na.rm = TRUE)  # 25th percentile
Q3 <- quantile(df$price, 0.75, na.rm = TRUE)  # 75th percentile
IQR_value <- Q3 - Q1  # Interquartile range

# 1.9.2 Defining bounds for outliers
lower_bound <- Q1 - (1.5 * IQR_value)
upper_bound <- Q3 + (1.5 * IQR_value)

# 1.9.3 Filtering the data (keeping values within bounds)
df <- df %>% filter(price >= lower_bound & price <= upper_bound)

# - Checking price variable to confirm that extreme values are removed
summary(df$price)  


## 2. Final Checks and Final Cleaned File
str(df)  # Verify correct data types
summary(df)  # Quick stats to check missing values and distribution

write.csv(df, "Cleaned_London.csv", row.names = FALSE)

