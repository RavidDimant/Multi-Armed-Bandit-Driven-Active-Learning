# Load necessary libraries
library(quantreg)
library(ggplot2)
library(dplyr)
library(reshape2)
library(dplyr)


# Load the dataset
CVD <- read.csv("C:/Users/user/Desktop/CVD_cleaned.csv")
CVD$General_Health <- as.factor(CVD$General_Health)
CVD$Checkup <- as.factor(CVD$Checkup)
CVD$Exercise <- as.factor(CVD$Exercise)
CVD$Heart_Disease <- as.factor(CVD$Heart_Disease)
CVD$Skin_Cancer <- as.factor(CVD$Skin_Cancer)
CVD$Other_Cancer <- as.factor(CVD$Other_Cancer)
CVD$Depression <- as.factor(CVD$Depression)
CVD$Diabetes <- as.factor(CVD$Diabetes)
CVD$Arthritis <- as.factor(CVD$Arthritis)
CVD$Sex <- as.factor(CVD$Sex)
CVD$Age_Category <- as.factor(CVD$Age_Category)
CVD$Smoking_History <- as.factor(CVD$Smoking_History)
CVD$BMI <- as.factor(CVD$BMI)

# Convert the "Diabetes" column to 1 or 0
CVD <- CVD %>%
  mutate(Diabetes = ifelse(Diabetes %in% c("Yes", "Yes, but female told only during pregnancy"), 1, 
                           ifelse(Diabetes %in% c("No", "No, pre-diabetes or borderline diabetes"), 0, Diabetes)))

# Display the first few rows of the dataset
head(CVD)

# Define the formula for the regression model
formula <- Diabetes ~ General_Health + Checkup  + Exercise + Heart_Disease + Skin_Cancer + Other_Cancer + Depression + BMI + Arthritis + Sex + Age_Category + Height_.cm. + Weight_.kg. + Smoking_History + Alcohol_Consumption + Fruit_Consumption + Green_Vegetables_Consumption + FriedPotato_Consumption 

# Fit the OLS model
ols_model <- lm(formula, data = CVD)
summary(ols_model)
