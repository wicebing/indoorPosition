library("writexl")
library(openxlsx)
library(readxl)
library(mgcv)
library(oddsratio)
library('vcd')
library(postHoc)

library(foreign)
library(ggplot2)
library(MASS)
require(boot)


indoorPosition <- read_excel("C:/Users/USER/Downloads/indoorPosition/checkThings.xlsx")
data_all <- indoorPosition