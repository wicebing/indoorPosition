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


indoorPosition <- read.csv("C:/Users/USER/Downloads/indoorPosition/areaPct.csv")
data_all <- indoorPosition

data_all$ev <- with(data_all, event==2 | event==1)
data_all_c <- subset(data_all , event==0 | event==1)
data_all_f <- subset(data_all , event==0 | event==2)