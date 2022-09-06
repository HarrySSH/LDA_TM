library(boot)
library(MASS)
table = read.csv('~/Box/NLP/Social_data/social_cohort_withage.csv')



label(table$Sex)<-"Sex"
label(table$Ethnicity)       <- "Ethnicity"
label(table$FirstRace)       <- "Race"
label(table$Age)       <- "Age"
table = table[,colnames(table)!='X']



library(table1)
my.render.cont <- function(x) {
  with(stats.apply.rounding(stats.default(x, ), digits = 2),
       c("",
         "median (Q1-Q3)" =
           sprintf(paste("%s (",Q1,"- %s)"), MEDIAN,Q3)))
}


table1(~ Sex +  Ethnicity + FirstRace +Age | factor(numNotesBins), data=table,render.continuous=my.render.cont)
help(table1)
table$Sex <- 
  factor(table$Sex, 
         levels=c(2,1,3),
         labels=c("Not Hispanic or Latino", # Reference
                  "Hispanic or Latino", 
                  "Unknown"))

table$Sex <- 
  factor(table$Sex, 
         levels=c(2,1,3),
         labels=c("Not Hispanic or Latino", # Reference
                  "Hispanic or Latino", 
                  "Unknown"))

melanoma2 <- melanoma
melanoma2$status <- 
  factor(melanoma2$status, 
         levels=c(2,1,3),
         labels=c("Alive", # Reference
                  "Melanoma death", 
                  "Non-melanoma death"))




