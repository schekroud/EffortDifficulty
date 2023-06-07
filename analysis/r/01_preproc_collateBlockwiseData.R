library(tidyverse)
library(magrittr)
dir <- '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty'
setwd(dir)

datapath <- paste0(dir, '/data/datafiles')

sublist <- seq(3,11, by = 1)

#this will collate all parts of the same subjects into one file per subject
dataFiles = list(NULL)
for(sub in sublist){
  dfiles = list.files(path = sprintf('%s/s%02d/', datapath, sub), full.names = TRUE) #list all files for that subject (we want to collate all blocks into one)
  
  df <- purrr::map_dfr(dfiles, function(x) read.csv(x, header = T, as.is = T, sep = ','))
  df$subid <- sub

  #add some variables here that will be used later in the analysis pipeline
  df %<>%
    dplyr::mutate(fbgiven = 'timed out') %>% #default
    dplyr::mutate(fbgiven = ifelse(fbtrig == 60, 'correct', fbgiven)) %>%
    dplyr::mutate(fbgiven = ifelse(fbtrig == 61, 'incorrect', fbgiven)) %>%
    dplyr::mutate(prevtrlfb = lag(fbgiven, 1)) %>% #get previous trial rewarded/unrewarded
    dplyr::mutate(rewarded = ifelse(fbgiven == 'correct', 1, 0),
                  unrewarded = ifelse(fbgiven == 'incorrect', 1, 0))
  
  outpath <- sprintf('%s/EffortDifficulty_s%02d_combined.csv', datapath, sub)
  write.csv(df, file = outpath, eol = '\n', col.names = T, row.names = F)
}