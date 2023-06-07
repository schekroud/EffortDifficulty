library(tidyverse)
library(magrittr)
dir <- '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty'
setwd(dir)
datapath <- paste0(dir, '/data/datafiles')

dfiles = list.files(path = datapath, pattern = '.csv', full.names = T) #gets full paths for all participant data files
df <- purrr::map_dfr(dfiles, function(x) read.csv(x, header = T, as.is = T, sep = ',')) #read them and rowbind into one dataframe
outpath <- sprintf('%s/data/EffortDifficulty_maindata.csv', dir)
write.csv(df, file = outpath, eol = '\n', col.names = T, row.names = F)

