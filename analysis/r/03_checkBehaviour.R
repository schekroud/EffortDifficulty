library(tidyverse)
library(magrittr)
dir <- '/Users/sammi/Desktop/postdoc/student_projects/EffortDifficulty'
setwd(dir)
fname <- sprintf('%s/data/EffortDifficulty_maindata.csv', dir)
##
#set theme for plots to standardise them and make cleaner
theme_set(theme_bw() +
            theme(
              strip.background = element_blank(),
              axis.text    = element_text(family = 'Source Sans Pro', colour = '#000000', size = 14),
              axis.title   = element_text(family = 'Source Sans Pro', colour = '#000000', size = 18),
              panel.border = element_rect(size = 1, color = '#000000'),
              legend.title = element_text(family = 'Source Sans Pro', colour = '#000000', size = 16),
              legend.text  = element_text(family = 'Source Sans Pro', colour = '#000000', size = 14),
              strip.text   = element_text(family = 'Source Sans Pro', colour = '#000000', size = 16) 
            ) 
)
# theme_set(theme_bw() + theme(strip.background = element_blank()))
se <- function(x) sd(x)/sqrt(length(x))
subs = seq(3,12,by=1) #these are the eeg subjects

df <- read.csv(fname, as.is = T, header = T)
df %<>% dplyr::filter(subid != 8) %>%
  dplyr::filter(subid != 4) %>% dplyr::filter(subid != 5)

df %>%
  dplyr::group_by(subid, difficultyOri) %>%
  dplyr::summarise_at(.vars = 'PerceptDecCorrect', .funs = 'mean', na.rm = T) %>% as.data.frame()

#get how many time out trials occurred
df %>% 
  dplyr::filter(fbtrig == 62) %>%
  dplyr::group_by(subid) %>% count()

df %>% dplyr::mutate(difficultyOri = as.factor(difficultyOri)) %>%
  #dplyr::filter(fbtrig != 62) %> #exclude timeout trials
  dplyr::group_by(subid, difficultyOri) %>%
  dplyr::summarise_at(.vars = 'PerceptDecCorrect', .funs = 'mean', na.rm = T) %>% as.data.frame() -> df.plotacc


df.plotacc %>%
  dplyr::group_by(difficultyOri) %>%
  dplyr::summarise_at(.vars = 'PerceptDecCorrect', .funs = c('mean', 'se')) %>% as.data.frame() %>%
  ggplot() +
  geom_bar(aes(x = difficultyOri, y = mean, fill = difficultyOri),
           stat = 'identity', width = .8, position=position_dodge(0.8)) +
  geom_errorbar(aes(x = difficultyOri, fill = difficultyOri,
                    ymin = mean-se, ymax = mean+se), width = 0.4, position = position_dodge(0.8)) +
  geom_point(inherit.aes=F, data = df.plotacc,
             aes(x = difficultyOri, y = PerceptDecCorrect), size = 2) +
  scale_fill_brewer(palette = 'Blues') +
  geom_hline(yintercept = 0.5, linetype = 'dashed') +
  labs(x = 'perceptual difficulty (°)', y = 'proportion correct') + theme(legend.position = 'none')

#plot as continuous x to see stepwise changes in acc ~ difficulty
df %>% dplyr::group_by(subid, difficultyOri) %>%
  dplyr::summarise_at(.vars = 'PerceptDecCorrect', .funs = 'mean', na.rm = T) %>% as.data.frame() -> df.acc

df.acc %>%
  dplyr::group_by(difficultyOri) %>%
  dplyr::summarise_at(.vars = 'PerceptDecCorrect', .funs = c('mean', 'se')) %>% as.data.frame() %>%
  ggplot() +
  geom_bar(aes(x = difficultyOri, y = mean, fill = difficultyOri),
           stat = 'identity', width = .8, position=position_dodge(0.8)) +
  geom_errorbar(aes(x = difficultyOri, fill = difficultyOri,
                    ymin = mean-se, ymax = mean+se), width = 0.4, position = position_dodge(0.8)) +
  geom_point(inherit.aes = F, data = df.acc, 
             aes(x = difficultyOri, y = PerceptDecCorrect), size = 2) +
  scale_fill_distiller(palette = 'Blues', type = 'seq', direction = 1) +
  scale_x_continuous(breaks = seq(1,12, by=1), minor_breaks = NULL) +
  geom_hline(yintercept = 0.5, linetype = 'dashed') +
  labs(x = 'perceptual difficulty (°)', y = 'proportion correct') + theme(legend.position = 'none')

df %>% dplyr::mutate(difficultyOri = as.factor(difficultyOri)) %>%
  dplyr::filter(PerceptDecCorrect == 1) %>%
  dplyr::group_by(subid, difficultyOri) %>%
  dplyr::summarise_at(.vars = 'rt', .funs = 'mean', na.rm = T) %>% as.data.frame() -> df.rt

df.rt %>%
  dplyr::group_by(difficultyOri) %>%
  dplyr::summarise_at(.vars = 'rt', .funs = c('mean', 'se')) %>% as.data.frame() %>%
  ggplot() +
  geom_bar(aes(x = difficultyOri, y = mean, fill = difficultyOri),
           stat = 'identity', width = .8, position=position_dodge(0.8)) +
  geom_errorbar(aes(x = difficultyOri, fill = difficultyOri,
                    ymin = mean-se, ymax = mean+se), width = 0.4, position = position_dodge(0.8)) +
  geom_point(inherit.aes = F, data = df.rt, 
             aes(x = difficultyOri, y = rt), size = 2) +
  scale_fill_brewer(palette = 'Blues') +
  labs(x = 'perceptual difficulty (°)', y = 'RT (s)') + theme(legend.position = 'none') +
  coord_cartesian(ylim = c(0.2, 0.6))

