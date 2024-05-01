library(tidyverse)
options(scipen=999)
library(xgboost)
library(modelr)

# read in data, bind, remove excess dfs
s2017 <- read_csv('savantpbp17.csv') %>% select(-1)
s2017$game_date <- as.Date(s2017$game_date, format = "%m/%d/%Y")
s2017$game_date <- format(s2017$game_date, "%Y-%m-%d")
s2018 <- read_csv('savantpbp18.csv') %>% select(-1)
s2018$game_date <- format(s2018$game_date, "%Y-%m-%d")
s2019 <- read_csv('savantpbp19.csv') %>% select(-1)
s2019$game_date <- format(s2019$game_date, "%Y-%m-%d")
s2021 <- read_csv('savantpbp21.csv') %>% select(-1)
s2021$game_date <- format(s2021$game_date, "%Y-%m-%d")
s2022 <- read_csv('savantpbp22.csv') %>% select(-1)
s2022$game_date <- format(s2022$game_date, "%Y-%m-%d")
s2023 <- read_csv("savantpbp23.csv") %>% select(-1)
s2023$game_date <- format(s2023$game_date, "%Y-%m-%d")

## Bring data together. Create filter for model that will be created (ex RHP vs. RHB)
mlbraw <- bind_rows(s2017, s2018, s2019, s2021, s2022, s2023) %>% distinct()
rm(s2022, s2021, s2019, s2018, s2017, s2023)


### Filter out pitchers hitting, besides Ohtani. This filters out any hitter that's thrown 75 or more pitches between 2017-2022
pitchers <-  mlbraw %>% group_by(pitcher) %>% summarize(pitches=n()) %>% ungroup() %>% filter(pitches > 74, pitcher != "660271")

table(mlbraw$events)
table(mlbraw$pitch_type)

## remove: NA / incorrect values, rare pitch types, outcomes that have nothing to do with pitch quality (like pickoffs).
# normalize pfx_x, spin and release points for righties, treat all field outs as the same, treat sacrifice plays as the same,
# group two seamers and sinkers, group curve and knuckle curve, group slider slurve and sweeper
mlbraw1 <- mlbraw %>% anti_join(pitchers, by = c("batter"="pitcher")) %>%
  filter(description != "pitchout",
         balls < 4, 
         strikes < 3, 
         outs_when_up < 3,
         !pitch_type %in% c("EP", "PO", "KN", "CS", "SC", "FA", "FO"),
         !str_detect(des, "pickoff"),
         !str_detect(des, "caught_stealing"),
         !str_detect(des, "stolen_"), 
         !des %in% c("game_advisory", "catcher_interf"),
         !events %in% c("game_advisory")) %>%
  mutate(
    # release_pos_x = ifelse(p_throws == "R", release_pos_x, -release_pos_x),
    #      pfx_x = ifelse(p_throws == "R", pfx_x, -pfx_x), 
    #      shadow_zone = if_else(plate_z < (sz_top + (1.45/12)) & plate_x > -((17 / 12) / 2) - (1.45/12) & plate_x < ((17 / 12) / 2) + (1.45/12) &
    #                            plate_z > (sz_top - (1.45/12)) & plate_x > -((17 / 12) / 2) - (1.45/12) & plate_x < ((17 / 12) / 2) + (1.45/12) |
    #                            plate_z > (sz_bot - (1.45/12)) & plate_x > -((17 / 12) / 2) - (1.45/12) & plate_x < ((17 / 12) / 2) + (1.45/12) &
    #                            plate_z < (sz_bot + (1.45/12)) & plate_x > -((17 / 12) / 2) - (1.45/12) & plate_x < ((17 / 12) / 2) + (1.45/12) |
    #                            plate_x < -((17 / 12) / 2) + (1.45/12) & plate_z < (sz_top + (1.45/12)) & plate_z > (sz_bot - (1.45/12)) &
    #                            plate_x > -((17 / 12) / 2) - (1.45/12) & plate_z < (sz_top + (1.45/12)) & plate_z > (sz_bot - (1.45/12))  |
    #                            plate_x > ((17 / 12) / 2) - (1.45/12) & plate_z < (sz_top + (1.45/12)) & plate_z > (sz_bot - (1.45/12))  & 
    #                            plate_x < ((17 / 12) / 2) + (1.45/12) & plate_z < (sz_top + (1.45/12)) & plate_z > (sz_bot - (1.45/12)), 1, 0),
    #      out_of_zone = if_else(zone > 9 & shadow_zone == 0, 1, 0),
    #      spin_axis = ifelse(p_throws == "R", spin_axis, -spin_axis), 
         year = year(as.Date(game_date)),
         events = case_when(
           events %in% c("field_out", "fielders_choice_out", "force_out", "other_out") ~ "out",
           events %in% c("double_play", "triple_play", "grounded_into_double_play", "sac_fly_double_play", "sac_bunt_double_play") ~ "double_play",
           events %in% c("strikeout", "strikeout_double_play") ~ "strikeout",
           events %in% c("sac_bunt", "sac_fly") ~ "sacrifice",
           TRUE ~ events),
         pitch_type = case_when(
           pitch_type %in% c("FT", "SI") ~ "SI",
           pitch_type %in% c("CU", "KC", "SV") ~ "CU",
           TRUE ~ pitch_type)) 

## Find average delta run values for each ball or strike (pitches not put in play)
bs_vals <- mlbraw1 %>% filter(type != "X") %>%  group_by(type) %>% summarize(dre_bs = mean(delta_run_exp, na.rm=T))

## same thing but for balls in play
ip_filt <- mlbraw1 %>% filter(type == 'X')

event_lm <- lm(delta_run_exp ~ events, data=ip_filt)
summary(event_lm)

## derive dre values for balls in play 
ip_vals <- ip_filt %>% add_predictions(event_lm, var = "pred_dre") %>%  group_by(events) %>% 
  summarize(dre_ip = mean(pred_dre, na.rm=T))

## merge values into main dataframe
mlbraw1 <- mlbraw1 %>% left_join(bs_vals, by = "type") %>%
  left_join(ip_vals, by = "events") %>% 
  mutate(dre_final = case_when(
    type != "X" ~ dre_bs,
    TRUE ~ dre_ip))

## get pitch distributions 
pitcher_dist <- mlbraw1 %>%
  group_by(pitcher, year, pitch_type) %>%
  summarise(
    pitches_thrown = n()
  )
total <- mlbraw1 %>% group_by(pitcher, year) %>% summarise(total_pitches = n())
pitcher_dist <- pitcher_dist %>% left_join(total, by=c("pitcher", "year"))
pitcher_dist$usage = pitcher_dist$pitches_thrown / pitcher_dist$total_pitches
mlbraw1 <- mlbraw1 %>% left_join(pitcher_dist, by=c("pitcher", "year", "pitch_type"))

## establish each pitcher's fastball metrics by season
pitcher_fastballs <- mlbraw1 %>% 
  filter(pitch_type %in% c("FF", "FC", "SI")) %>% 
  group_by(pitcher, year) %>% 
  slice_max(usage) %>%
  summarize(
    fb_velo = mean(release_speed, na.rm = TRUE), 
    fb_max_ivb = quantile(pfx_z, .8, na.rm = TRUE), 
    fb_max_x = quantile(pfx_x, .8, na.rm = TRUE), 
    fb_max_velo = quantile(release_speed, .8, na.rm = TRUE),
    fb_axis = mean(spin_axis, na.rm = TRUE)
  )

## join to main dataframe, create difference variables
mlbraw2 <- mlbraw1 %>% 
  left_join(pitcher_fastballs, by = c("year", "pitcher")) %>% 
  mutate(spin_dif = spin_axis - fb_axis, 
         velo_dif = release_speed-fb_velo,
         ivb_dif = fb_max_ivb-pfx_z, 
         break_dif = fb_max_x-pfx_x)

## creating lag variables
mlbraw2 <- mlbraw2 %>%
  arrange(game_pk, at_bat_number, pitch_number, pitcher) %>%
  group_by(game_pk, pitcher, at_bat_number) %>%
  mutate(
    lag_spin = if_else(row_number() == 1, 0, spin_axis - lag(spin_axis)),
    lag_velo = if_else(row_number() == 1, 0, release_speed - lag(release_speed)),
    lag_ivb = if_else(row_number() == 1, 0, pfx_z - lag(pfx_z)),
    lag_break = if_else(row_number() == 1, 0, pfx_x - lag(pfx_x))
  ) %>%
  ungroup() %>%
  mutate(
    lag_spin = coalesce(lag_spin, 0),
    lag_velo = coalesce(lag_velo, 0),
    lag_ivb = coalesce(lag_ivb, 0),
    lag_break = coalesce(lag_break, 0)
  )

mlbraw2 <- mlbraw2 %>%
  mutate(p_throws = if_else(p_throws == "L", 1, 0),
         stand = if_else(stand == "L", 1, 0))

## filter for only variables to be used in model
final_vars <- mlbraw2 %>% 
  filter(year == 2022) %>%
  select(dre_final, #starts_with("fb_"), 
         starts_with("lag_"), 
         release_speed, 
         release_spin_rate, release_extension, release_pos_x, release_pos_z, 
         pfx_x, pfx_z, pitch_type, spin_axis, spin_dif, velo_dif, ivb_dif, break_dif,
         p_throws, stand) %>%
  filter(!is.na(dre_final), !is.na(release_speed), !is.na(release_spin_rate), !is.na(release_extension), 
         !is.na(release_pos_x), !is.na(release_pos_z), !is.na(pfx_x), !is.na(pfx_z), !is.na(pitch_type), 
         !is.na(spin_axis), !is.na(spin_dif), !is.na(velo_dif), !is.na(ivb_dif), !is.na(break_dif))

d2023 <- mlbraw2 %>% 
  filter(year == 2023) %>%
  select(dre_final, #starts_with("fb_"), 
         starts_with("lag_"), 
         release_speed, 
         release_spin_rate, release_extension, release_pos_x, release_pos_z, 
         pfx_x, pfx_z, pitch_type, spin_axis, spin_dif, velo_dif, ivb_dif, break_dif,
         p_throws, stand) %>%
  filter(!is.na(dre_final), !is.na(release_speed), !is.na(release_spin_rate), !is.na(release_extension), 
         !is.na(release_pos_x), !is.na(release_pos_z), !is.na(pfx_x), !is.na(pfx_z), !is.na(pitch_type), 
         !is.na(spin_axis), !is.na(spin_dif), !is.na(velo_dif), !is.na(ivb_dif), !is.na(break_dif))


preds_data <- mlbraw2 %>% 
  filter(year == 2023) %>%
  select(dre_final, #starts_with("fb_"), 
         game_pk, pitcher, pitch_number, at_bat_number,
         starts_with("lag_"), release_speed, 
         release_spin_rate, release_extension, release_pos_x, release_pos_z, 
         pfx_x, pfx_z, pitch_type, spin_axis, spin_dif, velo_dif, ivb_dif, break_dif,
         p_throws, stand) %>%
  filter(!is.na(dre_final), !is.na(release_speed), !is.na(release_spin_rate), !is.na(release_extension), 
         !is.na(release_pos_x), !is.na(release_pos_z), !is.na(pfx_x), !is.na(pfx_z), !is.na(pitch_type), 
         !is.na(spin_axis), !is.na(spin_dif), !is.na(velo_dif), !is.na(ivb_dif), !is.na(break_dif))
  

#### Modeling #####
library(tidyverse)
library(openintro)
library(tidymodels)
library(finetune)
library(pbapply)
library(vip)
library(caret)

dmy <- dummyVars(" ~ pitch_type", data = final_vars)
dmy_d <- dummyVars(" ~ pitch_type", data = preds_data)
trsf <- data.frame(predict(dmy, newdata = final_vars))
trsf_d <- data.frame(predict(dmy_d, newdata = preds_data))

### join data. remove original variables which dummies have been created for. filter out any missing run values
vars <- cbind(final_vars, trsf) %>% select(-pitch_type) %>% filter(!is.na(dre_final))
vars_d <- cbind(d2023, trsf_d) %>% select(-pitch_type) %>% filter(!is.na(dre_final))
preds_data <- cbind(preds_data, trsf_d) %>% select(-pitch_type) %>% filter(!is.na(dre_final))

# Get a list of objects in the environment
objects <- ls()
objects_to_keep <- c("vars", "vars_d", "preds_data")
rm(list = setdiff(objects, objects_to_keep))

set.seed(1)
split = initial_split(vars, prop = 0.8)
train = training(split)
test = testing(split)

#10-fold Cross Validation. Normally dont have to go above 10
set.seed(2)
folds <- vfold_cv(train, v = 3)
folds

#Create the formula
formula <-
  recipe(dre_final ~ ., #. means everything
         data = vars
  )

# Specify our model
specifications <-
  boost_tree(
    trees = tune(),
    #min_n = tune(),
    mtry = tune(),
    learn_rate = tune(),
    #loss_reduction = tune(),
    tree_depth = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

#Match the specifications with the formula
workflow <- workflow(formula, specifications)

#This is the code to add a grid
xgb_grid <- workflow %>%
  parameters() %>%
  update(
    mtry = mtry(range = c(1, 20)),
    trees = trees(range = c(100, 2000)),
    tree_depth = tree_depth(range(1, 10)),
    #min_n = min_n(range = c(2, 100)),
    learn_rate = learn_rate(range = c(.1,.5)),
    #loss_reduction = loss_reduction(range = c(0, .2))
  ) %>%
  grid_max_entropy(size = 20)

#This allows you to process in parallel. Saves a lot of time!
doParallel::registerDoParallel(cores = 8)

#Use tune_race_anova to tune the model
metrics <- yardstick::metric_set(yardstick::rmse)
set.seed(3)

xgb_rs <- tune_race_anova( # tune race anova: Compares performance of all models after one fold, sees if it is significantly worse and stops training
  workflow,
  resamples = folds,
  grid = xgb_grid, # creates random grid, good place to start before a custom grid
  metrics = metrics,
  control = control_race(verbose_elim = FALSE,
                         burn_in = 2)
)
saveRDS(xgb_rs, file = "xgb_final.rds")

#Examine how the racing went/some of the hyperparameters
plot_race(xgb_rs)
# autoplot(xgb_rs) # Use these to set custom grid and find exactly what you are looking for

#Look at the best models
best = show_best(xgb_rs, n = 20)
View(best)

#Make the final model, you can use select_best or enter the hyperparameters manually
xgb_last <- workflow %>%
  finalize_workflow(
    select_best(xgb_rs, "rmse")
    # data.frame(
    #   mtry = 1,
    #   trees = 338,
    #   min_n = 4,
    #   tree_depth = 5,
    #   learn_rate = 0.018,
    #   loss_reduction = 5.57e-10,
    #   stop_iter = 19
    )
  # ) %>%
  # last_fit(split)
#)

fit_workflow <- fit(xgb_last, train)
predictions <- predict(fit_workflow, new_data = preds_data)
final_data <- cbind(preds_data, predictions)
saveRDS(final_data, file = "xgb_final.rds")

xgb_imp <- workflow %>%
  finalize_workflow(
    select_best(xgb_rs, "rmse")
    # data.frame(
    #   mtry = 1,
    #   trees = 338,
    #   min_n = 4,
    #   tree_depth = 5,
    #   learn_rate = 0.018,
    #   loss_reduction = 5.57e-10,
    #   stop_iter = 19
    # )
    ) %>%
    last_fit(split)

#Look at the metrics
xgb_imp$.metrics

importance_scores <- extract_workflow(xgb_imp) %>%
  extract_fit_parsnip() %>%
  vi()

importance_scores %>%
  mutate(Variable = case_when(
    grepl("^lag", Variable) ~ "Difference from Previous Pitch",
    grepl("_dif", Variable) ~ "Difference from Fastball",
    grepl("^fb", Variable) ~ "Primary Fastball Attributes",
    Variable == "pfx_x" ~ "Horizontal Movement",
    Variable == "pfx_z" ~ "Vertical Movement",
    Variable == "release_spin_rate" ~ "Spin Rate",
    Variable == "release_speed" ~ "Velocity",
    Variable == "spin_axis" ~ "Spin Axis",
    Variable == "release_pos_z" ~ "Vertical Release Point",
    Variable == "release_pos_x" ~ "Horizontal Release Point",
    Variable == "release_extension" ~ "Extension",
    Variable == "stand" ~ "Batter Handedness",
    Variable == "p_throws" ~ "Pitcher Handedness",
    grepl("^pitch_type", Variable) ~ "Pitch Type",
    TRUE ~ Variable)
  ) %>%
  group_by(Variable) %>%
  summarise(Importance = sum(Importance)) %>%
  ggplot(aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col(fill = "#041E42") +
  coord_flip() +
  labs(x = "Feature") +
  theme_minimal()

#Plot Test Set
ggplot(final_data, aes(x = dre_final, y = .pred)) +
  geom_point() +
  geom_abline() +
  theme_bw() + 
  coord_fixed() +
  xlim(-0.5, 2) + 
  ylim(-0.5, 2)

ggplot(final_data, aes(x = .pred)) +
  geom_density()

#Plot Histogram
ggplot(food, aes(x = calories)) + geom_histogram()
