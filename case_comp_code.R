# Code for the final XGBoost Model cumulating stuff, marginal effects, and game/situational effects

library(tidyverse)
library(dplyr)
library(caret)
library(xgboost)
library(caret)
library(openintro)
library(tidymodels)
library(finetune)
library(vip)

# setwd()

#### Initial Data Manipulation ####

# Read in 2023 PBP Data
read_csv("savantpbp23.csv") -> pbp_23

pbp_23 %>%
  filter(game_type == "R") -> pbp_23

# Read in predictions from Stuff Model

filename <- file.choose()
stuff_preds <- readRDS(filename)

pbp_23 <- pbp_23 %>%
  left_join(stuff_preds %>%
              rename(stuff = .pred) %>%
              dplyr::select(game_pk:pitch_number, stuff),
            by = c("game_pk", "pitcher", "at_bat_number", "pitch_number"))

# Graphs to look at data distribution

# pbp_23 %>%
#   ggplot(aes(stuff)) +
#   geom_density()

# pbp_23 %>%
#   arrange(delta_run_exp) -> pbp_23

pbp_23 %>%
  filter(description == "foul") -> fouls

# pbp_23 %>%
#   ggplot(aes(delta_run_exp)) +
#   geom_density()

# Filtering out non pitches
fouls %>%
  filter(is.na(events)) -> fouls_filtered

# DRE Distribution
pbp_23 %>%
  ggplot(aes(delta_run_exp)) +
  geom_density(fill = "#041E42", color = "#bf0d3e") +
  theme_minimal() +
  xlim(-.25, .25) +
  labs(x = "Delta Run Expectancy", title = "Delta Run Expectancy Density")
  
fouls_filtered %>%
  ggplot(aes(delta_run_exp)) +
  geom_density(fill = "#041E42", color = "#bf0d3e", lwd = 1) +
  theme_minimal() +
  labs(x = "Delta Run Expectancy", y = "Density",
       title = "Foul Ball Delta Run Expectancy Density") +
  theme(axis.text = element_text(size = 12),
        axis.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        plot.title = element_text(hjust = 0.5, size = 18, face = "bold")) +
  xlim(-0.2, 0)
  #xlim(-0.15, 0)

#### Modeling Process ####
pbp_23 %>%
  arrange(game_date, game_pk, inning, inning_topbot, at_bat_number, pitch_number) %>%
  group_by(game_pk, pitcher, at_bat_number) %>%
  mutate(nxt_pitch_dre = lead(delta_run_exp)) %>%
  dplyr::select(pitch_number, nxt_pitch_dre, delta_run_exp, description) %>%
  filter(!is.na(nxt_pitch_dre)) %>%
  mutate(diff_dre = nxt_pitch_dre - delta_run_exp) -> pn_data

pn_data %>%
  filter(description == "foul") -> pn_model

dum_model <- lm(diff_dre  ~ pitch_number, data = pn_model)
summary(dum_model)

names(pbp_23)
dum_model <- lm(delta_run_exp  ~ release_speed + release_spin_rate, data = pbp_23)

summary(dum_model)

options(scipen = 999)

fouls_filtered$fit_run_exp <- predict(dum_model, fouls_filtered)

fouls_filtered %>%
  dplyr::select(pitch_type:break_length_deprecated, release_spin_rate, delta_run_exp, fit_run_exp) -> temp

pbp_23 %>% 
  mutate(on_1b = if_else(!is.na(on_1b), 1, 0),
         on_2b = if_else(!is.na(on_2b), 1, 0),
         on_3b = if_else(!is.na(on_3b), 1, 0)) %>%
  mutate(count = paste0(balls, "-", strikes), .after = strikes) %>%
  mutate(base_state = as.factor(paste0(on_1b, "-", on_2b, "-", on_3b)),
         inning = as.factor(inning),
         outs_when_up = as.factor(outs_when_up)) -> pbp_23

# why are these in dataset
pbp_23 %>%
  filter(!count %in% c("4-1", "4-2")) %>%
  mutate(count = as.factor(count)) -> pbp_23

# add index 
pbp_23 %>%
  mutate(rowIndex = row_number()) -> pbp_23

# add indicator columns for foul balls and two strike foul balls
pbp_23 %>%
  mutate(foul_0s_fl = if_else(description == "foul" & strikes == 0, 1, 0),
         foul_1s_fl = if_else(description == "foul" & strikes == 1, 1, 0),
         foul_2s_fl = if_else(description == "foul" & strikes == 2, 1, 0)) %>%
  arrange(game_date, game_pk, inning, inning_topbot, at_bat_number, pitch_number) %>%
  group_by(game_pk, pitcher, at_bat_number) %>%
  mutate(total_2s_fl = cumsum(foul_2s_fl)) -> pbp_23

pbp_23 %>%
  mutate(perc_sz_top = percent_rank(sz_top),
         perc_sz_bot = percent_rank(sz_bot)) -> pbp_23

pbp_23 %>%
  filter(!is.na(sz_top)) %>%
  ggplot(aes(sz_top)) +
  geom_density()

pbp_23 %>%
  dplyr::select(
    # release_speed, release_pos_x, release_pos_y, release_pos_z,
    #             release_spin_rate, pfx_x, pfx_z, spin_axis,
                stuff, plate_x, plate_z, perc_sz_top, perc_sz_bot, 
                #pitch_number, 
                outs_when_up, count, base_state,
                delta_run_exp, rowIndex) -> data_oH


# pbp_23 %>%
#   filter(balls == 0 & strikes == 0 & outs_when_up == 0 &
#            on_1b == 0 & on_2b == 0 & on_3b == 0 &
#            #inning == 1 & inning_topbot == 'Top' &
#            events == "single") %>%
#   arrange(delta_run_exp) -> temp
# table(temp$delta_run_exp)
# table(temp$inning)

# one hot encode / convert categorical variables to 1/0 indicators
dum_cats <- dummyVars(" ~ .", data = data_oH)
data_oH <- data.frame(predict(dum_cats, newdata = data_oH))

# remove any observations with nas
data_oH <- data_oH[complete.cases(data_oH),]
data_model <- data_oH[, -which(names(data_oH) == "rowIndex")]


# Split the data
set.seed(1)
split <- initial_split(data_model, prop = 0.6)
train <- training(split)
test <- testing(split)

# 3-fold Cross Validation
set.seed(2)
folds <- vfold_cv(train, v = 5)
folds

# Create the formula
formula <-
  recipe(delta_run_exp ~ .,
         data = data_model)

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

# Match the specifications with the formula
workflow <- workflow(formula, specifications)

# This is the code to add a grid
# xgb_grid <- workflow %>%
#   parameters() %>%
#   update(
#     mtry = mtry(range = c(1, 10)),
#     trees = trees(range = c(10, 2000)),
#     min_n = min_n(range = c(2, 50)),
#     learn_rate = learn_rate(range = c(0,.1)),
#     loss_reduction = loss_reduction(range = c(0, .1))
#   ) %>%
#   grid_max_entropy(size = 50)

# This allows you to process in parallel. Saves a lot of time!
doParallel::registerDoParallel(cores = 8)

# Use tune_race_anova to tune the model
set.seed(3)

xgb_rs <- tune_race_anova(
  workflow,
  resamples = folds,
  grid = 30,
  metrics = metric_set(rmse),
  control = control_race(verbose_elim = F, 
                         burn_in = 2)
)

# Examine how the racing went/some of the hyperparameters
plot_race(xgb_rs)
autoplot(xgb_rs)

# optimal parameters
show_best(xgb_rs, n = 15)

# construct final model, you can use select_best or enter the hyperparameters manually
xgb_last <- workflow %>%
  finalize_workflow(
    select_best(xgb_rs, "rmse"))
    # data.frame(
    #   mtry = 3,
    #   trees = 844,
    #   min_n = 4,
    #   tree_depth = 11,
    #   learn_rate = 0.0138,
    #   loss_reduction = 5.57e-10,
    #   stop_iter = 19)
  # ) %>%
  # last_fit(split)

options(scipen = 999)

# generate predictions for entire dataset
fit_workflow <- fit(xgb_last, train)
data_oH$fitted <- predict(fit_workflow, data_oH) %>% pull()
cor(data_oH$delta_run_exp, data_oH$fitted)

data_oH %>%
  dplyr::select(rowIndex, fitted) %>%
  left_join(pbp_23, by = c("rowIndex")) -> data_pred

data_pred %>%
  group_by(count) %>%
  summarise(avg = mean(fitted)) %>%
  view()

# Marginal Effects
# 0 strikes: +0.006434
# 1 strike: +0.012616

# calculate marginal delta run expectancy effects
data_pred %>%
  mutate(me_dre = case_when(
    description == "foul" & strikes == 0 ~ 0.006,
    description == "foul" & strikes == 1 ~ 0.013,
    description == "foul" & strikes == 2 & total_2s_fl == 1 ~ 0.010648,
    description == "foul" & strikes == 2 & total_2s_fl == 2 ~ 0.00154,
    description == "foul" & strikes == 2 & total_2s_fl == 3 ~ 0.007908,
    description == "foul" & strikes == 2 & total_2s_fl == 4 ~ 0.017811,
    T ~ 0
  )) -> data_pred

# calculate difference in actual v. predicted delta run expectancy
# high values - good for batter ; low values - bad for batter
data_pred %>% 
  mutate(dre_diff = delta_run_exp - fitted + me_dre) -> data_pred

data_pred %>%
  filter(description == "foul") -> fb_pred

# fb_pred %>%
#   filter(strikes == 2) %>%
#   #mutate(FBR_plus = 100 * -dre_diff / mean(fb_pred$dre_diff)) %>%
#   group_by(batter) %>%
#   summarise(Total_2S_FBR = sum(dre_diff),
#             Mean_2S_FBR = mean(dre_diff),
#             #FBR_plus = mean(FBR_plus),
#             obs_2S = n()) %>%
#   arrange(desc(Total_2S_FBR)) -> fbr_2s_pred
# 
# fbr_total_pred %>%
#   left_join(fbr_2s_pred, by = c("batter")) %>%
#   write_csv("fbr_batter_preds.csv")

# data_pred %>%
#   group_by(batter) %>%
#   summarise(S0_FB = sum(foul_0s_fl),
#             S1_FB = sum(foul_1s_fl),
#             S2_FB = sum(foul_2s_fl)
#             ) -> fb_ratios
# 
# write_csv(fb_ratios, "fb_ratios.csv")

# all pitches; min 250 obs; 230 players
# 2-strike pitches; min 100 obs; 210 players

#### Creating Graphs and Analysis ####

fb_pred %>%
  filter(strikes == 2) %>%
  mutate(fb_oz = if_else(
    plate_x > 0.83101 | plate_x < -0.83101 | 
      plate_z > sz_top | plate_z < sz_bot, 1, 0),
    fb_z = if_else(fb_oz == 0, 1, 0),
    fb_heart = if_else(
      plate_x < 0.58566 & plate_x > -0.58566 &
        plate_z > sz_bot + 0.24536 & plate_z < sz_top - 0.24536, 1, 0 
        )) %>%
  group_by(batter) %>%
  summarise(fb_z = sum(fb_z),
            fb_oz = sum(fb_oz),
            fb_heart = sum(fb_heart)) %>%
  mutate(total_fb = fb_z + fb_oz,
         z_oz_ratio = fb_z/fb_oz,
         heart_fb_ratio = fb_heart/total_fb) %>%
  filter(total_fb >= 100) %>%
  arrange(z_oz_ratio) %>%
  view()

temp %>%
  filter(obs >= 125) %>%
  view()

#cor(temp$Mean_FBR, temp$FBR_plus)

data_pred %>%
  filter(base_state == "0-0-0" & count == "0-0" & 
           outs_when_up == 0 & description == "foul") %>%
  dplyr::select(delta_run_exp)

data_pred %>%
  filter(base_state == "0-0-0" & count == "0-0" & 
           outs_when_up == 0 & description == "foul")

# delta run expectancy: -0.038
  
# calculate mean dre diff by count
data_pred %>%
  filter(description == "foul") %>%
  group_by(balls, strikes) %>%
  summarise(dre_diff = mean(dre_diff)) -> count_plot

# plot dre diff by count using matrix 
count_plot %>%
  ggplot(aes(x = balls, y = strikes)) + 
  geom_raster(aes(fill = dre_diff)) +
  scale_fill_gradient2(low="#5064af",mid="white", high="#d82129",
                       midpoint=0, name = "FBR") +
  geom_text(aes(label = round(dre_diff, 3)), size = 6, fontface = "bold",
            color = c(rep("black", 4), "white", 
                      rep("black", 2), "white",
                     rep("black", 3), "white")) +
  theme_minimal() +
  labs(x = "Balls", y = "Strikes", 
       title = "Foul Ball Runs by Count") +
  theme(axis.text = element_text(size = 12),
        axis.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        plot.title = element_text(hjust = 0.5, size = 18, face = "bold"))


# Compute the breaks
breaks_x <- round(seq(min(data_pred$plate_x), max(data_pred$plate_x), length.out = 70 + 1), 3)
breaks_z <- seq(min(data_pred$plate_z), max(data_pred$plate_z), length.out = 70 + 1)

find_closest_value <- function(x, vector) {
  closest_val <- vector[which.min(abs(vector - x))]
  return(closest_val)
}
# 
# data_pred %>%
#   ggplot(aes(stuff)) +
#   geom_density()

avg_stuff <- mean(data_pred$stuff)

data_pred %>%
  filter(description == "foul") %>% #  & strikes == 2
  dplyr::select(plate_x, plate_z, fitted, dre_diff, stuff) %>%
  #mutate(stuff_scaled = -((stuff - avg_stuff) / avg_stuff * 100)) %>%
  mutate(plate_x = sapply(plate_x, find_closest_value, breaks_x),
         plate_z = sapply(plate_z, find_closest_value, breaks_z)) %>%
  mutate(stuff_group = case_when(
    stuff > quantile(stuff, 0.8) ~ "Bad",
    stuff < quantile(stuff, 0.2) ~ "Good",
    T ~ "Average")) %>%
  group_by(plate_x, plate_z, stuff_group) %>% # stuff_group
  summarise(fitted = mean(fitted),
            dre_diff = mean(dre_diff),
            stuff = mean(stuff)) -> p_heatmap

# (0.005089368  - 0.005089368)/0.005089368 
# 
p_heatmap %>%
  ggplot(aes(dre_diff)) +
  geom_density()

p_heatmap %>%
  #filter(fitted <= 0.1 & fitted >= -0.1) %>%
  mutate(dre_diff = if_else(dre_diff < -0.1, -0.1, dre_diff)) %>%
  ggplot(aes(x = plate_x, y = plate_z)) +
  geom_raster(aes(fill = dre_diff)) +
  facet_wrap(~factor(stuff_group, levels=c("Bad", "Average", "Good"))) +
  scale_fill_gradient2(low="#5064af",mid="white", high="#d82129", midpoint=0, name = "FBR") +
  #geom_contour(data=test, aes(x=plate_x, y=plate_z, z = fitted), colour = "white", )
  #geom_contour(aes(z = fitted))
  xlim(-1.5, 1.5) +
  ylim(1, 4) +
  geom_rect(aes(xmin = -0.833, xmax = 0.833, ymin = mean(pbp_23$sz_bot, na.rm = T), 
                ymax = mean(pbp_23$sz_top, na.rm = T)), fill = NA, color = "black",
            linetype = "dashed", lwd = 0.7) +
  theme_void() +
  theme(strip.text = element_text(size = 14, margin = margin(0, 0, 4, 0)),
        plot.margin = margin(0, 0.1, 0, 0, unit = "cm")
        )
        #axis.text = element_text(size = 12),)

# generate data heatmap example
heatmap_grid <- expand_grid(
  plate_x = seq(from=min(data_pred$plate_x), 
          to=max(data_pred$plate_x), 
          length.out = 200),
  plate_z = seq(from=min(data_pred$plate_z), 
          to=max(data_pred$plate_z), 
          length.out = 200),
  # stuff = c(unname(quantile(data_model$stuff, .1)),
  #           unname(quantile(data_model$stuff, .5)),
  #           unname(quantile(data_model$stuff, .9))
  #           )
) %>%
  # mutate(stuff_group = case_when(
  #   stuff == unname(quantile(data_model$stuff, .1)) ~ "Great",
  #   stuff == unname(quantile(data_model$stuff, .9)) ~ "Poor (Stuff)",
  #   T ~ "Average")) %>%
  mutate(id = 1)

data_oH %>%
  #dplyr::select(-c(plate_x, plate_z)) %>%
  summarise(across(c(stuff, sz_top:sz_bot), ~ mean(.))) %>%
  mutate(id = 1) -> num_cols

# data_pred %>%
#   filter(base_state == "0-0-0" & count == "1-2" &
#            outs_when_up == 0 & description == "foul")

# delta run expectancy: 
# 0-0: -0.038
# 0-2: 0

data_oH %>%
  dplyr::select(`outs_when_up.0`:`base_state.1.1.1`) %>%
  mutate(id = 1) %>%
  filter(`outs_when_up.0` == 1 & `base_state.0.0.0` == 1 &
           (`count.0.0` == 1 | `count.0.2` == 1 | `count.3.2` == 1)) %>%
  distinct(., .keep_all = T)  -> cat_cols

cat_cols$delta_run_exp <- c(-0.038, 0, 0)
cat_cols$count <- c("0-0", "0-2", "3-2")


# %>% mutate(delta_run_exp = 0)

heatmap_grid %>%
  left_join(num_cols, by = c("id")) %>%
  left_join(cat_cols, by = c("id")) -> heatmap_grid

heatmap_grid %>%
  filter(plate_x < 2 & plate_x > -2  & plate_z < 5 & plate_z > -1) -> heatmap_grid

heatmap_grid$fitted <- predict(fit_workflow, heatmap_grid) %>% pull()
#cor(data_oH$delta_run_exp, data_oH$fitted)
heatmap_grid %>%
  mutate(dre_diff = delta_run_exp - fitted) -> heatmap_grid

heatmap_grid %>%
  mutate(dre_diff = if_else(dre_diff < -0.07, -0.07, dre_diff)) %>%
  mutate(dre_diff = if_else(dre_diff > 0.07, 0.07, dre_diff)) %>%
  ggplot(aes(x = plate_x, y = plate_z)) +
  geom_raster(aes(fill = dre_diff)) +
  facet_wrap(~factor(count)) +
  scale_fill_gradient2(low="#5064af",mid="white", high="#d82129", midpoint=0, name = "FBR") +
  #geom_contour(data=test, aes(x=plate_x, y=plate_z, z = fitted), colour = "white", )
  #geom_contour(aes(z = fitted))
  xlim(-1.5, 1.5) +
  ylim(1, 4) +
  geom_rect(aes(xmin = -0.833, xmax = 0.833, ymin = mean(pbp_23$sz_bot, na.rm = T), 
                ymax = mean(pbp_23$sz_top, na.rm = T)), fill = NA, color = "black",
            linetype = "dashed", lwd = 0.7) +
  theme_void() +
  theme(#plot.title = element_text(hjust = 0.5, size = 18, face = "bold"),
        strip.text = element_text(size = 14, face = "bold", margin = margin(0, 0, 4, 0)),
        plot.margin = margin(0, 0.1, 0, 0, unit = "cm")
  )

data_pred %>%
  filter(batter == 592450 & strikes == 2 & description == "foul") %>%
  ggplot(aes(plate_x, plate_z)) +
  geom_point(aes(color = dre_diff)) +
  geom_rect(aes(xmin = -0.833, xmax = 0.833, ymin = 1.82, 
                ymax = 3.92), fill = NA, color = "black",
            linetype = "dashed", lwd = 0.7) +
  scale_fill_gradient2(low="#5064af",mid="white", high="#d82129", midpoint=0, name = "FBR") +
  xlim(-1.5, 1.5) +
  ylim(0, 5)

# judge: 592450
# nootbar: 663457

# generate re24

data_pred %>%
  filter(description == "foul") %>%
  group_by(base_state, outs_when_up) %>%
  summarise(dre_diff = mean(dre_diff)) %>%
  #pivot_wider(names_from = outs_when_up, values_from = dre_diff)
  ggplot(aes(x = outs_when_up, y = base_state)) + 
  geom_raster(aes(fill = dre_diff)) +
  scale_fill_gradient2(low="#5064af",mid="white", high="#d82129",
                       midpoint=0, name = "FBR") +
  geom_text(aes(label = round(dre_diff, 3)), size = 4, fontface = "bold") +
  theme_minimal()

data_oH %>%
  summarise(across(c(stuff:sz_bot), ~ mean(.))) %>%
  mutate(id = 1) -> num_cols

# data_pred %>%
#   filter(base_state == "0-0-0" & count == "1-2" &
#            outs_when_up == 0 & description == "foul")

# delta run expectancy: 
# 0-0: -0.038
# 0-2: 0

data_oH %>%
  dplyr::select(`outs_when_up.0`:`base_state.1.1.1`, delta_run_exp) %>%
  mutate(id = 1) %>%
  filter(`count.0.0` == 1) %>%
  distinct(across(c(`outs_when_up.0`:`base_state.1.1.1`)), .keep_all = T
           )  -> cat_cols


# %>% mutate(delta_run_exp = 0)

num_cols %>%
  left_join(cat_cols, by = c("id")) -> re24

re24$fitted <- predict(fit_workflow, re24) %>% pull()
#cor(data_oH$delta_run_exp, data_oH$fitted)
re24 %>%
  mutate(dre_diff = delta_run_exp - fitted) -> re24

# theme(plot.margin = margin(0, 15, 5, 5),
#       axis.text = element_text(size = 12),
#       axis.title = element_text(hjust = 0.5, size = 14, face = "bold"),
#       plot.title = element_text(hjust = 0.5, size = 18, face = "bold"))

data_pred %>%
  filter(description == "foul") %>%
  ggplot(aes(dre_diff)) +
  geom_density(fill = "#041E42", color = "#bf0d3e", lwd = 1.25) +
  theme_minimal() +
  labs(x = "FBR", y = "Density",
       title = "Foul Ball Runs (FBR) Density") +
  theme(axis.text = element_text(size = 12),
        axis.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        plot.title = element_text(hjust = 0.5, size = 18, face = "bold")) +
  xlim(-0.25, 0.25)


# extract for model importance
xgb_perf <- workflow %>%
  finalize_workflow(
    select_best(xgb_rs, "rmse")) %>%
  last_fit(split)
# data.frame(
#   mtry = 3,
#   trees = 844,
#   min_n = 4,
#   tree_depth = 11,
#   learn_rate = 0.0138,
#   loss_reduction = 5.57e-10,
#   stop_iter = 19)
# ) %>%
# last_fit(split)

# plot partial dependence of pitch number
# data_oH %>%
#   dplyr::select(c(stuff:sz_bot)) %>%
#   summarise(across(c(stuff:sz_bot), ~ mean(.))) %>%
#   mutate(id = 1) -> num_cols
# 
# data_oH %>%
#   dplyr::select(`outs_when_up.0`:`base_state.1.1.1`) %>%
#   mutate(id = 1) %>% 
#   dplyr::slice(94) -> cat_cols
# 
# data_oH %>%
#   distinct(pitch_number) %>%
#   arrange(pitch_number) %>%
#   mutate(id = 1) %>%
#   left_join(num_cols, by = c("id")) %>%
#   left_join(cat_cols, by = c("id")) -> pn_partial
# 
# pn_partial$fitted <- predict(fit_workflow, pn_partial) %>% pull()
# 
# pn_partial %>%
#   ggplot(aes(pitch_number, fitted)) +
#   geom_line()

#Look at the metrics
xgb_perf$.metrics

library(workflows)

#Look at the variable importance
importance_scores <- extract_workflow(xgb_perf) %>% 
  extract_fit_parsnip() %>%
  vi() 

importance_scores %>%
  mutate(Variable = case_when(
    grepl("^base_state", Variable) ~ "Base State",
    grepl("^count", Variable) ~ "Count",
    grepl("^outs_when_up", Variable) ~ "Outs When Up",
    Variable == "plate_x" ~ "Plate X",
    Variable == "plate_z" ~ "Plate Z",
    Variable == "sz_top" ~ "SZ Top",
    Variable == "sz_bot" ~ "SZ Bot",
    Variable == "stuff" ~ "Stuff",
    Variable == "pitch_number" ~ "Pitch Number",
    TRUE ~ Variable)
  ) %>%
  group_by(Variable) %>%
  summarise(Importance = sum(Importance)) %>%
  ggplot(aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col(fill = "#041E42") +
  coord_flip() +
  labs(x = "Feature") +
  theme_minimal() +
  theme(plot.margin = margin(0, 15, 5, 5),
        axis.text = element_text(size = 12),
        axis.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        plot.title = element_text(hjust = 0.5, size = 18, face = "bold"))
  
importance_scores %>%
  mutate(variable = gsub("\\..*$", "", variable)) %>% # Removing the suffix after dot in variable names
  group_by(variable) %>%
  summarise(importance = sum(importance)) %>%
  arrange(desc(importance))


#final_predictions <- collect_predictions(xgb_last)

# eta: 0.00710586129014271
# max_depth: 11
# nrounds: 727
# colsample_bynode: 0.428571428571429

library(xgboost)

set.seed(1)
split <- initial_split(data_model, prop = 0.7)
train_x <- training(split)
test_x <- testing(split)

xgb_train = xgb.DMatrix(data = data.matrix(train_x[, -50]), 
                        label = train_x$delta_run_exp)
model_xgboost <- xgboost(data = xgb_train, nrounds = 1000, max.depth = 8, 
                        colsample_bytree = 0.4, eta = 0.005,
                        verbose = 0)


cor(data_pred$pitch_number, data_pred$fitted)


