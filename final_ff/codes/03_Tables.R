# setwd('/Users/chelseachen/Dropbox/ActiveProject/PQCT')
setwd('/home/chenfy/projects/classifier_fragilefracture')
source('codes/functions.R')

dat = read.csv('data/meta/chivos_fragfrax_final.csv', row.names = 1)

vars = c('age','L1_L4_T','Neck_T','Total_T')
for (var in vars){print(summary(dat[[var]]))}

newvar = 'age_cat' # min 51, max 85
dat = dat %>%
  mutate(!!sym(newvar) := case_when(
    age >= 50 & age < 60 ~ 1,
    age >= 60 & age < 70 ~ 2,
    age >= 70 & age < 80 ~ 3,
    age >= 80 ~ 4,
    TRUE ~ NA # for unexpected cases
  ) %>% as.factor())
summary(dat[[newvar]])

newvar = 'DXA_l' # min -4.1, max 4.4
var = 'L1_L4_T'
summary(dat[[var]])
dat = dat %>%
  mutate(!!sym(newvar) := case_when(
    !!sym(var) >= -1 ~ 1,
    !!sym(var) < -1 & !!sym(var) > -2.5 ~ 2,
    !!sym(var) <= -2.5 ~ 3,
    TRUE ~ 0 # else
  ) %>% as.factor())
summary(dat[[newvar]])

newvar = 'DXA_n' # min -4.1, max 4.4
var = 'Neck_T'
dat = dat %>%
  mutate(!!sym(newvar) := case_when(
    !!sym(var) >= -1 ~ 1,
    !!sym(var) < -1 & !!sym(var) > -2.5 ~ 2,
    !!sym(var) <= -2.5 ~ 3,
    TRUE ~ 0 # else
  ) %>% as.factor())
summary(dat[[newvar]])

newvar = 'DXA_t' # min -4.1, max 4.4
var = 'Total_T'
dat = dat %>%
  mutate(!!sym(newvar) := case_when(
    !!sym(var) >= -1 ~ 1,
    !!sym(var) < -1 & !!sym(var) > -2.5 ~ 2,
    !!sym(var) <= -2.5 ~ 3,
    TRUE ~ 0 # else
  ) %>% as.factor())
summary(dat[[newvar]])

dat$ht = dat$height / 100 # cm -> m

dat <- dat %>%
  mutate(
    age_cat = as.factor(age_cat),
    sex = as.factor(sex),
    DXA_n = as.factor(DXA_n),
    DXA_l = as.factor(DXA_l),
    DXA_t = as.factor(DXA_t),
    Label_fracture = as.factor(Label_fracture)
  )

write.csv(dat, 'data/meta/Table1.csv')
#------------------------------------------------------
my.render.cont <- function(x) {
    with(stats.apply.rounding(stats.default(x), digits=2), c("",
        "Mean (SD)"=sprintf("%s &plusmn; %s", MEAN, SD)))
}
my.render.cat <- function(x) {
    c("", sapply(stats.default(x), function(y) with(y,
        sprintf("%d (%0.1f%%)", FREQ, PCT))))
}

vars_include = c('age_cat','sex','ht','weight',
'DXA_n','DXA_l','DXA_t',
'fragility_fracture','label',
'RADIUS_Tt_Ar','RADIUS_Tt_vBMD','RADIUS_Tb_Ar','RADIUS_Tb_vBMD','RADIUS_BV.TV','RADIUS_Tb_N','RADIUS_Tb_Th','RADIUS_Tb_Sp','RADIUS_Ct_Ar','RADIUS_Ct_vBMD','RADIUS_Ct_Pm','RADIUS_Ct_Po','RADIUS_Ct_Th',
'TIBIA_Tt_Ar','TIBIA_Tt_vBMD','TIBIA_Tb_Ar','TIBIA_Tb_vBMD','TIBIA_BV.TV','TIBIA_Tb_N','TIBIA_Tb_Th','TIBIA_Tb_Sp','TIBIA_Ct_Ar','TIBIA_Ct_vBMD','TIBIA_Ct_Pm','TIBIA_Ct_Po','TIBIA_Ct_Th'
)

vars_factor = c('fragility_fracture','label','sex')
for (var in vars_factor){dat[[var]] = as.factor(dat[[var]])}


# table, 2 classes, thus not anova but use MannWhitneyU
table1(
  ~ . | fragility_fracture,
  data = dat[,vars_include], overall=F,
  extra.col = list(`P-value` = pvalueMannWhitneyU),
  render.continuous = my.render.cont,  # Replace with your function or remove if not needed
  render.categorical = my.render.cat   # Replace with your function or remove if not needed
)

#------------------------------------------------------
# TABLE, cluster
dat = read.csv('data/cluster/dfp_cluster.csv', row.names=1)

my.render.cont <- function(x) {
    with(stats.apply.rounding(stats.default(x), digits=2), c("",
        "Mean (SD)"=sprintf("%s &plusmn; %s", MEAN, SD)))
}
my.render.cat <- function(x) {
    c("", sapply(stats.default(x), function(y) with(y,
        sprintf("%d (%0.1f%%)", FREQ, PCT))))
}

vars_include = c('age_cat','sex','ht','weight',
'DXA_n','DXA_l','DXA_t',
'fragility_fracture','label','cluster_new',
'RADIUS_Tt_Ar','RADIUS_Ct_Ar','RADIUS_Tb_Ar','RADIUS_Ct_Pm','RADIUS_Tt_vBMD','RADIUS_Ct_vBMD','RADIUS_Tb_vBMD','RADIUS_BV.TV','RADIUS_Tb_N','RADIUS_Tb_Th','RADIUS_Tb_Sp','RADIUS_Ct_Th','RADIUS_Ct_Po',
'TIBIA_Tt_Ar','TIBIA_Ct_Ar','TIBIA_Tb_Ar','TIBIA_Ct_Pm','TIBIA_Tt_vBMD','TIBIA_Ct_vBMD','TIBIA_Tb_vBMD','TIBIA_BV.TV','TIBIA_Tb_N','TIBIA_Tb_Th','TIBIA_Tb_Sp','TIBIA_Ct_Th','TIBIA_Ct_Po')

table1(
  ~ . | cluster_new,
  data = dat[,vars_include],
  extra.col = list(`P-value` = pvalueMannWhitneyU),
  render.continuous = my.render.cont,  # Replace with your function or remove if not needed
  render.categorical = my.render.cat   # Replace with your function or remove if not needed
)
