library(quanteda)
library(syuzhet)
library(mgcv)
library(reshape2)
library(readr)
library(stringr)
#devtools::install_github("kbenoit/LIWCalike")
library(LIWCalike)
library(lsa)
library(LSAfun)

#compute levenshtein distance
lev=list()
for (i in 1:nrow(data)){lev[i]=adist(data$post[i],data$response[i])}
data$lev=as.numeric(lev)

#estimate bipolar/univariate sentiment 
data$resp_val<-get_sentiment(data$response, method="nrc")
data$post_va<-get_sentiment(data$post, method="nrc")
data$ty_val<-get_sentiment(data$thankyou, method="nrc")

#run post and response texts through LIWC2007
liwc2007dict <- dictionary(file = "LIWC2007.cat", format = "wordstat", encoding = "WINDOWS-1252")
output_post=data.frame()
output_resp=data.frame()
output_post <- liwcalike(data$post, liwc2007dict)
output_resp <- liwcalike(data$response, liwc2007dict)

#grab post response function word vectors
post_funcwords=output_post[,c(8, 14, 15, 17, 21,22,23,24,25)]
resp_funcwords=output_resp[,c(8, 14, 15, 17, 21,22,23,24,25)]

#compute pennebaker-style LSM scores for function words
lsm1<- 1 - ( abs(as.numeric(post_funcwords[,1]) - as.numeric(resp_funcwords[,1])) / (as.numeric(post_funcwords[,1]) + as.numeric(resp_funcwords[,1]) + .0001))
lsm2<- 1 - ( abs(as.numeric(post_funcwords[,2]) - as.numeric(resp_funcwords[,2])) / (as.numeric(post_funcwords[,2]) + as.numeric(resp_funcwords[,2]) + .0001))
lsm3<- 1 - ( abs(as.numeric(post_funcwords[,3]) - as.numeric(resp_funcwords[,3])) / (as.numeric(post_funcwords[,3]) + as.numeric(resp_funcwords[,3]) + .0001))
lsm4<- 1 - ( abs(as.numeric(post_funcwords[,4]) - as.numeric(resp_funcwords[,4])) / (as.numeric(post_funcwords[,4]) + as.numeric(resp_funcwords[,4]) + .0001))
lsm5<- 1 - ( abs(as.numeric(post_funcwords[,5]) - as.numeric(resp_funcwords[,5])) / (as.numeric(post_funcwords[,5]) + as.numeric(resp_funcwords[,5]) + .0001))
lsm6<- 1 - ( abs(as.numeric(post_funcwords[,6]) - as.numeric(resp_funcwords[,6])) / (as.numeric(post_funcwords[,6]) + as.numeric(resp_funcwords[,6]) + .0001))
lsm7<- 1 - ( abs(as.numeric(post_funcwords[,7]) - as.numeric(resp_funcwords[,7])) / (as.numeric(post_funcwords[,7]) + as.numeric(resp_funcwords[,7]) + .0001))
lsm8<- 1 - ( abs(as.numeric(post_funcwords[,8]) - as.numeric(resp_funcwords[,8])) / (as.numeric(post_funcwords[,8]) + as.numeric(resp_funcwords[,8]) + .0001))
lsm9<- 1 - ( abs(as.numeric(post_funcwords[,9]) - as.numeric(resp_funcwords[,9])) / (as.numeric(post_funcwords[,9]) + as.numeric(resp_funcwords[,9]) + .0001))
data$lsm_tot= (lsm1 + lsm2 + lsm3 + lsm4 + lsm5 + lsm6 + lsm7 + lsm8 + lsm9)/9

#compute post response NRC emotional character vectors
post_sent<-get_nrc_sentiment(data$post, method="nrc")
resp_sent<-get_nrc_sentiment(data$response, method="nrc")

#compute pennebaker-style LSM scores for NRC emotional character vectors
esm1<- 1 - ( abs(as.numeric(post_sent[,1]) - as.numeric(resp_sent[,1])) / (as.numeric(post_sent[,1]) + as.numeric(resp_sent[,1]) + .0001))
esm2<- 1 - ( abs(as.numeric(post_sent[,2]) - as.numeric(resp_sent[,2])) / (as.numeric(post_sent[,2]) + as.numeric(resp_sent[,2]) + .0001))
esm3<- 1 - ( abs(as.numeric(post_sent[,3]) - as.numeric(resp_sent[,3])) / (as.numeric(post_sent[,3]) + as.numeric(resp_sent[,3]) + .0001))
esm4<- 1 - ( abs(as.numeric(post_sent[,4]) - as.numeric(resp_sent[,4])) / (as.numeric(post_sent[,4]) + as.numeric(resp_sent[,4]) + .0001))
esm5<- 1 - ( abs(as.numeric(post_sent[,5]) - as.numeric(resp_sent[,5])) / (as.numeric(post_sent[,5]) + as.numeric(resp_sent[,5]) + .0001))
esm6<- 1 - ( abs(as.numeric(post_sent[,6]) - as.numeric(resp_sent[,6])) / (as.numeric(post_sent[,6]) + as.numeric(resp_sent[,6]) + .0001))
esm7<- 1 - ( abs(as.numeric(post_sent[,7]) - as.numeric(resp_sent[,7])) / (as.numeric(post_sent[,7]) + as.numeric(resp_sent[,7]) + .0001))
esm8<- 1 - ( abs(as.numeric(post_sent[,8]) - as.numeric(resp_sent[,8])) / (as.numeric(post_sent[,8]) + as.numeric(resp_sent[,8]) + .0001))
data$esm_tot= (esm1 + esm2 + esm3 + esm4 + esm5 + esm6 + esm7 + esm8)/8


#compute post-response LSA cosine similarity
#download.file(url='http://www.lingexp.uni-tuebingen.de/z2/LSAspaces/EN_100k_lsa.rda',destfile='EN_100k_lsa.rda', method='curl')
load('EN_100k_lsa.rda')
lsa_cos=list()
for (i in 1:nrow(data)){lsa_cos[i]=costring(data$post[i], data$response[i], tvectors=EN_100k_lsa)}
#foreach(i=1:nrow(sampled_data)) %dopar% {lsa_cos[i]=costring(sampled_data$post[i], sampled_data$response[i], tvectors=EN_100k_lsa)}
data$lsa_cos=as.numeric(lsa_cos)

#compute post-level averages in resp_nchar, resp_val, lev dist, func word LSM,  emotional character similarity,  LSA similarity
#(for stressor-specific recovery analyses)
recov_data = data.frame(data$op_user_id, data$post_id, data$response_id, data$recovery_num, data$resp_nchar, data$resp_val, data$lev, data$lsm_tot, data$esm_tot, data$lsa_cos)
recov_data<- within(recov_data, {resp_nchar_postm = ave(data.resp_nchar, data.post_id, FUN=function(x) mean(x, na.rm=T))})
recov_data<- within(recov_data, {resp_val_postm = ave(data.resp_val, data.post_id, FUN=function(x) mean(x, na.rm=T))})
recov_data<- within(recov_data, {lev_postm = ave(data.lev, data.post_id, FUN=function(x) mean(x, na.rm=T))})
recov_data<- within(recov_data, {lsm_tot_postm = ave(data.lsm_tot, data.post_id, FUN=function(x) mean(x, na.rm=T))})
recov_data<- within(recov_data, {esm_tot_postm = ave(data.esm_tot, data.post_id, FUN=function(x) mean(x, na.rm=T))})
recov_data<- within(recov_data, {lsa_cos_postm = ave(data.lsa_cos, data.post_id, FUN=function(x) mean(x, na.rm=T))})
recov_data<-recov_data[c(1,2,4,11:16)]
recov_data<-unique(recov_data)

#label data and recov_data subsamples
data$subsample<- as.factor(data$op_user_id) %in%  sample(levels(droplevels(as.factor(data$op_user_id))),round(length(unique(data$op_user_id))/5), replace=F)
recov_data$subsample<- as.factor(recov_data$data.op_user_id) %in%  sample(levels(droplevels(as.factor(data$op_user_id))),round(length(unique(data$op_user_id))/5), replace=F)

#building large models that predict outcomes
b_allv_ratingt_vi<-bam(rating_num ~  s(resp_val, k=8,bs="cr") + s(lev, k=8, bs="cr") + s(lsm_tot, k=8, bs="cr") + s(esm_tot, k=8, bs="cr") + s(lsa_cos, k=8, bs="cr") + s(op_user_id, bs="re"), data=subset(data,subsample==TRUE))
b_allv_tyt_vi<-bam(ty ~ s(resp_val, k=8,bs="cr") + s(lev, k=7, bs="cr") + s(lsm_tot, k=8, bs="cr") + s(esm_tot, k=8, bs="cr") + s(lsa_cos, k=8, bs="cr") + s(op_user_id, bs="re"), data=subset(data,subsample==TRUE), family=binomial)
b_allv_ty_valt_vi<-bam(ty_val ~   s(resp_val, k=8,bs="cr") + s(lev, k=8, bs="cr") + s(lsm_tot, k=8, bs="cr") + s(esm_tot, k=8, bs="cr") + s(lsa_cos, k=8, bs="cr") + s(op_user_id, bs="re") , data=subset(data,subsample==TRUE))
b_allv_recoveryt_vi<-bam(data.recovery_num ~  s(resp_val_postm, k=8,bs="cr") + s(lev_postm, k=8, bs="cr") + s(lsm_tot_postm, k=8, bs="cr") + s(esm_tot_postm, k=8, bs="cr") + s(lsa_cos_postm, k=8, bs="cr") + s(data.op_user_id, bs="re"), data=subset(recov_data,subsample==TRUE))

#fit simple linear lme4 models for cross-referencing
b_allv_ratingt_vi<-lmer(rating_num ~  resp_val + lev + lsm_tot + esm_tot + lsa_cos + (1|op_user_id), data=subset(data,subsample==TRUE))
b_allv_tyt_vi<-glmer(ty ~ resp_val + lev + lsm_tot + esm_tot + lsa_cos + (1|op_user_id), data=subset(data,subsample==TRUE), family=binomial)
b_allv_ty_valt_vi<-lmer(ty_val ~  resp_val + lev + lsm_tot + esm_tot + lsa_cos + (1|op_user_id) , data=subset(data,subsample==TRUE))
b_allv_recoveryt_vi<-lmer(data.recovery_num ~  resp_val_postm + s(lev_postm, k=8, bs="cr") + s(lsm_tot_postm, k=8, bs="cr") + s(esm_tot_postm, k=8, bs="cr") + s(lsa_cos_postm, k=8, bs="cr") + s(data.op_user_id, bs="re"), data=subset(recov_data,subsample==TRUE))
