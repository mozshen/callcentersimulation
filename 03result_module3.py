
#%%

import get_results
from scipy.stats import t

#%%
#define basic parameters
parameter_dict1= {
    
    'special_agent_number': 2,
    'normal_agent_number': 3,
    'technical_agent_number': 2,
    
    
    'lambda1': 1.1,
    'lambda2': 1.1,
    'lambda3': 1.1,
    'lambda1_mal': 2,
    'lambda2_mal': 0.5,
    'lambda3_mal': 1,
    'D1': 180/60,
    'D2': 420/60,
    'D3': 600/60,
    'TC_prob': 0.15,  ## portion of technical service
    'special_prob': 0.4, ## special percentage
    'recall_prob': 0, ## ppl who choose to be recalled by the call center
    'churn_prob': 0.15, ## percentage of ppl who leave the queue unconditionally
    'patience_line' : 25, ## (minutes) higher bound of patience time
    'min_patience': 5, ##  (minutes) lower bound of patience time
    'recalAnswerProb' : 0.5
}

#%%
#resulting data
#part A
#We get result for system 1
a= get_results.batch_means(parameter_dict= parameter_dict1, malfunction_flag= False, seed= 10, time= 5*365*24*60, todelete_t0= 6*30*24*60, k= 30, alpha= 0.05)

#%%
#define basic parameters
#Defining System B Parameters
parameter_dict2= {
    
    'special_agent_number': 2,
    'normal_agent_number': 2,
    'technical_agent_number': 2,
    
    
    'lambda1': 1.1,
    'lambda2': 1.1,
    'lambda3': 1.1,
    
    'lambda1_mal': 2,
    'lambda2_mal': 0.5,
    'lambda3_mal': 1,
    
    'D1': 2.7,
    'D2': 5.8,
    'D3': 600/60,
    'TC_prob': 0.15,  ## portion of technical service
    'special_prob': 0.4, ## special percentage
    'recall_prob': 0, ## ppl who choose to be recalled by the call center
    'churn_prob': 0.15, ## percentage of ppl who leave the queue unconditionally
    'patience_line' : 25, ## (minutes) higher bound of patience time
    'min_patience': 5, ##  (minutes) lower bound of patience time
    'recalAnswerProb' : 0.5
}

#%%

#Replication Method

system_1= get_results.replication(parameter_dict= parameter_dict1, malfunction_flag= False, simulation_time= 2*365*24*60, todelete_t0= 6*30*24*60, r= 10, seeds= range(1, 11), alpha= 0.05)
system_2= get_results.replication(parameter_dict= parameter_dict2, malfunction_flag= False, simulation_time= 2*365*24*60, todelete_t0= 6*30*24*60, r= 10, seeds= range(1, 11), alpha= 0.05)

compare= system_1[['Metric', 'mean', 'std', 'count']]\
    .merge(system_2[['Metric', 'mean', 'std', 'count']],
           on= ['Metric'])

compare['t_statistic']= (compare['mean_x']- compare['mean_y'])/\
    (compare['std_x']**2/compare['count_x'] + compare['std_y']**2/ compare['count_y'])**0.5

compare['v']= ((compare['std_x']**2/compare['count_x'] + compare['std_y']**2/ compare['count_y'])**2/\
    ((compare['std_x']**2/compare['count_x'])**2/ (compare['count_x']- 1) + (compare['std_y']**2/compare['count_y'])**2/ (compare['count_y']- 1)))

compare['degf']= round(compare['v']).fillna(0).astype(int)


compare['p_value']= t.cdf(x= compare['t_statistic'], df= compare['degf'])
del system_1, system_2

#%%

#CRN Method

system_1= get_results.CRN(parameter_dict= parameter_dict1, malfunction_flag= False, simulation_time= 30*24*60, todelete_t0= 5*24*60, r= 5, seeds= range(1, 6), crn_seed= 10, alpha= 0.05)
system_2= get_results.CRN(parameter_dict= parameter_dict2, malfunction_flag= False, simulation_time= 30*24*60, todelete_t0= 5*24*60, r= 5, seeds= range(1, 6), crn_seed= 10, alpha= 0.05)

compare= system_1[['Metric', 'mean', 'std', 'count']]\
    .merge(system_2[['Metric', 'mean', 'std', 'count']],
           on= ['Metric'])

compare['t_statistic']= (compare['mean_x']- compare['mean_y'])/\
    (compare['std_x']**2/compare['count_x'] + compare['std_y']**2/ compare['count_y'])**0.5

compare['v']= ((compare['std_x']**2/compare['count_x'] + compare['std_y']**2/ compare['count_y'])**2/\
    ((compare['std_x']**2/compare['count_x'])**2/ (compare['count_x']- 1) + (compare['std_y']**2/compare['count_y'])**2/ (compare['count_y']- 1)))

compare['degf']= round(compare['v']).fillna(0).astype(int)


compare['p_value']= t.cdf(x= compare['t_statistic'], df= compare['degf'])
del system_1, system_2

#%%






