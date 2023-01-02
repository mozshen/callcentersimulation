
#%%

import base as base
import get_results

#%%
#define basic parameters
parameter_dict= {
    
    'special_agent_number': 2,
    'normal_agent_number': 3,
    'technical_agent_number': 2,
    
    
    'lambda1': 3,
    'lambda2': 1,
    'lambda3': 2,
    'lambda1_mal': 2,
    'lambda2_mal': 0.5,
    'lambda3_mal': 1,
    'D1': 180/60,
    'D2': 420/60,
    'D3': 600/60,
    'TC_prob': 0.15,  ## portion of technical service
    'special_prob': 0.3, ## special percentage
    'recall_prob': 0.5, ## ppl who choose to be recalled by the call center
    'churn_prob': 0.15, ## percentage of ppl who leave the queue unconditionally
    'patience_line' : 25, ## (minutes) higher bound of patience time
    'min_patience': 5, ##  (minutes) lower bound of patience time
    'recalAnswerProb' : 0.5
}

#%%
#resulting data
a= get_results.replication(parameter_dict,10, seeds= [1,2,3,4,5,6,7,8,9,10])
d= base.run_simulation(simulation_time= 20*30*24*60, parameter_dict= parameter_dict, seed= 71)


s= d[d.CustomerId== 307811]
a= d[(d.Time>490000) & (d.Time<500000)]


a.to_csv('Result_Data.csv')
#%%
#resulting 
#all function only returns table varieble
d= base.all(seed= 40)
d.to_csv('Table.csv')

#%%
#sesitivity analysis example:

get_results.sa(parameter_dict, get_results.Special_Customers_TotalTime,'Special Customers TotalTime', 'churn_prob', 0.08, 0.22, 7, 'recall_prob', 0.4, 0.6, 6, plot=True)


get_results.sa(parameter_dict, get_results.Special_Customers_NoLine,'Special Customers NoLine', 'lambda2', 0.5, 6, 15, plot=True)

get_results.sa(parameter_dict, get_results.Special_Customers_TotalTime,'Special Customers TotalTime', 'lambda1', 0.5, 35, 70, plot=True)

#%%