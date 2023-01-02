#%%

import base
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import scipy.stats
import seaborn as sns


#%%create metrics

def Special_Customers_TotalTime(d):
    
    a= d.sort_values(['CustomerId', 'Time'])
    a= a.groupby('CustomerId', as_index= False).agg({'CustomerType': 'first', 'Time': ['min', 'max'], 'Type': ['first', 'last'], 'answer_flag': 'first'})
    a= a[a[('answer_flag', 'first')].isna()]
    a= a[a['CustomerType']['first']== 'Special']

    a['duration'] = (a['Time']['max'] - a['Time']['min']).astype(float)

    mean = a['duration'].mean()
    
    return mean

#%%

def Special_Customers_NoLine(d):
    
    a= d.groupby(['CustomerId', 'Type'], as_index= False).agg({'Time': ['min', 'max', 'count'], 's_star': 'first'})
    a.columns= ['CustomerId', 'EventType', 'Time', 'time_max', 'event_count', 's_star']
    a= a[~a.EventType.isin(['StartMonth', 'Malfunction', 'EndMalfunction'])]
    a= a.drop(['time_max', 'event_count'], axis= 1)    
    
    df= pd.DataFrame({'CustomerId': list(a.CustomerId.unique())\
                      ,'EventType': [list(a['EventType'].unique())]* len(a.CustomerId.unique())})\
        .explode('EventType')
    
    a= a.merge(df,\
                         on= ['CustomerId', 'EventType'], how= 'right')
    
    b= a.pivot(index= ['CustomerId'], columns= ['EventType'], values= ['Time', 's_star'])
    b= b.reset_index()
    
    customers= d.groupby(['CustomerId'], as_index= False).agg({'CustomerType': 'first'})
    customers.columns= [('CustomerId',''), ('CustomerType','')]
    
    b= b.merge(customers, on= [('CustomerId', '')])
    b= b[b['CustomerType']== 'Special']
    
    b[('waiting_call', 'time')]= b[('Time', 'EndCall')]- b[('s_star', 'EndCall')]- b[('Time', 'Call')]
    b[('waiting_tc_call', 'time')]= b[('Time', 'TC_endofservice')]- b[('s_star', 'TC_endofservice')]- b[('Time', 'TC_arrival')]
    b[('waiting_leave', 'time')]= b[('Time', 'CustomerLeave')]- b[('Time', 'Call')]
    
    b[('waiting_total', 'time')]= (b[('waiting_call', 'time')].fillna(0)+ b[('waiting_tc_call', 'time')].fillna(0)+ b[('waiting_leave', 'time')].fillna(0)).round(5)
    b[('flag', 'wait')]= (b[('waiting_total', 'time')]== 0)
    
    return(b[('flag', 'wait')].sum()/ b.shape[0])

#%%

def line_metrics(d):
    
    line_max= pd.DataFrame(d[['QS', 'QN', 'QTS', 'QTN', 'RCN', 'RCS']].max()).reset_index()
    line_max.columns= ['Line', 'Max_Line']

    a= d[['Time', 'QS', 'QN', 'QTS', 'QTN', 'RCN', 'RCS']]
    a= pd.melt(a, value_vars=['QS', 'QN', 'QTS', 'QTN', 'RCN', 'RCS'], id_vars= ['Time'], value_name= 'Number', var_name= 'Line')
    a= a.sort_values(['Line', 'Time'])
    
    a['shift_id']= (a.Number- a.groupby('Line').Number.shift(1).fillna(0))
    a['shift_id_abs']= a['shift_id'].abs()
    a['shift_id_cum']= a.groupby('Line').shift_id_abs.cumsum()
    
    a= a.groupby(['Line', 'shift_id_cum'], as_index= False).agg({'Time': ['min', 'max'], 'Number': 'first'})
    a[('line', 'integrate')]= a[('Number', 'first')]* (a[('Time', 'max')]- a[('Time', 'min')])
    
    a= a.groupby([('Line', '')], as_index= False).agg({('line', 'integrate'): 'sum', ('Time', 'max'): 'max'})
    a[('Line', 'average')]= a[('line', 'integrate')]/ a[('Time', 'max')]
    
    a= a[[('Line', ''), ('Line', 'average')]]
    a.columns= ['Line', 'Average_Line']
    
    line_mean= a
    
    #average waiting time
    
    a= d.groupby(['CustomerId', 'Type'], as_index= False).agg({'Time': ['min', 'max', 'count'], 's_star': 'first'})
    a.columns= ['CustomerId', 'EventType', 'Time', 'time_max', 'event_count', 's_star']
    a= a[~a.EventType.isin(['StartMonth', 'Malfunction', 'EndMalfunction'])]
    a= a.drop(['time_max', 'event_count'], axis= 1)    
    a= a.reset_index(drop= True)
    
    
    df= pd.DataFrame({'CustomerId': list(a.CustomerId.unique())\
                      ,'EventType': [list(a['EventType'].unique())]* len(a.CustomerId.unique())})\
        .explode('EventType')
    
    a= a.merge(df,\
                         on= ['CustomerId', 'EventType'], how= 'right')
    
    b= a.pivot(index= ['CustomerId'], columns= ['EventType'], values= ['Time', 's_star'])
    b= b.reset_index()
    
    customers= d.groupby(['CustomerId'], as_index= False).agg({'CustomerType': 'first', 'answer_flag': 'max'})
    customers.columns= [('CustomerId',''), ('CustomerType',''), ('recall_flag', '')]
    
    
    b= b.merge(customers, on= [('CustomerId', '')])
    
    b[('waiting_call', 'time')]= b[('Time', 'EndCall')]- b[('s_star', 'EndCall')]- b[('Time', 'Call')]
    b[('waiting_tc_call', 'time')]= b[('Time', 'TC_endofservice')]- b[('s_star', 'TC_endofservice')]- b[('Time', 'TC_arrival')]
    b[('waiting_leave', 'time')]= b[('Time', 'CustomerLeave')]- b[('Time', 'Call')]
    
    b[('waiting_total', 'time')]= (b[('waiting_call', 'time')].fillna(0)+ b[('waiting_tc_call', 'time')].fillna(0)+ b[('waiting_leave', 'time')].fillna(0)).round(5)
    
    del df
    b= b[[('CustomerType', ''), ('recall_flag', ''), ('waiting_call', 'time'), ('waiting_tc_call', 'time'), ('waiting_leave', 'time')]]
    b.columns= ['CustomerType', 'recall_flag', 'waiting_call', 'waiting_tc_call', 'waiting_leave']
    
    
    a= pd.melt(b, id_vars= ['CustomerType', 'recall_flag'], value_vars= ['waiting_call', 'waiting_tc_call', 'waiting_leave'], var_name= 'Line', value_name= 'Time')
    a= a[a.CustomerType.notna()]
    a['Line']= a['Line']+ ' '+ a['CustomerType']+ ' '+np.where(a['recall_flag'].isna(), 'NR', 'R')
    a= a.groupby(['Line'], as_index= False).agg({'Time': ['mean', 'max']})
    
	
    line_dict= {'waiting_call Normal NR': 'QN',
                'waiting_call Special NR': 'QS',
                'waiting_call Special R': 'RCN',
                'waiting_tc_call Normal NR': 'QTN',
                'waiting_tc_call Special NR': 'QTS',
                'waiting_tc_call Special R': 'RCS'
                }

    a[('Line', '')]= a[('Line', '')].map(line_dict)
    a.columns= ['Line', 'Time_mean', 'Time_max']
    a= a[a.Line.notna()]    
    line_wait= a
    
    del a, b, customers, line_dict
    
    line_max= line_max.sort_values('Line').reset_index(drop= True)
    line_mean= line_mean.sort_values('Line').reset_index(drop= True)
    line_wait= line_wait.sort_values('Line').reset_index(drop= True)
    
    line_data= pd.concat([line_max, line_mean['Average_Line'], line_wait[['Time_mean', 'Time_max']]], axis= 1)
    return(line_data)


def QN_Average_Line(d):
    a= line_metrics(d).iloc[1]['Average_Line']
    return(a)


#%%

def Agents_Average_Productivity(d, agent_numbers= [3, 2, 2]):
    
    a= d[['Type', 'AgentType', 's_star', 'Time']]
    
    a= a[a.Type.isin(['EndCall', 'TC_endofservice'])]
    a['AgentType']= a['AgentType'].fillna(-1)
    a['s_star']= a['s_star'].astype(float)
    
    a= a.groupby(['Type', 'AgentType'], as_index= False).agg({'s_star': 'sum', 'Time': ['max', 'min']})
    
    a= a.sort_values(['Type', 'AgentType'])
    a['Agent_Number']= agent_numbers
    
    a['Agent']= ['SN', 'SS', 'TS']
    a['Productivity']= (a[('s_star', 'sum')]/a['Agent_Number'])/ (a[('Time', 'max')].max()- a[('Time', 'min')].min())
    a= a[['Agent', 'Productivity']]
    a.columns= ['Metric', 'Productivity']
    a['Metric']= a['Metric']+ '_Productivity'
    
    return(a)

#%%

def Call_Centre_Lost_Call_Proportion(d):
    
    a= d[['CustomerId', 'Type', 'Time', 'answer_flag']]
    a= a[~a.Type.isin(['StartMonth', 'Malfunction', 'EndMalfunction'])]
    
    b= a.set_index(['CustomerId'])
    b= b.pivot(columns= ['Type'], values= ['Time'])
    b= b.reset_index()
    
    customers= d.groupby(['CustomerId'], as_index= False).agg({'CustomerType': 'first', 'answer_flag': 'max'})
    customers.columns= [('CustomerId',''), ('CustomerType',''), ('recall_flag', '')]
    
    b= b.merge(customers, on= [('CustomerId', '')])
    b['Lost_Recall']= b['recall_flag'].map({0: 1}).fillna(0)
    b['Lost_Churn']= b[('Time', 'CustomerLeave')].where(b[('Time', 'CustomerLeave')].isna(), 1).fillna(0)
    b['Total_Lost']= b['Lost_Recall']+ b['Lost_Churn']
    
    a= b.groupby('CustomerType').agg({('Lost_Recall', ''): 'sum', ('Lost_Churn', ''): 'sum', ('Total_Lost', ''): 'sum', ('CustomerId', ''): 'count'}).reset_index()
    a.columns= ['CustomerType', 'Lost_Recall', 'Lost_Churn', 'Total_Lost', 'CustomerId']

    a['Total_Lost']= a['Total_Lost']/ a['CustomerId']
    a['Lost_Churn']= a['Total_Lost']/ a['CustomerId']
    a['Lost_Recall']= a['Total_Lost']/ a['CustomerId']
    
    return(a)
    
#%%

def get_all_metrics(d, agent_numbers= [3, 2, 2]):
    
    lines= line_metrics(d)
    lines= pd.melt(lines, id_vars= ['Line'], value_vars=['Max_Line', 'Average_Line', 'Time_mean', 'Time_max'], var_name= 'Metric')
    lines['Metric']= lines['Line']+ '_'+ lines['Metric']
    lines= lines.drop(['Line'], axis= 1)
    
    lost= Call_Centre_Lost_Call_Proportion(d)
    lost= pd.melt(lost, id_vars= ['CustomerType'], value_vars=['Total_Lost', 'Lost_Churn', 'Lost_Recall'], var_name= 'Metric')
    lost['Metric']= lost['CustomerType']+ '_'+ lost['Metric']
    lost= lost.drop(['CustomerType'], axis= 1)
    
    productivity= Agents_Average_Productivity(d)
    productivity= productivity.rename(columns= {'Productivity': 'value'})
    
    number_metrics=pd.DataFrame({'Metric': ['Special_Customers_TotalTime', 'Special_Customers_NoLine'],
                                 'value': [Special_Customers_TotalTime(d), Special_Customers_NoLine(d)]
                                 })
    
    all_metrics= pd.concat([number_metrics, lines, lost, productivity])
    return(all_metrics)


#%%
#sensitivity Analysis
def sa(parameter_dict, metricFunc, metricName, inputName, beginAt, stopAt, stepNumber, inputName2=None, beginAt2=None, stopAt2=None, stepNumber2=None, plot=False):
    ### inputName : string, could be the name of any globally defined variable

    if (inputName2, beginAt2, stopAt2, stepNumber2) == (None, None, None, None):

        outLine = list()
        inLine = list()
        length =  stopAt - beginAt

        for i in range(stepNumber + 1):
            inputPointer = beginAt + (i / stepNumber) * length
            inLine.append(inputPointer)

            globals()[str(inputName)] = inputPointer

            output = base.run_simulation(parameter_dict)
            metricResult = metricFunc(output)
            outLine.append(metricResult)

        if plot == False:
            return inLine, outLine
        
        elif plot == True:

            fig, ax = plt.subplots()

            mpl.rc('font', family='Times New Roman')
            mpl.rc('font', size=12)

            ax.plot(inLine, outLine, 'go--', linewidth=3, markersize=9)
            ax.set_title(str(metricName) + " per different values of " + inputName)

            ax.set_xlabel(inputName)
            ax.set_ylabel(metricName)

            ax.grid(True)

            return inLine, outLine, fig.show()

    elif (inputName2 != None and beginAt2 != None and stopAt2 != None and stepNumber2 != None):
        outLine = list()
        inLine1 = list()
        inLine2 = list()
        length = stopAt - beginAt
        length2 = stopAt2 - beginAt2

        for i in range(stepNumber + 1):
            outLine.append([])
            for j in range(stepNumber2 + 1):
                inputPointer = beginAt + (i / stepNumber) * length
                inputPointer2 = beginAt2 + (j / stepNumber2) * length2
                inLine1.append(inputPointer)
                inLine2.append(inputPointer2)

                globals()[str(inputName)] = inputPointer
                globals()[str(inputName2)] = inputPointer2

                output = base.run_simulation(parameter_dict)
                metricResult = metricFunc(output)
                outLine[i].append(metricResult)

        if plot == False:
            return inLine, outLine
        
        elif plot == True:
            fig = sns.heatmap(outLine, annot=True)
            fig.set(xlabel=inputName2, ylabel=inputName)
            return inLine1, inLine2, outLine, plt.show()

#%%
#creating confiedence interval for outputs
def replication(parameter_dict, simulation_time= 30*24*60, r= 3, malfunction_flag= False, seeds= None, alpha= 0.05, todelete_t0= 0):
    
    if seeds:
        pass
    else:
        seeds= [None]* r
    
    data= []

    for i in tqdm(range(1, 1+r)):
        d= base.run_simulation(simulation_time= simulation_time, parameter_dict= parameter_dict, malfunction_flag= malfunction_flag, seed= seeds[i-1])
        all_metrics= get_all_metrics(d)
        all_metrics['rep']= i
        data.append(all_metrics)
        
    
    a= pd.concat(data)
    a= a.groupby(['Metric']).agg({'value': ['mean', 'std', 'count']}).reset_index()
    a.columns= ['Metric', 'mean', 'std', 'count']
    a['alpha']= alpha
    a['qt']= scipy.stats.t.ppf(1- alpha/2, r-1)
    a['Lower_Bound']= a['mean']- a['std']/ (r)* a['qt']
    a['Upper_Bound']= a['mean']+ a['std']/ (r)* a['qt']
    
    return a

#%%

import statsmodels.api as sm

def batch_means(parameter_dict, time, todelete_t0, k, seed, malfunction_flag= False, alpha= 0.05):
    
    d= base.run_simulation(simulation_time= time, parameter_dict= parameter_dict, malfunction_flag= malfunction_flag, seed= seed)
    d= d[d.Time>todelete_t0]
    
    d_splitted= np.array_split(d, k)
    del d
    
    d_metrics= [get_all_metrics(d, agent_numbers= [parameter_dict['normal_agent_number'], parameter_dict['special_agent_number'], parameter_dict['technical_agent_number']]) for d in tqdm(d_splitted)]
    del d_splitted


    for i in range(len(d_metrics)):
        
        d_metrics[i]['batch']= i
    
    del i        
    m= pd.concat(d_metrics)    
    del d_metrics
    
    m= m.sort_values(['Metric', 'batch'])
    
    autocorr_coef= pd.DataFrame({
        'Metirc': m.Metric.unique(),
        'coef': [i[1] for i in m.groupby('Metric').value.apply(sm.tsa.acf).values]
        })
    
    
    a= m.groupby(['Metric']).agg({'value': ['mean', 'std', 'count']}).reset_index()
    a.columns= ['Metric', 'mean', 'std', 'count']
    a['alpha']= alpha
    a['qt']= scipy.stats.t.ppf(1- alpha/2, k-1)
    a['Lower_Bound']= a['mean']- a['std']/ (k)* a['qt']
    a['Upper_Bound']= a['mean']+ a['std']/ (k)* a['qt']
    
    return({
        
        'table_corr': autocorr_coef,
        'table_result': a
        
        })

#%%
import random
import math
def exponential(beta, n):
    #beta is E(X)
    rs= []
    
    for i in range(n):
        r = random.random()
        rs.append(-beta * math.log(r))
     
    return(rs)

def descrete_uniform(n, start= 0, end= 30):

    #creates int between start and end-1
    
    rs= []
    
    for i in range(n):
        r= random.random()
        rs.append(int(start+ r* (end- start)))
    
    return(rs)

def bern(p, n):
    
    rs= []
    
    for i in range(n):
        r= random.random()
        if r<=p:
            rs.append(1)
        else:
            rs.append(0)
    
    return(rs)

def uniform(a, b, n):
    
    rs= []
    for i in range(n):
        r = random.random()
        rs.append(a + (b - a) * r)
        
    return(rs) 


#%%
import copy
def CRN(parameter_dict, simulation_time= 30*24*60, r= 3, malfunction_flag= False, seeds= None, crn_seed= 10, alpha= 0.05, todelete_t0= 0):
    
    crn_data= {
        'shift1_times': [],
        'shift2_times': [],
        'shift3_times': [],
        
        'shift1_times_mal': [],
        'shift2_times_mal': [],
        'shift3_times_mal': [],
        
        
        'D1_times': [],
        'D2_times': [],
        'D3_times': [],
        
        'Special_Customers': [],
        'Churn_Customers': [],
        'Recall_Customers': [],
        
        }
    
    random.seed(crn_seed)
    
    crn_data['shift1_times']= exponential(parameter_dict['lambda1'], int(simulation_time/parameter_dict['lambda1'])*3)
    crn_data['shift2_times']= exponential(parameter_dict['lambda2'], int(simulation_time/parameter_dict['lambda2'])*3)
    crn_data['shift3_times']= exponential(parameter_dict['lambda3'], int(simulation_time/parameter_dict['lambda3'])*3)
    
    crn_data['shift1_times_mal']= exponential(parameter_dict['lambda1_mal'], int(simulation_time/parameter_dict['lambda1_mal'])*3)
    crn_data['shift2_times_mal']= exponential(parameter_dict['lambda2_mal'], int(simulation_time/parameter_dict['lambda2_mal'])*3)
    crn_data['shift3_times_mal']= exponential(parameter_dict['lambda3_mal'], int(simulation_time/parameter_dict['lambda3_mal'])*3)
    
    crn_data['D1_times']= exponential(parameter_dict['D1'], int(simulation_time/parameter_dict['D1']*8))
    crn_data['D2_times']= exponential(parameter_dict['D2'], int(simulation_time/parameter_dict['D2']*8))
    crn_data['D3_times']= exponential(parameter_dict['D3'], int(simulation_time/parameter_dict['D3']*8))
    
    crn_data['Special_Customers']= bern(parameter_dict['special_prob'], int(int(simulation_time)/ (parameter_dict['lambda1']+ parameter_dict['lambda2']+ parameter_dict['lambda3'])*10))
    crn_data['Churn_Customers']= bern(parameter_dict['churn_prob'], int(int(simulation_time)/ (parameter_dict['lambda1']+ parameter_dict['lambda2']+ parameter_dict['lambda3'])*10))
    crn_data['Recall_Customers']= bern(parameter_dict['recall_prob'], int(int(simulation_time)/ (parameter_dict['lambda1']+ parameter_dict['lambda2']+ parameter_dict['lambda3'])*10))
    crn_data['TC_Customers']= bern(parameter_dict['TC_prob'], int(int(simulation_time)/ (parameter_dict['lambda1']+ parameter_dict['lambda2']+ parameter_dict['lambda3'])*10))
    
    crn_data_= copy.deepcopy(crn_data)
    
    if seeds:
        pass
    else:
        seeds= [None]* r
    
    data= []

    for i in tqdm(range(1, 1+r)):
        d= base.run_simulation(simulation_time= simulation_time, crn_data= crn_data, parameter_dict= parameter_dict, malfunction_flag= malfunction_flag, seed= seeds[i-1])
        
        crn_data= copy.deepcopy(crn_data_)
        all_metrics= get_all_metrics(d)
        all_metrics['rep']= i
        data.append(all_metrics)
        
    
    a= pd.concat(data)
    a= a.groupby(['Metric']).agg({'value': ['mean', 'std', 'count']}).reset_index()
    a.columns= ['Metric', 'mean', 'std', 'count']
    a['alpha']= alpha
    a['qt']= scipy.stats.t.ppf(1- alpha/2, r-1)
    a['Lower_Bound']= a['mean']- a['std']/ (r)* a['qt']
    a['Upper_Bound']= a['mean']+ a['std']/ (r)* a['qt']
    
    return a

#%%





