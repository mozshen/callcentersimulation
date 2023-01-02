
"""
Discrete Event Simulation Project
6th semester
first half of 2022
"""

#%%

import random
import pandas as pd
import numpy as np
import math

#%%

import psutil, os

def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 
        
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

#%%

def run_simulation(parameter_dict, crn_data= None, malfunction_flag= True, simulation_time = 30 * 24 * 60, seed= None, store_fel= False):

    #set seed 
    if seed:
        random.seed(seed)
    else:
        random.seed()

    special_agent_number= parameter_dict['special_agent_number']
    normal_agent_number= parameter_dict['normal_agent_number']
    technical_agent_number= parameter_dict['technical_agent_number']
    
    lambda1= parameter_dict['lambda1']
    lambda2= parameter_dict['lambda2']
    lambda3= parameter_dict['lambda3']
    
    lambda1_mal= parameter_dict['lambda1_mal']
    lambda2_mal= parameter_dict['lambda2_mal']
    lambda3_mal= parameter_dict['lambda3_mal']
    
    D1= parameter_dict['D1']
    D2= parameter_dict['D2']
    D3= parameter_dict['D3']
    
    TC_prob= parameter_dict['TC_prob'] ## portion of technical service
    special_prob= parameter_dict['special_prob']## special percentage
    recall_prob= parameter_dict['recall_prob'] ## ppl who choose to be recalled by the call center
    churn_prob= parameter_dict['churn_prob'] ## percentage of ppl who leave the queue unconditionally
    
    patience_line = parameter_dict['patience_line'] ## (minutes) higher bound of patience time
    min_patience= parameter_dict['min_patience'] ##  (minutes) lower bound of patience time
    recalAnswerProb = parameter_dict['recalAnswerProb']

    
    #define all queues
    queues= {
        'QS': [],
        'QN': [],
        'RCN': [],
        'RCS': [],
        'QTS': [],
        'QTN': []
    
        }

    
    #  initializing: starting first event
    def set_state(start_state, first_event):

        state_variebles= start_state
        fel= first_event

        return state_variebles, fel


    #add to fel functions adds requiered events to current fel based on the
    #passed event
    #event is a dictionary with all pf that event's parameters
    
    def add_to_fel(fel, event, clock):

        if event['Type']== 'StartMonth':

            delta_time= 30*24*60
            event['Time']= clock+ delta_time
            fel.append(event)

        if event['Type']== 'Malfunction':

            delta_time= event['d']*24* 60
            #event.pop('d')

            event['Time']= clock+ delta_time
            fel.append(event)

        if event['Type']== 'EndMalfunction':

            delta_time= 24*60
            event['Time']= clock+ delta_time
            fel.append(event)

        if event['Type']== 'TC_endofservice':

            delta_time= event['s_star']
            #event.pop('s_star')

            event['Time']= clock+ delta_time
            fel.append(event)

        if event['Type']== 'TC_arrival':

            delta_time= event['s_star']
            #event.pop('s_star')

            event['Time']= clock+ delta_time
            fel.append(event)

        if event['Type']== 'EndCall':

            delta_time= event['s_star']
            #event.pop('s_star')

            event['Time']= clock+ delta_time
            fel.append(event)

        if event['Type']== 'CustomerLeave':

            delta_time= event['s_star']
            #event.pop('s_star')

            event['Time']= clock+ delta_time
            fel.append(event)

        if event['Type']== 'Call':

            delta_time= event['s_star']
            #event.pop('s_star')

            event['Time']= clock+ delta_time
            fel.append(event)

    #functions for creating random variebles
    def exponential(beta):
        #beta is E(X)
        r = random.random()
        return -beta * math.log(r)

    def descrete_uniform(start= 0, end= 30):

        #creates int between start and end-1
        r= random.random()
        return(int(start+ r* (end- start)))

    def bern(p):

        r= random.random()
        if r<=p:
            return(1)
        else:
            return(0)

    def uniform(a, b):
        r = random.random()
        return a + (b - a) * r


    #below are all events. each functions related to only one event.


    def startmonth(fel, state_variebles, clock):

        d= descrete_uniform()

        event= {'Type':'Malfunction', 'd': d, 'CustomerId': -1}
        add_to_fel(fel, event, clock)

        event= {'Type':'StartMonth', 'CustomerId': -1}
        add_to_fel(fel, event, clock)

    
    ### setting lambdas as defualt variables to be remembered

    def malfunction(fel, state_variebles, clock, lambda1_mal, lambda2_mal, lambda3_mal):

        state_variebles['Mal']= 1

        state_variebles['Lambda1']= lambda1_mal
        state_variebles['Lambda2']= lambda2_mal
        state_variebles['Lambda3']= lambda3_mal

        event= {'Type':'EndMalfunction', 'CustomerId': -1}
        add_to_fel(fel, event, clock)

    def end_malfunction(fel, state_variebles, clock, lambda1= lambda1, lambda2= lambda2, lambda3= lambda3):

        state_variebles['Mal']= 0

        state_variebles['Lambda1']= lambda1
        state_variebles['Lambda2']= lambda2
        state_variebles['Lambda3']= lambda3

    def TC_endofservice(fel, state_variebles, clock, D3):

        state_variebles['TS']= state_variebles['TS']- 1

        if state_variebles['QTS']>0:
            
            customer= sorted(queues['QTS'], key=lambda x: (x['Time'], x['CustomerId']))[0]
            customerid= customer['CustomerId']
            customertype= customer['CustomerType']
            queues['QTS'].remove(customer)
            
            state_variebles['TS']+=1
            state_variebles['QTS']-=1
            
            if crn_data:
                s_star= crn_data['D3_times'].pop(0)
            
            else:
                s_star= exponential(D3)

            event= {'Type':'TC_endofservice', 'CustomerType': customertype, 'CustomerId': customerid, 's_star': s_star}
            add_to_fel(fel, event, clock)

        else:

            if state_variebles['QTN']>0:

                customer= sorted(queues['QTN'], key=lambda x: (x['Time'], x['CustomerId']))[0]
                customerid= customer['CustomerId']
                customertype= customer['CustomerType']
                queues['QTN'].remove(customer)
                
                state_variebles['TS']+=1
                state_variebles['QTN']-=1

                
                if crn_data:
                    s_star= crn_data['D3_times'].pop(0)
                
                else:
                    s_star= exponential(D3)

                
                event= {'Type':'TC_endofservice', 'CustomerType': customertype, 'CustomerId': customerid, 's_star': s_star}
                add_to_fel(fel, event, clock)

    def TC_arrival(fel, state_variebles, clock, customerid, customertype, D3):

        if state_variebles['TS']<technical_agent_number:

            state_variebles['TS']+=1
            
            if crn_data:
                s_star= crn_data['D3_times'].pop(0)
            
            else:
                s_star= exponential(D3)

            
            event= {'Type':'TC_endofservice', 'CustomerType': customertype, 'CustomerId': customerid, 's_star': s_star}
            add_to_fel(fel, event, clock)

        else:

            if customertype== 'Normal':

                state_variebles['QTN']+=1
                queues['QTN'].append({'CustomerId': customerid, 'CustomerType': customertype, 'Time': clock})

            else:

                state_variebles['QTS']+=1
                queues['QTS'].append({'CustomerId': customerid,'CustomerType': customertype, 'Time': clock})



    def EndCall(fel, state_variebles, clock, customerid, customertype, agenttype, TC_prob, D2, D1):
        
        
        if crn_data:
            e= crn_data['TC_Customers'].pop(0)
        
        else:
            e= bern(TC_prob)        
        
        ## e = 1: customer needs tc
        ## e = 0: exit system

        if e==1:

            s_star=0 #we assume that there is no delay at refering to technical centre
            event= {'Type':'TC_arrival', 'CustomerType': customertype, 'CustomerId': customerid, 's_star': s_star}
            add_to_fel(fel, event, clock)

        ## normal agent

        if agenttype== 'Normal':

            state_variebles['SN']-=1

            ## no recall / continueing service
            if state_variebles['QN']>0:

                state_variebles['SN']+=1
                state_variebles['QN']-=1
                
                customer= sorted(queues['QN'], key=lambda x: (x['Time'], x['CustomerId']))[0]
                customerid= customer['CustomerId']
                customertype= customer['CustomerType']
                queues['QN'].remove(customer)
                
                if crn_data:
                    s_star= crn_data['D2_times'].pop(0)
                
                else:
                    s_star= exponential(D2)

                event= {'Type':'EndCall', 'CustomerType': customertype, 'AgentType': agenttype, 'CustomerId': customerid, 's_star': s_star}
                add_to_fel(fel, event, clock)
                
                try:
                    #leave events are removed since the service is started for the customer who wanted to leave
                    leave_event= [item for item in fel if ((item['CustomerId']== customerid) & (item['Type']== 'CustomerLeave'))][0]
                    fel.remove(leave_event)
                
                except:
                    pass
                    
                
            ## recalling ppl
            else:
                
                shift_number = int((clock% (24*60))/ (8*60))+1

                if shift_number >= 2:
                    
                    if state_variebles['RCN']>0:

                        state_variebles['RCN']-=1
                        
                        
                        answer = bern(recalAnswerProb)
                        
                        if answer == 1:
                            
                            state_variebles['SN']+=1
                            
                            if crn_data:
                                s_star= crn_data['D2_times'].pop(0)
                            
                            else:
                                s_star= exponential(D2)

                        else:
                            state_variebles['SN']+=1
                            s_star = 0

                        customer= sorted(queues['RCN'], key=lambda x: (x['Time'], x['CustomerId']))[0]
                        customerid= customer['CustomerId']
                        customertype= customer['CustomerType']
                        queues['RCN'].remove(customer)
                        
                        event= {'Type':'EndCall', 'CustomerType': customertype, 'AgentType': agenttype, 'CustomerId': customerid, 's_star': s_star, 'answer_flag': answer}
                        
                        add_to_fel(fel, event, clock)
                        
                
                
        ## special agent
        else:

            state_variebles['SS']-=1

            if state_variebles['QS']>0:

                state_variebles['SS']+=1
                state_variebles['QS']-=1
                
                customer= sorted(queues['QS'], key=lambda x: (x['Time'], x['CustomerId']))[0]
                customerid= customer['CustomerId']
                customertype= customer['CustomerType']
                queues['QS'].remove(customer)
                
                if crn_data:
                    s_star= crn_data['D1_times'].pop(0)
                
                else:
                    s_star= exponential(D1)

                event= {'Type':'EndCall', 'CustomerType': customertype, 'AgentType': agenttype, 'CustomerId': customerid, 's_star': s_star}
                add_to_fel(fel, event, clock)
                
                try:
                    #leave events are removed since the service is started for the customer who wanted to leave
                    leave_event= [item for item in fel if ((item['CustomerId']== customerid) & (item['Type']== 'CustomerLeave'))][0]
                    fel.remove(leave_event)
                
                except:
                    pass
                

            else:

                if state_variebles['QN']>0:

                    state_variebles['SS']+=1
                    state_variebles['QN']-=1
                    
                    customer= sorted(queues['QN'], key=lambda x: (x['Time'], x['CustomerId']))[0]
                    customerid= customer['CustomerId']
                    customertype= customer['CustomerType']
                    queues['QN'].remove(customer)
                    
                    if crn_data:
                        s_star= crn_data['D1_times'].pop(0)
                    
                    else:
                        s_star= exponential(D1)

                    event= {'Type':'EndCall', 'CustomerType': customertype, 'AgentType': agenttype, 'CustomerId': customerid, 's_star': s_star}
                    add_to_fel(fel, event, clock)
                    
                    try:
                        #leave events are removed since the service is started for the customer who wanted to leave
                        leave_event= [item for item in fel if ((item['CustomerId']== customerid) & (item['Type']== 'CustomerLeave'))][0]
                        fel.remove(leave_event)
                    
                    except:
                        pass
                    

                else:

                    shift_number= int((clock% (24*60))/ (8*60))+1

                    if shift_number>=2:

                        if state_variebles['RCS']>0:

                            state_variebles['RCS']-=1
                            
                            customer= sorted(queues['RCS'], key=lambda x: (x['Time'], x['CustomerId']))[0]
                            customerid= customer['CustomerId']
                            customertype= customer['CustomerType']
                            queues['RCS'].remove(customer)
                            
                            answer = bern(recalAnswerProb)
                            if answer == 1:
                                state_variebles['SS']+=1
                                
                                if crn_data:
                                    s_star= crn_data['D1_times'].pop(0)
                                
                                else:
                                    s_star= exponential(D1)

                                
                                
                            else:
                                state_variebles['SS']+=1
                                s_star = 0

                            event= {'Type':'EndCall', 'CustomerType': customertype, 'AgentType': agenttype, 'CustomerId': customerid, 's_star': s_star, 'answer_flag': answer}
                            add_to_fel(fel, event, clock)

                        else:

                            if state_variebles['RCN']>0:

                                state_variebles['RCN']-=1

                                customer= sorted(queues['RCN'], key=lambda x: (x['Time'], x['CustomerId']))[0]
                                customerid= customer['CustomerId']
                                customertype= customer['CustomerType']
                                queues['RCN'].remove(customer)

                                answer = bern(recalAnswerProb)
                                if answer == 1:
                                    
                                    if crn_data:
                                        s_star= crn_data['D1_times'].pop(0)
                                    
                                    else:
                                        s_star= exponential(D1)

                                    
                                    state_variebles['SS']+=1
                                else:
                                    state_variebles['SS']+=1
                                    s_star = 0


                                event= {'Type':'EndCall', 'CustomerType': customertype, 'AgentType': agenttype, 'CustomerId': customerid, 's_star': s_star, 'answer_flag': answer}
                                add_to_fel(fel, event, clock)



    def CustomerLeave(fel, state_variebles, clock, customerid, customertype):
        
        if customertype== 'Normal':
            
            left_customer= [item for item in queues['QN'] if ((item['CustomerId']== customerid))][0]
            queues['QN'].remove(left_customer)
            
            state_variebles['QN']-=1
                
        else:
            
            left_customer= [item for item in queues['QS'] if ((item['CustomerId']== customerid))][0]
            queues['QS'].remove(left_customer)
            
            
            state_variebles['QS']-=1
                
    def Call(fel, state_variebles, clock, customerid, special_prob, recall_prob, churn_prob, patience_line, min_patience, lambda1, lambda2, lambda3):

        
        if crn_data:
            f= crn_data['Special_Customers'].pop(0)
        
        else:
            f= bern(special_prob)


        if f==1:

            customertype= 'Special'

            if state_variebles['SS']<special_agent_number:

                state_variebles['SS']+=1

                if crn_data:
                    s_star= crn_data['D1_times'].pop(0)
                
                else:
                    s_star= exponential(D1)

                event= {'Type':'EndCall', 'CustomerType': customertype, 'AgentType': 'Special', 'CustomerId': customerid, 's_star': s_star}
                add_to_fel(fel, event, clock)

            else:

                state_variebles['QS']+=1
                
                #we add customer to line and remove it if he/she wants to recall or leave or  him/her service ends
                queues['QS'].append({'CustomerId': customerid, 'CustomerType': customertype, 'Time': clock})

                if state_variebles['QS']>4:
                        
                    if crn_data:
                        g= crn_data['Recall_Customers'].pop(0)
                    
                    else:
                        g= bern(recall_prob)

                    
                    

                    if g==1:

                        state_variebles['RCS']+=1
                        queues['RCS'].append({'CustomerId': customerid, 'CustomerType': customertype, 'Time': clock})
                        
                        state_variebles['QS']-=1
                        queues['QS'].pop(-1) #the last element is the customer sinece there has been no sorting

                    else:
                        
                        if crn_data:
                            h= crn_data['Churn_Customers'].pop(0)
                        
                        else:
                            h= bern(churn_prob)



                        if h==1:

                            x= max(patience_line, state_variebles['QS'])

                            s_star= uniform(min_patience, x)

                            event= {'Type':'CustomerLeave', 'CustomerType': customertype, 'CustomerId': customerid, 's_star': s_star}
                            add_to_fel(fel, event, clock)

                else:

                    if crn_data:
                        h= crn_data['Churn_Customers'].pop(0)
                    
                    else:
                        h= bern(churn_prob)


                    if h==1:

                        x= max(patience_line, state_variebles['QS'])

                        s_star= uniform(min_patience, x)

                        event= {'Type':'CustomerLeave', 'CustomerType': customertype, 'CustomerId': customerid, 's_star': s_star}
                        add_to_fel(fel, event, clock)

        else:

            customertype= 'Normal'


            if state_variebles['SN']<normal_agent_number:

                state_variebles['SN']+=1

                if crn_data:
                    s_star= crn_data['D2_times'].pop(0)
                
                else:
                    s_star= exponential(D2)

                event= {'Type':'EndCall', 'CustomerType': customertype, 'AgentType': 'Normal', 'CustomerId': customerid, 's_star': s_star}
                add_to_fel(fel, event, clock)

            else:

                if state_variebles['SS']<special_agent_number:

                    if state_variebles['QS']>0:

                        state_variebles['QN']+=1
                        queues['QN'].append({'CustomerId': customerid, 'CustomerType': customertype, 'Time': clock})

                        if state_variebles['QN']>4:

                            if crn_data:
                                g= crn_data['Recall_Customers'].pop(0)
                            
                            else:
                                g= bern(recall_prob)


                            if g==1:

                                state_variebles['RCN']+=1
                                queues['RCN'].append({'CustomerId': customerid, 'CustomerType': customertype, 'Time': clock})
                                
                                state_variebles['QN']-=1
                                queues['QN'].pop(-1)

                            else:

                                h= bern(churn_prob)

                                if h==1:

                                    x= max(patience_line, state_variebles['QN'])

                                    s_star= uniform(min_patience, x)

                                    event= {'Type':'CustomerLeave', 'CustomerType': customertype, 'CustomerId': customerid, 's_star': s_star}
                                    add_to_fel(fel, event, clock)

                        else:

                            if crn_data:
                                g= crn_data['Churn_Customers'].pop(0)
                            
                            else:
                                h= bern(churn_prob)


                            if h==1:

                                x= max(patience_line, state_variebles['QN'])

                                s_star= uniform(min_patience, x)

                                event= {'Type':'CustomerLeave', 'CustomerType': customertype, 'CustomerId': customerid, 's_star': s_star}
                                add_to_fel(fel, event, clock)

                    else:

                        state_variebles['SS']+=1

                        if crn_data:
                            s_star= crn_data['D1_times'].pop(0)
                        
                        else:
                            s_star= exponential(D1)

                        event= {'Type':'EndCall', 'CustomerType': customertype, 'AgentType': 'Special', 'CustomerId': customerid, 's_star': s_star}
                        add_to_fel(fel, event, clock)

                else:

                    state_variebles['QN']+=1
                    queues['QN'].append({'CustomerId': customerid, 'CustomerType': customertype, 'Time': clock})


                    if state_variebles['QN']>4:

                        if crn_data:
                            g= crn_data['Recall_Customers'].pop(0)
                        
                        else:
                            g= bern(recall_prob)


                        if g==1:

                            state_variebles['RCN']+=1
                            queues['RCN'].append({'CustomerId': customerid, 'CustomerType': customertype, 'Time': clock})

                            state_variebles['QN']-=1
                            queues['QN'].pop(-1)

                        else:

                            if crn_data:
                                h= crn_data['Churn_Customers'].pop(0)
                            
                            else:
                                h= bern(churn_prob)


                            if h==1:

                                x= max(patience_line, state_variebles['QN'])

                                s_star= uniform(min_patience, x)

                                event= {'Type':'CustomerLeave', 'CustomerType': customertype, 'CustomerId': customerid, 's_star': s_star}
                                add_to_fel(fel, event, clock)

                    else:

                        if crn_data:
                            h= crn_data['Churn_Customers'].pop(0)
                        
                        else:
                            h= bern(churn_prob)


                        if h==1:

                            x= max(patience_line, state_variebles['QN'])

                            s_star= uniform(min_patience, x)

                            event= {'Type':'CustomerLeave', 'CustomerType': customertype, 'CustomerId': customerid, 's_star': s_star}
                            add_to_fel(fel, event, clock)


        shift_number= int((clock% (24*60))/ (8*60))+1

        if shift_number == 1:
            
            if crn_data:
                s_star = exponential(state_variebles['Lambda1'])
            
            else:
                s_star= exponential(D3)
                
        elif shift_number == 2:
            
            if crn_data:
                s_star= crn_data['shift2_times'].pop(0)
            
            else:
                s_star = exponential(state_variebles['Lambda2'])
            
        else:
            if crn_data:
                s_star= crn_data['shift3_times'].pop(0)
            
            else:
                s_star = exponential(state_variebles['Lambda3'])

        event= {'Type':'Call', 'CustomerId': customerid+1, 's_star': s_star}
        add_to_fel(fel, event, clock)




    start_state= dict({
        'QN': 0,
        'QS': 0,
        'SS': 0,
        'SN': 0,
        'QTN': 0,
        'QTS': 0,
        'TS': 0,
        'RCS': 0,
        'RCN': 0,
        'Mal': 0,
        'Lambda1': lambda1,
        'Lambda2': lambda2,
        'Lambda3': lambda3,

        })
    
    if malfunction_flag:
    
        first_event= [{'Type': 'Call', 'Time': 0, 'CustomerId': 0},
                      {'Type': 'StartMonth', 'Time': 0, 'CustomerId': -1}
                      ]
    else:
        
        first_event= [{'Type': 'Call', 'Time': 0, 'CustomerId': 0}
                      ]
    
        
        
    
    def simulation(simulation_time):

        table= dict({
            'fel': [],
            'state_variebles': [],
            'current_event': []
            })

        customerid= 0
        state_variebles, fel = set_state(start_state, first_event)

        clock = 0
        fel.append({'Type': 'End of Simulation', 'Time': simulation_time, 'CustomerId': -1})

        w= 1 #week counter
        sorted_fel = sorted(fel, key=lambda x: (x['Time'], x['CustomerId']))

        while (clock < simulation_time):

            current_event = sorted_fel[0]  # find imminent event

            clock = current_event['Time']


            try:
                customertype = current_event['CustomerType']
            except:
                pass

            try:
                agenttype = current_event['AgentType']
            except:
                pass

            customerid=current_event['CustomerId']

            if clock < simulation_time:

                if current_event['Type'] == 'StartMonth':
                    startmonth(fel, state_variebles, clock)

                if current_event['Type'] == 'Malfunction':
                    malfunction(fel, state_variebles, clock, lambda1_mal, lambda2_mal, lambda3_mal)

                if current_event['Type'] == 'EndMalfunction':
                    end_malfunction(fel, state_variebles, clock, lambda1= lambda1, lambda2= lambda2, lambda3= lambda3)

                if current_event['Type'] == 'TC_endofservice':
                    TC_endofservice(fel, state_variebles, clock, D3)

                if current_event['Type'] == 'TC_arrival':

                    TC_arrival(fel, state_variebles, clock, customerid, customertype, D3)

                if current_event['Type'] == 'EndCall':
                    EndCall(fel, state_variebles, clock, customerid, customertype, agenttype, TC_prob, D2, D1)

                if current_event['Type'] == 'CustomerLeave':
                    CustomerLeave(fel, state_variebles, clock, customerid, customertype)

                if current_event['Type'] == 'Call':
                    Call(fel, state_variebles, clock, customerid, special_prob, recall_prob, churn_prob, patience_line, min_patience, lambda1, lambda2, lambda3)



                table['current_event'].append(current_event)
                table['state_variebles'].append(list(state_variebles.values()))
                
                try:
                    fel.remove(current_event)
                except:
                    print(current_event)
                
                sorted_fel = sorted(fel, key=lambda x: (x['Time'], x['CustomerId']))
                if store_fel:
                    table['fel'].append(list(sorted_fel ))

                if clock//(7*24*60)>=w:

                    print(f'{w} weeks done')
                    w= w+1
                
                #print(queues)



        table['current_event']= sorted(table['current_event'], key=lambda x: x['Time'])
        
        d= pd.concat([pd.DataFrame(table['current_event']),\
                      pd.DataFrame(table['state_variebles'])],\
                      axis= 1)
        
        del table
        
        d.columns= list(d.columns[:(len(d.columns)- len(state_variebles.keys()))])+\
                    list(state_variebles.keys())
        
        #d= reduce_mem_usage(d)
        
        if recall_prob== 0:
            d['answer_flag']= np.nan
            
        return d

    
    return simulation(simulation_time)


#%%

