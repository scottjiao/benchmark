

import time
import subprocess
import multiprocessing
from threading import main_thread
from pipeline_utils import get_best_hypers,run_command_in_parallel,config_study_name,Run,proc_yes,get_best_hypers_from_csv
import os
import copy
#time.sleep(60*60*4)

metric_mapping={"IMDB_corrected":"2_valMaF1","ACM_corrected":"2_valAcc","DBLP_corrected":"2_valAcc","pubmed_HNE_complete":"2_valAcc"}


dataset_to_evaluate=[("ACM_corrected",1,1),("IMDB_corrected",1,1),("DBLP_corrected",1,1),("pubmed_HNE_complete",1,1),]  # dataset,worker_num,repeat

prefix="ablation_newCsv";specified_args=["dataset",   "net",      "slot_aggregator","predicted_by_slot","get_out"]

fixed_info={"task_property":prefix,"net":"slotGAT","slot_aggregator":"average","using_optuna":"False","predicted_by_slot":"max","get_out":"getMaxSlot"}
task_space={"feats-type":1,}

search_property="technique";yes_names=[f"slot_aggregator","predicted_by_slot"]
no=["attantion_average","attention_average","attention_mse","edge_feat_0","oracle"]

def get_tasks(task_space):
    tasks=[{}]
    for k,v in task_space.items():
        tasks=expand_task(tasks,k,v)
    return tasks

def expand_task(tasks,k,v):
    temp_tasks=[]
    if type(v) is str and type(eval(v)) is list:
        for value in eval(v):
            if k.startswith("search_"):
                value=str([value])
            for t in tasks:
                temp_t=copy.deepcopy(t)
                temp_t[k]=value
                temp_tasks.append(temp_t)
    elif type(v) is list:
        for value in v:
            for t in tasks:
                temp_t=copy.deepcopy(t)
                temp_t[k]=value
                temp_tasks.append(temp_t)
    else:
        for t in tasks:
            temp_t=copy.deepcopy(t)
            temp_t[k]=v
            temp_tasks.append(temp_t)
    return temp_tasks
    #if k.startswith("search_"):
        ##list

        
task_to_evaluate=get_tasks(task_space)
print(task_to_evaluate)
gpus=["1"]
total_trial_num=1


for dataset,worker_num,repeat in dataset_to_evaluate:
    for task in task_to_evaluate:
        args_dict={}
        for dict_to_add in [task,fixed_info]:
            for k,v in dict_to_add.items():
                args_dict[k]=v
        net=args_dict['net']
        ##################################
        ##edit yes and no for filtering!##
        ##################################
        yes=proc_yes(yes_names,args_dict)
        yes.append(search_property)
        best_hypers=get_best_hypers_from_csv(dataset,net,yes,no,metric=metric_mapping[dataset])
        for dict_to_add in [best_hypers]:
            for k,v in dict_to_add.items():
                args_dict[k]=v
        trial_num=int(total_trial_num/ (len(gpus)*worker_num) )
        if trial_num<=1:
            trial_num=1

        args_dict['dataset']=dataset
        args_dict['trial_num']=trial_num
        args_dict['repeat']=repeat

        study_name,study_storage=config_study_name(prefix=prefix,specified_args=specified_args,extract_dict=args_dict)
        #study_name=f"get_embeddings_{dataset}_net_{task['net']}_feats_type_{task['feats-type']}_slot_aggregator{task['slot_aggregator']}"
        #study_storage=f"sqlite:///db/{study_name}.db"
        
        args_dict['study_name']=study_name
        args_dict['study_storage']=study_storage



        run_command_in_parallel(args_dict,gpus,worker_num)

