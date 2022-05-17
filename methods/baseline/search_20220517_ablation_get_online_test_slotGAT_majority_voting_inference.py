

import time
import subprocess
import multiprocessing
from threading import main_thread
from pipeline_utils import get_best_hypers,run_command_in_parallel,config_study_name,Run
import os

dataset_to_evaluate=[("DBLP",1,5),("ACM",1,5),("IMDB",1,5),("Freebase",1,5)]  # dataset,worker_num,repeat

#dataset_to_evaluate=[("Freebase",1,5)]  # dataset,worker_num,repeat

prefix="online_evaluation";specified_args=["dataset",   "net",    "feats-type",     "slot_aggregator",     "predicted_by_slot"]


fixed_info={"task_property":prefix,"net":"slotGAT","slot_aggregator":"average","get_test_for_online":"True"}
task_to_evaluate=[
{"feats-type":"1","predicted_by_slot":"majority_voting"},
]
gpus=["0"]
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
        yes=["technique",dataset,f"feat_type_{args_dict['feats-type']}",f"aggr_{args_dict['slot_aggregator']}"]
        no=["attantion_average","attention_average","attention_mse","edge_feat_0","oracle"]
        best_hypers=get_best_hypers(dataset,net,yes,no)
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

