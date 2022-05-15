

import time
import subprocess
import multiprocessing
from threading import main_thread
import os
class Run( multiprocessing.Process):
    def __init__(self,command):
        super().__init__()
        self.command=command
    def run(self):
        
        subprocess.run(self.command,shell=True)

def get_best_hypers(dataset,net,yes,no):
    print(f"yes: {yes}, no: {no}")
    #get search best hypers
    best={}
    fns=[]
    for root, dirs, files in os.walk("./log", topdown=False):
        for name in files:
            FLAG=1
            if "old" in root:
                continue
            if ".py" in name:
                continue
            if ".txt" in name:
                continue
            for n in no:
                if n in name:
                    FLAG=0
            for y in yes:
                if y not in name:
                    FLAG=0
            if FLAG==0:
                continue

            if dataset in name:
                name0=name.replace("_GTN","",1) if "kdd" not in name else name
                if net in name0 :

                    fn=os.path.join(root, name)
                    fns.append(fn)
    score_max=0
    print(fns)
    if fns==[]:
        raise Exception
    for fn in fns:
        path=fn
        FLAG0=False
        FLAG1=False
        with open(fn,"r") as f:
            for line in f:
                if "Best trial" in line and FLAG0==False:
                    FLAG0=True
                    FLAG1=False
                    continue
                if FLAG0==True:
                    if "Value" in line:
                        _,score=line.strip("\n").replace(" ","").split(":")
                        score=float(score)
                        continue
                    if "Params:" in line:
                        FLAG1=True
                        count=0
                        continue
                if FLAG1==True and score>=score_max and "    " in line and count<=5:

                    param,value=line.strip("\n").replace(" ","").split(":")
                    best[param]=value
                    score_max=score
                    FLAG0=False
                    count+=1
        print(best)
        best_hypers={}
        for key in best.keys():
            best_hypers["search_"+key]=f"""[{best[key]}]"""
    return best_hypers

def run_command_in_parallel(args_dict,gpus,worker_num):


    command='python run_analysis.py  '
    for key in args_dict.keys():
        command+=f" --{key} {args_dict[key]} "


    process_queue=[]
    for gpu in gpus:
        
        command+=f" --gpu {gpu} "
        command+=f"   > ./log/{args_dict['study_name']}.txt  "
        for _ in range(worker_num):
            
            command+=f"   > ./log/{args_dict['study_name']}.txt  "
            print(f"running: {command}")
            p=Run(command)
            p.daemon=True
            p.start()
            process_queue.append(p)
            time.sleep(5)

    for p in process_queue:
        p.join()

def config_study_name(prefix,specified_args,extract_dict):
    study_name=prefix
    for k in specified_args:
        v=extract_dict[k]
        study_name+=f"_{k}_{v}"
    if study_name[0]=="_":
        study_name=study_name.replace("_","",1)
    study_storage=f"sqlite:///db/{study_name}.db"
    return study_name,study_storage

if __name__ == '__main__':
    
    dataset_to_evaluate=[("IMDB_corrected",1,1),("IMDB_corrected_oracle",1,1),]  # dataset,worker_num,repeat

    fixed_info={"get_out":"True",}
    task_to_evaluate=[
    {"task_property":"technique","net":"slotGAT","feats-type":"0","slot_aggregator":"by_slot_0"},
    {"task_property":"technique","net":"slotGAT","feats-type":"0","slot_aggregator":"by_slot_1"},
    {"task_property":"technique","net":"slotGAT","feats-type":"0","slot_aggregator":"by_slot_2"},
    {"task_property":"technique","net":"slotGAT","feats-type":"0","slot_aggregator":"by_slot_3"},
    {"task_property":"technique","net":"slotGAT","feats-type":"0","slot_aggregator":"slot_majority_voting"},
    {"task_property":"technique","net":"slotGAT","feats-type":"1","slot_aggregator":"by_slot_0"},
    {"task_property":"technique","net":"slotGAT","feats-type":"1","slot_aggregator":"by_slot_1"},
    {"task_property":"technique","net":"slotGAT","feats-type":"1","slot_aggregator":"by_slot_2"},
    {"task_property":"technique","net":"slotGAT","feats-type":"1","slot_aggregator":"by_slot_3"},
    {"task_property":"technique","net":"slotGAT","feats-type":"1","slot_aggregator":"slot_majority_voting"},
    ]
    gpus=["0"]
    total_trial_num=1



    for dataset,worker_num,repeat in dataset_to_evaluate:
        for task in task_to_evaluate:
            net=args_dict['net']
            yes=["technique","feat_type",f"feat_type_{task['feats-type']}",f"aggr_{task['slot_aggregator']}"]
            no=[]
            best_hypers=get_best_hypers(dataset,net,yes,no)
            trial_num=int(total_trial_num/ (len(gpus)*worker_num) )
            if trial_num<=1:
                trial_num=1

            args_dict={}
            args_dict['dataset']=dataset
            args_dict['trial_num']=trial_num
            args_dict['repeat']=repeat
            for dict_to_add in [best_hypers,task,fixed_info]:
                for k,v in dict_to_add.items():
                    args_dict[k]=v

            prefix="get_embeddings";specified_args=["dataset","feats-type","slot_aggregator"]
            study_name,study_storage=config_study_name(prefix=prefix,specified_args=specified_args,extract_dict=args_dict)
            #study_name=f"get_embeddings_{dataset}_net_{task['net']}_feats_type_{task['feats-type']}_slot_aggregator{task['slot_aggregator']}"
            #study_storage=f"sqlite:///db/{study_name}.db"
            
            args_dict['study_name']=study_name
            args_dict['study_storage']=study_storage



            run_command_in_parallel(args_dict,gpus,worker_num)


