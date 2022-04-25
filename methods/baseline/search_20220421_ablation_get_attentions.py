

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

dataset_to_evaluate=[("IMDB_corrected",1,1),]

fixed_info={"get_out":"True",}
task_to_evaluate=[
{"net":"slotGAT","feats-type":"0","slot_aggregator":"last_fc",},
{"net":"slotGAT","feats-type":"1","slot_aggregator":"last_fc",},
{"net":"slotGAT","feats-type":"2","slot_aggregator":"last_fc",},
{"net":"slotGAT","feats-type":"3","slot_aggregator":"last_fc",},
{"net":"slotGAT","feats-type":"0","slot_aggregator":"average",},
{"net":"slotGAT","feats-type":"1","slot_aggregator":"average",},
{"net":"slotGAT","feats-type":"2","slot_aggregator":"average",},
{"net":"slotGAT","feats-type":"3","slot_aggregator":"average",},
]
gpus=["0"]
total_trial_num=1




for dataset,worker_num,repeat in dataset_to_evaluate:
    for task in task_to_evaluate:

        yes=["technique","kdd","feat_type",f"feat_type_{task['feats-type']}",f"aggr_{task['slot_aggregator']}"]
        no=[]
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
                    if task["net"] in name0 :

                        fn=os.path.join(root, name)
                        fns.append(fn)
        score_max=0
        print(fns)
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

        best_hypers={}
        for key in best.keys():
            best_hypers["search_"+key]=f"""[{best[key]}]"""
        best_str=''
        for key in best_hypers.keys():
            best_str+=f" --{key} {best_hypers[key]} "
        task_str=''
        for key in task.keys():
            task_str+=f" --{key} {task[key]} "
        fixed_str=''
        for key in fixed_info.keys():
            fixed_str+=f" --{key} {fixed_info[key]} "


        study_name=f"get_attentions_{dataset}_net_{task['net']}_feats_type_{task['feats-type']}_slot_aggregator{task['slot_aggregator']}"
        study_storage=f"sqlite:///db/{study_name}.db"
        #trial_num=int(total_trial_num/ (len(gpus)*worker_num) )
        trial_num=1

        process_queue=[]
        for gpu in gpus:
            for _ in range(worker_num):
                command=f"""python run_analysis.py --feats-type 0 --dataset {dataset} --gpu {gpu} --trial_num {trial_num} --study_name {study_name} --study_storage {study_storage}  --repeat {repeat}   {task_str}  {best_str}  {fixed_str}   > ./log/{study_name}.txt   """
                print(f"running: {command}")
                p=Run(command)
                p.daemon=True
                p.start()
                process_queue.append(p)
                time.sleep(5)

        for p in process_queue:
            p.join()


