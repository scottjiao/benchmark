

import time
import subprocess
import multiprocessing
from threading import main_thread

class Run( multiprocessing.Process):
    def __init__(self,command):
        super().__init__()
        self.command=command
    def run(self):
        
        subprocess.run(self.command,shell=True)

#dataset_to_evaluate=[("DBLP_GTN",2,20),("ACM_GTN",2,20),("IMDB_GTN",1,20),("pubmed_HNE_complete",1,30)]

dataset_to_evaluate=[("DBLP_GTN",3,20),("ACM_GTN",3,20),("IMDB_GTN",3,20),("pubmed_HNE_complete",2,30)]

fixed_info={"search_num_layers":"[2,3]","search_hidden_dim":"[32,64]","slot_aggregator":"last_fc"}
task_to_evaluate=[
{"feats-type":"0","net":"slotGCN"},
{"feats-type":"1","net":"slotGCN"},
{"feats-type":"2","net":"slotGCN"},
{"feats-type":"3","net":"slotGCN"},
{"feats-type":"0","net":"slotGAT"},
{"feats-type":"1","net":"slotGAT"},
{"feats-type":"2","net":"slotGAT"},
{"feats-type":"3","net":"slotGAT"},
]
gpus=["0","1"]
total_trial_num=20
task_str=''
fixed_str=''
for key in sorted(fixed_info.keys()):
    fixed_str+=f" --{key} {fixed_info[key]} "

for dataset,worker_num,re in dataset_to_evaluate:
    for task in task_to_evaluate:
        for key in sorted(task.keys()):
            task_str+=f" --{key} {task[key]} "
        study_name=f"ablation_{dataset}_net_{task['net']}_feat_type_{task['feats-type']}"
        study_storage=f"sqlite:///db/{study_name}.db"
        trial_num=int(total_trial_num/ (len(gpus)*worker_num) )

        process_queue=[]
        for gpu in gpus:
            for _ in range(worker_num):
                command=f"""python run_analysis.py --dataset {dataset} --gpu {gpu} --repeat {re} --trial_num {trial_num} --study_name {study_name} --study_storage {study_storage} {task_str} {fixed_str} > ./log/{study_name}.txt """
                p=Run(command)
                p.daemon=True
                p.start()
                process_queue.append(p)
                time.sleep(2)

        for p in process_queue:
            p.join()


