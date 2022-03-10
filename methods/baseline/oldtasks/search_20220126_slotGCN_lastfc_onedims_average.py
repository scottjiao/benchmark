

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

dataset_to_evaluate=[("DBLP_GTN",2,20),("ACM_GTN",2,20),("IMDB_GTN",2,20),("pubmed_HNE_complete",1,30)]

fixed_info={}
task_to_evaluate=[{"net":"slotGCN","search_num_layers":"[2,3,4,5]","slot_aggregator":"last_fc","search_hidden_dim":"[64,128]"},{"net":"slotGCN","search_num_layers":"[2,3,4,5]","slot_aggregator":"average","search_hidden_dim":"[64,128]"},{"net":"slotGCN","search_num_layers":"[2,3,4,5]","slot_aggregator":"onedimconv","search_hidden_dim":"[64,128]"},]
gpus=["0","1"]
total_trial_num=50
task_str=''

for dataset,worker_num,re in dataset_to_evaluate:
    for task in task_to_evaluate:
        for key in sorted(task.keys()):
            task_str+=f" --{key} {task[key]} "


        study_name=f"techniques_{dataset}_net_{task['net']}_{task['slot_aggregator']}"
        study_storage=f"sqlite:///db/{study_name}.db"
        trial_num=int(total_trial_num/ (len(gpus)*worker_num) )

        process_queue=[]
        for gpu in gpus:
            for _ in range(worker_num):
                command=f"""python run_search.py --feats-type 0 --dataset {dataset} --gpu {gpu} --repeat {re} --trial_num {trial_num} --study_name {study_name} --study_storage {study_storage} {task_str} > ./log/{study_name}.txt """
                p=Run(command)
                p.daemon=True
                p.start()
                process_queue.append(p)
                time.sleep(5)

        for p in process_queue:
            p.join()


