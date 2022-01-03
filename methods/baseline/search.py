

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

dataset_to_evaluate=[("pubmed_HNE_complete",1),("DBLP_HNE_sampled",1)]
task_to_evaluate=[{"n_type_mappings":"True", "res_n_type_mappings":"True"},{"n_type_mappings":"True", "res_n_type_mappings":"False"},{"n_type_mappings":"False", "res_n_type_mappings":"False"}]
gpus=["0","1"]
total_trial_num=30

for dataset,worker_num in dataset_to_evaluate:
    for task in task_to_evaluate:
        study_name=f"node_added_simpleHGN_{dataset}_n_type_mappings_{task['n_type_mappings']}_res_{task['res_n_type_mappings']}"
        study_storage=f"sqlite:///{study_name}.db"
        trial_num=int(total_trial_num/ (len(gpus)*worker_num) )

        process_queue=[]
        for gpu in gpus:
            for _ in range(worker_num):
                command=f"python run_search.py --net changedGAT  --feats-type 0 --dataset {dataset} --gpu {gpu} --trial_num {trial_num} --study_name {study_name} --study_storage {study_storage} --n_type_mappings {task['n_type_mappings']} --res_n_type_mappings {task['res_n_type_mappings']}  "
                p=Run(command)
                p.daemon=True
                p.start()
                process_queue.append(p)
                time.sleep(5)

        for p in process_queue:
            p.join()


