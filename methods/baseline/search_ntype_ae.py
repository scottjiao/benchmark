

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
task_to_evaluate=[{"ae_layer":"last_hidden", "ae_sampling_factor":"0.01"},{"ae_layer":"last_hidden", "ae_sampling_factor":"0.2"},{"ae_layer":"None", "ae_sampling_factor":"0.00"},]
gpus=["0","1"]
total_trial_num=50



for dataset,worker_num in dataset_to_evaluate:
    for task in task_to_evaluate:
        study_name=f"ntype_ae_simpleHGN_{dataset}_ae_layer_{task['ae_layer']}_ae_sampling_factor_{task['ae_sampling_factor']}"
        study_storage=f"sqlite:///{study_name}.db"
        trial_num=int(total_trial_num/ (len(gpus)*worker_num) )

        process_queue=[]
        for gpu in gpus:
            for _ in range(worker_num):
                command=f"python run_search.py --net changedGAT  --feats-type 0 --dataset {dataset} --gpu {gpu} --trial_num {trial_num} --study_name {study_name} --study_storage {study_storage} --ae_layer {task['ae_layer']} --ae_sampling_factor {task['ae_sampling_factor']}  "
                p=Run(command)
                p.daemon=True
                p.start()
                process_queue.append(p)
                time.sleep(5)

        for p in process_queue:
            p.join()


