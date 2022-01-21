

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

dataset_to_evaluate=[("pubmed_HNE_complete",2)]
task_to_evaluate=[{"net":"GTN","search_num_layers":"[2]"},{"net":"GTN","search_num_layers":"[4]"},{"net":"GTN","search_num_layers":"[8]"},]
gpus=["0","1"]
total_trial_num=27



for dataset,worker_num in dataset_to_evaluate:
    for task in task_to_evaluate:
        study_name=f"baselines_data_{dataset}_net_{task['net']}_{task['search_num_layers']}"
        study_storage=f"sqlite:///{study_name}.db"
        trial_num=int(total_trial_num/ (len(gpus)*worker_num) )

        process_queue=[]
        for gpu in gpus:
            for _ in range(worker_num):
                command=f"""python run_search.py --feats-type 0 --dataset {dataset} --gpu {gpu} --trial_num {trial_num} --study_name {study_name} --study_storage {study_storage} --net {task['net']} --search_num_layers "{task['search_num_layers']}" --search_hidden_dim "[64]" --repeat 30  --search_lr_times_on_filter_GTN "[1,10,100]"   > ./log/{study_name}.txt   """
                p=Run(command)
                p.daemon=True
                p.start()
                process_queue.append(p)
                time.sleep(5)

        for p in process_queue:
            p.join()


