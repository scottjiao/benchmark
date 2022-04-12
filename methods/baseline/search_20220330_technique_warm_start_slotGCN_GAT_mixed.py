

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

dataset_to_evaluate=[("ACM_GTN",2,20),("DBLP_GTN",2,20),("IMDB_GTN",2,20),("pubmed_HNE_complete",1,30)]

fixed_info={"search_num_layers":"[2,3]",
            "slot_aggregator":"last_fc",
            "selection_weight_average":"True",
            "selection_types":"0_1",
            "get_out":"True",
            "epoch":"800",
            "lr_times_on_feat_average":"100"}
task_to_evaluate=[
                {"net":"slotGAT"},
                {"net":"slotGCN"},
                ]
gpus=["0","1"]
total_trial_num=18
task_str=''
fixed_str=''
for key in sorted(fixed_info.keys()):
    fixed_str+=f" --{key} {fixed_info[key]} "

for dataset,worker_num,re in dataset_to_evaluate:
    for task in task_to_evaluate:
        for key in sorted(task.keys()):
            task_str+=f" --{key} {task[key]} "
        study_name=f"technique_{dataset}_net_{task['net']}_warm_uped_mix_auto_feat"
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


"""
for key in f_1.keys():
    assert key in f_2.keys()
    if f_1[key]!=f_2[key]:
        print(f"key: {key}, f_1: {f_1[key]}| f_2: {f_2[key]}")
"""