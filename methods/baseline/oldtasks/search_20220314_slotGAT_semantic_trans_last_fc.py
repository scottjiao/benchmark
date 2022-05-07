

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
task_to_evaluate=[
{"net":"slotGAT","search_num_layers":"[2]","search_hidden_dim":"[64]","semantic_trans_normalize":"row","semantic_trans":"True","slot_aggregator":"last_fc"},
{"net":"slotGAT","search_num_layers":"[4]","search_hidden_dim":"[64]","semantic_trans_normalize":"row","semantic_trans":"True","slot_aggregator":"last_fc"},
{"net":"slotGAT","search_num_layers":"[8]","search_hidden_dim":"[64]","semantic_trans_normalize":"row","semantic_trans":"True","slot_aggregator":"last_fc"},
{"net":"slotGAT","search_num_layers":"[2]","search_hidden_dim":"[64]","semantic_trans_normalize":"col","semantic_trans":"True","slot_aggregator":"last_fc"},
{"net":"slotGAT","search_num_layers":"[4]","search_hidden_dim":"[64]","semantic_trans_normalize":"col","semantic_trans":"True","slot_aggregator":"last_fc"},
{"net":"slotGAT","search_num_layers":"[8]","search_hidden_dim":"[64]","semantic_trans_normalize":"col","semantic_trans":"True","slot_aggregator":"last_fc"},
]
gpus=["0","1"]
total_trial_num=20
task_str=''

for dataset,worker_num,re in dataset_to_evaluate:
    for task in task_to_evaluate:
        for key in sorted(task.keys()):
            task_str+=f" --{key} {task[key]} "


        study_name=f"techniques_{dataset}_net_semantic_trans{task['net']}_last_fc_layer_{task['search_num_layers']}_normalize_{task['semantic_trans_normalize']}"
        study_storage=f"sqlite:///db/{study_name}.db"
        trial_num=int(total_trial_num/ (len(gpus)*worker_num) )

        process_queue=[]
        for gpu in gpus:
            for _ in range(worker_num):
                command=f"""python run_analysis.py --feats-type 0 --dataset {dataset} --gpu {gpu} --repeat {re} --trial_num {trial_num} --study_name {study_name} --study_storage {study_storage} {task_str} > ./log/{study_name}.txt """
                p=Run(command)
                p.daemon=True
                p.start()
                process_queue.append(p)
                time.sleep(5)

        for p in process_queue:
            p.join()


