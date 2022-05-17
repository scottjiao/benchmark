

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

dataset_to_evaluate=[("IMDB_corrected",1,10),]#("Freebase_corrected",1,20)]

fixed_info={}
task_to_evaluate=[
{"task_property":"technique","net":"slotGAT","feats-type":"1","slot_aggregator":"average","search_hidden_dim":"[64,128]","search_num_layers":"[2,3]","attention_mse_sampling_factor":0.1,"attention_mse_weight_factor":0.1,"attention_0_type_bigger_constraint":0,"attention_1_type_bigger_constraint":0},
{"task_property":"technique","net":"slotGAT","feats-type":"1","slot_aggregator":"average","search_hidden_dim":"[64,128]","search_num_layers":"[2,3]","attention_mse_sampling_factor":0.1,"attention_mse_weight_factor":1,"attention_0_type_bigger_constraint":0,"attention_1_type_bigger_constraint":0},
{"task_property":"technique","net":"slotGAT","feats-type":"1","slot_aggregator":"average","search_hidden_dim":"[64,128]","search_num_layers":"[2,3]","attention_mse_sampling_factor":0.1,"attention_mse_weight_factor":0.1,"attention_0_type_bigger_constraint":1,"attention_1_type_bigger_constraint":0},
{"task_property":"technique","net":"slotGAT","feats-type":"1","slot_aggregator":"average","search_hidden_dim":"[64,128]","search_num_layers":"[2,3]","attention_mse_sampling_factor":0.1,"attention_mse_weight_factor":1,"attention_0_type_bigger_constraint":1,"attention_1_type_bigger_constraint":0},
{"task_property":"technique","net":"slotGAT","feats-type":"1","slot_aggregator":"average","search_hidden_dim":"[64,128]","search_num_layers":"[2,3]","attention_mse_sampling_factor":0.1,"attention_mse_weight_factor":0.1,"attention_0_type_bigger_constraint":0.1,"attention_1_type_bigger_constraint":0},
{"task_property":"technique","net":"slotGAT","feats-type":"1","slot_aggregator":"average","search_hidden_dim":"[64,128]","search_num_layers":"[2,3]","attention_mse_sampling_factor":0.1,"attention_mse_weight_factor":1,"attention_0_type_bigger_constraint":0.1,"attention_1_type_bigger_constraint":0},
{"task_property":"technique","net":"slotGAT","feats-type":"1","slot_aggregator":"average","search_hidden_dim":"[64,128]","search_num_layers":"[2,3]","attention_mse_sampling_factor":0.1,"attention_mse_weight_factor":0.1,"attention_0_type_bigger_constraint":0.01,"attention_1_type_bigger_constraint":0},
{"task_property":"technique","net":"slotGAT","feats-type":"1","slot_aggregator":"average","search_hidden_dim":"[64,128]","search_num_layers":"[2,3]","attention_mse_sampling_factor":0.1,"attention_mse_weight_factor":1,"attention_0_type_bigger_constraint":0.01,"attention_1_type_bigger_constraint":0},
{"task_property":"technique","net":"slotGAT","feats-type":"1","slot_aggregator":"average","search_hidden_dim":"[64,128]","search_num_layers":"[2,3]","attention_mse_sampling_factor":0,"attention_mse_weight_factor":0,"attention_0_type_bigger_constraint":0,"attention_1_type_bigger_constraint":1},
{"task_property":"technique","net":"slotGAT","feats-type":"1","slot_aggregator":"average","search_hidden_dim":"[64,128]","search_num_layers":"[2,3]","attention_mse_sampling_factor":0,"attention_mse_weight_factor":0,"attention_0_type_bigger_constraint":0,"attention_1_type_bigger_constraint":0.1},
{"task_property":"technique","net":"slotGAT","feats-type":"1","slot_aggregator":"average","search_hidden_dim":"[64,128]","search_num_layers":"[2,3]","attention_mse_sampling_factor":0,"attention_mse_weight_factor":0,"attention_0_type_bigger_constraint":0,"attention_1_type_bigger_constraint":0.01},
]
gpus=["0","1"]
total_trial_num=30
task_str=''
fixed_str=''
for key in sorted(fixed_info.keys()):
    fixed_str+=f" --{key} {fixed_info[key]} "

for dataset,worker_num,re in dataset_to_evaluate:
    for task in task_to_evaluate:
        task_property=task["task_property"]
        for key in sorted(task.keys()):
            task_str+=f" --{key} {task[key]} "
        study_name=f"{task_property}_{dataset}_net_{task['net']}_feat_type_{task['feats-type']}_aggr_{task['slot_aggregator']}_attention_mse_sampling_{task['attention_mse_sampling_factor']}_weight_{task['attention_mse_weight_factor']}_0_type_bigger_{task['attention_0_type_bigger_constraint']}_1_type_bigger_{task['attention_1_type_bigger_constraint']}"
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


