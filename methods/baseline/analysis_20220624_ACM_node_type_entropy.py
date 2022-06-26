from re import I
import sys
import pickle
from numpy.core.numeric import identity
sys.path.append('../../')
import time
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from utils.pytorchtools import EarlyStopping
from utils.data import load_data
from utils.tools import func_args_parse,single_feat_net,multi_feat_net,vis_data_collector,blank_profile,writeIntoCsvLogger
#from utils.tools import index_generator, evaluate_results_nc, parse_minibatch
from GNN import myGAT,HeteroCGNN,changedGAT,GAT,GCN,NTYPE_ENCODER,GTN,attGTN,slotGAT,slotGCN,LabelPropagation,MLP,slotGTN
import dgl
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import tensorboard_trace_handler
from sklearn.manifold import TSNE
#import wandb

from tqdm import tqdm

import json



feature_usage_dict={0:"loaded features",
1:"only target node features (zero vec for others)",
2:"only target node features (id vec for others)",
3:"all id vec. Default is 2",
4:"only term features (id vec for others)",
5:"only term features (zero vec for others)",
}

ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
ap.add_argument('--feats-type', type=int, default=0,
                help='Type of the node features used. ' +
                        '0 - loaded features; ' +
                        '1 - only target node features (zero vec for others); ' +
                        '2 - only target node features (id vec for others); ' +
                        '3 - all id vec. Default is 2;' +
                    '4 - only term features (id vec for others);' + 
                    '5 - only term features (zero vec for others).')
#ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
ap.add_argument('--com_dim', type=int, default=64 )
ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
ap.add_argument('--patience', type=int, default=30, help='Patience.')
ap.add_argument('--repeat', type=int, default=30, help='Repeat the training and testing for N times. Default is 1.')
#ap.add_argument('--num-layers', type=int, default=2)
#ap.add_argument('--lr', type=float, default=5e-4)
ap.add_argument('--dropout', type=float, default=0.5)
#ap.add_argument('--weight-decay', type=float, default=1e-4)
ap.add_argument('--slope', type=float, default=0.05)
ap.add_argument('--residual', type=str, default="True")
ap.add_argument('--dataset', type=str)
ap.add_argument('--edge-feats', type=int, default=64)
ap.add_argument('--run', type=int, default=1)
ap.add_argument('--gpu', type=str, default="0")
#ap.add_argument('--hiddens', type=str, default="64_32")
ap.add_argument('--activation', type=str, default="elu")
ap.add_argument('--bias', type=str, default="true")
ap.add_argument('--net', type=str, default="myGAT")
ap.add_argument('--n_type_mappings', type=str, default="False")
ap.add_argument('--res_n_type_mappings', type=str, default="False")

ap.add_argument('--task_property', type=str, default="notSpecified")
ap.add_argument('--study_name', type=str, default="temp")
ap.add_argument('--study_storage', type=str, default="sqlite:///db/temp.db")
ap.add_argument('--trial_num', type=int, default=50)
ap.add_argument('--etype_specified_attention', type=str, default="False")
ap.add_argument('--verbose', type=str, default="False")
ap.add_argument('--ae_layer', type=str, default="None")  #"last_hidden", "None"
ap.add_argument('--ae_sampling_factor', type=float, default=0.01)  
ap.add_argument('--slot_aggregator', type=str, default="None")
ap.add_argument('--slot_attention', type=str, default="False") #
ap.add_argument('--predicted_by_slot', type=str, default="None")
ap.add_argument('--relevant_passing', type=str, default="False") #



ap.add_argument('--attention_average', type=str, default="False")
ap.add_argument('--attention_mse_sampling_factor', type=float, default=0)  
ap.add_argument('--attention_mse_weight_factor', type=float, default=0)  
ap.add_argument('--attention_1_type_bigger_constraint', type=float, default=0)
ap.add_argument('--attention_0_type_bigger_constraint', type=float, default=0)  
ap.add_argument('--slot_trans', type=str, default="all")  #all, one
ap.add_argument('--LP_alpha', type=float, default=0.5)  #1,0.99,0.5
ap.add_argument('--get_out', default="False")  
ap.add_argument('--profile', default="False")  
ap.add_argument('--get_out_tsne', default="False")  
ap.add_argument('--using_optuna', default="True")  



ap.add_argument('--get_test_for_online', default="False")  
ap.add_argument('--addLogitsEpsilon', type=float, default=1e-5)  #
ap.add_argument('--addLogitsTrain', type=str, default="False")  #
ap.add_argument('--predictionCorrectionTrainBeta', type=float, default=0)  #
ap.add_argument('--predictionCorrectionLabelLength', type=str, default="False")  #
ap.add_argument('--predictionCorrectionRelu', type=str, default="True")  #
ap.add_argument('--predictionCorrectionBetaAbs', type=str, default="True")  #
ap.add_argument('--predictionCorrectionGammaAbs', type=str, default="True")  #
ap.add_argument('--predictionCorrectionTrainGamma', type=float, default=0)  #
ap.add_argument('--predCorIgnoreOneLabel', type=str, default="False")  #
ap.add_argument('--LossCorrectionAbsDiff', type=str, default="False")  #



ap.add_argument('--delete_type_nodes', default="None")  

ap.add_argument('--normalize', default="True")  
ap.add_argument('--semantic_trans', default="False")  
ap.add_argument('--semantic_trans_normalize', default="row")  #row,col

ap.add_argument('--selection_types', default="0_1")  #feats x_y_z
ap.add_argument('--selection_weight_average', default="False")  #feat type '0 1' by default
ap.add_argument('--lr_times_on_feat_average',type=int, default=1)  #1 100

ap.add_argument('--ablation_deletion', default="None")   # None, 1-N

ap.add_argument('--search_num_heads', type=str, default="[8]")
ap.add_argument('--search_lr', type=str, default="[1e-3,5e-4,1e-4]")
ap.add_argument('--search_weight_decay', type=str, default="[5e-4,1e-4,1e-5]")
ap.add_argument('--search_hidden_dim', type=str, default="[64,128]")
ap.add_argument('--search_num_layers', type=str, default="[2]")
ap.add_argument('--search_lr_times_on_filter_GTN', type=str, default="[100]")

torch.set_num_threads(4)



args = ap.parse_args()



os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)




import optuna
from optuna.trial import TrialState
torch.manual_seed(1234)






def run_model_DBLP(trial=None):
    #data preparation
    #dataset="ACM_corrected"
    #dataset="IMDB_corrected"
    #dataset="DBLP_corrected"
    dataset="pubmed_HNE_complete"
    num_layers=1
    
    
    
    

    """num_heads=trial.suggest_categorical("num_heads", [8])
    lr=trial.suggest_categorical("lr", [1e-3,5e-4,1e-4])
    weight_decay=trial.suggest_categorical("weight_decay", [5e-4,1e-4,1e-5])
    hidden_dim=trial.suggest_categorical("hidden_dim", [64,128])
    num_layers=trial.suggest_categorical("num_layers", [2])"""
    if trial!=None:
        num_heads=trial.suggest_categorical("num_heads", eval(args.search_num_heads))
        lr=trial.suggest_categorical("lr", eval(args.search_lr))
        weight_decay=trial.suggest_categorical("weight_decay", eval(args.search_weight_decay))
        hidden_dim=trial.suggest_categorical("hidden_dim", eval(args.search_hidden_dim))
        num_layers=trial.suggest_categorical("num_layers", eval(args.search_num_layers))
        lr_times_on_filter_GTN=trial.suggest_categorical("lr_times_on_filter_GTN", eval(args.search_lr_times_on_filter_GTN))
    else:
        num_heads=int(eval(args.search_num_heads)[0]);assert len(eval(args.search_num_heads))==1
        lr=float(eval(args.search_lr)[0]);assert len(eval(args.search_lr))==1
        weight_decay=float(eval(args.search_weight_decay)[0]);assert len(eval(args.search_weight_decay))==1
        hidden_dim=int(eval(args.search_hidden_dim)[0]);assert len(eval(args.search_hidden_dim))==1
        num_layers=int(eval(args.search_num_layers)[0]);assert len(eval(args.search_num_layers))==1
        lr_times_on_filter_GTN=float(eval(args.search_lr_times_on_filter_GTN)[0]);assert len(eval(args.search_lr_times_on_filter_GTN))==1
    
    if True:
        feats_type = args.feats_type
        com_dim=args.com_dim
        L2_norm=True
        semantic_trans=args.semantic_trans
        semantic_trans_normalize=args.semantic_trans_normalize
        slot_aggregator=args.slot_aggregator
        slot_trans=args.slot_trans
        """num_heads=args.num_heads
        lr=args.lr
        weight_decay=args.weight_decay
        hidden_dim=args.hidden_dim
        num_layers=args.num_layers"""
        lr_times_on_feat_average=args.lr_times_on_feat_average
        ae_layer=args.ae_layer
        ae_sampling_factor=args.ae_sampling_factor
        delete_type_nodes=args.delete_type_nodes
        attention_average=args.attention_average
        predicted_by_slot=args.predicted_by_slot
        slot_attention=args.slot_attention
        relevant_passing=args.relevant_passing

        attention_mse_sampling_factor=args.attention_mse_sampling_factor
        attention_mse_weight_factor=args.attention_mse_weight_factor
        attention_1_type_bigger_constraint=args.attention_1_type_bigger_constraint
        attention_0_type_bigger_constraint=args.attention_0_type_bigger_constraint
        addLogitsEpsilon=args.addLogitsEpsilon   #addLogitsEpsilon,addLogitsTrain
        addLogitsTrain=args.addLogitsTrain
        predictionCorrectionTrainBeta=args.predictionCorrectionTrainBeta
        predictionCorrectionRelu=args.predictionCorrectionRelu
        predictionCorrectionTrainGamma=args.predictionCorrectionTrainGamma
        predictionCorrectionLabelLength=args.predictionCorrectionLabelLength
        predCorIgnoreOneLabel=args.predCorIgnoreOneLabel
        LossCorrectionAbsDiff=args.LossCorrectionAbsDiff
        predictionCorrectionGammaAbs=args.predictionCorrectionGammaAbs
        predictionCorrectionBetaAbs=args.predictionCorrectionBetaAbs
        n_type_mappings=eval(args.n_type_mappings)
        res_n_type_mappings=eval(args.res_n_type_mappings)
        get_out_tsne=args.get_out_tsne
        if res_n_type_mappings:
            assert n_type_mappings 
        multi_labels=True if dataset in ["IMDB","IMDB_corrected","IMDB_corrected_oracle"] else False
        dl_mode='multi' if multi_labels else 'bi'

        #num_heads=1
        #hiddens=[int(i) for i in args.hiddens.split("_")]
        features_list, adjM, labels, train_val_test_idx, dl = load_data(dataset,multi_labels=multi_labels,delete_type_nodes=delete_type_nodes)
        class_num=max(labels)+1 if not multi_labels else len(labels[0])
        exp_info=f"dataset information :\n\tnode num: {adjM.shape[0]}\n\t\tattribute num: {features_list[0].shape[1]}\n\t\tnode type_num: {len(features_list)}\n\t\tnode type dist: {dl.nodes['count']}"+\
                    f"\n\tedge num: {adjM.nnz}"+\
                    f"\n\tclass num: {class_num}"+\
                    f"\n\tlabel num: {len(train_val_test_idx['train_idx'])+len(train_val_test_idx['val_idx'])+len(train_val_test_idx['test_idx'])} \n\t\ttrain labels num: {len(train_val_test_idx['train_idx'])}\n\t\tval labels num: {len(train_val_test_idx['val_idx'])}\n\t\ttest labels num: {len(train_val_test_idx['test_idx'])}"+"\n"+f"feature usage: {feature_usage_dict[args.feats_type]}"+"\n"+f"exp setting: {vars(args)}"+"\n"
        if multi_labels:
            for num,ct in enumerate([(torch.LongTensor(labels).sum(1)==i).int().sum().item() for i in range(class_num+1)]):
                exp_info+=f"\n\t{ct} nodes has {num} labels"
        print(exp_info) if args.verbose else None
        vis_data_saver=vis_data_collector()
        vis_data_saver.save_meta(exp_info,"exp_info")
        running_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(running_time)
        
        #check
        """
        if np.sum(np.sum(labels,dim=1)>1)>1 and not multi_labels:
            raise Exception("Multi labels check failed!)
        
        """

        torch.manual_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        features_list = [mat2tensor(features).to(device) for features in features_list]
        if args.selection_weight_average=="True":
            assert feats_type==0 
        if feats_type == 0:
            in_dims = [features.shape[1] for features in features_list]
        elif feats_type == 1 or feats_type == 5:
            save = 0 if feats_type == 1 else 2
            in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
            for i in range(0, len(features_list)):
                if i == save:
                    in_dims.append(features_list[i].shape[1])
                else:
                    in_dims.append(10)
                    features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
        elif feats_type == 2 or feats_type == 4:
            save = feats_type - 2
            in_dims = [features.shape[0] for features in features_list]
            for i in range(0, len(features_list)):
                if i == save:
                    in_dims[i] = features_list[i].shape[1]
                    continue
                dim = features_list[i].shape[0]
                indices = np.vstack((np.arange(dim), np.arange(dim)))
                indices = torch.LongTensor(indices)
                values = torch.FloatTensor(np.ones(dim))
                features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
        elif feats_type == 3:
            in_dims = [features.shape[0] for features in features_list]
            for i in range(len(features_list)):
                dim = features_list[i].shape[0]
                indices = np.vstack((np.arange(dim), np.arange(dim)))
                indices = torch.LongTensor(indices)
                values = torch.FloatTensor(np.ones(dim))
                features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
        labels = torch.LongTensor(labels).to(device)  if not multi_labels else  torch.FloatTensor(labels).to(device)
        train_idx = train_val_test_idx['train_idx']
        train_idx = np.sort(train_idx)
        val_idx = train_val_test_idx['val_idx']
        val_idx = np.sort(val_idx)
        test_idx = train_val_test_idx['test_idx']
        test_idx = np.sort(test_idx)
        
        """if os.path.exists(f"./temp/{dataset}_delete_ntype_{delete_type_nodes}.ett"):
            with open(f"./temp/{dataset}_delete_ntype_{delete_type_nodes}.ett","rb") as f:
                #edge2type=pickle.load(f)
                pass
        else:"""
        edge2type = {}
        for k in dl.links['data']:
            for u,v in zip(*dl.links['data'][k].nonzero()):
                edge2type[(u,v)] = k
        count_self=0
        for i in range(dl.nodes['total']):
            FLAG=0
            if (i,i) not in edge2type:
                edge2type[(i,i)] = len(dl.links['count'])
                FLAG=1
        count_self+=FLAG
        count_reverse=0
        for k in dl.links['data']:
            FLAG=0
            for u,v in zip(*dl.links['data'][k].nonzero()):
                if (v,u) not in edge2type:
                    edge2type[(v,u)] = count_reverse+1+len(dl.links['count'])
                    FLAG=1
            count_reverse+=FLAG
        num_etype=len(dl.links['count'])+count_self+count_reverse
        #this operation will make gap of etype ids.
        #with open(f"./temp/{dataset}_delete_ntype_{delete_type_nodes}.ett","wb") as f:
            #pickle.dump(edge2type,f)
            #pass
            
        

        g = dgl.DGLGraph(adjM+(adjM.T))
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        g = g.to(device)
        #reorganize the edge ids
        """if os.path.exists(f"./temp/{dataset}_delete_ntype_{delete_type_nodes}.eft"):
            with open(f"./temp/{dataset}_delete_ntype_{delete_type_nodes}.eft","rb") as f:
                #e_feat=pickle.load(f)
                pass
        else:"""
        e_feat = []
        count=0
        count_mappings={}
        counted_dict={}
        eid=0
        etype_ids={}
        g_=g.cpu()
        for u, v in tqdm(zip(*g_.edges())):
            
            u =u.item() #u.cpu().item()
            v =v.item() #v.cpu().item()
            if not counted_dict.setdefault(edge2type[(u,v)],False) :
                count_mappings[edge2type[(u,v)]]=count
                counted_dict[edge2type[(u,v)]]=True
                count+=1
            e_feat.append(count_mappings[edge2type[(u,v)]])
            if edge2type[(u,v)] in etype_ids.keys():
                etype_ids[edge2type[(u,v)]].append(eid)
            else:
                etype_ids[edge2type[(u,v)]]=[eid]
            eid+=1
        e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)
        with open(f"./temp/{dataset}_delete_ntype_{delete_type_nodes}.eft","wb") as f:
            pickle.dump(e_feat,f)
        g.etype_ids=etype_ids


        if LossCorrectionAbsDiff=="True":
            reduc="none"
        elif LossCorrectionAbsDiff=="False":
            reduc="mean"
        loss = nn.BCELoss(reduction=reduc) if multi_labels else F.nll_loss
        loss_val = nn.BCELoss() if multi_labels else F.nll_loss
        g.edge_type_indexer=F.one_hot(e_feat).to(device)

        """if os.path.exists(f"./temp/{dataset}.nec"):
            with open(f"./temp/{dataset}.nec","rb") as f:
                g.node_etype_collector=pickle.load(f).to(device)
        else:
            g.node_etype_collector=torch.zeros(dl.nodes['total'],g.edge_type_indexer.shape[1]).to(device)
            for u, v,etype in zip(*g.edges(),e_feat):
                u = u.cpu().item()
                v = v.cpu().item()
                etype=etype.cpu().item()
                g.node_etype_collector[u,etype]=1
            with open(f"./temp/{dataset}.nec","wb") as f:
                pickle.dump(g.node_etype_collector,f)"""
        
        #num_etype=g.edge_type_indexer.shape[1]
        #num_etype=len(dl.links['count'])*2+1
        num_ntypes=len(features_list)
        #num_layers=len(hiddens)-1
        num_nodes=dl.nodes['total']
        g.node_idx_by_ntype=[]
        g.num_ntypes=num_ntypes
        g.node_ntype_indexer=torch.zeros(num_nodes,num_ntypes).to(device)
        ntype_dims=[]
        idx_count=0
        ntype_count=0
        for feature in features_list:
            temp=[]
            for _ in feature:
                temp.append(idx_count)
                g.node_ntype_indexer[idx_count][ntype_count]=1
                idx_count+=1

            g.node_idx_by_ntype.append(temp)
            ntype_dims.append(feature.shape[1])
            ntype_count+=1
        ntypes=g.node_ntype_indexer.argmax(1)

        if args.activation=="elu":
            activation=F.elu
        else:
            activation=torch.nn.Identity()
        


        etype_specified_attention=eval(args.etype_specified_attention)
        #eindexer=g.edge_type_indexer.unsqueeze(1).unsqueeze(1)    #  num_edges*1*1*num_etype
        eindexer=None
        if args.get_out=="True":
            
            f = open("./analysis/"+f"dataset_info_{dataset}"+".json", 'w')
            json.dump({"node_idx_by_ntype":g.node_idx_by_ntype}, f, indent=4)
            f.close()

    normalize=args.normalize
    LP_alpha=args.LP_alpha
    ntype_indexer=g.node_ntype_indexer
    ntype_acc=0
    collector={}
    ma_F1s=[]
    mi_F1s=[]
    val_accs=[]
    val_losses_neg=[]
    toCsvRepetition=[]
    def trace_handler(p):
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)
        print(output)
        p.export_chrome_trace("./profiler/trace_"+args.study_name+"_" + str(p.step_num) + ".json")


    #wandb.init(project=args.study_name, 
    #        name=f"trial_num_{trial.number}",
    #        # Track hyperparameters and run metadata
    #        config={
    #        "learning_rate": lr,
    #        "architecture": args.net,
    #        "dataset": dataset,
    #        "num_heads": num_heads, 
    #        "weight_decay": weight_decay, 
     #       "hidden_dim":hidden_dim , 
     #       "num_layers": num_layers, 
    #        "lr_times_on_filter_GTN":lr_times_on_filter_GTN , 
    #        },
    #        reinit=True)
    #re-id the train-validation in each repeat
    tr_len,val_len=len(train_idx),len(val_idx)
    total_idx=np.concatenate([train_idx,val_idx])
    total_idx=np.random.permutation(total_idx)
    train_idx,val_idx=total_idx[0:tr_len],total_idx[tr_len:tr_len+val_len]

    selection_types=args.selection_types

    if args.selection_weight_average=="True":
        net_wrapper=multi_feat_net
    elif args.selection_weight_average=="False":
        net_wrapper=single_feat_net

    t_re0=time.time()
    num_classes = dl.labels_train['num_classes']
    heads = [num_heads] * num_layers + [1]
    #if args.net=='LabelPropagation':
    #net=LabelPropagation(num_layers, LP_alpha)
    GNN=LabelPropagation
    for num_layers in [1,2,3,4,5]:
        fargs,fkargs=func_args_parse(num_layers, LP_alpha)

        net=net_wrapper(selection_types,features_list,GNN,*fargs,**fkargs)
            
        print(f"model using: {net.__class__.__name__}")  if args.verbose=="True" else None

        result=net(g,labels,mask=train_idx)


        for ntype_id,node_idx in enumerate(g.node_idx_by_ntype):
            dist=result[0][node_idx]+1e-8
            log_dist=torch.log2(dist)
            E=(-log_dist*dist).sum(dim=-1).mean(0)
            print(f"node type: {ntype_id}, mean entropy: {E}\tin {num_layers}-th propagation")




    score=None





    return score

def remove_ckp_files(ckp_dname):
    import shutil
    shutil.rmtree(ckp_dname)
    #os.mkdir(ckp_dname)

if __name__ == '__main__':
    

    #torch.cuda.set_device(int(args.gpu))
    #device=torch.device(f"cuda:{int(args.gpu)}")
    if args.using_optuna=="True":
        if args.study_name=="temp":
            if os.path.exists("./db/temp.db"):
                os.remove("./db/temp.db")
        print("start search---------------")
        study = optuna.create_study(study_name=args.study_name, storage=args.study_storage,direction="maximize",pruner=optuna.pruners.MedianPruner(),load_if_exists=True)
        study.optimize(run_model_DBLP, n_trials=args.trial_num)
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial




        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        fn=os.path.join("log",args.study_name)
        if os.path.exists(fn):
            with open(fn,"a") as f:
                
                f.write("Best trial:\n")
                f.write(f"  Value: {trial.value}\n", )
                f.write("  Params: \n")
                for key, value in trial.params.items():
                    f.write("    {}: {}\n".format(key, value))
    else:
        run_model_DBLP(trial=None)




    
