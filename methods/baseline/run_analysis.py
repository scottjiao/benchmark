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
        multi_labels=True if args.dataset in ["IMDB","IMDB_corrected","IMDB_corrected_oracle"] else False

        #num_heads=1
        #hiddens=[int(i) for i in args.hiddens.split("_")]
        features_list, adjM, labels, train_val_test_idx, dl = load_data(args.dataset,multi_labels=multi_labels,delete_type_nodes=delete_type_nodes)
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
        
        """if os.path.exists(f"./temp/{args.dataset}_delete_ntype_{delete_type_nodes}.ett"):
            with open(f"./temp/{args.dataset}_delete_ntype_{delete_type_nodes}.ett","rb") as f:
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
        #with open(f"./temp/{args.dataset}_delete_ntype_{delete_type_nodes}.ett","wb") as f:
            #pickle.dump(edge2type,f)
            #pass
            
        

        g = dgl.DGLGraph(adjM+(adjM.T))
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        g = g.to(device)
        #reorganize the edge ids
        """if os.path.exists(f"./temp/{args.dataset}_delete_ntype_{delete_type_nodes}.eft"):
            with open(f"./temp/{args.dataset}_delete_ntype_{delete_type_nodes}.eft","rb") as f:
                #e_feat=pickle.load(f)
                pass
        else:"""
        e_feat = []
        count=0
        count_mappings={}
        counted_dict={}
        eid=0
        etype_ids={}
        for u, v in tqdm(zip(*g.edges())):
            
            u = u.cpu().item()
            v = v.cpu().item()
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
        with open(f"./temp/{args.dataset}_delete_ntype_{delete_type_nodes}.eft","wb") as f:
            pickle.dump(e_feat,f)
        g.etype_ids=etype_ids


        if LossCorrectionAbsDiff=="True":
            reduc="none"
        elif LossCorrectionAbsDiff=="False":
            reduc="mean"
        loss = nn.BCELoss(reduction=reduc) if multi_labels else F.nll_loss
        loss_val = nn.BCELoss() if multi_labels else F.nll_loss
        g.edge_type_indexer=F.one_hot(e_feat).to(device)

        """if os.path.exists(f"./temp/{args.dataset}.nec"):
            with open(f"./temp/{args.dataset}.nec","rb") as f:
                g.node_etype_collector=pickle.load(f).to(device)
        else:
            g.node_etype_collector=torch.zeros(dl.nodes['total'],g.edge_type_indexer.shape[1]).to(device)
            for u, v,etype in zip(*g.edges(),e_feat):
                u = u.cpu().item()
                v = v.cpu().item()
                etype=etype.cpu().item()
                g.node_etype_collector[u,etype]=1
            with open(f"./temp/{args.dataset}.nec","wb") as f:
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
            
            f = open("./analysis/"+f"dataset_info_{args.dataset}"+".json", 'w')
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
    #        "dataset": args.dataset,
    #        "num_heads": num_heads, 
    #        "weight_decay": weight_decay, 
     #       "hidden_dim":hidden_dim , 
     #       "num_layers": num_layers, 
    #        "lr_times_on_filter_GTN":lr_times_on_filter_GTN , 
    #        },
    #        reinit=True)
    for re in range(args.repeat):
        
        
        
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
        if args.net=='myGAT':
            #net = myGAT(g, args.edge_feats, len(dl.links['count'])*2+1, in_dims, hidden_dim, num_classes, num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, True, 0.05)
            #net = myGAT(g, args.edge_feats, num_etype, in_dims, hidden_dim, num_classes, num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, True, 0.05)
            GNN=myGAT
            fargs,fkargs=func_args_parse(g, args.edge_feats, num_etype, in_dims, hidden_dim, num_classes, num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, True, 0.05)
        elif args.net=='changedGAT':
            #net = changedGAT(g, args.edge_feats, num_etype, in_dims, hidden_dim, num_classes, num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, True, 0.05,num_ntype=num_ntypes,n_type_mappings=n_type_mappings,res_n_type_mappings=res_n_type_mappings,etype_specified_attention=etype_specified_attention,eindexer=eindexer,ae_layer=ae_layer)
            GNN=changedGAT
            fargs,fkargs=func_args_parse(g, args.edge_feats, num_etype, in_dims, hidden_dim, num_classes, num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, True, 0.05,num_ntype=num_ntypes,n_type_mappings=n_type_mappings,res_n_type_mappings=res_n_type_mappings,etype_specified_attention=etype_specified_attention,eindexer=eindexer,ae_layer=ae_layer,get_out=args.get_out)
        elif args.net=='slotGAT':
            GNN=slotGAT
            fargs,fkargs=func_args_parse(g, args.edge_feats, num_etype, in_dims, hidden_dim, num_classes, num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, True, 0.05,num_ntype=num_ntypes,n_type_mappings=n_type_mappings,res_n_type_mappings=res_n_type_mappings,etype_specified_attention=etype_specified_attention,eindexer=eindexer,ae_layer=ae_layer,aggregator=slot_aggregator,semantic_trans=semantic_trans,semantic_trans_normalize=semantic_trans_normalize,attention_average=attention_average,attention_mse_sampling_factor=attention_mse_sampling_factor,attention_mse_weight_factor=attention_mse_weight_factor,attention_1_type_bigger_constraint=attention_1_type_bigger_constraint,attention_0_type_bigger_constraint=attention_0_type_bigger_constraint,
            predicted_by_slot=predicted_by_slot,addLogitsEpsilon=addLogitsEpsilon,addLogitsTrain=addLogitsTrain,get_out=args.get_out,slot_attention=slot_attention)
            #net = slotGAT()
        elif args.net=='GAT':
            #net=GAT(g, in_dims, hidden_dim, num_classes, num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, True)
            GNN=GAT
            fargs,fkargs=func_args_parse(g, in_dims, hidden_dim, num_classes, num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, True,get_out=args.get_out)
        elif args.net=='GCN':
            #net=GCN(g, in_dims, hidden_dim, num_classes, num_layers, F.relu, args.dropout)
            GNN=GCN
            fargs,fkargs=func_args_parse(g, in_dims, hidden_dim, num_classes, num_layers, F.relu, args.dropout,get_out=args.get_out)
        elif args.net=="slotGCN":
            #net=slotGCN(g, in_dims, hidden_dim, num_classes, num_layers, F.relu, args.dropout,num_ntype=num_ntypes,aggregator=slot_aggregator,slot_trans=slot_trans,ntype_indexer=ntype_indexer,semantic_trans=semantic_trans,semantic_trans_normalize=semantic_trans_normalize)
            GNN=slotGCN
            fargs,fkargs=func_args_parse(g, in_dims, hidden_dim, num_classes, num_layers, F.relu, args.dropout,num_ntype=num_ntypes,aggregator=slot_aggregator,slot_trans=slot_trans,ntype_indexer=ntype_indexer,semantic_trans=semantic_trans,semantic_trans_normalize=semantic_trans_normalize,get_out=args.get_out)
        elif args.net=='GTN':
            #net=GTN(g,num_etype, in_dims, hidden_dim, num_classes, num_layers,num_heads, F.relu, args.dropout)
            GNN=GTN
            fargs,fkargs=func_args_parse(g,num_etype, in_dims, hidden_dim, num_classes, num_layers,num_heads, F.relu, args.dropout,get_out=args.get_out)
        elif args.net=='attGTN':
            #net=attGTN(g,num_etype, in_dims, hidden_dim, num_classes, num_layers,num_heads, F.relu, args.dropout,args.residual)
            GNN=attGTN
            fargs,fkargs=func_args_parse(g,num_etype, in_dims, hidden_dim, num_classes, num_layers,num_heads, F.relu, args.dropout,args.residual,get_out=args.get_out)
        elif args.net=='LabelPropagation':
            #net=LabelPropagation(num_layers, LP_alpha)
            GNN=LabelPropagation
            fargs,fkargs=func_args_parse(num_layers, LP_alpha)
        elif args.net=='MLP':
            #net=MLP(g,in_dims,hidden_dim,num_classes,num_layers,F.relu,args.dropout)
            GNN=MLP
            fargs,fkargs=func_args_parse(g,in_dims,hidden_dim,num_classes,num_layers,F.relu,args.dropout)
        elif args.net=="slotGTN":
            #net=slotGTN(g,num_etype, in_dims, hidden_dim, num_classes, num_layers,num_heads, F.relu, args.dropout,num_ntype=num_ntypes,normalize=normalize,ntype_indexer=ntype_indexer)
            GNN=slotGTN
            fargs,fkargs=func_args_parse(g,num_etype, in_dims, hidden_dim, num_classes, num_layers,num_heads, F.relu, args.dropout,num_ntype=num_ntypes,normalize=normalize,ntype_indexer=ntype_indexer,get_out=args.get_out)
        else:
            raise NotImplementedError()

        net=net_wrapper(selection_types,features_list,GNN,*fargs,**fkargs)
            
        print(f"model using: {net.__class__.__name__}")  if args.verbose=="True" else None
        #print(net)  if args.verbose=="True" else None
        #net=HeteroCGNN(g=g,num_etype=num_etype,num_ntypes=num_ntypes,num_layers=num_layers,hiddens=hiddens,dropout=args.dropout,num_classes=num_classes,bias=args.bias,activation=activation,com_dim=com_dim,ntype_dims=ntype_dims,L2_norm=L2_norm,negative_slope=args.slope,num_heads=num_heads)
        net.to(device)
        if args.net in ['GTN','slotGTN']:
            ad_list=[]
            other_list=[]
            for pname, p in net.named_parameters():
                #print(f"named pn: {pname} p: {p.shape}")
            
                if "convs" in pname:
                    ad_list.append(p)
                else:
                    other_list.append(p)
            optimizer = torch.optim.Adam([{'params':other_list,"lr":lr,"weight_decay":weight_decay},
                                            {'params':ad_list,"lr":lr_times_on_filter_GTN*lr,"weight_decay":weight_decay},])
        elif args.net=="LabelPropagation":
            pass
        elif args.selection_weight_average=="True":
            ad_list=[]
            other_list=[]
            for pname, p in net.named_parameters():
                #print(f"named pn: {pname} p: {p.shape}")
            
                if "average_weight" in pname:
                    ad_list.append(p)
                else:
                    other_list.append(p)
            optimizer = torch.optim.Adam([{'params':other_list,"lr":lr,"weight_decay":weight_decay},
                                            {'params':ad_list,"lr":lr_times_on_feat_average*lr,"weight_decay":weight_decay},])
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        #print(optimizer) if args.verbose=="True" else None
        # training loop
        net.train()
        t=time.localtime()
        str_t=f"{t.tm_year:0>4d}{t.tm_mon:0>2d}{t.tm_hour:0>2d}{t.tm_min:0>2d}{t.tm_sec:0>2d}{int(time.time()*1000)%1000}"
        ckp_dname=os.path.join('checkpoint',str_t)
        os.mkdir(ckp_dname)
        ckp_fname=os.path.join(ckp_dname,'checkpoint_{}_{}_re_{}_feat_{}_heads_{}_{}.pt'.format(args.dataset, num_layers,re,args.feats_type,num_heads,net.__class__.__name__))
        early_stopping = EarlyStopping(patience=args.patience, verbose=False, save_path=ckp_fname)

        
        dec_dic={}
        #ntypes=None   #N*num_ntype
        if args.profile=="True":
            profile_func=profile
        elif args.profile=="False":
            profile_func=blank_profile

        #wandb.watch(net, log_freq=5)
        with profile_func(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,schedule=torch.profiler.schedule(
                wait=2,
                warmup=2,
                active=6,
                repeat=1),on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler/trace_"+args.study_name)) as prof:
            for epoch in range(args.epoch):
                if args.net=="LabelPropagation"  :
                    continue
                t_0_start = time.time()
                # training
                net.train()
                with record_function("model_inference"):
                    logits,encoded_embeddings = net(features_list, e_feat) 
                logp = F.log_softmax(logits, 1) if not multi_labels else F.sigmoid(logits)
                if LossCorrectionAbsDiff=="True":
                    trainLabels=labels[train_idx]
                    with torch.no_grad():
                        trainPred = logits[train_idx].argmax(axis=1) if not multi_labels else (logits[train_idx]>0).int()
                        dif=trainLabels.sum(1)-trainPred.sum(1)
                        dif=torch.abs(dif)
                        sample_weights_one=torch.ones_like(dif)
                        sample_weights=sample_weights_one+dif
                    train_loss = ((loss(logp[train_idx], labels[train_idx]).mean(1))*sample_weights).mean(0)
                elif LossCorrectionAbsDiff=="False":
                    train_loss = loss(logp[train_idx], labels[train_idx])# if not multi_labels else loss(logp[train_idx], labels[train_idx])

                if predictionCorrectionTrainBeta!=0:
                    #on training
                    expLogits=torch.exp(-logits[train_idx].mean(1)) #核心就在于对logits均值的操控
                    expLogits_neg=torch.exp(logits[train_idx].mean(1)) 
                    trainLabels=labels[train_idx]
                    with torch.no_grad():
                        trainPred = logits[train_idx].argmax(axis=1) if not multi_labels else (logits[train_idx]>0).int()
                        dif=trainLabels.sum(1)-trainPred.sum(1)
                        dif_neg=trainPred.sum(1)-trainLabels.sum(1)
                        #dif_1= dif
                        if predictionCorrectionRelu=="True":
                            dif=F.relu( dif)
                            #dif_neg=F.relu(dif)
                            
                        labelLength=trainLabels.sum(1)
                        labelOneFlag=(trainLabels.sum(1)==1).int()
                        notLabelOneFlag=1-labelOneFlag
                    if predictionCorrectionTrainGamma==0:
                        if predCorIgnoreOneLabel=="True":
                            sampleFilter=notLabelOneFlag  #n
                        elif predCorIgnoreOneLabel=="False":
                            sampleFilter=1
                        else:
                            raise Exception
                            
                        if predictionCorrectionLabelLength=="True":
                            correctionLoss=predictionCorrectionTrainBeta*(expLogits*labelLength*dif*sampleFilter).mean(0)
                        elif predictionCorrectionLabelLength=="False":
                            correctionLoss=predictionCorrectionTrainBeta*(expLogits*dif*sampleFilter).mean(0)
                        else:
                            raise Exception

                    elif predictionCorrectionTrainGamma>0:
                        correctionLoss=(predictionCorrectionTrainBeta*(expLogits*dif*notLabelOneFlag)+  predictionCorrectionTrainGamma*(expLogits_neg*dif_neg*labelOneFlag)  ).mean(0)
                    print(f"Epoch\t{epoch}|train_loss\t{train_loss}|correctionLoss\t{correctionLoss}")  if args.verbose=="True" else None
                    train_loss +=correctionLoss
                    





                if attention_mse_weight_factor>0 or attention_1_type_bigger_constraint>0 or attention_0_type_bigger_constraint>0:
                    mse=0
                    t0_bigger_mse=0
                    t1_bigger_mse=0
                    for l in net.gat_layers:
                        mse+=attention_mse_weight_factor*l.mse
                        t1_bigger_mse+=attention_1_type_bigger_constraint*l.t1_bigger_mse
                        t0_bigger_mse+=attention_0_type_bigger_constraint*l.t0_bigger_mse
                    print(f"Epoch:{epoch} loss: {train_loss.item()}, attention mse: {mse.item()}, t1 bigger mse: {t1_bigger_mse.item()}, t0 bigger mse: {t0_bigger_mse.item()}") if args.verbose=="True" else None
                    train_loss +=mse
                    train_loss +=t0_bigger_mse
                    train_loss +=t1_bigger_mse
                #autoencoder for ntype
                if ae_layer!="None":
                    if "decoder" not in dec_dic.keys():
                        dec_dic["decoder"]=NTYPE_ENCODER(in_dim=encoded_embeddings.shape[1],hidden_dim=hidden_dim,out_dim=num_ntypes,dropout=args.dropout).to(device)
                        
                        print(dec_dic["decoder"])  if args.verbose=="True" else None
                    ntype_decoder=dec_dic["decoder"]
                    
                    #produce ntype logits
                    ntype_logits=ntype_decoder(encoded_embeddings)
                    #compute ntype loss
                    ntype_idx=torch.randperm(encoded_embeddings.shape[0])[:int(ntype_logits.shape[0]*ae_sampling_factor)].to(device)

                    logp_ntype = F.log_softmax(ntype_logits, 1)
                    ntype_acc=(logp_ntype[ntype_idx].argmax(1)==ntypes[ntype_idx]).float().mean()
                    train_loss+=F.nll_loss(logp_ntype[ntype_idx], ntypes[ntype_idx])

                # autograd
                optimizer.zero_grad()
                with record_function("model_backward"):
                    train_loss.backward()
                    optimizer.step()

                t_0_end = time.time()

                # 
                #print('Epoch {:05d} '.format(epoch, )

                t_1_start = time.time()
                # validation
                net.eval()
                with torch.no_grad():
                    logits,_ = net(features_list, e_feat)
                    logp = F.log_softmax(logits, 1) if not multi_labels else F.sigmoid(logits)
                    val_loss = loss_val(logp[val_idx], labels[val_idx])
                t_1_end = time.time()
                
                # print validation info
                if not  multi_labels:
                    
                    val_logits = logits[val_idx]
                    pred = val_logits.argmax(axis=1)
                    val_acc=((pred==labels[val_idx]).int().sum()/(pred==labels[val_idx]).shape[0]).item()
                    #wandb.log({f"val_acc_{re}": val_acc, f"val_loss_{re}": val_loss.item(),f"Train_Loss_{re}":train_loss.item()})
                    print('Epoch {:05d} | Train_Loss: {:.4f} | train Time: {:.4f} | Val_Loss {:.4f} | val Time(s) {:.4f} val acc: {:.4f}'.format(
                    epoch, train_loss.item(), t_0_end-t_0_start,val_loss.item(), t_1_end - t_1_start ,     val_acc     )      ) if (args.verbose=="True" and epoch%5==0) else None
                if args.get_out=="True":
                    if args.selection_weight_average=="True":
                        w=net.W.flatten(0).cpu().tolist()
                        if args.verbose=="True":
                            print(w)
                        vis_data_saver.collect_in_training(w[0],"w0",re,epoch);vis_data_saver.collect_in_training(w[1],"w1",re,epoch)
                        vis_data_saver.collect_in_training(val_loss.item(),"val_loss",re,epoch)
                        vis_data_saver.collect_in_training(val_acc,"val_acc",re,epoch)
                        vis_data_saver.collect_in_training(train_loss.item(),"train_loss",re,epoch)
                    
                # early stopping
                early_stopping(val_loss, net)
                if epoch>args.epoch/2 and early_stopping.early_stop:
                    #print('Early stopping!')
                    break
                prof.step()
            
        # validation with evaluate_results_nc
        if not multi_labels:
            if args.net!="LabelPropagation":
                net.load_state_dict(torch.load(ckp_fname))
            net.eval()
            with torch.no_grad():
                logits,_ = net(features_list, e_feat,get_out=args.get_out) if args.net!="LabelPropagation"  else net(g,labels,mask=train_idx)
                val_logits = logits[val_idx]
                pred = val_logits.argmax(axis=1);all_pred=logits.argmax(axis=1)
                val_acc=((pred==labels[val_idx]).int().sum()/(pred==labels[val_idx]).shape[0]).item()
            val_accs.append(val_acc)

            val_results=dl.evaluate_by_group(all_pred,val_idx,train=True)
            test_results=dl.evaluate_by_group(all_pred,test_idx,train=False)
            toCsv={ "1_featType":feats_type,
                    "1_numLayers":num_layers,
                    "1_hiddenDim":hidden_dim,
                    "1_numOfHeads":num_heads,
                    "1_Lr":lr,
                    "1_Wd":weight_decay,
                    "2_valAcc":val_results["acc"],
                    "2_valMiPre":val_results["micro-pre"],
                    "2_valMaPre":val_results["macro-pre"],
                    "2_valMiRec":val_results["micro-rec"],
                    "2_valMaRec":val_results["macro-rec"],
                    "2_valMiF1":val_results["micro-f1"],
                    "2_valMaF1":val_results["macro-f1"],
                    "3_testAcc":test_results["acc"],
                    "3_testMiPre":test_results["micro-pre"],
                    "3_testMaPre":test_results["macro-pre"],
                    "3_testMiRec":test_results["micro-rec"],
                    "3_testMaRec":test_results["macro-rec"],
                    "3_testMiF1":test_results["micro-f1"],
                    "3_testMaF1":test_results["macro-f1"], }
            toCsvRepetition.append(toCsv)
            #writeIntoCsvLogger(toCsv,f"./log/{args.study_name}.csv")
            score=sum(val_accs)/len(val_accs)
        else:
            val_losses_neg.append(1/(1+val_loss))
            score=sum(val_losses_neg)/len(val_losses_neg)
        if trial:
            trial.report(score, re)
        # Handle pruning based on the intermediate value.
        if trial and trial.should_prune():
            raise optuna.exceptions.TrialPruned()


        # testing with evaluate_results_nc
        if args.net!="LabelPropagation":
            net.load_state_dict(torch.load(ckp_fname)) 
        net.eval()
        test_logits = []
        with torch.no_grad():
            logits,_ = net(features_list, e_feat,get_out=args.get_out) if args.net!="LabelPropagation"  else net(g,labels,mask=train_idx)
            if re==0:
                logits_save=logits
                featW=net.W.flatten(0).cpu().tolist() if args.selection_weight_average=="True" else None
            else:
                logits_save=torch.cat((logits_save,logits),0)
                featW.append(net.W.flatten(0).cpu().tolist()) if args.selection_weight_average=="True" else None
            test_logits = logits[test_idx]
            pred = test_logits.cpu().numpy().argmax(axis=1) if not multi_labels else (test_logits.cpu().numpy()>0).astype(int)
            onehot = np.eye(num_classes, dtype=np.int32)
            dl_mode='multi' if multi_labels else 'bi'
            if args.get_test_for_online=="True":
                assert args.repeat==5
                assert args.trial_num==1
                if not os.path.exists(f"./testout/{ args.study_name.replace(args.dataset+'_','')}"):
                    os.mkdir(f"./testout/{ args.study_name.replace(args.dataset+'_','')}")
                dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_path=f"./testout/{ args.study_name.replace(args.dataset+'_','')}/{args.dataset}_{re+1}.txt",mode=dl_mode)
            pred = onehot[pred] if not multi_labels else  pred
            d=dl.evaluate(pred,mode=dl_mode)
            print(d) if args.verbose=="True" else None

        #wandb.log({f"macro-f1_{re}": d["macro-f1"], f"micro-f1_{re}": d["micro-f1"]})
        ma_F1s.append(d["macro-f1"])
        mi_F1s.append(d["micro-f1"])
        t_re1=time.time();t_re=t_re1-t_re0
        #print(f" this round cost {t_re}(s)")
        if args.net!="LabelPropagation":
            remove_ckp_files(ckp_dname=ckp_dname)
    #wandb.log({"macro-f1_total_mean":round(float(100*np.mean(np.array(ma_F1s)) ),2) ,
    #"macro-f1_total_std":round(float(100*np.std(np.array(ma_F1s)) ),2) ,
    # "micro-f1_total_mean": round(float(100*np.mean(np.array(mi_F1s)) ),2),
    # "micro-f1_total_std": round(float(100*np.std(np.array(mi_F1s)) ),2),
    #})
    #wandb.finish()
    vis_data_saver.collect_whole_process(round(float(100*np.mean(np.array(ma_F1s)) ),2),name="macro-f1-mean");vis_data_saver.collect_whole_process(round(float(100*np.std(np.array(ma_F1s)) ),2),name="macro-f1-std");vis_data_saver.collect_whole_process(round(float(100*np.mean(np.array(mi_F1s)) ),2),name="micro-f1-mean");vis_data_saver.collect_whole_process(round(float(100*np.std(np.array(mi_F1s)) ),2),name="micro-f1-std")
    print(f"mean and std of macro-f1: {  100*np.mean(np.array(ma_F1s)) :.1f}\u00B1{  100*np.std(np.array(ma_F1s)) :.1f}");print(f"mean and std of micro-f1: {  100*np.mean(np.array(mi_F1s)) :.1f}\u00B1{  100*np.std(np.array(mi_F1s)) :.1f}")
    print(exp_info);#print(net) if args.verbose=="True" else None
    if trial:
        print(f"trial.params: {str(trial.params)}")
    #print(optimizer) if args.verbose=="True" else None
    toCsvAveraged={}
    for tocsv in toCsvRepetition:
        for name in tocsv.keys():
            if name.startswith("1_"):
                toCsvAveraged[name]=tocsv[name]
            else:
                if name not in toCsvAveraged.keys():
                    toCsvAveraged[name]=[]
                toCsvAveraged[name].append(tocsv[name])
    
    for name in toCsvAveraged.keys():
        if not name.startswith("1_"):
            toCsvAveraged[name]=sum(toCsvAveraged[name])/len(toCsvAveraged[name])
    #toCsvAveraged["5_expInfo"]=exp_info

    writeIntoCsvLogger(toCsvAveraged,f"./log/{args.study_name}.csv")

    if args.get_out=="True":

        if args.net=="slotGAT":


            logitsNegDist=[(f"number of samples with {i}/{logits.shape[1]} negative logit",int(((logits<0).float().sum(1)==i).float().sum())) for i in range(logits.shape[1]+1)]
            vis_data_saver.collect_whole_process(logitsNegDist   ,name=f"logitsNegDist")



            #print(net.logits_mean)
            #print(net.scale_analysis)
            #net.majority_voting_analysis["pattern_counts"]
            #net.majority_voting_analysis["ties_first_labels"]
            """ties_labels=net.majority_voting_analysis["ties_labels"]
            for choice_pos,ties_order_labels in enumerate([net.majority_voting_analysis["ties_first_labels"],net.majority_voting_analysis["ties_second_labels"],net.majority_voting_analysis["ties_third_labels"],net.majority_voting_analysis["ties_fourth_labels"]]):
                choice_pos+=1
                #training 
                count=0
                t_count=0
                for node in train_idx.tolist()+val_idx.tolist():
                    if node in ties_labels.keys():
                        tie_lb=ties_order_labels[node]
                        if dl.labels_train['data'][node][tie_lb]:
                            count+=1
                        t_count+=1
                print(f"training correct count of {choice_pos}-th choice: {count}/{t_count}")
                vis_data_saver.collect_whole_process(f"{count}/{t_count}",name=f"training correct count of {choice_pos}-th choice")

                count=0
                t_count=0
                for node in test_idx.tolist():
                    if node in ties_labels.keys():
                        tie_lb=ties_order_labels[node]
                        if dl.labels_test['data'][node][tie_lb]:
                            count+=1
                        t_count+=1
                print(f"test correct count of {choice_pos}-th choice: {count}/{t_count}")
                vis_data_saver.collect_whole_process(f"{count}/{t_count}",name=f"test correct count of {choice_pos}-th choice")
            vis_data_saver.collect_whole_process(str(net.majority_voting_analysis['pattern_counts'])   ,name=f"pattern_counts")
            slot_counts=net.votings_int.squeeze(-1)[g.node_idx_by_ntype[0]].sum(0).flatten().tolist()
            vis_data_saver.collect_whole_process(slot_counts   ,name=f"slot_counts")
            slots_correlations=(net.votings_int.squeeze(-1)[g.node_idx_by_ntype[0]].transpose(1,0).float().matmul( net.votings_int.squeeze(-1)[g.node_idx_by_ntype[0]].float()  )).tolist()
            vis_data_saver.collect_whole_process(slots_correlations   ,name=f"slots_correlations")


            #grouping the patterns
            pattern_counts=net.majority_voting_analysis['pattern_counts']
            
            for pattern in pattern_counts.keys():
                #train_pattern
                train_pat_ids=[]
                for i in train_idx.tolist()+val_idx.tolist():
                    if tuple(net.voting_patterns[i].flatten().tolist())==pattern:
                        train_pat_ids.append(i)
                result=dl.evaluate_by_group(pred,train_pat_ids,train=True,mode="multi")
                vis_data_saver.collect_whole_process(result   ,name=f"results of pattern {pattern}(train+val)")
                #test_pattern
                test_pat_ids=[]
                for i in test_idx.tolist():
                    if tuple(net.voting_patterns[i].flatten().tolist())==pattern:
                        test_pat_ids.append(i)
                result=dl.evaluate_by_group(pred,test_pat_ids,train=False,mode="multi")
                vis_data_saver.collect_whole_process(result   ,name=f"results of pattern {pattern}(test)")"""

            if multi_labels:    
                pred = logits.cpu().numpy().argmax(axis=1) if not multi_labels else (logits.cpu().numpy()>0).astype(int)
                #different predicted number of labels
                #diff_len_dict={"train":{},"test":{}}
                diff_len_idx={"train":{},"test":{}}
                diff_len_results={"train":{},"test":{}}
                for i in train_idx.tolist()+val_idx.tolist():
                    lbs=dl.labels_train['data'][i].nonzero()[0]
                    pds=pred[i].nonzero()[0]
                    true_label_length=len(lbs)
                    diff_length=len(pds)-len(lbs)
                    k=f"true_label_length:{true_label_length}/diff_length: {diff_length}"
                    if k not in diff_len_idx["train"].keys():
                        #diff_len_dict["train"][k]=0
                        diff_len_idx["train"][k]=[]
                    #diff_len_dict["train"][k]+=1
                    diff_len_idx["train"][k].append(i)
                for k,idx_list in diff_len_idx["train"].items():
                    diff_len_results["train"][k]=dl.evaluate_by_group(pred,idx_list,train=True,mode="multi")
                #diff_len_dict["train"]=sorted([(k,v) for k,v in diff_len_dict["train"].items()])
                diff_len_results["train"]=sorted([(k,v) for k,v in diff_len_results["train"].items()])
                
                for i in test_idx.tolist():
                    lbs=dl.labels_test['data'][i].nonzero()[0]
                    pds=pred[i].nonzero()[0]
                    true_label_length=len(lbs)
                    diff_length=len(pds)-len(lbs)
                    k=f"true_label_length:{true_label_length}/diff_length: {diff_length}"
                    k_length=f"true_label_length:{true_label_length}"
                    if k not in diff_len_idx["test"].keys():
                        diff_len_idx["test"][k]=[]
                    if k_length not in diff_len_idx["test"].keys():
                        diff_len_idx["test"][k_length]=[]
                    #diff_len_dict["test"][k]+=1
                    diff_len_idx["test"][k].append(i)
                    diff_len_idx["test"][k_length].append(i)
                for k,idx_list in diff_len_idx["test"].items():
                    diff_len_results["test"][k]=dl.evaluate_by_group(pred,idx_list,train=False,mode="multi")
                #diff_len_dict["test"]=sorted([(k,v) for k,v in diff_len_dict["test"].items()])
                diff_len_results["test"]=sorted([(k,v) for k,v in diff_len_results["test"].items()])
                
                #vis_data_saver.collect_whole_process(diff_len_dict   ,name=f"diff_len_dict")
                vis_data_saver.collect_whole_process(diff_len_results   ,name=f"diff_len_results")

            # dataset analysis
            #train
            
            """for X,Y,name,flg in [(train_idx.tolist()+val_idx.tolist(),dl.labels_train,"train",True),(test_idx.tolist(),dl.labels_test,"test",False)]:
                class_idx_dict={}
                class_idx_results={}
                cooccurence_1={}
                cooccurence_2={}
                cooccurence_3={}
                for i in X:
                    lbs=Y['data'][i].nonzero()[0]
                    for l in lbs:
                        l=int(l)
                        if len(lbs) not in class_idx_dict.keys():
                            class_idx_dict[len(lbs)]={}
                            class_idx_results[len(lbs)]={}
                        if l not in class_idx_dict[len(lbs)].keys():
                            class_idx_dict[len(lbs)][l]=[]
                            class_idx_results[len(lbs)][l]=[]
                        class_idx_dict[len(lbs)][l].append(i)
                        #class_idx_results[len(lbs)][l]=dl.evaluate_by_group(pred,idx_list,train=True,mode="multi")
                    for l in lbs:
                        l=int(l)
                        if l not in cooccurence_1.keys():
                            cooccurence_1[l]=0
                        if l not in cooccurence_2.keys():
                            cooccurence_2[l]={}
                        if l not in cooccurence_3.keys():
                            cooccurence_3[l]={}
                        if len(lbs)==1:
                            cooccurence_1[l]+=1
                        for l_ in lbs:
                            l_=int(l_)
                            if l_ not in cooccurence_2[l].keys():
                                cooccurence_2[l][l_]=0
                            if l_ not in cooccurence_3[l].keys():
                                cooccurence_3[l][l_]={}
                            if len(lbs)==2:
                                cooccurence_2[l][l_]+=1
                            for l__ in lbs:
                                l__=int(l__)
                                if l__ not in cooccurence_3[l][l_].keys():
                                    cooccurence_3[l][l_][l__]=0
                                if len(lbs)==3:
                                    cooccurence_3[l][l_][l__]+=1
                for length,l_dicts in class_idx_dict.items():
                    for l,idx in l_dicts.items():
                        class_idx_results[length][l]=dl.evaluate_by_group(pred,idx,train=flg,mode="multi")
                vis_data_saver.collect_whole_process(class_idx_results   ,name=f"class_idx_results_by_label_length({name})")
                vis_data_saver.collect_whole_process(cooccurence_1   ,name=f"cooccurence_1({name})")
                vis_data_saver.collect_whole_process(cooccurence_2   ,name=f"cooccurence_2({name})")
                vis_data_saver.collect_whole_process(cooccurence_3   ,name=f"cooccurence_3({name})")"""


                    

            #vis_data_saver.collect_whole_process([x.tolist() for x in net.scale_analysis ],name=f"scale_analysis")
            #vis_data_saver.collect_whole_process(net.ties_ratio_1,name=f"ties_ratio_1")
            #vis_data_saver.collect_whole_process(net.ties_ratio_2,name=f"ties_ratio_2")
            if get_out_tsne=="True":
                for i in range(num_layers):
                    #for head in range(num_heads):
                    #    attentions_i_head=net.gat_layers[i].attentions[:,head,:].squeeze(-1).cpu().numpy()
                    #    attention_hist_i_head=[int(x) for x in list(np.histogram(attentions_i_head,bins=10,range=(0,1))[0])]
                    #    vis_data_saver.collect_whole_process( attention_hist_i_head ,name=f"attention_hist_layer_{i}_head_{head}")
                    #for etype in range(num_etype):
                    #    attentions_i_et=net.gat_layers[i].attentions[etype_ids[etype],:,:].flatten().cpu().numpy()
                     #   attention_hist_i_head=[int(x) for x in list(np.histogram(attentions_i_et,bins=10,range=(0,1))[0])]
                    #    vis_data_saver.collect_whole_process( attention_hist_i_head ,name=f"attention_hist_layer_{i}_et_{etype}")
                    #vis_data_saver.collect_whole_process( net.gat_layers[i].attn_correlation ,name=f"attn_correlation_layer_{i}_et_0_1")
                    for nt in tqdm(range(net.gat_layers[i].emb.shape[0])):
                        x_emb=TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(net.gat_layers[i].emb[nt].squeeze(0).numpy())
                        #vis_data_saver.collect_whole_process_tensor(  ,name=f"emb_layer_{i}")
                        vis_data_saver.collect_whole_process( x_emb.tolist() ,name=f"tsne_emb_layer_{i}_slot_{nt}")




        """out_fn=os.path.join("analysis",args.study_name+".out")
        logits_save=logits_save.cpu().numpy()
        np.save(out_fn, logits_save, allow_pickle=True, fix_imports=True)"""
        #if args.selection_weight_average=="True":
            #W_fn=os.path.join("analysis",args.study_name+".w");featW=np.array(featW);np.save(W_fn, featW, allow_pickle=True, fix_imports=True)
        if not os.path.exists(f"./analysis"):
            os.mkdir("./analysis")
        if not os.path.exists(f"./analysis/{args.study_name}"):
            os.mkdir(f"./analysis/{args.study_name}")
        vis_data_saver.save(os.path.join(f"./analysis/{args.study_name}",args.study_name+".visdata"))
        #vis_data_saver.visualize_tsne(dn=f"./analysis/{args.study_name}",node_idx_by_ntype=g.node_idx_by_ntype)


        






    fn=os.path.join("log",args.study_name)
    if os.path.exists(fn):
        m="a"
    else:
        m="w"
    
    with open(fn,m) as f:
        
        f.write(f"  multi_feat_weight: {str(net.W.flatten(0).cpu().tolist())} \n") if args.selection_weight_average=="True" else None
        f.write(f"score {  score :.4f}  mean and std of macro-f1: {  100*np.mean(np.array(ma_F1s)) :.1f}\u00B1{  100*np.std(np.array(ma_F1s)) :.1f} micro-f1: {  100*np.mean(np.array(mi_F1s)) :.1f}\u00B1{  100*np.std(np.array(mi_F1s)) :.1f}\n")
        f.write(str(exp_info)+"\n")
        if trial:
            f.write(f"trial.params: {str(trial.params)}"+"\n")
        #f.write(str(net)+"\n")
    """ else:
        with open(fn,"w") as f:
            f.write(f"score {  score :.4f}  mean and std of macro-f1: {  100*np.mean(np.array(ma_F1s)) :.1f}\u00B1{  100*np.std(np.array(ma_F1s)) :.1f} micro-f1: {  100*np.mean(np.array(mi_F1s)) :.1f}\u00B1{  100*np.std(np.array(mi_F1s)) :.1f}\n")
            f.write(str(exp_info)+"\n")
            f.write(f"trial.params: {str(trial.params)}"+"\n")
            #f.write(str(net)+"\n")"""

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




    
