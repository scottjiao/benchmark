import sys
import pickle
from numpy.core.numeric import identity
sys.path.append('../../')
import time
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from utils.pytorchtools import EarlyStopping
from utils.data import load_data
from utils.tools import vis_data_collector
#from utils.tools import index_generator, evaluate_results_nc, parse_minibatch
from GNN import myGAT,HeteroCGNN,changedGAT,GAT,GCN,NTYPE_ENCODER,GTN,attGTN,slotGAT,slotGCN
import dgl
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import networkx
from sklearn.metrics import f1_score
import json

feature_usage_dict={0:"loaded features",
1:"only target node features (zero vec for others)",
2:"only target node features (id vec for others)",
3:"all id vec. Default is 2",
4:"only term features (id vec for others)",
5:"only term features (zero vec for others)",
}
"""
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
ap.add_argument('--dataset', type=str)"""

torch.set_num_threads(4)


"""
args = ap.parse_args()
"""

os.environ["CUDA_VISIBLE_DEVICES"]="0"


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




def get_heterophily_score(labeled_idx,netx_g,labels,kmax=2):
    het_score={}
    set_all_labeled_nodes=set(labeled_idx)
    for node in labeled_idx:
        k_neigh=[]
        for k in range(1,kmax+1):
            k_neigh_dict=networkx.single_source_shortest_path_length(netx_g, node, cutoff=k)
            for idx in k_neigh_dict:
                if k_neigh_dict[idx]==k:
                    k_neigh.append(idx)
        neighbor_with_labels=set(k_neigh).intersection(set_all_labeled_nodes)
        labels_of_neighbors=labels[list(neighbor_with_labels)]
        if len(labels_of_neighbors)==0:
            continue
        else:
            label_of_node=labels[node]
            het_score[node]=(((labels_of_neighbors==label_of_node).int().sum())/(len(labels_of_neighbors))).item()
    return het_score


def get_hetero_distribution(dataset,labeled_idx,netx_g,labels,k=2):
    scores=list(get_heterophily_score(labeled_idx,netx_g,labels,kmax=k).values())
    hist, edges = np.histogram(
        scores,
        bins=10,
        range=(0, 1),
        density=False)
    print(f"\thist: {hist} of dataset {dataset}")




def analysis(dataset,net,get_logits_way="average"):
    feats_type = 0
    """num_heads=args.num_heads
    lr=args.lr
    weight_decay=args.weight_decay
    hidden_dim=args.hidden_dim
    num_layers=args.num_layers"""



    #num_heads=1
    #hiddens=[int(i) for i in args.hiddens.split("_")]
    features_list, adjM, labels, train_val_test_idx, dl = load_data(dataset)
    exp_info=f"dataset information :\n\tnode num: {adjM.shape[0]}\n\t\tattribute num: {features_list[0].shape[1]}\n\t\tnode type_num: {len(features_list)}\n\t\tnode type dist: {dl.nodes['count']}"+\
                f"\n\tedge num: {adjM.nnz}"+\
                f"\n\tclass num: {max(labels)+1}"+\
                f"\n\tlabel num: {len(train_val_test_idx['train_idx'])+len(train_val_test_idx['val_idx'])+len(train_val_test_idx['test_idx'])} \n\t\ttrain labels num: {len(train_val_test_idx['train_idx'])}\n\t\tval labels num: {len(train_val_test_idx['val_idx'])}\n\t\ttest labels num: {len(train_val_test_idx['test_idx'])}"+"\n"+f"feature usage: {feature_usage_dict[feats_type]}"+"\n"+f"exp setting: {dataset}"+"\n"
    print(exp_info)
    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    device = torch.device('cpu')
    features_list = [mat2tensor(features).to(device) for features in features_list]
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
    labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)
    edge2type = {}
    deg_dist_by_edge_type={}
    for k in dl.links['data']:
        deg_dist_by_edge_type[f"{k}_out"]={}
        deg_dist_by_edge_type[f"{k}_in"]={}
        for u,v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u,v)] = k
            if u not in deg_dist_by_edge_type[f"{k}_out"]:
                deg_dist_by_edge_type[f"{k}_out"][u]=0
            if v not in deg_dist_by_edge_type[f"{k}_in"]:
                deg_dist_by_edge_type[f"{k}_in"][v]=0
            deg_dist_by_edge_type[f"{k}_out"][u]+=1
            deg_dist_by_edge_type[f"{k}_in"][v]+=1
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
                edge2type[(v,u)] = count+1+len(dl.links['count'])
                FLAG=1
                et=count+1+len(dl.links['count'])
                if u not in deg_dist_by_edge_type[f"{et}_out"]:
                    deg_dist_by_edge_type[f"{et}_out"][u]=0
                if v not in deg_dist_by_edge_type[f"{et}_in"]:
                    deg_dist_by_edge_type[f"{et}_in"][v]=0
                deg_dist_by_edge_type[f"{et}_out"][u]+=1
                deg_dist_by_edge_type[f"{et}_in"][v]+=1
        count_reverse+=FLAG
    num_etype=len(dl.links['count'])+count_self+count_reverse
    #this operation will make gap of etype ids.
    #with open(f"./temp/{args.dataset}_delete_ntype_{delete_type_nodes}.ett","wb") as f:
        #pickle.dump(edge2type,f)
        #pass
    print("the meta information")   
    print(dl.links['count'])


    g = dgl.DGLGraph(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    #reorganize the edge ids
    
    e_feat = []
    count=0
    count_mappings={}
    counted_dict={}
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        if not counted_dict.setdefault(edge2type[(u,v)],False) :
            count_mappings[edge2type[(u,v)]]=count
            counted_dict[edge2type[(u,v)]]=True
            count+=1
        e_feat.append(count_mappings[edge2type[(u,v)]])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)



    netx_g=dgl.to_networkx(g.cpu()).to_undirected()

    g.edge_type_indexer=F.one_hot(e_feat).to(device)

    if os.path.exists(f"./temp/{dataset}.nec"):
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
            pickle.dump(g.node_etype_collector,f)

    num_etype=g.edge_type_indexer.shape[1]
    num_ntypes=len(features_list)
    #num_layers=len(hiddens)-1
    num_nodes=dl.nodes['total']
    g.node_idx_by_ntype=[]
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
    f = open("./analysis/"+f"dataset_info_{dataset}"+".json", 'w')
    json.dump({"node_idx_by_ntype":g.node_idx_by_ntype}, f, indent=4)
    f.close()
    eindexer=g.edge_type_indexer.unsqueeze(1).unsqueeze(1)    #  num_edges*1*1*num_etype
    ntype_indexer=g.node_ntype_indexer


    labeled_idx=[]
    for values in train_val_test_idx.values():
        labeled_idx.extend(values)
    set_all_labeled_nodes=set(labeled_idx)

    #exp info
    #print(exp_info)


    
    #edge density
    edgeDens=adjM.nnz/(adjM.shape[0])**2
    print(f"edge density: {edgeDens}")

    #distribution of degrees

    def count_hist(input):
        #input must be int numbers
        m=max(input)
        if m<20:
            m=20
        x=torch.zeros([m+1])
        for i in input:
            x[i]+=1
        return x

    dist_deg=count_hist(g.in_degrees().cpu()).int()
    print(f"degree dist of top 20 :{dist_deg[:20]}")
    plt.title("degree distribution from deg 1 to deg 20")
    plt.plot(list(range(20)),dist_deg[:20])
    x_l=MultipleLocator(1)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_l)
    plt.xlim(-0.5,20.5)
    plt.savefig(f"deg_{dataset}")

    for k,v in deg_dist_by_edge_type.items():
        et_dir=k
        dist_deg=count_hist(list(v.values()))
        plt.title(f"degree distribution of et {et_dir} from deg 1 to deg 20")
        plt.plot(list(range(20)),dist_deg[:20])
        x_l=MultipleLocator(1)
        ax=plt.gca()
        ax.xaxis.set_major_locator(x_l)
        plt.xlim(-0.5,20.5)
        plt.savefig(f"./dataset_analysis/deg_{dataset}_{et_dir}")
        plt.cla()




    #spectrum of laplacian


    #spectrum=networkx.linalg.spectrum.laplacian_spectrum(netx_g)
    #print(f"Top 5 of spectrum are {spectrum[0:5]}, and the last 5 are {spectrum[-5:-1]}")
    print(f"The 'connected' status: {networkx.algorithms.components.is_connected(netx_g)}")
    comps_ids=sorted(networkx.algorithms.components.connected_components(netx_g), key=len, reverse=True)
    comps_lens=[len(c) for c in comps_ids]
    print(f"the size of largest connected components is {comps_lens}")
    diams=[]
    eigs=[]
    for comps_id in comps_ids:
        if len(comps_id)<10:
            continue
        subg=netx_g.subgraph(comps_id)
        #print(f"computing of diameter of comp of size {len(comps_id)}")
        #eigs.append(networkx.linalg.spectrum.laplacian_spectrum(subg)[1:3])
        diams.append(networkx.approximation.diameter(subg))
    #print(f"the smallest 3 of largest connected components is {eigs}")
    print(f"the diagram of largest connected components is {diams}")


    # heterophily

    ks=[1,2,3,4,5,6]
    labeled_idx=[]
    for values in train_val_test_idx.values():
        labeled_idx.extend(values)
    set_all_labeled_nodes=set(labeled_idx)
    for k in ks:
        ratio=0
        for node in labeled_idx:
            k_neigh=[]
            k_neigh_dict=networkx.single_source_shortest_path_length(netx_g, node, cutoff=k)
            for idx in k_neigh_dict:
                if k_neigh_dict[idx]==k:
                    k_neigh.append(idx)
            neighbor_with_labels=set(k_neigh).intersection(set_all_labeled_nodes)
            labels_of_neighbors=labels[list(neighbor_with_labels)]
            if len(labels_of_neighbors)==0:
                continue
            else:
                label_of_node=labels[node]
                ratio+=((labels_of_neighbors==label_of_node).int().sum())/(len(labels_of_neighbors))
        ratio=ratio/len(labeled_idx)
        print(f"homophily ratio of {k}-order: {ratio}")
    print("*"*20)
    print(dataset,net)
    print("*"*20)
    get_hetero_distribution(dataset,labeled_idx,netx_g,labels,k=2)

    """node_num=adjM.shape[0]
    if get_logits_way=="average":
        logits_cum=torch.from_numpy(np.load(f"./analysis/get_best_outs_{dataset}_net_{net}.out.npy"))
        logits_cum=F.softmax(logits_cum,dim=1)
        re=(logits_cum.shape[0]/node_num)
        assert int(re)-re==0
        re=int(re)
        logits=0
        for i in range(re):
            logits+=logits_cum[i*node_num:(i+1)*node_num,:]
        logits=logits/re
    else :
        logits_cum=torch.from_numpy(np.load(f"./analysis/get_best_outs_{dataset}_net_{net}.out.npy"))
        logits_cum=F.softmax(logits_cum,dim=1)
        logits=logits_cum[get_logits_way*node_num:(get_logits_way+1)*node_num,:]

    ####################!!!!!!!!!!!!!!!!!!! labels没有test的信息！
    labels[test_idx] = torch.from_numpy(dl.labels_test['data'][test_idx]).argmax(axis=1)
    #每一个类的自己的acc
    class_recall=torch.zeros([max(labels)+1])
    class_recall_count=torch.zeros([max(labels)+1])
    class_precision=torch.zeros([max(labels)+1])
    class_precision_count=torch.zeros([max(labels)+1])
    for node in labeled_idx:
        class_recall_count[labels[node]]+=1
        class_recall[labels[node]]+=int(torch.argmax(logits[node])==labels[node])
        class_precision_count[torch.argmax(logits[node])]+=1
        class_precision[torch.argmax(logits[node])]+=int(torch.argmax(logits[node])==labels[node])
    #test每一个类的自己的acc
    test_class_recall=torch.zeros([max(labels)+1])
    test_class_recall_count=torch.zeros([max(labels)+1])
    test_class_precision=torch.zeros([max(labels)+1])
    test_class_precision_count=torch.zeros([max(labels)+1])
    for node in test_idx:
        test_class_recall_count[labels[node]]+=1
        test_class_recall[labels[node]]+=int(torch.argmax(logits[node])==labels[node])
        test_class_precision_count[torch.argmax(logits[node])]+=1
        test_class_precision[torch.argmax(logits[node])]+=int(torch.argmax(logits[node])==labels[node])
    mi_f1,ma_f1=f1_score(labels[labeled_idx], torch.argmax(logits[labeled_idx],dim=1).cpu().numpy(), average='micro'),f1_score(labels[labeled_idx], torch.argmax(logits[labeled_idx],dim=1).cpu().numpy(), average='macro')
    test_mi_f1,test_ma_f1=f1_score(labels[test_idx], torch.argmax(logits[test_idx],dim=1).cpu().numpy(), average='micro'),f1_score(labels[test_idx], torch.argmax(logits[test_idx],dim=1).cpu().numpy(), average='macro')
    test_class_precision=np.around((test_class_precision/test_class_precision_count).numpy(),decimals=3)
    test_class_recall=np.around((test_class_recall/test_class_recall_count).numpy(),decimals=3)
    class_precision=np.around((class_precision/class_precision_count).numpy(),decimals=3)
    class_recall=np.around((class_recall/class_recall_count).numpy(),decimals=3)
    onehot = np.eye(logits.shape[1], dtype=np.int32)
    pred = onehot[logits[test_idx].cpu().numpy().argmax(axis=1)]
    d=dl.evaluate(pred)
    
    print(f"\tthe test class count:  {test_class_recall_count.numpy()}")
    print(f"\tthe test predicted class count:  {test_class_precision_count.numpy()}")
    print(f"\tthe test class precision:  {test_class_precision}")
    print(f"\tthe test class recall:  {test_class_recall}")


    assert sum(test_class_recall_count.numpy())==sum(test_class_precision_count.numpy())
    total=sum(test_class_recall_count.numpy())
    fig, ax_left = plt.subplots()
    ax_right = ax_left.twinx()
    ax_left.plot(test_class_recall, color='black',label="recall")
    ax_left.plot(test_class_precision, color='black', linestyle='dashed',label="precision")
    ax_left.tick_params(axis='y', labelcolor='black')
    ax_left.set_ylabel("accs", color='black')
    ax_right.plot(test_class_recall_count.numpy()/total, color='red',label="true count")
    ax_right.plot(test_class_precision_count.numpy()/total, color='red', linestyle='dashed',label="predicted count")
    ax_right.tick_params(axis='y', labelcolor='red')
    ax_right.set_ylabel("percentages", color='red')
    x=list(range(len(test_class_recall)))
    plt.xticks(x,x)
    plt.title("accs_test_"+dataset+"_"+net)
    #plt.ylim([0,1])
    ax_left.legend(loc=1)
    ax_right.legend(loc=2)
    plt.savefig("./analysis/"+"accs_test_"+dataset+"_"+net+".png")
    plt.cla()





    print(f"\tthe all class count:  {class_recall_count.numpy()}")
    print(f"\tthe all predicted class count:  {class_precision_count.numpy()}")
    print(f"\tthe all class precision:  {class_precision}")
    print(f"\tthe all class recall:  {class_recall}")
    print(f"\treported test microf1: {d['micro-f1']:.3f}, macrof1: {d['macro-f1']:.3f}")
    print(f"\tcomputed test microf1: {test_mi_f1:.3f}, macrof1: {test_ma_f1:.3f}")
    print(f"\tcomputed all  microf1: {mi_f1:.3f}, macrof1: {ma_f1:.3f}")


    #class labels distribution跟node type有关系吗？
    label_dist_ntype=torch.zeros([num_ntypes,max(labels)+1])
    label_dist_ntype_count=torch.zeros([num_ntypes,1])
    for node in range(node_num):
        label_dist_ntype[ntypes[node]]+=logits[node]
        label_dist_ntype_count[ntypes[node]]+=1
    print(f"\tthe label preference of ntypes {np.around((label_dist_ntype/label_dist_ntype_count).numpy(),decimals=3)}")
    

    #prediction accuracy 跟 heterophily程度有关系吗？
    het_score=get_heterophily_score(labeled_idx,netx_g,labels,kmax=2)
    labels=labels.cpu()
    ntypes=ntypes.cpu()
    def vis(text,set_of_nodes):
        bins=np.zeros(10)
        bins_flag=np.zeros(10)
        for node in set_of_nodes:
            if node not in het_score:
                continue
            flag=   int(torch.argmax(logits[node])==labels[node])
            het_pos=int(het_score[node]*10)
            if het_pos==10:
                het_pos=9
            bins[het_pos]+=1
            bins_flag[het_pos]+=flag
        print(f"\t{text} bins: {bins} of dataset {dataset}")
        print(f"\t{text} the acc of bins:  {np.around(bins_flag/bins,decimals=3)} of dataset {dataset}")
        acc_datas=np.around(bins_flag/bins,decimals=3)
        bins_data=bins
        #vis
        x=[j/10 for j in  range(len(acc_datas))]

        left_data = acc_datas
        total=sum(bins_data)
        right_data =[ j/total for j in  bins_data]

        fig, ax_left = plt.subplots()
        ax_right = ax_left.twinx()

        ax_left.plot([j+0.05 for j in x],left_data, color='black')
        ax_left.tick_params(axis='y', labelcolor='black')
        ax_left.set_ylabel("accs", color='black')
        ax_right.plot([j+0.05 for j in x],right_data, color='red')
        ax_right.tick_params(axis='y', labelcolor='red')
        ax_right.set_ylabel("percentages", color='red')

        plt.title(text+"_"+dataset+"_"+net)
        #plt.ylim([0,1])
        #plt.legend()
        plt.xticks(x+[1],x+[1])
        plt.savefig("./analysis/"+text+"_"+dataset+"_"+net+".png")
        plt.cla()

    
    vis("all",labeled_idx)
    vis("test",test_idx)"""




#for dataset in ["pubmed_HNE_complete",
#                        "DBLP_GTN",
#                        "ACM_GTN",
#                        "IMDB_GTN",]:
#    for net in ["slotGTN"]:
#        analysis(dataset,net)
#analysis("pubmed_HNE_complete","LabelPropagation")

for dataset in ["ACM"]:
                        #"IMDB_corrected_oracle","IMDB_corrected",]:
    for net in ["slotGAT"]:
        analysis(dataset,net)
#analysis("pubmed_HNE_complete","LabelPropagation")