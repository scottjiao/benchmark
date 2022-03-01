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
#from utils.tools import index_generator, evaluate_results_nc, parse_minibatch
from GNN import myGAT,HeteroCGNN,changedGAT,GAT,GCN,NTYPE_ENCODER,GTN,attGTN,slotGAT,slotGCN
import dgl
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import networkx


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







def analysis(dataset):
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

    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
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

    if os.path.exists(f"./temp/{dataset}.ett"):
        with open(f"./temp/{dataset}.ett","rb") as f:
            edge2type=pickle.load(f)
    else:
        edge2type = {}
        for k in dl.links['data']:
            for u,v in zip(*dl.links['data'][k].nonzero()):
                edge2type[(u,v)] = k
        for i in range(dl.nodes['total']):
            if (i,i) not in edge2type:
                edge2type[(i,i)] = len(dl.links['count'])
        count=1
        for k in dl.links['data']:
            for u,v in zip(*dl.links['data'][k].nonzero()):
                if (v,u) not in edge2type:
                    edge2type[(v,u)] = count+1+len(dl.links['count'])
                    count+=1

        #this operation will make gap of etype ids.
        with open(f"./temp/{dataset}.ett","wb") as f:
            pickle.dump(edge2type,f)
        


    g = dgl.DGLGraph(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    #reorganize the edge ids
    if os.path.exists(f"./temp/{dataset}.eft"):
        with open(f"./temp/{dataset}.eft","rb") as f:
            e_feat=pickle.load(f)
    else:
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
        with open(f"./temp/{dataset}.eft","wb") as f:
            pickle.dump(e_feat,f)



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

    eindexer=g.edge_type_indexer.unsqueeze(1).unsqueeze(1)    #  num_edges*1*1*num_etype
    ntype_indexer=g.node_ntype_indexer


    labeled_idx=[]
    for values in train_val_test_idx.values():
        labeled_idx.extend(values)
    set_all_labeled_nodes=set(labeled_idx)

    #exp info
    #print(exp_info)


    """
    #edge density
    edgeDens=adjM.nnz/(adjM.shape[0])**2
    print(f"edge density: {edgeDens}")

    #distribution of degrees

    def count_hist(input):
        #input must be int numbers
        m=max(input)
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
        print(f"homophily ratio of {k}-order: {ratio}")"""
    print(dataset)
    get_hetero_distribution(dataset,labeled_idx,netx_g,labels,k=2)

    node_num=adjM.shape[0]
    logits_cum=torch.from_numpy(np.load(f"./analysis/get_best_outs_{dataset}_net_slotGCN.out.npy"))
    logits_cum=F.softmax(logits_cum,dim=1)
    re=(logits_cum.shape[0]/node_num)
    assert int(re)-re==0
    re=int(re)
    logits=0
    for i in range(re):
        logits+=logits_cum[i*node_num:(i+1)*node_num,:]
    logits=logits/re

    het_score=get_heterophily_score(labeled_idx,netx_g,labels,kmax=2)
    labels=labels.cpu()
    ntypes=ntypes.cpu()

    #每一个类的自己的acc
    class_acc=torch.zeros([max(labels)+1])
    class_count=torch.zeros([max(labels)+1])
    for node in labeled_idx:
        class_count[labels[node]]+=1
        class_acc[labels[node]]+=int(torch.argmax(logits[node])==labels[node])
    print(f"\tthe class count:  {class_count.numpy()}")
    print(f"\tthe class acc:  {np.around((class_acc/class_count).numpy(),decimals=3)}")


    #class labels distribution跟node type有关系吗？
    label_dist_ntype=torch.zeros([num_ntypes,max(labels)+1])
    label_dist_ntype_count=torch.zeros([num_ntypes,1])
    for node in range(node_num):
        label_dist_ntype[ntypes[node]]+=logits[node]
        label_dist_ntype_count[ntypes[node]]+=1
    print(f"\tthe label preference of ntypes {np.around((label_dist_ntype/label_dist_ntype_count).numpy(),decimals=3)}")
    

    #prediction accuracy 跟 heterophily程度有关系吗？
    bins=np.zeros(10)
    bins_flag=np.zeros(10)
    for node in labeled_idx:
        if node not in het_score:
            continue
        flag=   int(torch.argmax(logits[node])==labels[node])
        het_pos=int(het_score[node]*10)
        if het_pos==10:
            het_pos=9
        bins[het_pos]+=1
        bins_flag[het_pos]+=flag
    print(f"\tbins: {bins} of dataset {dataset}")
    print(f"\tthe acc of bins:  {np.around(bins_flag/bins,decimals=3)} of dataset {dataset}")



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

for dataset in ["pubmed_HNE_complete",
                        "DBLP_GTN",
                        "ACM_GTN",
                        "IMDB_GTN",]:
    analysis(dataset)