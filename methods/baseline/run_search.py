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
from GNN import myGAT,HeteroCGNN,changedGAT,GAT,GCN,NTYPE_ENCODER,GTN,attGTN
import dgl

feature_usage_dict={0:"loaded features",
1:"only target node features (zero vec for others)",
2:"only target node features (id vec for others)",
3:"all id vec. Default is 2",
4:"only term features (id vec for others)",
5:"only term features (zero vec for others)",
}

ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
ap.add_argument('--feats-type', type=int, default=3,
                help='Type of the node features used. ' +
                        '0 - loaded features; ' +
                        '1 - only target node features (zero vec for others); ' +
                        '2 - only target node features (id vec for others); ' +
                        '3 - all id vec. Default is 2;' +
                    '4 - only term features (id vec for others);' + 
                    '5 - only term features (zero vec for others).')
ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
ap.add_argument('--com_dim', type=int, default=64 )
ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
ap.add_argument('--patience', type=int, default=30, help='Patience.')
ap.add_argument('--repeat', type=int, default=30, help='Repeat the training and testing for N times. Default is 1.')
ap.add_argument('--num-layers', type=int, default=2)
ap.add_argument('--lr', type=float, default=5e-4)
ap.add_argument('--dropout', type=float, default=0.5)
ap.add_argument('--weight-decay', type=float, default=1e-4)
ap.add_argument('--slope', type=float, default=0.05)
ap.add_argument('--residual', type=str, default="True")
ap.add_argument('--dataset', type=str)
ap.add_argument('--edge-feats', type=int, default=64)
ap.add_argument('--run', type=int, default=1)
ap.add_argument('--gpu', type=str, default="0")
ap.add_argument('--hiddens', type=str, default="64_32")
ap.add_argument('--activation', type=str, default="elu")
ap.add_argument('--bias', type=str, default="true")
ap.add_argument('--net', type=str, default="myGAT")
ap.add_argument('--n_type_mappings', type=str, default="False")
ap.add_argument('--res_n_type_mappings', type=str, default="False")
ap.add_argument('--study_name', type=str, default="temp")
ap.add_argument('--study_storage', type=str, default="sqlite:///temp.db")
ap.add_argument('--trial_num', type=int, default=50)
ap.add_argument('--etype_specified_attention', type=str, default="False")
ap.add_argument('--verbose', type=str, default="False")
ap.add_argument('--ae_layer', type=str, default="None")  #"last_hidden", "None"
ap.add_argument('--ae_sampling_factor', type=float, default=0.01)  


ap.add_argument('--search_num_heads', type=str, default="[8]")
ap.add_argument('--search_lr', type=str, default="[1e-3,5e-4,1e-4]")
ap.add_argument('--search_weight_decay', type=str, default="[5e-4,1e-4,1e-5]")
ap.add_argument('--search_hidden_dim', type=str, default="[64,128]")
ap.add_argument('--search_num_layers', type=str, default="[2]")
ap.add_argument('--search_lr_times_on_filter_GTN', type=str, default="[100]")





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
    num_heads=trial.suggest_categorical("num_heads", eval(args.search_num_heads))
    lr=trial.suggest_categorical("lr", eval(args.search_lr))
    weight_decay=trial.suggest_categorical("weight_decay", eval(args.search_weight_decay))
    hidden_dim=trial.suggest_categorical("hidden_dim", eval(args.search_hidden_dim))
    num_layers=trial.suggest_categorical("num_layers", eval(args.search_num_layers))
    lr_times_on_filter_GTN=trial.suggest_categorical("lr_times_on_filter_GTN", eval(args.search_lr_times_on_filter_GTN))
    if True:
        feats_type = args.feats_type
        com_dim=args.com_dim
        L2_norm=True


        """num_heads=args.num_heads
        lr=args.lr
        weight_decay=args.weight_decay
        hidden_dim=args.hidden_dim
        num_layers=args.num_layers"""
        
        ae_layer=args.ae_layer
        ae_sampling_factor=args.ae_sampling_factor

        n_type_mappings=eval(args.n_type_mappings)
        res_n_type_mappings=eval(args.res_n_type_mappings)
        if res_n_type_mappings:
            assert n_type_mappings 



        #num_heads=1
        hiddens=[int(i) for i in args.hiddens.split("_")]
        features_list, adjM, labels, train_val_test_idx, dl = load_data(args.dataset)
        exp_info=f"dataset information :\n\tnode num: {adjM.shape[0]}\n\t\tattribute num: {features_list[0].shape[1]}\n\t\tnode type_num: {len(features_list)}\n\t\tnode type dist: {dl.nodes['count']}"+\
                    f"\n\tedge num: {adjM.nnz}"+\
                    f"\n\tclass num: {max(labels)+1}"+\
                    f"\n\tlabel num: {len(train_val_test_idx['train_idx'])+len(train_val_test_idx['val_idx'])+len(train_val_test_idx['test_idx'])} \n\t\ttrain labels num: {len(train_val_test_idx['train_idx'])}\n\t\tval labels num: {len(train_val_test_idx['val_idx'])}\n\t\ttest labels num: {len(train_val_test_idx['test_idx'])}"+"\n"+f"feature usage: {feature_usage_dict[args.feats_type]}"+"\n"+f"exp setting: {vars(args)}"+"\n"
        #print(exp_info)

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
        
        if os.path.exists(f"./temp/{args.dataset}.ett"):
            with open(f"./temp/{args.dataset}.ett","rb") as f:
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
            with open(f"./temp/{args.dataset}.ett","wb") as f:
                pickle.dump(edge2type,f)
            
        

        g = dgl.DGLGraph(adjM+(adjM.T))
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        g = g.to(device)
        #reorganize the edge ids
        if os.path.exists(f"./temp/{args.dataset}.eft"):
            with open(f"./temp/{args.dataset}.eft","rb") as f:
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
            with open(f"./temp/{args.dataset}.eft","wb") as f:
                pickle.dump(e_feat,f)




        g.edge_type_indexer=F.one_hot(e_feat).to(device)

        if os.path.exists(f"./temp/{args.dataset}.nec"):
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

        if args.activation=="elu":
            activation=F.elu
        else:
            activation=torch.nn.Identity()
        


        etype_specified_attention=eval(args.etype_specified_attention)
        eindexer=g.edge_type_indexer.unsqueeze(1).unsqueeze(1)    #  num_edges*1*1*num_etype





    ntype_acc=0
    collector={}
    ma_F1s=[]
    mi_F1s=[]
    val_accs=[]
    for re in range(args.repeat):


        #re-id the train-validation in each repeat
        tr_len,val_len=len(train_idx),len(val_idx)
        total_idx=np.concatenate([train_idx,val_idx])
        total_idx=np.random.permutation(total_idx)
        train_idx,val_idx=total_idx[0:tr_len],total_idx[tr_len:tr_len+val_len]





        t_re0=time.time()
        num_classes = dl.labels_train['num_classes']
        heads = [num_heads] * num_layers + [1]
        if args.net=='myGAT':
            #net = myGAT(g, args.edge_feats, len(dl.links['count'])*2+1, in_dims, hidden_dim, num_classes, num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, True, 0.05)
            net = myGAT(g, args.edge_feats, num_etype, in_dims, hidden_dim, num_classes, num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, True, 0.05)
        elif args.net=='changedGAT':
            net = changedGAT(g, args.edge_feats, num_etype, in_dims, hidden_dim, num_classes, num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, True, 0.05,num_ntype=num_ntypes,n_type_mappings=n_type_mappings,res_n_type_mappings=res_n_type_mappings,etype_specified_attention=etype_specified_attention,eindexer=eindexer,ae_layer=ae_layer)
        elif args.net=='GAT':
            net=GAT(g, in_dims, hidden_dim, num_classes, num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, True)
        elif args.net=='GCN':
            net=GCN(g, in_dims, hidden_dim, num_classes, num_layers, F.relu, args.dropout)
        elif args.net=='GTN':
            net=GTN(g,num_etype, in_dims, hidden_dim, num_classes, num_layers,num_heads, F.relu, args.dropout)
        elif args.net=='attGTN':
            net=attGTN(g,num_etype, in_dims, hidden_dim, num_classes, num_layers,num_heads, F.relu, args.dropout,args.residual)

            
        print(f"model using: {net.__class__.__name__}")  if args.verbose=="True" else None
        #print(net)  if args.verbose=="True" else None
        #net=HeteroCGNN(g=g,num_etype=num_etype,num_ntypes=num_ntypes,num_layers=num_layers,hiddens=hiddens,dropout=args.dropout,num_classes=num_classes,bias=args.bias,activation=activation,com_dim=com_dim,ntype_dims=ntype_dims,L2_norm=L2_norm,negative_slope=args.slope,num_heads=num_heads)
        net.to(device)
        if args.net=='GTN':
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
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        print(optimizer) if args.verbose=="True" else None
        # training loop
        net.train()
        t=time.localtime()
        str_t=f"{t.tm_year:0>4d}{t.tm_mon:0>2d}{t.tm_hour:0>2d}{t.tm_min:0>2d}{t.tm_sec:0>2d}{int(time.time()*1000)%1000}"
        ckp_dname=os.path.join('checkpoint',str_t)
        os.mkdir(ckp_dname)
        ckp_fname=os.path.join(ckp_dname,'checkpoint_{}_{}_re_{}_feat_{}_heads_{}_{}.pt'.format(args.dataset, args.num_layers,re,args.feats_type,args.num_heads,net.__class__.__name__))
        early_stopping = EarlyStopping(patience=args.patience, verbose=False, save_path=ckp_fname)

        
        dec_dic={}
        #ntypes=None   #N*num_ntype
        for epoch in range(args.epoch):
            t_0_start = time.time()
            # training
            net.train()

            logits,encoded_embeddings = net(features_list, e_feat)
            logp = F.log_softmax(logits, 1)
            train_loss = F.nll_loss(logp[train_idx], labels[train_idx])
            

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
                logp = F.log_softmax(logits, 1)
                val_loss = F.nll_loss(logp[val_idx], labels[val_idx])
            t_1_end = time.time()
            # print validation info
            if epoch%5==0:
                print('Epoch {:05d} | Train_Loss: {:.4f} | train Time: {:.4f} | Val_Loss {:.4f} | train Time(s) {:.4f} ntype acc: {:.4f}'.format(
                epoch, train_loss.item(), t_0_end-t_0_start,val_loss.item(), t_1_end - t_1_start ,     ntype_acc      )      ) if args.verbose=="True" else None
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                #print('Early stopping!')
                break
        
        # validation with evaluate_results_nc
        net.load_state_dict(torch.load(ckp_fname))
        net.eval()
        with torch.no_grad():
            logits,_ = net(features_list, e_feat)
            val_logits = logits[val_idx]
            pred = val_logits.argmax(axis=1)
            val_acc=((pred==labels[val_idx]).int().sum()/(pred==labels[val_idx]).shape[0]).item()
        val_accs.append(val_acc)

        
        score=sum(val_accs)/len(val_accs)
        trial.report(score, re)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()






        # testing with evaluate_results_nc
        net.load_state_dict(torch.load(ckp_fname))
        net.eval()
        test_logits = []
        with torch.no_grad():
            logits,_ = net(features_list, e_feat)
            test_logits = logits[test_idx]
            pred = test_logits.cpu().numpy().argmax(axis=1)
            onehot = np.eye(num_classes, dtype=np.int32)
            dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_path=f"{args.dataset}_{args.run}.txt")
            pred = onehot[pred]
            d=dl.evaluate(pred)
            print(d) if args.verbose=="True" else None
        ma_F1s.append(d["macro-f1"])
        mi_F1s.append(d["micro-f1"])
        t_re1=time.time();t_re=t_re1-t_re0
        #print(f" this round cost {t_re}(s)")
        remove_ckp_files(ckp_dname=ckp_dname)
    print(f"mean and std of macro-f1: {  100*np.mean(np.array(ma_F1s)) :.1f}\u00B1{  100*np.std(np.array(ma_F1s)) :.1f}")
    print(f"mean and std of micro-f1: {  100*np.mean(np.array(mi_F1s)) :.1f}\u00B1{  100*np.std(np.array(mi_F1s)) :.1f}")
    print(exp_info)
    print(net)
    print(optimizer) if args.verbose=="True" else None

    fn=os.path.join("log",args.study_name)
    if os.path.exists(fn):
        with open(fn,"a") as f:
            f.write(f"score {  score :.4f}  mean and std of macro-f1: {  100*np.mean(np.array(ma_F1s)) :.1f}\u00B1{  100*np.std(np.array(ma_F1s)) :.1f} micro-f1: {  100*np.mean(np.array(mi_F1s)) :.1f}\u00B1{  100*np.std(np.array(mi_F1s)) :.1f}\n")
            f.write(str(exp_info)+"\n")
            f.write(str(net)+"\n")
    else:
        with open(fn,"w") as f:
            f.write(f"score {  score :.4f}  mean and std of macro-f1: {  100*np.mean(np.array(ma_F1s)) :.1f}\u00B1{  100*np.std(np.array(ma_F1s)) :.1f} micro-f1: {  100*np.mean(np.array(mi_F1s)) :.1f}\u00B1{  100*np.std(np.array(mi_F1s)) :.1f}\n")
            f.write(str(exp_info)+"\n")
            f.write(str(net)+"\n")

    return score

def remove_ckp_files(ckp_dname):
    import shutil
    shutil.rmtree(ckp_dname)
    #os.mkdir(ckp_dname)

if __name__ == '__main__':
    

    #torch.cuda.set_device(int(args.gpu))
    #device=torch.device(f"cuda:{int(args.gpu)}")
    if args.study_name=="temp":
        if os.path.exists("./temp.db"):
            os.remove("./temp.db")
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




    
