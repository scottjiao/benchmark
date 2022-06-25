import torch
import torch as th
import torch.nn.functional as F
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv

import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
from conv import myGATConv,HCGNNConv,changedGATConv,slotGATConv,slotGCNConv
from torch.profiler import profile, record_function, ProfilerActivity

from dgl._ffi.base import DGLError





class slotGTN(nn.Module):
    def __init__(self,
                    g,
                    num_etypes,
                    in_dims,
                    num_hidden,
                    num_classes,
                    num_convs,
                    heads,
                    activation,
                    dropout,
                    num_ntype,
                    normalize,
                    ntype_indexer,get_out="False"
                    ):
        super(slotGTN,self).__init__()
        
        self.g=g
        self.num_etypes=num_etypes
        self.in_dims=in_dims
        self.num_hidden=num_hidden
        self.num_classes=num_classes
        self.num_convs=num_convs
        self.heads=heads
        self.activation=activation
        self.dropout=nn.Dropout(dropout)
        
        self.num_ntype=num_ntype
        #self.fc1_list=nn.Linear(in_dims,num_hidden,bias=True)
        self.fc1_list=nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc1_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
            
        self.fc2_list=nn.ModuleList([nn.Linear(num_hidden, num_classes, bias=True) for in_dim in in_dims])
        for fc in self.fc2_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        
        #self.fc2 = nn.Parameter(torch.Tensor(self.num_ntype,num_hidden*heads, num_hidden))#self.fc2=nn.Linear(num_hidden*heads,num_hidden,bias=True)
        self.fc3 = nn.Parameter(torch.Tensor(self.num_ntype,num_classes*heads, num_classes))#self.fc2=nn.Linear(num_hidden*heads,num_hidden,bias=True)
        nn.init.xavier_uniform_(self.fc3)
        #self.fc3=nn.Linear(num_hidden*self.num_ntype,num_classes,bias=True)
        self.fc4=nn.Linear(num_classes*self.num_ntype,num_classes,bias=True)
        nn.init.xavier_uniform_(self.fc4.weight)
        self.normalize=normalize
        self.convs=[]
        for _ in range(heads):
            self.convs.append(nn.Sequential (   *[  GTNConv(num_etypes,g) for _ in range(num_convs)     ]))
        self.ntype_indexer=ntype_indexer
        self.convs=nn.ModuleList(self.convs)
        self.count=0
        

    def forward(self,features_list, e_feat):
        
        h = []
        for nt_id,(fc,fc_l, feature) in enumerate(zip(self.fc1_list,self.fc2_list, features_list)):
            nt_ft=fc_l(F.elu(self.dropout(fc(feature))))
            emsen_ft=torch.zeros([nt_ft.shape[0],nt_ft.shape[1]*self.num_ntype]).to(feature.device)
            emsen_ft[:,nt_ft.shape[1]*nt_id:nt_ft.shape[1]*(nt_id+1)]=nt_ft
            h.append(emsen_ft)   # the id is decided by the node types
        h = torch.cat(h, 0)        #  num_nodes*(num_type*hidden_dim)
        #只有变成h.view(-1,num_ntypes,hidden_dim)才安全


        #h=self.fc1(h)
        z=[]
        for convs_head in self.convs:
            ones=torch.ones([h.shape[0],1]).to(h.device)
            deg=convs_head(ones)
            norm=torch.pow(deg,-1)
            norm[norm == float("Inf")] = 0
            if self.normalize=="True":
                z.append(norm*convs_head(h))
            else:
                z.append(convs_head(h))
        z=torch.cat(z,1)#  num_nodes*(num_type*hidden_dim*num_heads)
        z=F.relu(self.dropout(z))


        ntype_indexer=self.ntype_indexer.permute(1,0)  #需要num_ntypes*num_nodes的0-1 one hot indexer
        z=z.view(-1,self.heads,self.num_ntype,self.num_classes).permute(0,2,1,3).flatten(2,3)  # num_nodes*num_ntypes*(num_heads*num_hidden)
        z=z.permute(1,0,2) # num_ntypes*num_nodes*(num_heads*num_hidden)
        """
        
        y1=tensor([[1., 2., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        y2=tensor([[-1., -2., -3., -0., -0., -0., -0., -0., -0., -0., -0., -0.],
        [-0., -0., -0., -0., -0., -0., -3., -4., -5., -0., -0., -0.]])
        
        yt=torch.cat([y1,y2],1)
        yt.view(-1,2,4,3).permute(0,2,1,3).flatten(2,3)
            #>>>   tensor([[[ 1.,  2.,  3., -1., -2., -3.],
                            [ 0.,  0.,  0., -0., -0., -0.],
                            [ 0.,  0.,  0., -0., -0., -0.],
                            [ 0.,  0.,  0., -0., -0., -0.]],

                            [[ 0.,  0.,  0., -0., -0., -0.],
                            [ 0.,  0.,  0., -0., -0., -0.],
                            [ 3.,  4.,  5., -3., -4., -5.],
                            [ 0.,  0.,  0., -0., -0., -0.]]])"""

        z=torch.bmm(z,self.fc3)
        #z=torch.bmm(z,self.fc2)*ntype_indexer+z*(1-ntype_indexer) # num_ntypes*num_nodes*num_hidden
        z=z.permute(1,0,2).flatten(1,2)# num_nodes*(num_ntypes*num_hidden)
        z=F.relu(self.dropout(z))

             #
        z=self.fc4(z)
        logits=z                  #
        encoded_embeddings=z 
        """if self.count%100==0:
            for i,convs_head in enumerate(self.convs):
                print(f"channel {i}") if i==0 else None
                for j,mod in enumerate(convs_head):
                    print(f"\tlayer {j} "+str(mod.filter.cpu().tolist())) if i==0 else None"""


        self.count+=1

        return logits, encoded_embeddings    #hidden_logits


class slotGTNConv(nn.Module):
    def __init__(self,
                num_etypes,
                g,
                    allow_zero_in_degree = False):
        super(slotGTNConv,self).__init__()
        self.g=g
        g.edge_type_indexer

        self._allow_zero_in_degree = allow_zero_in_degree
        self.w=nn.Parameter(torch.Tensor( 1,num_etypes ) )
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.normal_(self.w, std=0.01)
    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self,h):
        #w=self.w.to(h.device)
        w=self.w
        graph=self.g
        with graph.local_scope():
            node_idx_by_ntype=graph.node_idx_by_ntype
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            self.filter = F.softmax(w, dim=1)
            ew=(self.g.edge_type_indexer*self.filter).sum(1).unsqueeze(-1)

            #normalization
            """graph.srcdata.update({'ones': torch.ones([h.shape[0],1]).to(h.device)})
            graph.edata.update({'ew': ew})
            graph.update_all(fn.u_mul_e('ones', 'ew', 'm'),
                             fn.sum('m', 'deg'))
            deg = graph.dstdata['deg']
            norm=torch.pow(deg,-1)
            norm[norm == float("Inf")] = 0"""

            #propagate

            graph.srcdata.update({'ft': h})
            graph.edata.update({'ew': ew})
            graph.update_all(fn.u_mul_e('ft', 'ew', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
        

        return rst





class MLP(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(MLP, self).__init__()
        self.num_classes=num_classes
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(nn.Linear(num_hidden, num_hidden, bias=True))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(num_hidden, num_hidden))
        # output layer
        self.layers.append(nn.Linear(num_hidden, num_classes))
        for ly in self.layers:
            nn.init.xavier_normal_(ly.weight, gain=1.414)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features_list, e_feat):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)

        for i, layer in enumerate(self.layers):
            encoded_embeddings=h
            h = self.dropout(h)
            h = layer(h)
            h=F.relu(h) if i<len(self.layers) else h

        return h,encoded_embeddings



class LabelPropagation(nn.Module):
    r"""
    Description
    -----------
    Introduced in `Learning from Labeled and Unlabeled Data with Label Propagation <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.3864&rep=rep1&type=pdf>`_
    .. math::
        \mathbf{Y}^{\prime} = \alpha \cdot \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \mathbf{Y} + (1 - \alpha) \mathbf{Y},
    where unlabeled data is inferred by labeled data via propagation.
    Parameters
    ----------
        num_layers: int
            The number of propagations.
        alpha: float
            The :math:`\alpha` coefficient.
    """
    def __init__(self, num_layers, alpha):
        super(LabelPropagation, self).__init__()

        self.num_layers = num_layers
        self.alpha = alpha
    
    @torch.no_grad()
    def forward(self, g, labels, mask,get_out="False"):    # labels.shape=(number of nodes of type 0)  may contain false labels, therefore here the mask argument which provides the training nodes' idx is important
        with g.local_scope():
            if labels.dtype == torch.long:
                labels = F.one_hot(labels.view(-1)).to(torch.float32)
            y=torch.zeros((g.num_nodes(),labels.shape[1])).to(labels.device)
            y[mask] = labels[mask]
            
            last = (1 - self.alpha) * y
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).to(labels.device).unsqueeze(1)

            for _ in range(self.num_layers):
                # Assume the graphs to be undirected
                g.ndata['h'] = y * norm
                g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                y = last + self.alpha * g.ndata.pop('h') * norm
                y=F.normalize(y,p=1,dim=1)   #normalize y by row with p-1-norm
                y[mask] = labels[mask]
                last = (1 - self.alpha) * y
            
            return y,None



class slotGCN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout,num_ntype,aggregator,slot_trans,ntype_indexer,semantic_trans,semantic_trans_normalize,get_out="False"):
        super(slotGCN, self).__init__()
        self.g = g
        self.num_ntype=num_ntype
        self.aggregator=aggregator
        self.slot_trans=slot_trans
        self.num_classes=num_classes
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(slotGCNConv(num_hidden, num_hidden, activation=activation, weight=False,num_ntype=self.num_ntype,slot_trans=slot_trans,ntype_indexer=ntype_indexer,semantic_trans=semantic_trans,semantic_trans_normalize=semantic_trans_normalize))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(slotGCNConv(num_hidden, num_hidden, activation=activation,num_ntype=self.num_ntype,slot_trans=slot_trans,ntype_indexer=ntype_indexer,semantic_trans=semantic_trans,semantic_trans_normalize=semantic_trans_normalize))
        # output layer
        self.layers.append(slotGCNConv(num_hidden, num_classes,num_ntype=self.num_ntype,aggregator=self.aggregator,ntype_indexer=ntype_indexer,semantic_trans=semantic_trans,semantic_trans_normalize=semantic_trans_normalize))
        negative_slope=0.2
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(p=dropout)
        if self.aggregator=="onedimconv":
            self.nt_aggr=nn.Parameter(torch.FloatTensor(1,self.num_ntype,1));nn.init.normal_(self.nt_aggr,std=1)

    def forward(self, features_list, e_feat):
        h = []
        for nt_id,(fc, feature) in enumerate(zip(self.fc_list, features_list)):
            nt_ft=fc(feature)
            emsen_ft=torch.zeros([nt_ft.shape[0],nt_ft.shape[1]*self.num_ntype]).to(feature.device)
            emsen_ft[:,nt_ft.shape[1]*nt_id:nt_ft.shape[1]*(nt_id+1)]=nt_ft
            h.append(emsen_ft)   # the id is decided by the node types
        h = torch.cat(h, 0)        #  num_nodes*(num_type*hidden_dim)


        """h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)"""
        for i, layer in enumerate(self.layers):
            encoded_embeddings=h
            h = self.dropout(h)
            h = layer(self.g, h)

        logits=h
        if self.aggregator=="average":
            logits=logits.view(-1,self.num_ntype,self.num_classes).mean(1)
        elif self.aggregator=="onedimconv":
            logits=(logits.view(-1,self.num_ntype,self.num_classes)*F.softmax(self.leaky_relu(self.nt_aggr),dim=1)).sum(1)
        elif self.aggregator=="last_fc":
            logits=logits
        else:
            raise NotImplementedError()
        h=logits

        return h,encoded_embeddings






       
class slotGAT(nn.Module):
    def __init__(self,
                 g,
                 edge_dim,
                 num_etypes,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha,
                 num_ntype,
                 n_type_mappings,
                 res_n_type_mappings,
                 etype_specified_attention,
                 eindexer,
                 ae_layer,aggregator="average",semantic_trans="False",semantic_trans_normalize="row",attention_average="False",attention_mse_sampling_factor=0,attention_mse_weight_factor=0,attention_1_type_bigger_constraint=0,attention_0_type_bigger_constraint=0,predicted_by_slot="None",
                 addLogitsEpsilon=0,addLogitsTrain="None",get_out="False",slot_attention="False",relevant_passing="False"):
        super(slotGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        self.ae_layer=ae_layer
        self.num_ntype=num_ntype
        self.num_classes=num_classes
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.attention_mse_sampling_factor=attention_mse_sampling_factor
        self.attention_mse_weight_factor=attention_mse_weight_factor
        self.attention_1_type_bigger_constraint=attention_1_type_bigger_constraint
        self.attention_0_type_bigger_constraint=attention_0_type_bigger_constraint
        self.predicted_by_slot=predicted_by_slot
        self.addLogitsEpsilon=addLogitsEpsilon
        self.addLogitsTrain=addLogitsTrain
        self.slot_attention=slot_attention
        self.relevant_passing=relevant_passing
        if relevant_passing=="True":
            assert slot_attention=="True"

        #self.ae_drop=nn.Dropout(feat_drop)
        #if ae_layer=="last_hidden":
            #self.lc_ae=nn.ModuleList([nn.Linear(num_hidden * heads[-2],num_hidden, bias=True),nn.Linear(num_hidden,num_ntype, bias=True)])
        self.last_fc = nn.Parameter(th.FloatTensor(size=(num_classes*self.num_ntype, num_classes))) ;nn.init.xavier_normal_(self.last_fc, gain=1.414)
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(slotGATConv(edge_dim, num_etypes,
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha,num_ntype=num_ntype,n_type_mappings=n_type_mappings,res_n_type_mappings=res_n_type_mappings,etype_specified_attention=etype_specified_attention,eindexer=eindexer,inputhead=True,semantic_trans=semantic_trans,semantic_trans_normalize=semantic_trans_normalize,attention_average=attention_average,attention_mse_sampling_factor=attention_mse_sampling_factor,attention_mse_weight_factor=attention_mse_weight_factor,attention_1_type_bigger_constraint=attention_1_type_bigger_constraint,attention_0_type_bigger_constraint=attention_0_type_bigger_constraint,slot_attention=slot_attention,relevant_passing=relevant_passing))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(slotGATConv(edge_dim, num_etypes,
                num_hidden* heads[l-1] , num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha,num_ntype=num_ntype,n_type_mappings=n_type_mappings,res_n_type_mappings=res_n_type_mappings,etype_specified_attention=etype_specified_attention,eindexer=eindexer,semantic_trans=semantic_trans,semantic_trans_normalize=semantic_trans_normalize,attention_average=attention_average,attention_mse_sampling_factor=attention_mse_sampling_factor,attention_mse_weight_factor=attention_mse_weight_factor,attention_1_type_bigger_constraint=attention_1_type_bigger_constraint,attention_0_type_bigger_constraint=attention_0_type_bigger_constraint,slot_attention=slot_attention,relevant_passing=relevant_passing))
        # output projection
        self.gat_layers.append(slotGATConv(edge_dim, num_etypes,
            num_hidden* heads[-2] , num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha,num_ntype=num_ntype,n_type_mappings=n_type_mappings,res_n_type_mappings=res_n_type_mappings,etype_specified_attention=etype_specified_attention,eindexer=eindexer,semantic_trans=semantic_trans,semantic_trans_normalize=semantic_trans_normalize,attention_average=attention_average,attention_mse_sampling_factor=attention_mse_sampling_factor,attention_mse_weight_factor=attention_mse_weight_factor,attention_1_type_bigger_constraint=attention_1_type_bigger_constraint,attention_0_type_bigger_constraint=attention_0_type_bigger_constraint,slot_attention=slot_attention,relevant_passing=relevant_passing))
        self.aggregator=aggregator
        self.by_slot=[f"by_slot_{nt}" for nt in range(g.num_ntypes)]
        assert aggregator in (["onedimconv","average","last_fc","slot_majority_voting","max"]+self.by_slot)
        if self.aggregator=="onedimconv":
            self.nt_aggr=nn.Parameter(torch.FloatTensor(1,1,self.num_ntype,1));nn.init.normal_(self.nt_aggr,std=1)
        self.get_out=get_out
        self.epsilon = torch.FloatTensor([1e-12]).cuda()

    def forward(self, features_list,e_feat, get_out="False"):
        with record_function("model_forward"):
            encoded_embeddings=None
            h = []
            for nt_id,(fc, feature) in enumerate(zip(self.fc_list, features_list)):
                nt_ft=fc(feature)
                emsen_ft=torch.zeros([nt_ft.shape[0],nt_ft.shape[1]*self.num_ntype]).to(feature.device)
                emsen_ft[:,nt_ft.shape[1]*nt_id:nt_ft.shape[1]*(nt_id+1)]=nt_ft
                h.append(emsen_ft)   # the id is decided by the node types
            h = torch.cat(h, 0)        #  num_nodes*(num_type*hidden_dim)
            res_attn = None
            for l in range(self.num_layers):
                h, res_attn = self.gat_layers[l](self.g, h, e_feat,get_out=get_out, res_attn=res_attn)   #num_nodes*num_heads*(num_ntype*hidden_dim)
                h = h.flatten(1)#num_nodes*(num_heads*num_ntype*hidden_dim)
                #if self.ae_layer=="last_hidden":
                encoded_embeddings=h
            # output projection
            logits, _ = self.gat_layers[-1](self.g, h, e_feat,get_out=get_out, res_attn=None)   #num_nodes*num_heads*num_ntype*hidden_dim
        #average across the ntype info
        if self.predicted_by_slot!="None" and self.training==False:
            with record_function("predict_by_slot"):
                logits=logits.view(-1,1,self.num_ntype,self.num_classes)
                self.scale_analysis=torch.std_mean(logits.squeeze(1).mean(dim=-1).detach().cpu(),dim=0) if self.get_out=="True" else None
                if self.predicted_by_slot in ["majority_voting","majority_voting_max"] :
                    logits=logits.squeeze(1)           # num_nodes * num_ntypes*num_classes
                    with torch.no_grad():
                        slot_votings=torch.argmax(logits,dim=-1)   # num_nodes * num_ntypes
                        if self.get_out=="True":
                            slot_votings_onehot=F.one_hot(slot_votings)## num_nodes * num_ntypes *num_classes
                            votings_count=slot_votings_onehot.sum(1) ## num_nodes  *num_classes
                            votings_max_count=votings_count.max(1)[0] ## num_nodes 
                            ties_flags_pos=(votings_max_count.unsqueeze(-1)==votings_count)   ## num_nodes  *num_classes
                            ties_flags=ties_flags_pos.sum(-1)>1   ## num_nodes 
                            ties_ids=ties_flags.int().nonzero().flatten().tolist()   ## num_nodes 
                            voting_patterns=torch.sort(votings_count,descending=True,dim=-1)[0]  #num_nodes  *num_classes
                            pattern_counts={}
                            ties_labels={}
                            ties_first_labels={}
                            ties_second_labels={}
                            ties_third_labels={}
                            ties_fourth_labels={}
                            for i in range(voting_patterns.shape[0]):
                                if i in self.g.node_idx_by_ntype[0]:
                                    pattern=tuple(voting_patterns[i].flatten().tolist())
                                    if pattern not in pattern_counts.keys():
                                        pattern_counts[pattern]=0
                                    pattern_counts[pattern]+=1

                            for i in ties_ids:
                                ties_labels[i]=ties_flags_pos[i].nonzero().flatten().tolist()
                                ties_first_labels[i]=ties_labels[i][0]
                                ties_second_labels[i]=ties_first_labels[i] if len(ties_labels[i])<2 else ties_labels[i][1]
                                ties_third_labels[i]=ties_second_labels[i] if len(ties_labels[i])<3 else ties_labels[i][2]
                                ties_fourth_labels[i]=ties_third_labels[i] if len(ties_labels[i])<4 else ties_labels[i][3]
                                    
                            self.majority_voting_analysis={"pattern_counts":pattern_counts,"ties_first_labels":ties_first_labels,"ties_second_labels":ties_second_labels,"ties_third_labels":ties_third_labels,"ties_fourth_labels":ties_fourth_labels,"ties_labels":ties_labels,"ties_ids":ties_ids}

                        ## num_nodes *num_classes
                        votings=torch.argmax(F.one_hot(torch.argmax(logits,dim=-1)).sum(1),dim=-1)  #num_nodes
                        #num_nodes*1
                        votings_int=(slot_votings==(votings.unsqueeze(1))).int().unsqueeze(-1)   # num_nodes *num_ntypes *1
                        self.votings_int=votings_int
                        self.voting_patterns=voting_patterns  if self.get_out=="True" else None


                    if self.predicted_by_slot=="majority_voting_max":
                        logits=(logits*votings_int).max(1,keepdim=True)[0] #num_nodes *  1 *num_classes
                    else:
                        logits=(logits*votings_int).sum(1,keepdim=True) #num_nodes *  1 *num_classes
                elif self.predicted_by_slot=="max":
                    logits=logits.max(2)[0]
                else:
                    target_slot=int(self.predicted_by_slot)
                    logits=logits[:,:,target_slot,:].squeeze(2)
        else:
            with record_function("slot_aggregation"):
                if self.aggregator=="average":
                    logits=logits.view(-1,1,self.num_ntype,self.num_classes).mean(2)
                elif self.aggregator=="onedimconv":
                    logits=(logits.view(-1,1,self.num_ntype,self.num_classes)*F.softmax(self.leaky_relu(self.nt_aggr),dim=2)).sum(2)
                elif self.aggregator=="last_fc":
                    logits=logits.view(-1,1,self.num_ntype,self.num_classes)
                    logits=logits.flatten(1)
                    logits=logits.matmul(self.last_fc).unsqueeze(1)
                elif self.aggregator=="max":
                    logits=logits.view(-1,1,self.num_ntype,self.num_classes).max(2)[0]



                else:
                    raise NotImplementedError()
        #average across the heads
        ### logits = [num_nodes *  num_of_heads *num_classes]
        self.logits_mean=logits.flatten().mean()
        logits = logits.mean(1)
        if self.addLogitsTrain=="True" or (self.addLogitsTrain=="False" and self.training==False):
            logits+=self.addLogitsEpsilon
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        return logits, encoded_embeddings    #hidden_logits




class attGTN(nn.Module):
    def __init__(self,
                    g,
                    num_etypes,
                    in_dims,

                    num_hidden,
                    num_classes,
                    num_convs,
                    heads,
                    activation,
                    dropout,
                    residual,get_out="False"
                    ):
        super(attGTN,self).__init__()
        
        self.g=g
        self.num_etypes=num_etypes
        self.in_dims=in_dims
        self.num_hidden=num_hidden
        self.num_classes=num_classes
        self.num_convs=num_convs
        self.heads=heads
        self.activation=activation
        self.dropout=nn.Dropout(dropout)
        self.edgedrop=nn.Dropout(dropout)
        #self.fc1_list=nn.Linear(in_dims,num_hidden,bias=True)
        self.fc1_list=nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        self.fc2=nn.Linear(num_hidden*heads,num_hidden,bias=True)
        self.fc3=nn.Linear(num_hidden,num_classes,bias=True)
        self.residual=residual
        self.convs=[]
        dim=num_hidden
        for _ in range(heads):
            self.convs.append(nn.Sequential (   *[  attGTNConv(dim,g,self.edgedrop) for _ in range(num_convs)     ]))
        
        self.convs=nn.ModuleList(self.convs)
        self.count=0
        

    def forward(self,features_list, e_feat):
        
        h = []
        for fc, feature in zip(self.fc1_list, features_list):
            h.append(fc(feature))   # the id is decided by the node types
        h = torch.cat(h, 0) #num_nodes*hidden


        #h=self.fc1(h)
        z=[]
        for convs_head in self.convs:
            #ones=torch.ones([h.shape[0],1]).to(h.device)
            #deg=convs_head(ones)
            #norm=torch.pow(deg,-1)
            #norm[norm == float("Inf")] = 0
            #z.append(norm*convs_head(h))
            if self.residual=="True":
                z.append(h+convs_head(h))
            else:
                z.append(convs_head(h))

        z=torch.cat(z,1)
        z=F.relu(self.dropout(z))
        z=self.fc2(z)
        encoded_embeddings=z      #
        z=F.relu(self.dropout(z))
        z=self.fc3(z)
        logits=z                  #

        """if self.count%100==0:
            for i,convs_head in enumerate(self.convs):
                print(f"channel {i}") if i==0 else None
                for j,mod in enumerate(convs_head):
                    print(f"\tlayer {j} "+str(mod.filter.cpu().tolist())) if i==0 else None"""


        self.count+=1

        return logits, encoded_embeddings    #hidden_logits

class attGTNConv(nn.Module):
    def __init__(self,
                dim,
                g,
                dropout,
                    allow_zero_in_degree = False):
        super(attGTNConv,self).__init__()
        self.g=g
        #g.edge_type_indexer
        self.edgedrop=dropout
        self.al=nn.Parameter(torch.Tensor( 1,dim ) )
        self.ar=nn.Parameter(torch.Tensor( 1,dim ) )
        self._allow_zero_in_degree = allow_zero_in_degree
        negative_slope=0.2;self.leaky_relu = nn.LeakyReLU(negative_slope)
        #self.w=nn.Parameter(torch.Tensor( 1,num_etypes ) )
        self.reset_parameters()


    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.ar, gain=gain)
        nn.init.xavier_normal_(self.al, gain=gain)
    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self,h):
        #w=self.w.to(h.device)
        #w=self.w
        graph=self.g
        with graph.local_scope():
            node_idx_by_ntype=graph.node_idx_by_ntype
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            #self.filter = F.softmax(w, dim=1)
            #ew=(self.g.edge_type_indexer*self.filter).sum(1).unsqueeze(-1)

            #normalization
            """graph.srcdata.update({'ones': torch.ones([h.shape[0],1]).to(h.device)})
            graph.edata.update({'ew': ew})
            graph.update_all(fn.u_mul_e('ones', 'ew', 'm'),
                             fn.sum('m', 'deg'))
            deg = graph.dstdata['deg']
            norm=torch.pow(deg,-1)
            norm[norm == float("Inf")] = 0"""

            #propagate

            al_s=self.al*h.sum(-1).unsqueeze(-1)
            ar_s=self.ar*h.sum(-1).unsqueeze(-1)
            graph.srcdata.update({'al': al_s,'ar':ar_s})
            graph.apply_edges(fn.u_add_v('al', 'ar', 'a'))
            a = self.leaky_relu(graph.edata.pop('a')    )
            # compute softmax
            #graph.edata['a'] = self.edgedrop(edge_softmax(graph, a))
            graph.edata['a'] = self.edgedrop(edge_softmax(graph, a))

            graph.srcdata.update({'ft': h})
            #graph.edata.update({'ew': ew})
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
        

        return rst





class GTN(nn.Module):
    def __init__(self,
                    g,
                    num_etypes,
                    in_dims,
                    num_hidden,
                    num_classes,
                    num_convs,
                    heads,
                    activation,
                    dropout,get_out="False"
                    ):
        super(GTN,self).__init__()
        
        self.g=g
        self.num_etypes=num_etypes
        self.in_dims=in_dims
        self.num_hidden=num_hidden
        self.num_classes=num_classes
        self.num_convs=num_convs
        self.heads=heads
        self.activation=activation
        self.dropout=nn.Dropout(dropout)
        #self.fc1_list=nn.Linear(in_dims,num_hidden,bias=True)
        self.fc1_list=nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        self.fc2=nn.Linear(num_hidden*heads,num_hidden,bias=True)
        self.fc3=nn.Linear(num_hidden,num_classes,bias=True)
        self.convs=[]
        for _ in range(heads):
            self.convs.append(nn.Sequential (   *[  GTNConv(num_etypes,g) for _ in range(num_convs)     ]))
        
        self.convs=nn.ModuleList(self.convs)
        self.count=0
        

    def forward(self,features_list, e_feat):
        
        h = []
        for fc, feature in zip(self.fc1_list, features_list):
            h.append(fc(feature))   # the id is decided by the node types
        h = torch.cat(h, 0)



        #h=self.fc1(h)
        z=[]
        for convs_head in self.convs:
            ones=torch.ones([h.shape[0],1]).to(h.device)
            deg=convs_head(ones)
            norm=torch.pow(deg,-1)
            norm[norm == float("Inf")] = 0
            z.append(norm*convs_head(h))
        z=torch.cat(z,1)
        z=F.relu(self.dropout(z))
        z=self.fc2(z)
        encoded_embeddings=z      #
        z=F.relu(self.dropout(z))
        z=self.fc3(z)
        logits=z                  #

        """if self.count%100==0:
            for i,convs_head in enumerate(self.convs):
                print(f"channel {i}") if i==0 else None
                for j,mod in enumerate(convs_head):
                    print(f"\tlayer {j} "+str(mod.filter.cpu().tolist())) if i==0 else None"""


        self.count+=1

        return logits, encoded_embeddings    #hidden_logits


class GTNConv(nn.Module):
    def __init__(self,
                num_etypes,
                g,
                    allow_zero_in_degree = False):
        super(GTNConv,self).__init__()
        self.g=g
        g.edge_type_indexer

        self._allow_zero_in_degree = allow_zero_in_degree
        self.w=nn.Parameter(torch.Tensor( 1,num_etypes ) )
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.normal_(self.w, std=0.01)
    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self,h):
        #w=self.w.to(h.device)
        w=self.w
        graph=self.g
        with graph.local_scope():
            node_idx_by_ntype=graph.node_idx_by_ntype
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            self.filter = F.softmax(w, dim=1)
            ew=(self.g.edge_type_indexer*self.filter).sum(1).unsqueeze(-1)

            #normalization
            """graph.srcdata.update({'ones': torch.ones([h.shape[0],1]).to(h.device)})
            graph.edata.update({'ew': ew})
            graph.update_all(fn.u_mul_e('ones', 'ew', 'm'),
                             fn.sum('m', 'deg'))
            deg = graph.dstdata['deg']
            norm=torch.pow(deg,-1)
            norm[norm == float("Inf")] = 0"""

            #propagate

            graph.srcdata.update({'ft': h})
            graph.edata.update({'ew': ew})
            graph.update_all(fn.u_mul_e('ft', 'ew', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
        

        return rst







class NTYPE_ENCODER(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim,dropout):
        super(NTYPE_ENCODER,self).__init__()
        self.lc_ae=nn.ModuleList([nn.Linear(in_dim,hidden_dim, bias=True),nn.Linear(hidden_dim,out_dim, bias=True)])
        self.ae_drop=nn.Dropout(dropout)
    def forward(self,h):
        for i in range(len(self.lc_ae)):
            h=self.lc_ae[i](h)
            if i==0:
                h=self.ae_drop(h)
                h=F.relu(h)
            hidden_logits=h
        return hidden_logits
        
class changedGAT(nn.Module):
    def __init__(self,
                 g,
                 edge_dim,
                 num_etypes,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha,
                 num_ntype,
                 n_type_mappings,
                 res_n_type_mappings,
                 etype_specified_attention,
                 eindexer,
                 ae_layer,get_out="False"):
        super(changedGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        self.ae_layer=ae_layer
        #self.ae_drop=nn.Dropout(feat_drop)
        #if ae_layer=="last_hidden":
            #self.lc_ae=nn.ModuleList([nn.Linear(num_hidden * heads[-2],num_hidden, bias=True),nn.Linear(num_hidden,num_ntype, bias=True)])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(changedGATConv(edge_dim, num_etypes,
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha,num_ntype=num_ntype,n_type_mappings=n_type_mappings,res_n_type_mappings=res_n_type_mappings,etype_specified_attention=etype_specified_attention,eindexer=eindexer))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(changedGATConv(edge_dim, num_etypes,
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha,num_ntype=num_ntype,n_type_mappings=n_type_mappings,res_n_type_mappings=res_n_type_mappings,etype_specified_attention=etype_specified_attention,eindexer=eindexer))
        # output projection
        self.gat_layers.append(changedGATConv(edge_dim, num_etypes,
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha,num_ntype=num_ntype,n_type_mappings=n_type_mappings,res_n_type_mappings=res_n_type_mappings,etype_specified_attention=etype_specified_attention,eindexer=eindexer))
        self.epsilon = torch.FloatTensor([1e-12]).cuda()

    def forward(self, features_list, e_feat,get_out="False"):

        hidden_logits=None
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))   # the id is decided by the node types
        h = torch.cat(h, 0)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](self.g, h, e_feat, res_attn=res_attn)
            h = h.flatten(1)
            #if self.ae_layer=="last_hidden":
            encoded_embeddings=h
            """for i in range(len(self.lc_ae)):
                _h=self.lc_ae[i](_h)
                if i==0:
                    _h=self.ae_drop(_h)
                    _h=F.relu(_h)
            hidden_logits=_h"""
        # output projection
        logits, _ = self.gat_layers[-1](self.g, h, e_feat, res_attn=None)
        logits = logits.mean(1)
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        return logits, encoded_embeddings    #hidden_logits






class myGAT(nn.Module):
    def __init__(self,
                 g,
                 edge_dim,
                 num_etypes,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha,get_out="False"):
        super(myGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(myGATConv(edge_dim, num_etypes,
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(myGATConv(edge_dim, num_etypes,
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha))
        # output projection
        self.gat_layers.append(myGATConv(edge_dim, num_etypes,
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha))
        self.epsilon = torch.FloatTensor([1e-12]).cuda()

    def forward(self, features_list, e_feat):


        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))   # the id is decided by the node types
        h = torch.cat(h, 0)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](self.g, h, e_feat, res_attn=res_attn)
            h = h.flatten(1)
            encoded_embeddings=h
        # output projection
        logits, _ = self.gat_layers[-1](self.g, h, e_feat, res_attn=None)
        logits = logits.mean(1)
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        return logits,encoded_embeddings

class RGAT(nn.Module):
    def __init__(self,
                 gs,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,get_out="False"):
        super(GAT, self).__init__()
        self.gs = gs
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList([nn.ModuleList() for i in range(len(gs))])
        self.activation = activation
        self.weights = nn.Parameter(torch.zeros((len(in_dims), num_layers+1, len(gs))))
        self.sm = nn.Softmax(2)
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        for i in range(len(gs)):
            # input projection (no residual)
            self.gat_layers[i].append(GATConv(
                num_hidden, num_hidden, heads[0],
                feat_drop, attn_drop, negative_slope, False, self.activation))
            # hidden layers
            for l in range(1, num_layers):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers[i].append(GATConv(
                    num_hidden * heads[l-1], num_hidden, heads[l],
                    feat_drop, attn_drop, negative_slope, residual, self.activation))
            # output projection
            self.gat_layers[i].append(GATConv(
                num_hidden * heads[-2], num_classes, heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, features_list):
        nums = [feat.size(0) for feat in features_list]
        weights = self.sm(self.weights)
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for l in range(self.num_layers):
            out = []
            for i in range(len(self.gs)):
                out.append(torch.split(self.gat_layers[i][l](self.gs[i], h).flatten(1), nums))
            h = []
            for k in range(len(nums)):
                tmp = []
                for i in range(len(self.gs)):
                    tmp.append(out[i][k]*weights[k,l,i])
                h.append(sum(tmp))
            h = torch.cat(h, 0)
        out = []
        for i in range(len(self.gs)):
            out.append(torch.split(self.gat_layers[i][-1](self.gs[i], h).mean(1), nums))
        logits = []
        for k in range(len(nums)):
            tmp = []
            for i in range(len(self.gs)):
                tmp.append(out[i][k]*weights[k,-1,i])
            logits.append(sum(tmp))
        logits = torch.cat(logits, 0)
        return logits

class GAT(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,get_out="False"):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, features_list, e_feat):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
            encoded_embeddings=h
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits,encoded_embeddings


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout,get_out="False"):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(num_hidden, num_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features_list, e_feat,get_out="False"):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            encoded_embeddings=h
            h = self.dropout(h)
            h = layer(self.g, h)
        return h,encoded_embeddings












class HeteroCGNN(nn.Module):  #heterogeneous communication graph neural networks
    def __init__(self,g,num_etype,num_ntypes,num_layers,hiddens,dropout,num_classes,bias,activation,com_dim,ntype_dims,L2_norm,num_heads,negative_slope,**kwargs):
        super(HeteroCGNN, self).__init__()

        self.g=g
        self.num_layers = num_layers
        self.num_etype=num_etype
        self.dropout=nn.Dropout(dropout)
        ntype_dims=ntype_dims
        com_dim=com_dim
        self.input_projections=nn.ModuleList([nn.Linear(in_dim, hiddens[0], bias=True) for in_dim in ntype_dims])
        self.ntypeLinear=[]
        self.activation=activation
        for i in range(num_ntypes):
            temp=[]
            #temp.append(nn.Linear(ntype_dims[i],hiddens[0], bias=True))
            for j in range(num_layers):
                temp.append(nn.Linear(hiddens[j],hiddens[j+1]))
            temp=nn.ModuleList(temp)
            self.ntypeLinear.append(temp)
        self.ntypeLinear=nn.ModuleList(self.ntypeLinear)
        #self.ntypeLinear=nn.ModuleList([  nn.ModuleList( [nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims] )     for num_hidden in hiddens      ])
        self.convs=[]
        for i in range(num_layers):
            self.convs.append(HCGNNConv(in_dim=hiddens[i],com_dim=com_dim,dropout=dropout,bias=bias,activation=activation,num_etype=num_etype,num_heads=num_heads,negative_slope=negative_slope))
        self.convs=nn.ModuleList(self.convs)
        self.prediction_linear=nn.Linear(hiddens[-1]+com_dim,num_classes)
        self.epsilon = torch.FloatTensor([1e-12]).cuda()
        #initialize communication signals
        self.com_signal=torch.zeros(self.g.num_nodes(),com_dim)
        self.L2_norm=L2_norm


    def forward(self, features_list,e_feat):
        node_idx_by_ntype=self.g.node_idx_by_ntype
        node_ntype_indexer =self.g.node_ntype_indexer #N*num_ntype    #每个node可不可以有属于其他node的语义
        h = []
        for fc, feature in zip(self.input_projections, features_list):
            h.append(fc(feature))   # the id is decided by the node types
        h = torch.cat(h, 0)
        com_signal=self.com_signal.to(features_list[0].device)
        for i in range(self.num_layers):
            com_signal=self.convs[i](self.g,h,com_signal)
            h_new=[]
            for type_count,idx in enumerate(node_idx_by_ntype):
                h_new.append(self.activation(self.dropout(self.ntypeLinear[type_count][i](h[idx,:]))))
            h = torch.cat(h_new, 0)
        h=torch.cat((h,com_signal),dim=1)
        logits=self.prediction_linear(h)
        if self.L2_norm:
            logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        return logits
            
