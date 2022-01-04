import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv

import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
from conv import myGATConv,HCGNNConv,changedGATConv

        
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
                 ae_layer):
        super(changedGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        self.ae_layer=ae_layer
        self.ae_drop=nn.Dropout(feat_drop)
        if ae_layer=="last_hidden":
            self.lc_ae=nn.ModuleList([nn.Linear(num_hidden * heads[-2],num_hidden, bias=True),nn.Linear(num_hidden,num_ntype, bias=True)])
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

    def forward(self, features_list, e_feat):

        hidden_logits=None
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))   # the id is decided by the node types
        h = torch.cat(h, 0)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](self.g, h, e_feat, res_attn=res_attn)
            h = h.flatten(1)
            if self.ae_layer=="last_hidden":
                _h=h
                for i in range(len(self.lc_ae)):
                    _h=self.lc_ae[i](_h)
                    if i==0:
                        _h=self.ae_drop(_h)
                        _h=F.relu(_h)
                hidden_logits=_h
        # output projection
        logits, _ = self.gat_layers[-1](self.g, h, e_feat, res_attn=None)
        logits = logits.mean(1)
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        return logits,hidden_logits









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
                 alpha):
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
        # output projection
        logits, _ = self.gat_layers[-1](self.g, h, e_feat, res_attn=None)
        logits = logits.mean(1)
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        return logits

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
                 residual):
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
                 residual):
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

    def forward(self, features_list):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
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

    def forward(self, features_list):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        return h












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
            
