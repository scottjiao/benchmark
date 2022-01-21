import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv

import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
from conv import myGATConv,HCGNNConv,changedGATConv,slotGATConv

from dgl._ffi.base import DGLError

       
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
                 ae_layer):
        super(slotGAT, self).__init__()
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
        self.gat_layers.append(slotGATConv(edge_dim, num_etypes,
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha,num_ntype=num_ntype,n_type_mappings=n_type_mappings,res_n_type_mappings=res_n_type_mappings,etype_specified_attention=etype_specified_attention,eindexer=eindexer))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(slotGATConv(edge_dim, num_etypes,
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha,num_ntype=num_ntype,n_type_mappings=n_type_mappings,res_n_type_mappings=res_n_type_mappings,etype_specified_attention=etype_specified_attention,eindexer=eindexer))
        # output projection
        self.gat_layers.append(slotGATConv(edge_dim, num_etypes,
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha,num_ntype=num_ntype,n_type_mappings=n_type_mappings,res_n_type_mappings=res_n_type_mappings,etype_specified_attention=etype_specified_attention,eindexer=eindexer))
        self.epsilon = torch.FloatTensor([1e-12]).cuda()

    def forward(self, features_list, e_feat):

        encoded_embeddings=None
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
                    residual
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
                    dropout
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
                 ae_layer):
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

    def forward(self, features_list, e_feat):
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
            
