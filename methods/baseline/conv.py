"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
from shutil import ExecError
import torch as th
from torch import nn
import torch
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair


class slotGATConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """
    def __init__(self,
                 edge_feats,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 alpha=0.,
                 num_ntype=None,n_type_mappings=False,res_n_type_mappings=False,etype_specified_attention=False,eindexer=None,aggregate_slots=False,inputhead=False):
        super(slotGATConv, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Embedding(num_etypes, edge_feats)
        self.n_type_mappings=n_type_mappings
        self.res_n_type_mappings=res_n_type_mappings
        self.etype_specified_attention=etype_specified_attention
        self.eindexer=eindexer
        self.aggregate_slots=aggregate_slots
        self.num_ntype=num_ntype
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
            raise Exception("!!!")
        else:
            if not n_type_mappings:
                #self.fc = nn.Linear(
                    #self._in_src_feats, out_feats * num_heads, bias=False)
                self.fc = nn.Parameter(th.FloatTensor(size=(self.num_ntype, self._in_src_feats, out_feats * num_heads)))
            """else:
                self.fc =nn.ModuleList([nn.Linear(
                    self._in_src_feats, out_feats * num_heads, bias=False)  for _ in range(num_ntype)] )
                raise Exception("!!!")"""
        self.fc_e = nn.Linear(edge_feats, edge_feats*num_heads, bias=False)
        if self.etype_specified_attention:
            self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats,num_etypes)))
            self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats,num_etypes)))
        else:
            self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats*self.num_ntype)))
            self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats*self.num_ntype)))
            self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                if not self.res_n_type_mappings:

                    self.res_fc =nn.Parameter(th.FloatTensor(size=(self.num_ntype, self._in_src_feats, out_feats * num_heads)))
                    """self.res_fc = nn.Linear(
                        self._in_dst_feats, num_heads * out_feats, bias=False)"""
                else:
                    raise NotImplementedError()
                    self.res_fc =nn.ModuleList([nn.Linear(
                        self._in_dst_feats, num_heads * out_feats, bias=False)  for _ in range(num_ntype)] )
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            raise NotImplementedError()
            if not self.n_type_mappings:
                self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
            else:
                self.bias_param=nn.ModuleList([ nn.Parameter(th.zeros((1, num_heads, out_feats)))  for _ in range(num_ntype) ])
        self.alpha = alpha
        self.inputhead=inputhead

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc, gain=gain)
            """if self.n_type_mappings:
                for m in self.fc:
                    nn.init.xavier_normal_(m.weight, gain=gain)
            else:
                nn.init.xavier_normal_(self.fc.weight, gain=gain)"""
        else:
            raise Exception("!!!")
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if not self.etype_specified_attention:
            nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            if self.res_n_type_mappings:
                for m in self.res_fc:
                    nn.init.xavier_normal_(m.weight, gain=gain)
            else:
                nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        elif isinstance(self.res_fc, Identity):
            pass
        elif isinstance(self.res_fc, nn.Parameter):
            nn.init.xavier_normal_(self.res_fc, gain=gain)

        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None):
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

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
                raise Exception("!!!")
            else:
                #feature transformation first
                h_src = h_dst = self.feat_drop(feat)   #num_nodes*(num_heads*num_ntype*input_dim)
                if self.n_type_mappings:
                    raise Exception("!!!")
                    h_new=[]
                    for type_count,idx in enumerate(node_idx_by_ntype):
                        h_new.append(self.fc[type_count](h_src[idx,:]).view(
                        -1, self._num_heads, self._out_feats))
                    feat_src = feat_dst = torch.cat(h_new, 0)
                else:
                    if self.inputhead:
                        h_src=h_src.view(-1,1,self.num_ntype,self._in_src_feats)
                    else:
                        h_src=h_src.view(-1,self._num_heads,self.num_ntype,int(self._in_src_feats/self._num_heads))
                    h_dst=h_src=h_src.permute(2,0,1,3).flatten(2)  #num_ntype*num_nodes*(in_feat_dim)
                    #self.fc with num_ntype*(in_feat_dim)*(out_feats * num_heads)
                    
                    feat_src = feat_dst = torch.bmm(h_src,self.fc)  #num_ntype*num_nodes*(out_feats * num_heads)
                    feat_src = feat_dst =feat_src.permute(1,0,2).view(                 #num_nodes*num_heads*(num_ntype*hidden_dim)
                        -1,self.num_ntype ,self._num_heads, self._out_feats).permute(0,2,1,3).flatten(2)


                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
        
            if self.etype_specified_attention:
                el = (feat_src.unsqueeze(-1) * self.attn_l).sum(dim=2).unsqueeze(2) #num_nodes*heads*dim*num_etype   1*heads*dim*1   
                er = (feat_dst.unsqueeze(-1) * self.attn_r).sum(dim=2).unsqueeze(2)
                graph.srcdata.update({'ft': feat_src, 'el': el})
                graph.dstdata.update({'er': er})
                graph.apply_edges(fn.u_add_v('el', 'er', 'e'))  #  num_edges*heads*1*num_etype
                e=self.leaky_relu((graph.edata.pop('e')*self.eindexer).sum(-1))
            else:
                e_feat = self.edge_emb(e_feat)
                e_feat = self.fc_e(e_feat).view(-1, self._num_heads, self._edge_feats)
                ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1)
                el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.srcdata.update({'ft': feat_src, 'el': el})
                graph.dstdata.update({'er': er})
                graph.edata.update({'ee': ee})
                graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
                e = self.leaky_relu(graph.edata.pop('e')+graph.edata.pop('ee'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            if res_attn is not None:
                graph.edata['a'] = graph.edata['a'] * (1-self.alpha) + res_attn * self.alpha
            # then message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
                             
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                if not self.res_n_type_mappings:
                    if self._in_dst_feats != self._out_feats:
                        resval =torch.bmm(h_src,self.res_fc).permute(1,0,2).view(                 #num_nodes*num_heads*(num_ntype*hidden_dim)
                        -1,self.num_ntype ,self._num_heads, self._out_feats).permute(0,2,1,3).flatten(2)
                        #resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                    else:
                        resval = self.res_fc(h_src).view(h_dst.shape[0], -1, self._out_feats*self.num_ntype)  #Identity
                else:
                    raise NotImplementedError()
                    res_new=[]
                    for type_count,idx in enumerate(node_idx_by_ntype):
                        res_new.append(self.res_fc[type_count](h_dst[idx,:]).view(
                        h_dst[idx,:].shape[0], -1, self._out_feats))
                    resval = torch.cat(res_new, 0)
                rst = rst + resval
            # bias
            if self.bias:
                if self.n_type_mappings:
                    rst_new=[]
                    for type_count,idx in enumerate(node_idx_by_ntype):
                        rst_new.append(        rst[idx]+ self.bias_param[type_count]    )
                    rst = torch.cat(rst_new, 0)
                else:

                    rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()



class changedGATConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """
    def __init__(self,
                 edge_feats,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 alpha=0.,
                 num_ntype=None,n_type_mappings=False,res_n_type_mappings=False,etype_specified_attention=False,eindexer=None):
        super(changedGATConv, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Embedding(num_etypes, edge_feats)
        self.n_type_mappings=n_type_mappings
        self.res_n_type_mappings=res_n_type_mappings
        self.etype_specified_attention=etype_specified_attention
        self.eindexer=eindexer
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
            raise Exception("!!!")
        else:
            if not n_type_mappings:
                self.fc = nn.Linear(
                    self._in_src_feats, out_feats * num_heads, bias=False)
            else:
                self.fc =nn.ModuleList([nn.Linear(
                    self._in_src_feats, out_feats * num_heads, bias=False)  for _ in range(num_ntype)] )
        self.fc_e = nn.Linear(edge_feats, edge_feats*num_heads, bias=False)
        if self.etype_specified_attention:
            self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats,num_etypes)))
            self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats,num_etypes)))
        else:
            self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
            self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
            self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                if not self.res_n_type_mappings:
                    self.res_fc = nn.Linear(
                        self._in_dst_feats, num_heads * out_feats, bias=False)
                else:
                    self.res_fc =nn.ModuleList([nn.Linear(
                        self._in_dst_feats, num_heads * out_feats, bias=False)  for _ in range(num_ntype)] )
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            if not self.n_type_mappings:
                self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
            else:
                self.bias_param=nn.ModuleList([ nn.Parameter(th.zeros((1, num_heads, out_feats)))  for _ in range(num_ntype) ])
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            if self.n_type_mappings:
                for m in self.fc:
                    nn.init.xavier_normal_(m.weight, gain=gain)
            else:
                nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            raise Exception("!!!")
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if not self.etype_specified_attention:
            nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            if self.res_n_type_mappings:
                for m in self.res_fc:
                    nn.init.xavier_normal_(m.weight, gain=gain)
            else:
                nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None):
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

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
                raise Exception("!!!")
            else:
                #feature transformation first
                h_src = h_dst = self.feat_drop(feat)

                if self.n_type_mappings:
                    h_new=[]
                    for type_count,idx in enumerate(node_idx_by_ntype):
                        h_new.append(self.fc[type_count](h_src[idx,:]).view(
                        -1, self._num_heads, self._out_feats))
                    feat_src = feat_dst = torch.cat(h_new, 0)
                else:
                    feat_src = feat_dst = self.fc(h_src).view(
                        -1, self._num_heads, self._out_feats)



                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            
            if self.etype_specified_attention:
                el = (feat_src.unsqueeze(-1) * self.attn_l).sum(dim=2).unsqueeze(2) #num_nodes*heads*dim*num_etype   1*heads*dim*1   
                er = (feat_dst.unsqueeze(-1) * self.attn_r).sum(dim=2).unsqueeze(2)
                graph.srcdata.update({'ft': feat_src, 'el': el})
                graph.dstdata.update({'er': er})
                graph.apply_edges(fn.u_add_v('el', 'er', 'e'))  #  num_edges*heads*1*num_etype
                e=self.leaky_relu((graph.edata.pop('e')*self.eindexer).sum(-1))



            else:
                e_feat = self.edge_emb(e_feat)
                e_feat = self.fc_e(e_feat).view(-1, self._num_heads, self._edge_feats)
                ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1)
                el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.srcdata.update({'ft': feat_src, 'el': el})
                graph.dstdata.update({'er': er})
                graph.edata.update({'ee': ee})
                graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
                e = self.leaky_relu(graph.edata.pop('e')+graph.edata.pop('ee'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            if res_attn is not None:
                graph.edata['a'] = graph.edata['a'] * (1-self.alpha) + res_attn * self.alpha
            # then message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
                             
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                if not self.res_n_type_mappings:
                    resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                else:
                    res_new=[]
                    for type_count,idx in enumerate(node_idx_by_ntype):
                        res_new.append(self.res_fc[type_count](h_dst[idx,:]).view(
                        h_dst[idx,:].shape[0], -1, self._out_feats))
                    resval = torch.cat(res_new, 0)
                rst = rst + resval
            # bias
            if self.bias:
                if self.n_type_mappings:
                    rst_new=[]
                    for type_count,idx in enumerate(node_idx_by_ntype):
                        rst_new.append(        rst[idx]+ self.bias_param[type_count]    )
                    rst = torch.cat(rst_new, 0)
                else:

                    rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()




class HCGNNConv(nn.Module):
    def __init__(self,in_dim,com_dim,dropout,bias,activation,num_etype,num_heads,negative_slope,allow_zero_in_degree=False):
        super(HCGNNConv, self).__init__()
        self.in_dim=in_dim
        self.com_dim=com_dim
        self.dropout=dropout
        self.attn_drop = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.bias=bias
        self.activation=activation
        self.num_etype=num_etype
        self.multi_linear=nn.Parameter(th.FloatTensor(size=(num_etype, in_dim, com_dim)))     #num_etype*D*D_0
        self._allow_zero_in_degree = allow_zero_in_degree
        self.attn_l=nn.Parameter(th.FloatTensor(size=(num_etype,1, num_heads, com_dim)))
        self.attn_r=nn.Parameter(th.FloatTensor(size=(num_etype,1, num_heads, com_dim)))
        assert com_dim % num_heads==0
        self.fc=nn.Parameter(th.FloatTensor(size=(com_dim*num_heads, com_dim)))
        self.reset_parameters()
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.multi_linear, gain=gain)
    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value
    def forward(self,graph,feat,com_signal):
        with graph.local_scope():
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
            
            
            #message=cat((com_signal, processed_feat),dim=1)
            etype_indexer=graph.edge_type_indexer.T.unsqueeze(-1).float()  #num_etype*E*1     #(0,0,...,1,....,0)   #每个位置对应一种edge type，1处代表这个edge是对应type
            etype_indexer=etype_indexer.unsqueeze(2)          #for multi-head

            node_etype_collector=graph.node_etype_collector.T.unsqueeze(-1)   #num_etype*N*1   #(0,1,...,1,....,0)  每个位置对应一种edge type，1处代表这个node有对应type的边
            feature_multi_modal=feat.unsqueeze(0).repeat(self.num_etype,1,1)*node_etype_collector  #num_etype*N*D
            processed_feats=th.bmm(feature_multi_modal,self.multi_linear)*node_etype_collector  #num_etype*N*D_0
            com_signal=com_signal.unsqueeze(0)   #1*N*D_0



            mess=processed_feats+com_signal  #num_etype*N*D_0
            #mess=mess@self.fc   #num_etype*N*(D_0/num_heads)
            mess=mess.unsqueeze(2) #num_etype*N*1*D_0

            #是跟s_i做还是跟x_i做attention的计算？
            el = (mess * self.attn_l).sum(dim=-1).unsqueeze(-1)#num_etype*N*num_head*1
            er = (mess * self.attn_r).sum(dim=-1).unsqueeze(-1) #num_etype*N*num_head*1
            #mess=th.bmm(decoder,processed_feats||com_signal)
            #construct attention 
            graph.srcdata.update({ 'el': th.permute(el,[1,2,3,0])})
            graph.dstdata.update({'er':th.permute(er,[1,2,3,0])})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e=self.attn_drop(edge_softmax(graph,self.leaky_relu(graph.edata.pop('e'))))
            e=e*th.permute(etype_indexer,[1,2,3,0])


            graph.edata.update({'e': e}) # E*  num_heads * 1 *num_etype
            graph.srcdata.update({"ft":th.permute(mess,[1,2,3,0])})
            graph.update_all(fn.u_mul_e('ft', 'e', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            rst=rst.sum(-1)  # N * num_heads * D_0
            rst=rst.sum(1) # N * (num_heads * D_0)
            #rst=rst@self.fc

            # bias
            #if self.bias:
                #rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst

# pylint: enable=W0235
class myGATConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """
    def __init__(self,
                 edge_feats,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 alpha=0.):
        super(myGATConv, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Embedding(num_etypes, edge_feats)
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
            raise Exception("!!!")
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.fc_e = nn.Linear(edge_feats, edge_feats*num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None):
        with graph.local_scope():
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

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
                raise Exception("!!!")
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            e_feat = self.edge_emb(e_feat)
            e_feat = self.fc_e(e_feat).view(-1, self._num_heads, self._edge_feats)
            ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1)
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.edata.update({'ee': ee})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e')+graph.edata.pop('ee'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            if res_attn is not None:
                graph.edata['a'] = graph.edata['a'] * (1-self.alpha) + res_attn * self.alpha
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
                             
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias:
                rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()
