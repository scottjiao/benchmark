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
import torch
import torch.nn.functional as F
import numpy as np
# pylint: disable=W0235
class slotGCNConv(nn.Module):
    
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=False,num_ntype=None,aggregator=None,slot_trans=None,ntype_indexer=None,semantic_trans="False",semantic_trans_normalize="row"):
        super(slotGCNConv, self).__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.num_ntype=num_ntype
        self.aggregator=aggregator
        self.slot_trans=slot_trans
        self.ntype_indexer=ntype_indexer
        if slot_trans=="one":
            assert self.aggregator in ["last_fc",None]
        
        self.semantic_transition_matrix=nn.Parameter(th.Tensor(self.num_ntype , self.num_ntype))
        self.semantic_trans=semantic_trans
        self.semantic_trans_normalize=semantic_trans_normalize

        if weight:
            if self.aggregator=="last_fc":
                self.weight = nn.Parameter(th.Tensor(self.num_ntype*in_feats, out_feats))
            else:
                self.weight = nn.Parameter(th.Tensor(self.num_ntype,in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            if self.aggregator=="last_fc":
                self.bias = nn.Parameter(th.Tensor(out_feats))
            else:
                self.bias = nn.Parameter(th.Tensor(self.num_ntype*out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.semantic_transition_matrix)
            

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None, edge_weight=None):
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
            aggregate_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ['left', 'both']:
                degs = graph.out_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight
            if self.semantic_trans=="True":
                if self.semantic_trans_normalize=="row":
                    dim_flag=1
                elif self.semantic_trans_normalize=="col":
                    dim_flag=0
                st_m= F.softmax( self.semantic_transition_matrix,dim=dim_flag ).unsqueeze(0)
                #st_m= F.softmax( torch.randn_like(self.semantic_transition_matrix),dim=dim_flag ).unsqueeze(0)# ruin exp!!! 
                
            elif self.semantic_trans=="False":
                st_m=torch.eye(self.num_ntype).to(self.semantic_transition_matrix.device)
            
            #st_m=torch.zeros_like()  # ruin exp!!! 
            #print(st_m)
            if self._in_feats > self._out_feats:
                if self.aggregator=="last_fc":
                    feat_src=torch.matmul(st_m, feat_src.view(-1,self.num_ntype,self._in_feats)).flatten(1)
                    feat_src = th.mm(feat_src, weight)
                elif self.slot_trans=="one":
                    ntype_indexer=self.ntype_indexer.permute(1,0)  #需要num_ntypes*num_nodes的0-1 one hot indexer
                    #feat_src=feat_src.view(-1,self.num_ntype,self._in_feats).permute(1,0,2)
                    feat_src=torch.matmul(st_m, feat_src.view(-1,self.num_ntype,self._in_feats)).permute(1,0,2)
                    if weight is not None:
                        feat_src = th.bmm(feat_src, weight)*ntype_indexer+feat_src*(1-ntype_indexer)
                    
                    feat_src=feat_src.permute(1,0,2).flatten(1)
                else:
                    ###reshape feat_src
                    #feat_src=feat_src.view(-1,self.num_ntype,self._in_feats).permute(1,0,2)
                    feat_src=torch.matmul(st_m, feat_src.view(-1,self.num_ntype,self._in_feats)).permute(1,0,2)
                    # mult W first to reduce the feature size for aggregation.
                    if weight is not None:
                        feat_src = th.bmm(feat_src, weight)
                    feat_src=feat_src.permute(1,0,2).flatten(1)
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if self.aggregator=="last_fc":

                    rst=torch.matmul(st_m,rst.view(-1,self.num_ntype,self._in_feats)).flatten(1)

                    rst = th.mm(rst, weight)
                elif self.slot_trans=="one":
                    ntype_indexer=self.ntype_indexer.permute(1,0).unsqueeze(-1)  #需要num_ntypes*num_nodes的0-1 one hot indexer
                    #rst=rst.view(-1,self.num_ntype,self._in_feats).permute(1,0,2)
                    rst=torch.matmul(st_m,rst.view(-1,self.num_ntype,self._in_feats)).permute(1,0,2)
                    
                    if weight is not None:
                        rst = th.bmm(rst, weight)*ntype_indexer+rst*(1-ntype_indexer)
                    rst=rst.permute(1,0,2).flatten(1)
                else:
                    #######reshape feat_src
                    rst=torch.matmul(st_m,rst.view(-1,self.num_ntype,self._in_feats)).permute(1,0,2)
                    if weight is not None:
                        rst = th.bmm(rst, weight)
                    rst=rst.permute(1,0,2).flatten(1)

            if self._norm in ['right', 'both']:
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

    def extra_repr(self):
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)


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
                 num_ntype=None,n_type_mappings=False,res_n_type_mappings=False,etype_specified_attention=False,eindexer=None,aggregator=None,inputhead=False,semantic_trans="False",semantic_trans_normalize="row",attention_average="False",attention_mse_sampling_factor=0,attention_mse_weight_factor=0,attention_1_type_bigger_constraint=0,attention_0_type_bigger_constraint=0):
        super(slotGATConv, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Embedding(num_etypes, edge_feats) if edge_feats else None
        self.n_type_mappings=n_type_mappings
        self.res_n_type_mappings=res_n_type_mappings
        self.etype_specified_attention=etype_specified_attention
        self.eindexer=eindexer
        self.aggregator=aggregator
        self.num_ntype=num_ntype 
        self.semantic_transition_matrix=nn.Parameter(th.Tensor(self.num_ntype , self.num_ntype))
        self.semantic_trans=semantic_trans
        self.semantic_trans_normalize=semantic_trans_normalize
        self.attentions=None
        self.attention_average=attention_average
        self.attention_mse_sampling_factor=attention_mse_sampling_factor
        self.attention_mse_weight_factor=attention_mse_weight_factor
        self.attention_1_type_bigger_constraint=attention_1_type_bigger_constraint
        self.attention_0_type_bigger_constraint=attention_0_type_bigger_constraint

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
                if self.aggregator=="last_fc":
                    self.fc = nn.Parameter(th.FloatTensor(size=(self._in_src_feats*self.num_ntype, out_feats * num_heads))) #num_heads=1 in last layer
                else:

                    self.fc = nn.Parameter(th.FloatTensor(size=(self.num_ntype, self._in_src_feats, out_feats * num_heads)))
            """else:
                self.fc =nn.ModuleList([nn.Linear(
                    self._in_src_feats, out_feats * num_heads, bias=False)  for _ in range(num_ntype)] )
                raise Exception("!!!")"""
        self.fc_e = nn.Linear(edge_feats, edge_feats*num_heads, bias=False) if edge_feats else None
        if self.etype_specified_attention:
            self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats,num_etypes)))
            self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats,num_etypes)))

        elif self.aggregator=="last_fc" or (self.aggregator and "by_slot" in self.aggregator) or self.aggregator=="slot_majority_voting":
            self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
            self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
            self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats))) if edge_feats else None
        else:
            
            self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats*self.num_ntype)))
            self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats*self.num_ntype)))
            self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats))) if edge_feats else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                if not self.res_n_type_mappings:
                    if self.aggregator=="last_fc":
                        self.res_fc =nn.Parameter(th.FloatTensor(size=( self._in_src_feats*self.num_ntype, out_feats * num_heads)))#num_heads=1 in last layer
                    else:
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
        if not self.etype_specified_attention and self._edge_feats:
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
        if self._edge_feats:
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
                raise Exception("!!!")
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                if self.semantic_trans=="True":
                    if self.semantic_trans_normalize=="row":
                        dim_flag=1
                    elif self.semantic_trans_normalize=="col":
                        dim_flag=0
                    st_m= F.softmax( self.semantic_transition_matrix,dim=dim_flag ).unsqueeze(0).unsqueeze(0)
                    #st_m= F.softmax( torch.randn_like(self.semantic_transition_matrix),dim=dim_flag ).unsqueeze(0).unsqueeze(0)# ruin exp!!! 
                    #st_m= torch.zeros_like(self.semantic_transition_matrix).unsqueeze(0).unsqueeze(0)# ruin exp!!! 
                elif self.semantic_trans=="False":
                    st_m=torch.eye(self.num_ntype).to(self.semantic_transition_matrix.device)
                #feature transformation first
                h_src = h_dst = self.feat_drop(feat)   #num_nodes*(num_ntype*input_dim)
                if self.n_type_mappings:
                    raise Exception("!!!")
                    h_new=[]
                    for type_count,idx in enumerate(node_idx_by_ntype):
                        h_new.append(self.fc[type_count](h_src[idx,:]).view(
                        -1, self._num_heads, self._out_feats))
                    feat_src = feat_dst = torch.cat(h_new, 0)
                elif self.aggregator=="last_fc":
                    h_src=h_src.view(-1,1,self.num_ntype,self._in_src_feats)
                    h_src=torch.matmul(st_m,h_src)
                    h_src=h_src.flatten(1)

                    feat_src = feat_dst = torch.mm(h_src,self.fc).view(-1,1,self._out_feats)


                else:
                    if self.inputhead:
                        h_src=h_src.view(-1,1,self.num_ntype,self._in_src_feats)
                        h_src=torch.matmul(st_m,h_src)
                    else:
                        h_src=h_src.view(-1,self._num_heads,self.num_ntype,int(self._in_src_feats/self._num_heads))
                        h_src=torch.matmul(st_m,h_src)
                    h_dst=h_src=h_src.permute(2,0,1,3).flatten(2)  #num_ntype*num_nodes*(in_feat_dim)
                    self.emb=h_dst.cpu().detach()
                    #self.fc with num_ntype*(in_feat_dim)*(out_feats * num_heads)
                    
                    feat_dst = torch.bmm(h_src,self.fc)  #num_ntype*num_nodes*(out_feats * num_heads)
                    
                    if self.aggregator and "by_slot" in self.aggregator:
                        target_slot=int(self.aggregator.split("_")[2])
                        feat_src = feat_dst =feat_dst[target_slot,:,:].unsqueeze(1)
                    elif  self.aggregator=="slot_majority_voting":
                        with torch.no_grad():
                            nnt_nn=torch.argmax(feat_dst,dim=-1).permute(1,0)   # num_nodes * num_ntypes
                            votings=torch.argmax(F.one_hot(torch.argmax(feat_dst,dim=-1).permute(1,0)).sum(1),dim=-1)  #num_nodes
                            votings_int=(nnt_nn==(votings.unsqueeze(1))).int().permute(1,0).unsqueeze(-1)   #num_ntypes* num_nodes *1
                        feat_src = feat_dst =(votings_int*feat_dst).sum(0).unsqueeze(1)
                    else:
                        feat_src = feat_dst =feat_dst.permute(1,0,2).view(                 #num_nodes*num_heads*(num_ntype*hidden_dim)
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
                e_feat = self.edge_emb(e_feat) if self._edge_feats else None
                e_feat = self.fc_e(e_feat).view(-1, self._num_heads, self._edge_feats)  if self._edge_feats else None
                ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1) if self._edge_feats else 0
                el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.srcdata.update({'ft': feat_src, 'el': el})
                graph.dstdata.update({'er': er})
                graph.edata.update({'ee': ee}) if self._edge_feats else None
                graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
                e_=graph.edata.pop('e')
                ee=graph.edata.pop('ee') if self._edge_feats else 0
                e=e_+ee
                
                e = self.leaky_relu(e)
            # compute softmax
            a=self.attn_drop(edge_softmax(graph, e))
            graph.apply_edges(fn.u_add_v('er', 'el', 'e_reverse'))
            e_reverse=graph.edata.pop('e_reverse')
            e_reverse=e_reverse+ee
            e_reverse = self.leaky_relu(e_reverse)
            a_reverse=self.attn_drop(edge_softmax(graph, e_reverse,norm_by='src'))
            if self.attention_average=="True":
                

                a[graph.etype_ids[1]]=(a[graph.etype_ids[1]]+a_reverse[graph.etype_ids[1]])/2
            if self.attention_mse_sampling_factor>0:
                #choosed_nodes_indices= torch.randperm(len(graph.etype_ids[1]))[:int(len(graph.etype_ids[1])*self.attention_mse_sampling_factor)]
                choosed_edges=np.random.choice(graph.etype_ids[1],int(len(graph.etype_ids[1])*self.attention_mse_sampling_factor), replace=False)
                #choosed_nodes=graph.etype_ids[1][choosed_nodes_indices]
                mse=torch.mean((a[choosed_edges]-a_reverse[choosed_edges])**2,dim=[0,1,2])
                self.mse=mse
                
            else:
                self.mse=torch.tensor(0)
            if self.attention_1_type_bigger_constraint>0:
                self.t1_bigger_mse=torch.mean((1-a[graph.etype_ids[1]])**2,dim=[0,1,2])
            else:
                self.t1_bigger_mse=torch.tensor(0)
            if self.attention_0_type_bigger_constraint>0:
                self.t0_bigger_mse=torch.mean((1-a[graph.etype_ids[0]])**2,dim=[0,1,2])
            else:
                self.t0_bigger_mse=torch.tensor(0)
            cor_vec=torch.stack([a[graph.etype_ids[1]].flatten(),a_reverse[graph.etype_ids[1]].flatten()],dim=0)
            self.attn_correlation=np.corrcoef(cor_vec.detach().cpu())[0,1]
            
            #print("a mean",[round(a[graph.etype_ids[i]].mean().item(),3) for i in range(7)],"a_reverse mean",[round(a_reverse[graph.etype_ids[i]].mean().item(),3) for i in range(7)])

            graph.edata['a'] = a
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
                        if self.aggregator=="last_fc":
                            resval =torch.mm(h_src,self.res_fc).view(-1,1,self._out_feats)
                        else:
                            resval =torch.bmm(h_src,self.res_fc)
                            if self.aggregator and "by_slot" in self.aggregator:
                                resval =resval[target_slot,:,:].unsqueeze(1)
                            elif self.aggregator=="slot_majority_voting":
                                resval =(resval*votings_int).sum(0).unsqueeze(1)
                            else:
                                resval =resval.permute(1,0,2).view(                 #num_nodes*num_heads*(num_ntype*hidden_dim)
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
            self.attentions=graph.edata.pop('a').detach()
            torch.cuda.empty_cache()
            return rst, self.attentions



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
            if self.n_type_mappings:
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
