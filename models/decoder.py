import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from dataset.dataset_utils import IMG_SIZE, BONE_LENGTH
from utils.utils import projection_batch, get_dense_color_path, get_graph_dict_path, get_upsample_path
from models.model_zoo import GCN_vert_convert, graph_upsample, graph_avg_pool
from models.model_attn import DualGraph
import torchvision
from torch.nn import functional as F
import collections.abc
from einops import rearrange
from timm.models.vision_transformer import Block
from functools import partial
from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=False, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class linear_block(nn.Module):
    def __init__(self, ch_in, ch_out, drop=0.1):
        super(linear_block, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(ch_in, ch_out),
            nn.GELU(),
            nn.Dropout(drop)
        )
    def forward(self, x):
        x = self.linear(x)
        return x
class uMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        self.linear512_256 = linear_block(512, 256, drop)
        self.linear256_256 = linear_block(256, 256, drop)
        self.linear256_512 = linear_block(256, 512, drop)

    def forward(self, x):

        x = self.linear512_256(x)
        res_256 = x

        x = self.linear256_256(x)
        x = x + res_256

        x = self.linear256_512(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key):
        B, Nq, C = query.shape
        _, Nk, _ = key.shape
        q = self.q(query).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B  head 21 c//head
        k = self.k(key).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(key).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Blockzidingyi(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm3 = norm_layer(dim)
        self.mlp = uMLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, q, k):
        q = q + self.attn(self.norm1(q), self.norm2(k))
        q = q + self.mlp(self.norm3(q))
        return q


class CrossTransformer(nn.Module):
    def __init__(self, in_chans1=512, in_chans2=512, depth=4, num_heads=4, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, 64, in_chans1))
        self.pos_embed2 = nn.Parameter(torch.randn(1, 1 + 2 * 64, in_chans2))
        self.blocks = nn.ModuleList([
            Blockzidingyi(in_chans1, num_heads, mlp_ratio, qkv_bias=False, norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, q, k):
        q = q + self.pos_embed
        k = k + self.pos_embed2
        for blk in self.blocks:
            q = blk(q, k)
        return q


def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))

        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_conv1d_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv1d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))

        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_deconv_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.ConvTranspose2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False))


        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)
class Transformer(nn.Module):
    def __init__(self, in_chans=512, joint_num=21, depth=4, num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, joint_num, in_chans))
        self.blocks = nn.ModuleList([
            Blockzidingyi(in_chans, num_heads, mlp_ratio, qkv_bias=False, norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return x


class IAM(nn.Module):
    def __init__(self):
        super(IAM, self).__init__()
        self.FC = nn.Linear(512 * 2, 512)
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + (2 * 8 * 8), 512))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.SA_T = nn.ModuleList([
            Block(512, 4, 4.0, qkv_bias=False, norm_layer=nn.LayerNorm)
            for i in range(4)])
        self.FC2 = nn.Linear(512, 512)

        self.CA_T = CrossTransformer()
        self.FC3 = nn.Linear(512, 512)

    def forward(self, feat1, feat2):
        B, C, H, W = feat1.shape
        feat1 = rearrange(feat1, 'B C H W -> B (H W) C')
        feat2 = rearrange(feat2, 'B C H W -> B (H W) C')

        token_j = self.FC(torch.cat((feat1, feat2), dim=-1))


        token_s = torch.cat((feat1, feat2), dim=1) + self.pos_embed[:, 1:]
        cls_token = (self.cls_token + self.pos_embed[:, :1]).expand(B, -1, -1)
        token_s = torch.cat((cls_token, token_s), dim=1)
        for blk in self.SA_T:
            token_s = blk(token_s)
        token_s = self.FC2(token_s)

        output = self.CA_T(token_j, token_s)
        output = self.FC3(output)
        output = rearrange(output, 'B (H W) C -> B C H W', H=H, W=W)
        return output


class FIAM(nn.Module):
    def __init__(self):
        super(FIAM, self).__init__()
        self.conv_l = make_conv_layers([2048, 512], kernel=1, stride=1, padding=0)
        self.conv_r = make_conv_layers([2048, 512], kernel=1, stride=1, padding=0)
        self.Extract = IAM()
        self.Adapt_r = IAM()
        self.Adapt_l = IAM()
        self.conv_l2 = make_conv_layers([512 * 2, 512 * 2], kernel=1, stride=1, padding=0)
        self.conv_r2 = make_conv_layers([512 * 2, 512 * 2], kernel=1, stride=1, padding=0)

    def forward(self, x):
        rhand_feat = self.conv_r(x)
        lhand_feat = self.conv_l(x)
        inter_feat = self.Extract(rhand_feat, lhand_feat)
        rinter_feat = self.Adapt_r(rhand_feat, inter_feat)
        linter_feat = self.Adapt_l(lhand_feat, inter_feat)
        rhand_feat = self.conv_r2(torch.cat((rhand_feat, rinter_feat), dim=1))
        lhand_feat = self.conv_l2(torch.cat((lhand_feat, linter_feat), dim=1))
        return rhand_feat, lhand_feat

def weights_init(layer):
    classname = layer.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0.0)


class decoder(nn.Module):
    def __init__(self,
                 global_feature_dim=2048,
                 f_in_Dim=[256, 256, 256, 256],
                 f_out_Dim=[128, 64, 32],
                 gcn_in_dim=[256, 128, 128],
                 gcn_out_dim=[128, 128, 64],
                 graph_k=2,
                 graph_layer_num=4,
                 left_graph_dict={},
                 right_graph_dict={},
                 vertex_num=778,
                 dense_coor=None,
                 num_attn_heads=4,
                 upsample_weight=None,
                 dropout=0.05):
        super(decoder, self).__init__()
        assert len(f_in_Dim) == 4
        f_in_Dim = f_in_Dim[:-1]
        assert len(gcn_in_dim) == 3
        for i in range(len(gcn_out_dim) - 1):
            assert gcn_out_dim[i] == gcn_in_dim[i + 1]

        graph_dict = {'left': left_graph_dict, 'right': right_graph_dict}
        graph_dict['left']['coarsen_graphs_L'].reverse()
        graph_dict['right']['coarsen_graphs_L'].reverse()
        graph_L = {}
        for hand_type in ['left', 'right']:
            graph_L[hand_type] = graph_dict[hand_type]['coarsen_graphs_L']

        self.vNum_in = graph_L['left'][0].shape[0]
        self.vNum_out = graph_L['left'][2].shape[0]
        self.vNum_all = graph_L['left'][-1].shape[0]
        self.vNum_mano = vertex_num
        self.gf_dim = global_feature_dim
        self.gcn_in_dim = gcn_in_dim
        self.gcn_out_dim = gcn_out_dim

        self.FIAM = FIAM()

        if dense_coor is not None:
            dense_coor = torch.from_numpy(dense_coor).float()
            self.register_buffer('dense_coor', dense_coor)

        self.converter = {}
        for hand_type in ['left', 'right']:
            self.converter[hand_type] = GCN_vert_convert(vertex_num=self.vNum_mano,
                                                         graph_perm_reverse=graph_dict[hand_type]['graph_perm_reverse'],
                                                         graph_perm=graph_dict[hand_type]['graph_perm'])

        self.dual_gcn = DualGraph(verts_in_dim=self.gcn_in_dim,
                                  verts_out_dim=self.gcn_out_dim,
                                  graph_L_Left=graph_L['left'][:3],
                                  graph_L_Right=graph_L['right'][:3],
                                  graph_k=[graph_k, graph_k, graph_k],
                                  graph_layer_num=[graph_layer_num, graph_layer_num, graph_layer_num],
                                  img_size=[8, 16, 32],
                                  img_f_dim=f_in_Dim,
                                  grid_size=[8, 8, 8],
                                  grid_f_dim=f_out_Dim,
                                  n_heads=num_attn_heads,
                                  dropout=dropout)


        self.gf_layer_left = nn.Sequential(*(nn.Linear(1024, self.gcn_in_dim[0] - 3),
                                             nn.LayerNorm(self.gcn_in_dim[0] - 3, eps=1e-6)))
        self.gf_layer_right = nn.Sequential(*(nn.Linear(1024, self.gcn_in_dim[0] - 3),
                                              nn.LayerNorm(self.gcn_in_dim[0] - 3, eps=1e-6)))
                                              
        self.unsample_layer = nn.Linear(self.vNum_out, self.vNum_mano, bias=False)
        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
        )

        self.coord_head = nn.Linear(self.gcn_out_dim[-1], 3)
        self.avg_head = nn.Linear(self.vNum_out, 1)
        self.params_head = nn.Linear(self.gcn_out_dim[-1], 3)

        weights_init(self.gf_layer_left)
        weights_init(self.gf_layer_right)
        weights_init(self.coord_head)
        weights_init(self.avg_head)
        weights_init(self.params_head)

        if upsample_weight is not None:
            state = {'weight': upsample_weight.to(self.unsample_layer.weight.data.device)}
            self.unsample_layer.load_state_dict(state)
        else:
            weights_init(self.unsample_layer)

    def get_upsample_weight(self):
        return self.unsample_layer.weight.data

    def get_converter(self):
        return self.converter

    def get_hand_pe(self, bs, num=None):
        if num is None:
            num = self.vNum_in
        dense_coor = self.dense_coor.repeat(bs, 1, 1) * 2 - 1
        pel = self.converter['left'].vert_to_GCN(dense_coor)
        pel = graph_avg_pool(pel, p=pel.shape[1] // num)
        per = self.converter['right'].vert_to_GCN(dense_coor)
        per = graph_avg_pool(per, p=per.shape[1] // num)
        return pel, per




    def forward(self, x, fmaps):
        assert x.shape[1] == self.gf_dim
        fmaps = fmaps[:-1]
        bs = x.shape[0]

        rhand_feat, lhand_feat = self.FIAM(x)
        rhand_feat1 = self.output_layer(rhand_feat)
        lhand_feat1 = self.output_layer(lhand_feat)

        pel, per = self.get_hand_pe(bs, num=self.vNum_in)
        LF = self.gf_layer_left(lhand_feat1).unsqueeze(1).repeat(1,self.vNum_in,1)
        RF = self.gf_layer_left(rhand_feat1).unsqueeze(1).repeat(1,self.vNum_in,1)

        Lf = torch.cat([LF, pel], dim=-1)
        Rf = torch.cat([RF, per], dim=-1)

        Lf, Rf = self.dual_gcn(Lf, Rf, fmaps)

        scale = {}
        trans2d = {}
        temp = self.avg_head(Lf.transpose(-1, -2))[..., 0]
        temp = self.params_head(temp)
        scale['left'] = temp[:, 0]
        trans2d['left'] = temp[:, 1:]
        temp = self.avg_head(Rf.transpose(-1, -2))[..., 0]
        temp = self.params_head(temp)
        scale['right'] = temp[:, 0]
        trans2d['right'] = temp[:, 1:]

        handDictList = []

        paramsDict = {'scale': scale, 'trans2d': trans2d}
        verts3d = {'left': self.coord_head(Lf), 'right': self.coord_head(Rf)}
        verts2d = {}
        result = {'verts3d': {}, 'verts2d': {}}
        for hand_type in ['left', 'right']:
            verts2d[hand_type] = projection_batch(scale[hand_type], trans2d[hand_type], verts3d[hand_type], img_size=IMG_SIZE)
            result['verts3d'][hand_type] = self.unsample_layer(verts3d[hand_type].transpose(1, 2)).transpose(1, 2)
            result['verts2d'][hand_type] = projection_batch(scale[hand_type], trans2d[hand_type], result['verts3d'][hand_type], img_size=IMG_SIZE)
        handDictList.append({'verts3d': verts3d, 'verts2d': verts2d})

        otherInfo = {}
        otherInfo['verts3d_MANO_list'] = {'left': [], 'right': []}
        otherInfo['verts2d_MANO_list'] = {'left': [], 'right': []}
        for i in range(len(handDictList)):
            for hand_type in ['left', 'right']:
                v = handDictList[i]['verts3d'][hand_type]
                v = graph_upsample(v, p=self.vNum_all // v.shape[1])
                otherInfo['verts3d_MANO_list'][hand_type].append(self.converter[hand_type].GCN_to_vert(v))
                v = handDictList[i]['verts2d'][hand_type]
                v = graph_upsample(v, p=self.vNum_all // v.shape[1])
                otherInfo['verts2d_MANO_list'][hand_type].append(self.converter[hand_type].GCN_to_vert(v))

        return result, paramsDict, handDictList, otherInfo


def load_decoder(cfg, encoder_info):
    graph_path = get_graph_dict_path()
    with open(graph_path['left'], 'rb') as file:
        left_graph_dict = pickle.load(file)
    with open(graph_path['right'], 'rb') as file:
        right_graph_dict = pickle.load(file)

    dense_path = get_dense_color_path()
    with open(dense_path, 'rb') as file:
        dense_coor = pickle.load(file)

    upsample_path = get_upsample_path()
    with open(upsample_path, 'rb') as file:
        upsample_weight = pickle.load(file)
    upsample_weight = torch.from_numpy(upsample_weight).float()

    model = decoder(
        global_feature_dim=encoder_info['global_feature_dim'],
        f_in_Dim=encoder_info['fmaps_dim'],
        f_out_Dim=cfg.MODEL.IMG_DIMS,
        gcn_in_dim=cfg.MODEL.GCN_IN_DIM,
        gcn_out_dim=cfg.MODEL.GCN_OUT_DIM,
        graph_k=cfg.MODEL.graph_k,
        graph_layer_num=cfg.MODEL.graph_layer_num,
        vertex_num=778,
        dense_coor=dense_coor,
        left_graph_dict=left_graph_dict,
        right_graph_dict=right_graph_dict,
        num_attn_heads=4,
        upsample_weight=upsample_weight,
        dropout=cfg.TRAIN.dropout
    )

    return model
