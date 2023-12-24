import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from itertools import combinations

def kendall_ranking_correlation(support, query):
    '''Kendall's rank correlation.
    '''
    c = support.shape[-1]
    pair_num = int(c * (c - 1) // 2)
    c_pair = random.sample(sorted(combinations(list(range(c)), 2)), pair_num)
    support_prank = support[:, c_pair].diff().sign().squeeze()
    query_prank = query[:, c_pair].diff().sign().squeeze()
    score = torch.mm(query_prank, support_prank.T)
    return score / pair_num

def sampled_kendall(support, query, iter_num = 5):
    '''A linear time complexity alternative for Kendall's rank correlation.
    '''
    c = support.shape[-1]
    pair_num = (c - 1) * (iter_num + 1)
    support_prank = support.diff().sign().squeeze(-1)
    query_prank = query.diff().sign().squeeze(-1)
    for i in range(iter_num):
        p1 = torch.randperm(c)
        s_prank = support[:, p1].diff().sign().squeeze(-1)
        q_prank = query[:, p1].diff().sign().squeeze(-1)
        support_prank = torch.cat([support_prank, s_prank], dim = -1)
        query_prank = torch.cat([query_prank, q_prank], dim = -1)
    score = torch.mm(query_prank, support_prank.T)
    return score / pair_num

def diffkendall(support, query, beta = 1, T = 0.0125):
    '''The differentiable Kendall's rank correlation.
    '''
    c = support.shape[-1]
    pair_num = int(c * (c - 1) // 2)
    c_pair = random.sample(sorted(combinations(list(range(c)), 2)), pair_num)
    support_prank = support[:, c_pair].diff().squeeze()
    query_prank = query[:, c_pair].diff().squeeze(-1)
    score = support_prank.repeat([query.shape[0], 1, 1]) * query_prank.unsqueeze(1).repeat([1, support.shape[0], 1])
    score = 1 / (1 + (-score * beta).exp())
    score = 2 * score - 1
    score = score.mean(dim=-1)
    score = score / T
    return score

class Kendall(nn.Module):
    def __init__(self, args, mode='kendall', config = None):
        super().__init__()
        self.mode = mode
        if self.mode == 'ce_pre_train':
            self.fc = nn.Linear(64, args.num_class)
        self.args = args
        if self.args.backbone_name == 'resnet12':
            from .resnet import ResNet
            self.encoder = ResNet(args=args)
        elif self.args.backbone_name == 'resnet18':
            from .res18 import ResNet
            self.encoder = ResNet()
        elif self.args.backbone_name == 'WRN_28_10':
            from .wrn28 import Wide_ResNet
            self.encoder = Wide_ResNet(28, 10, 0.5)
        elif self.args.backbone_name == 'conv-4':
            from .convnet import ConvNet
            self.encoder = ConvNet()
        elif self.args.backbone_name == 'S2M2':
            from .S2M2_WRN28 import WideResNet28_10
            self.encoder = WideResNet28_10()
    def forward(self, input):
        if self.mode == 'get_feat':
            return self.encoder(input)
        elif self.mode == 'ce_pre_train':
            x = F.adaptive_avg_pool2d(self.encoder(input), [1, 1]).squeeze()
            return self.fc(x)
        elif self.mode == 'cosine':
            support, query = input
            support = F.adaptive_avg_pool2d(support, [1, 1]).squeeze()
            query = F.adaptive_avg_pool2d(query, [1, 1]).squeeze()
            c_num = support.shape[-1]
            support = support.view(self.args.shot, -1, c_num).mean(dim=0)
            c_logits = torch.mm(F.normalize(query, dim=-1), F.normalize(support, dim=-1).T)
            return c_logits
        elif self.mode == 'kendall':
            support, query = input
            support = F.adaptive_avg_pool2d(support, [1, 1]).squeeze()
            query = F.adaptive_avg_pool2d(query, [1, 1]).squeeze()
            c_num = support.shape[-1]
            support = support.view(self.args.shot, -1, c_num).mean(dim=0)
            score = kendall_ranking_correlation(support, query)
            return score
        elif self.mode == 'diffkendall':
            support, query = input
            support = F.adaptive_avg_pool2d(support, [1, 1]).squeeze()
            query = F.adaptive_avg_pool2d(query, [1, 1]).squeeze()
            c_num = support.shape[1]
            support = support.view(self.args.shot, -1, c_num).mean(dim=0)
            sig_logits = diffkendall(support, query, beta=self.args.beta, T=0.0125,pair_num='full')
            k_logits = kendall_ranking_correlation(support, query)
            return sig_logits, k_logits
        else:
            raise ValueError('Unknown mode')






