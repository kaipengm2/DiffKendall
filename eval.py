import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
from Models.dataloader.samplers import CategoriesSampler
from Models.utils import *
from Models.models.DiffKendall import Kendall
import tqdm
from dataset import MiniImageNet, tieredImageNet, general_dataset, OxfordFlowers102Dataset, CategoriesSampler
from dataset import MiniImageNet, tieredImageNet, general_dataset, OxfordFlowers102Dataset
from torchvision import transforms

PRETRAIN_DIR=''
DATA_DIR=''

parser = argparse.ArgumentParser()
#about dataset
datasets = ['miniimagenet', 'tieredimagenet', 'CUB', 'Traffic_Signs','VGG_Flower','QuickDraw','Fungi']
parser.add_argument('-dataset_name', type=str, default='miniimagenet', choices=datasets)
parser.add_argument('-data_dir', type=str, default=DATA_DIR,help='dir of datasets')
# about model
parser.add_argument('-pretrain_dir', type=str, default=PRETRAIN_DIR)
backbones = ['resnet12', 'resnet18', 'WRN_28_10', 'conv-4', 'S2M2']
parser.add_argument('-backbone_name', type=str, default='resnet12', choices=backbones)
parser.add_argument('-metric', type=str, default='kendall', choices=['cosine','kendall'])
#about task
parser.add_argument('-way', type=int, default=5)
parser.add_argument('-shot', type=int, default=1)
parser.add_argument('-query', type=int, default=15,help='number of query image per class')
parser.add_argument('-test_episode', type=int, default=2000, help='number of testing episodes after training')
# OTHERS
parser.add_argument('-gpu', default='7')

args = parser.parse_args()
pprint(vars(args))
num_gpu = set_gpu(args)

# Dataset Setup
if args.dataset_name == 'miniimagenet':
    test_set = MiniImageNet('test', args)
elif args.dataset_name == 'tieredimagenet':
    test_set = tieredImageNet('test', args)
else:
    resize_sz = 92
    crop_sz = 84
    normalize = transforms.Normalize(mean=[0.4712, 0.4499, 0.4031],
                                        std=[0.2726, 0.2634, 0.2794])
    transform = transforms.Compose([
        transforms.Resize([resize_sz, resize_sz]),
        transforms.CenterCrop(crop_sz),
        transforms.ToTensor(),
        normalize])
    if args.dataset_name == 'VGG_Flower':
        test_set = OxfordFlowers102Dataset(args.data_dir, transform)
    else:
        test_set = general_dataset(args.data_dir, transform)
sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
loader = DataLoader(test_set, batch_sampler=sampler, pin_memory=True)

# Model Setup
model = Kendall(args)
model = load_model(args, model)
model = nn.DataParallel(model, list(range(num_gpu)))
model = model.cuda()
model.eval()

# Test Phase
ave_acc = Averager()
test_acc_record = np.zeros((args.test_episode,))
label = torch.arange(args.way).repeat(args.query)
if torch.cuda.is_available():
    label = label.type(torch.cuda.LongTensor)
else:
    label = label.type(torch.LongTensor)
tqdm_gen = tqdm.tqdm(loader)
with torch.no_grad():
    for i, batch in enumerate(tqdm_gen, 1):
        data, _ = [_.cuda() for _ in batch]
        k = args.way * args.shot
        model.module.mode = 'get_feat'
        data = model(data)
        data_shot, data_query = data[:k], data[k:]
        model.module.mode = args.metric
        score = model((data_shot, data_query))
        acc = count_acc(score, label)* 100
        ave_acc.add(acc)
        test_acc_record[i-1] = acc
        m, pm = compute_confidence_interval(test_acc_record[:i])
        tqdm_gen.set_description('batch {}: This episode:{:.2f}  average: {:.4f}+{:.4f}'.format(i, acc, m, pm))
m, pm = compute_confidence_interval(test_acc_record)
