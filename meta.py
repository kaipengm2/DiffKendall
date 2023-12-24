import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from Models.dataloader.samplers import CategoriesSampler
from Models.utils import *
from Models.models.DiffKendall import Kendall
from torch.utils.tensorboard import SummaryWriter
import tqdm
import time

DATA_DIR=''
PRETRAIN_DIR=''

parser = argparse.ArgumentParser()
#about dataset
parser.add_argument('-dataset', type=str, default='miniimagenet', choices=['miniimagenet', 'tieredimagenet'])
parser.add_argument('-data_dir', type=str, default=DATA_DIR,help='dir of datasets')
parser.add_argument('-set',type=str,default='val',choices=['val', 'test'],help='the set used for validation')
#about model
parser.add_argument('-pretrain_dir', type=str, default=PRETRAIN_DIR)
parser.add_argument('-backbone_name', type=str, default='resnet12')
#about training
parser.add_argument('-beta', type=int, default=1,help='the hyparameter of DiffKendall')
parser.add_argument('-bs', type=int, default=4,help='batch size of tasks')
parser.add_argument('-max_epoch', type=int, default=60)
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-step_size', type=int, default=10)
parser.add_argument('-gamma', type=float, default=0.5)
parser.add_argument('-val_frequency',type=int,default=50)
parser.add_argument('-random_val_task',action='store_true',help='random samples tasks for validation at each epoch')
parser.add_argument('-save_all',action='store_true',help='save models on each epoch')
#about task
parser.add_argument('-way', type=int, default=5)
parser.add_argument('-shot', type=int, default=1)
parser.add_argument('-query', type=int, default=15,help='number of query image per class')
parser.add_argument('-val_episode', type=int, default=200, help='number of validation episode')
# OTHERS
parser.add_argument('-gpu', default='7')

args = parser.parse_args()
pprint(vars(args))
num_gpu = set_gpu(args)
args.save_path = 'diffkendall/%s/%dshot-%dway/'%(args.dataset,args.shot,args.way)
args.save_path = osp.join('checkpoint',args.save_path)
ensure_path(args.save_path)
def save_model(name):
    torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))

# Model Setup
model = Kendall(args)
model = load_model(args, model)
model = nn.DataParallel(model, list(range(num_gpu)))
model = model.cuda()

# Dataset Setup
if args.dataset == 'miniimagenet':
    from dataset import MiniImageNet
    trainset = MiniImageNet('train', args)
    valset = MiniImageNet(args.set, args)
elif args.dataset == 'tieredimagenet':
    from dataset import tieredImageNet
    trainset = tieredImageNet('train', args)
    valset = tieredImageNet(args.set, args)
else:
    raise ValueError('Unknown dataset')
train_sampler = CategoriesSampler(trainset.label, args.val_frequency*args.bs, args.way, args.shot + args.query)
train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, pin_memory=True)
val_sampler = CategoriesSampler(valset.label, args.val_episode, args.way, args.shot + args.query)
val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, pin_memory=True)
args.random_val_task = False
if not args.random_val_task:
    print ('fix val set for all epochs')
    val_loader=[x for x in val_loader]
print('save all checkpoint models:', (args.save_all is True))

label = torch.arange(args.way, dtype=torch.int8).repeat(args.query)
label = label.type(torch.LongTensor)
label = label.cuda()
optimizer = torch.optim.SGD([{'params': model.parameters(),'lr':args.lr}], momentum=0.9, nesterov=True, weight_decay=0.0005)
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

trlog = {}
trlog['args'] = vars(args)
trlog['train_loss'] = []
trlog['val_loss'] = []
trlog['train_acc'] = []
trlog['val_acc'] = []
trlog['max_acc'] = 0.0
trlog['max_acc_epoch'] = 0
global_count = 0
writer = SummaryWriter(osp.join(args.save_path,'tf'))
result_list=[args.save_path]
for epoch in range(1, args.max_epoch + 1):
    print (args.save_path)
    start_time=time.time()
    tl = Averager()
    ta = Averager()
    tqdm_gen = tqdm.tqdm(train_loader)
    model.train()
    optimizer.zero_grad()
    for i, batch in enumerate(tqdm_gen, 1):
        global_count = global_count + 1
        data, _ = [_.cuda() for _ in batch]
        k = args.way * args.shot
        model.module.mode = 'get_feat'
        data = model(data)
        data_shot, data_query = data[:k], data[k:]
        model.module.mode = 'diffkendall'
        sig_score, k_score = model((data_shot, data_query))
        loss = F.cross_entropy(sig_score, label)
        acc = count_acc(k_score, label)
        writer.add_scalar('data/loss', float(loss), global_count)
        writer.add_scalar('data/acc', float(acc), global_count)
        total_loss = loss/args.bs
        writer.add_scalar('data/total_loss', float(total_loss), global_count)
        tqdm_gen.set_description('epo {}, total loss={:.4f} acc={:.4f}'
              .format(epoch, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)
        total_loss.backward()
        if i%args.bs==0:
            optimizer.step()
            optimizer.zero_grad()
    tl = tl.item()
    ta = ta.item()
    vl = Averager()
    va = Averager()

    #validation
    model.eval()
    with torch.no_grad():
        tqdm_gen = tqdm.tqdm(val_loader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, _ = [_.cuda() for _ in batch]
            k = args.way * args.shot
            model.module.mode = 'get_feat'
            data = model(data)
            data_shot, data_query = data[:k], data[k:]
            model.module.mode = 'kendall'
            k_score = model((data_shot, data_query))
            loss = F.cross_entropy(k_score, label)
            acc = count_acc(k_score, label)
            vl.add(loss.item())
            va.add(acc)
    vl = vl.item()
    va = va.item()
    writer.add_scalar('data/val_loss', float(vl), epoch)
    writer.add_scalar('data/val_acc', float(va), epoch)
    tqdm_gen.set_description('epo {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
    print ('val acc:%.4f'%va)
    if va >= trlog['max_acc']:
        print ('*********A better model is found*********')
        trlog['max_acc'] = va
        trlog['max_acc_epoch'] = epoch
        save_model('max_acc')
    trlog['train_loss'].append(tl)
    trlog['train_acc'].append(ta)
    trlog['val_loss'].append(vl)
    trlog['val_acc'].append(va)
    result_list.append('epoch:%03d,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f'%(epoch,tl,ta,vl,va))
    torch.save(trlog, osp.join(args.save_path, 'trlog'))
    if args.save_all:
        save_model('epoch-%d'%epoch)
        torch.save(optimizer.state_dict(), osp.join(args.save_path,'optimizer_latest.pth'))
    print('best epoch {}, best val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
    print ('This epoch takes %d seconds'%(time.time()-start_time),'\nstill need %.2f hour to finish'%((time.time()-start_time)*(args.max_epoch-epoch)/3600))
    #lr_scheduler.step()
writer.close()
save_list_to_txt(os.path.join(args.save_path,'results.txt'),result_list)
