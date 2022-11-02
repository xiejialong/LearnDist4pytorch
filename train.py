from utils.misc import *
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import MultiStepLR
import torch.cuda.amp as amp
from functools import partial
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from models import build_model
from datasets import build_dataset
from criterion import build_criterion
from tqdm import tqdm

class Trainer:
    def __init__(self, model, criterion, train_data: DataLoader, valid_data: DataLoader, gpu_id: int, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.valid_data = valid_data
        self.model = DDP(model, device_ids=[gpu_id])
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = amp.GradScaler()  # prevent zero garient(underflow)
        self.start_epoch = 0
        self.criterion = criterion

    def _save_checkpoint(self, epoch, args):
        if not os.path.exists(args.checkpoint_save_path):
            os.mkdir(args.checkpoint_save_path)
        PATH=  f"{args.checkpoint_save_path}/checkpoint_{epoch}.pt"
        torch.save(
                {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()
                }, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
    
    def _resume_checkpoint(self, ckp_path):
        if os.path.isfile(ckp_path):
            checkpoint = torch.load(ckp_path, map_location="cuda")
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            raise ValueError(
                "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
                .format(ckp_path))
    def _run_batch(self, sources, targets, args):
        # forward
        with amp.autocast(enabled=args.enable_amp):
            outputs = self.model(sources)
            loss, num_correct = self.criterion(outputs, targets)
        

        # backward
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        if args.max_norm:
            self.scaler.unscale_(self.optimizer)  # gradients must be unscaled before clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(). args.max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss, num_correct

    def _run_epoch(self, epoch, args):
        self.model.train()
        train_correct_num = 0
        train_total = 0
        train_loss = 0
        self.train_data.sampler.set_epoch(epoch)
        tqdm_bar = tqdm(self.train_data, desc="Train", ncols=100)
        for sources, targets in tqdm_bar:
            sources = sources.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss, correct_num_batch = self._run_batch(sources, targets, args)
            train_correct_num += correct_num_batch
            train_total += len(targets)
            train_loss += loss
        acc = train_correct_num / train_total
        loss = train_loss/np.ceil(train_total/args.batch_size)
        self.scheduler.step()
        torch.cuda.empty_cache()
        print(f"Epoch: {epoch}, Training Loss: {loss}, Traing Acc: {acc}, Learning rate: {self.scheduler.get_last_lr()}")
    
    @torch.no_grad()
    def _valuate_model(self, epoch, args):
        self.model.eval()
        train_correct_num = 0
        train_total = 0
        train_loss = 0
        tqdm_bar = tqdm(self.valid_data, desc="Valid", ncols=100)
        for sources, targets in tqdm_bar:
            sources = sources.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            outputs = self.model(sources)
            loss, correct_num_batch = self.criterion(outputs, targets)
            train_correct_num += correct_num_batch
            train_total += len(targets)
            train_loss += loss
        acc = train_correct_num / train_total
        loss = train_loss/np.ceil(train_total/args.batch_size)
        print(f"Epoch: {epoch}, Validation Loss: {loss}, Validation Acc: {acc}")

    def train(self, args):
        if args.resume:
            self._resume_checkpoint(args.resume)
        for epoch in range(self.start_epoch, args.max_epochs):
            self._run_epoch(epoch, args)
            if self.gpu_id == 0 and epoch%args.save_every==0:
                self._save_checkpoint(epoch, args)
                self._valuate_model(epoch, args)
                

def main_worker(gpu,  args):
    """args: rank, ngpus_per_node, gpu, world_size, sync_bn, base_lr, weight_decay, milestones, lr_decay, batch_size, batch_size_val, workers, workers_val, manual_seed"""
    # local rank & global rank
    args.gpu = gpu
    args.rank = args.rank * args.ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)
    print("initialize group!")
    dist.init_process_group(backend=args.backend, init_method=args.dist_url, rank=args.rank, world_size=args.world_size)
    dist.barrier()  # Synchronizes all processes.
    print("build model!")
    model, param_list = build_model(args) # 构建自己的模型
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model = DDP(model.to(gpu), device_ids=[gpu], find_unused_parameters=True)

    # build optimizer and lr scheduler
    optimizer = torch.optim.AdamW(params=param_list, lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_decay)
    print()

    # bulid datasets
    args.batch_size = int(args.batch_size / args.ngpus_per_node)
    args.batch_size_val = int(args.batch_size_val / args.ngpus_per_node)
    args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
    train_data = build_dataset(data_type="train", args=args)
    val_data = build_dataset(data_type="valid", args=args)

    # build dataloader
    init_fn = partial(worker_init_fn,
                      num_workers=args.workers,
                      rank=args.rank,
                      seed=args.manual_seed)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_data, 
                                                batch_size=args.batch_size, 
                                                shuffle=False, 
                                                num_workers=args.workers, 
                                                pin_memory=True, 
                                                worker_init_fn=init_fn,
                                                # collate_fn=collate_fn,   # map-type dataset
                                                sampler=train_sampler, 
                                                drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, 
                                            batch_size=args.batch_size_val, 
                                            shuffle=False, 
                                            num_workers=args.workers_val, 
                                            pin_memory=True, 
                                            sampler=val_sampler, 
                                            drop_last=False)
    criterion = build_criterion
    manager = Trainer(model, criterion, train_loader, val_loader, gpu, optimizer, scheduler)
    manager.train(args)
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="simple distributed training job")
    parser.add_argument('--model_name', type=str, default='res', help="vgg or resnet")
    parser.add_argument('--feature', type=int, default=18, help="layer feature of the model, vgg includes 11,13,16,19, res includes 18,34,50,101,152")
    parser.add_argument('--base_lr', type=float, default=0.001, help="learing rate of the backpropagation")
    parser.add_argument('--weight_decay', type=float, default=0., help="weight decay")
    parser.add_argument('--lr_decay', type=float, default=0.1, help="lr decay")
    parser.add_argument('--milestones', type=list, default=[5, 10, 13], help="milestones")
    parser.add_argument('--workers', type=int, default=8, help="weight decay")
    parser.add_argument('--workers_val', type=int, default=8, help="weight decay")
    parser.add_argument('--manual_seed', type=int, default=0, help="weight decay")
    parser.add_argument('--max_epochs', type=int, default=15, help='number of the epoch')
    parser.add_argument('--max_norm', type=float, default=0, help='number of the epoch')
    parser.add_argument('--classes', type=int, default=2, help='number of the classes')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=16, help='batch size')
    parser.add_argument('--rank', type=int, default=0, help='rank')
    parser.add_argument('--ngpus_per_node', type=int, default=2, help='number of gpu')
    parser.add_argument('--world_size', type=int, default=2, help='number of processes')
    parser.add_argument('--sync_bn', type=bool, default=False, help='sync_bn')
    parser.add_argument('--num_class', type=int, default=2, help='batch size')
    parser.add_argument('--resume', type=str, default=None, help='batch size')
    parser.add_argument('--num_devices', type=int, default=2, help='batch size')
    parser.add_argument('--backend', type=str, default="nccl", help='batch size')
    parser.add_argument('--dist_url', type=str, default="tcp://localhost:3681", help='batch size')
    parser.add_argument('--num_layer', type=int, default=18, help='batch size')
    parser.add_argument('--backbone', type=str, default="resnet", help='batch size')
    parser.add_argument('--dataset', type=str, default="dogsandcats", help='batch size')
    parser.add_argument('--enable_amp', type=bool, default=True, help='batch size')
    parser.add_argument('--save_every', type=int, default=2, help='batch size')
    parser.add_argument('--checkpoint_save_path', type=str, default="checkpoint", help='batch size')

    args = parser.parse_args()


    mp.spawn(main_worker, nprocs=args.num_devices, args=(args,))

if __name__=="__main__":
    main()
