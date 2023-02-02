import copy
import os
from turtle import distance
import warnings
import random
warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from eval_metrics import evaluate, evaluate_clothes, evaluate_locations, evaluate_scales


import torch
import torch.nn.functional as F
from torch.autograd.variable import Variable
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader

from dataloader import MEVID, MEVID_Video default_collate
from model import build_model
import pickle


class WeightEMA(object):
    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr
        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)


def compute_metric(feature, label, action, seq_type, probe_seqs, gallery_seqs):
    action_list = list(set(action))
    action_list.sort()
    action_num = len(action_list)
    num_rank = 1
    acc = np.zeros([len(probe_seqs), num_rank])
    for (p, probe_seq) in enumerate(probe_seqs):
        for gallery_seq in gallery_seqs:
            gseq_mask = np.isin(seq_type, gallery_seq)
            gallery_x = feature[gseq_mask, :]
            gallery_y = label[gseq_mask]
            pseq_mask = np.isin(seq_type, probe_seq)
            probe_x = feature[pseq_mask, :]
            probe_y = label[pseq_mask]
            dist = cuda_dist(probe_x, gallery_x)
            idx = dist.sort(1)[1].cpu().numpy()
            if len(gallery_x) == 0 or len(probe_x) == 0:
                acc[p, :] = np.zeros(num_rank)
            else:
                acc[p, :] = np.round(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0, 0) * 100 / dist.shape[0], 2)
    return acc[:, None, None, :]


# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    acc = acc.squeeze()
    result = np.mean(np.mean(acc - np.diag(np.diag(acc)), 1))
    diag_mean = np.mean(np.diag(np.diag(acc)))
    print(f'non_diag_mean: {result}', flush=True)
    print(f'diag_mean: {diag_mean}', flush=True)
    return result


def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    return dist


def train_epoch(epoch, data_loader, model, optimizer, ema_optimizer, criterion, writer, use_cuda, args, accumulation_steps=1):
    print('train at epoch {}'.format(epoch), flush=True)
    
    losses = []
    supervised_subject_losses = []
    supervised_distance_losses = []
    
    ground_truth_subject, ground_truth_distance = [], []
    predictions_subject, predictions_distance = [], []
    
    model.train()

    for i, (clips, target_subjects, target_distances, keys) in enumerate(tqdm(data_loader)):
        assert len(clips) == len(target_subjects)

        if use_cuda:
            clips = Variable(clips.type(torch.FloatTensor)).cuda()
            target_subjects = Variable(target_subjects.type(torch.LongTensor)).cuda()
            target_distances = Variable(target_distances.type(torch.LongTensor)).cuda()
        else:
            clips = Variable(clips.type(torch.FloatTensor))
            target_subjects = Variable(target_subjects.type(torch.LongTensor))
            target_distances = Variable(target_distances.type(torch.LongTensor))

        optimizer.zero_grad()
        
        output_subjects, output_distances, features = model(clips)
        
        subject_loss = criterion(output_subjects, target_subjects)
        distance_loss = criterion(output_distances, target_distances)

        output_subjects = torch.argmax(output_subjects, dim=1)
        output_distances = torch.argmax(output_distances, dim=1)

        predictions_distance.extend(output_distances.cpu().data.numpy())
        predictions_subject.extend(output_subjects.cpu().data.numpy())

        ground_truth_distance.extend(target_distances.cpu().data.numpy())
        ground_truth_subject.extend(target_subjects.cpu().data.numpy())
        
        loss = subject_loss + distance_loss
    
        losses.append(loss.item())
        supervised_distance_losses.append(distance_loss.item())
        supervised_subject_losses.append(subject_loss.item())

        loss = loss / accumulation_steps

        loss.backward()

        if (i+1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            ema_optimizer.step()

        losses.append(loss.item())

        del loss, output_distances, output_subjects, clips, target_distances, target_subjects, features, subject_loss, distance_loss

    subject_accuracy = accuracy_score(ground_truth_subject, predictions_subject)
    distance_accuracy = accuracy_score(ground_truth_distance, predictions_distance)

    print('Training Epoch: %d, Loss: %.4f, SL: %.4f, DL: %.4f' % (epoch, np.mean(losses), np.mean(supervised_subject_losses),  np.mean(supervised_distance_losses)), flush=True)
    print('Training Epoch: %d, Subject Accuracy: %.4f' % (epoch, subject_accuracy), flush=True)
    print('Training Epoch: %d, Distance Accuracy: %.4f' % (epoch, distance_accuracy), flush=True)
        
    writer.add_scalar('Training Loss', np.mean(losses), epoch)
    writer.add_scalar('Subject Loss', np.mean(supervised_subject_losses), epoch)
    writer.add_scalar('Distance Loss', np.mean(supervised_distance_losses), epoch)
      
    return model
      

def val_epoch(cfg, epoch, query_dataloader, gallery_dataloader, model, writer, use_cuda, args):
    print('validation at epoch {}'.format(epoch))
    model.eval()

    ground_truth_distance = []
    predictions_distance = []

    qf, q_pids, q_camids, q_oids = [], [], [], []
    for i, (clips, target_subjects, target_distances, keys) in enumerate(tqdm(query_dataloader)):
        assert len(clips) == len(target_subjects)

        with torch.no_grad():
            if use_cuda:
                clips = Variable(clips.type(torch.FloatTensor)).cuda()
                target_subjects = Variable(target_subjects.type(torch.LongTensor)).cuda()
                target_distances = Variable(target_distances.type(torch.LongTensor)).cuda()
            else:
                clips = Variable(clips.type(torch.FloatTensor))
                target_subjects = Variable(target_subjects.type(torch.LongTensor))
                target_distances = Variable(target_distances.type(torch.LongTensor))

            output_subjects, output_distances, features = model(clips)

            output_subjects = torch.argmax(output_subjects, dim=1)
            output_distances = torch.argmax(output_distances, dim=1)
            
            predictions_distance.extend(output_distances.cpu().data.numpy())
            ground_truth_distance.extend(target_distances.cpu().data.numpy())
            
            pid, camid, oid = keys[0][0:4], keys[0][4:8], keys[0][8:12]
            q_pids.append(pid)
            q_camids.append(camid)
            q_oids.append(oid)
            qf.append(features[0].cpu())

    distance_accuracy = accuracy_score(ground_truth_distance, predictions_distance)

    print('Validation Epoch: %d, Distance Accuracy - Probe: %.4f' % (epoch, distance_accuracy), flush=True)
    
    ground_truth_distance = []
    predictions_distance = []

    gf, g_pids, g_camids, g_oids = [], [], [], []
    for i, (clips, target_subjects, target_distances, keys) in enumerate(tqdm(gallery_dataloader)):
        assert len(clips) == len(target_subjects)

        with torch.no_grad():
            if use_cuda:
                clips = Variable(clips.type(torch.FloatTensor)).cuda()
                target_subjects = Variable(target_subjects.type(torch.LongTensor)).cuda()
                target_distances = Variable(target_distances.type(torch.LongTensor)).cuda()
            else:
                clips = Variable(clips.type(torch.FloatTensor))
                target_subjects = Variable(target_subjects.type(torch.LongTensor))
                target_distances = Variable(target_distances.type(torch.LongTensor))

            output_subjects, output_distances, features = model(clips)

            output_subjects = torch.argmax(output_subjects, dim=1)
            output_distances = torch.argmax(output_distances, dim=1)
            
            predictions_distance.extend(output_distances.cpu().data.numpy())
            ground_truth_distance.extend(target_distances.cpu().data.numpy())
            
            pid, camid, oid = keys[0][0:4], keys[0][4:8], keys[0][8:12]
            g_pids.append(pid)
            g_camids.append(camid)
            g_oids.append(oid)
            gf.append(features[0].cpu())

    distance_accuracy = accuracy_score(ground_truth_distance, predictions_distance)
    
    print('Validation Epoch: %d, Distance Accuracy - Gallery: %.4f' % (epoch, distance_accuracy), flush=True)
        
    qf = torch.stack(qf)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    q_oids = np.asarray(q_oids)
    print("Extracted features for query set, obtained {} matrix".format(qf.shape))

    gf = torch.stack(gf)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    g_oids = np.asarray(g_oids)
    print("Extracted features for gallery set, obtained {} matrix".format(gf.shape))

    print("Computing distance matrix")
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m,n))

    distmat = - torch.mm(qf, gf.t())
    distmat = distmat.data.cpu()
    distmat = distmat.numpy()

    ranks=[1, 5, 10, 20]

    np.savez('eval.npz', distmat=distmat, q_pids=q_pids, g_pids=g_pids, q_camids=q_camids, g_camids=g_camids, q_oids=q_oids, g_oids=g_oids)

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Overall Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    cmc, mAP = evaluate_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_oids, g_oids, mode='SC')
    print("Same Clothes Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    cmc, mAP = evaluate_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_oids, g_oids, mode='CC')
    print("Changing Clothes Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    cmc, mAP = evaluate_locations(cfg, distmat, q_pids, g_pids, q_camids, g_camids, mode='SL')
    print("Same Locations Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    cmc, mAP = evaluate_locations(cfg, distmat, q_pids, g_pids, q_camids, g_camids, mode='DL')
    print("Changing Locations Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    cmc, mAP = evaluate_scales(cfg, distmat, q_pids, g_pids, q_camids, g_camids, mode='SS')
    print("Same Scales Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    cmc, mAP = evaluate_scales(cfg, distmat, q_pids, g_pids, q_camids, g_camids, mode='DS')
    print("Changing Scales Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    return cmc[0]             
    

def train_model(cfg, run_id, save_dir, use_cuda, args, writer):
    shuffle = True
    print("Run ID : " + args.run_id)
   
    print("Parameters used : ")
    print("batch_size: " + str(args.batch_size))
    print("lr: " + str(args.learning_rate))
    
    tracks_train = np.loadtxt(cfg.tracks_train_file).astype(np.int) 
    tracks_test = np.loadtxt(cfg.tracks_test_file).astype(np.int) 
    query_ID = np.loadtxt(cfg.query_file).astype(np.int) 
    tracks_query = tracks_test[query_ID,:]
    gallery_ID = [i for i in range(tracks_test.shape[0]) if i not in query_ID]
    tracks_gallery = tracks_test[gallery_ID,:]

    train_data_gen = MEVID_Video(cfg, cfg.train_file, tracks_train, 'train', args.input_dim, args.batch_size) 
    query_data_gen = MEVID_Video(cfg, cfg.test_file, tracks_query, 'test', args.input_dim, args.batch_size)
    gallery_data_gen = MEVID_Video(cfg, cfg.test_file, tracks_gallery, 'test', args.input_dim, args.batch_size)
    
    train_dataloader = DataLoader(train_data_gen, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, drop_last=False, collate_fn=default_collate) 
    query_dataloader = DataLoader(query_data_gen, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False, collate_fn=default_collate)
    gallery_dataloader = DataLoader(gallery_data_gen, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False, collate_fn=default_collate)

    print("Number of training samples : " + str(len(train_data_gen)))
    print("Number of query samples : " + str(len(query_data_gen)))
    print("Number of gallery samples : " + str(len(gallery_data_gen)))
    
    steps_per_epoch = len(train_data_gen) / args.batch_size
    print("Steps per epoch: " + str(steps_per_epoch))

    model = build_model(args.model_version, args.input_dim, cfg.num_subjects, cfg.num_scales, 0, args.hidden_dim, args.num_heads, args.num_layers)
    
    #####################################################################################################################
    num_gpus = len(args.gpu.split(','))
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    
    if use_cuda:
       model.cuda()
    #####################################################################################################################
    
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        pretrained_weights = torch.load(args.checkpoint)['state_dict']
        model.load_state_dict(pretrained_weights, strict=True)
        print("loaded", flush=True)

    if args.optimizer == 'ADAM':
        print("Using ADAM optimizer")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'ADAMW':
        print("Using ADAMW optimizer")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        print("Using SGD optimizer")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
        
    ema_model = copy.deepcopy(model)
    ema_optimizer = WeightEMA(model, ema_model, args.learning_rate, alpha=args.ema_decay)
    
    criterion = CrossEntropyLoss()
    
    max_fmap_score, fmap_score = -1, -1
    # loop for each epoch
    for epoch in range(args.num_epochs):
        model = train_epoch(epoch, train_dataloader, model, optimizer, ema_optimizer, criterion, writer, use_cuda, args, accumulation_steps=args.steps)
        if epoch % args.validation_interval == 0:
            score1 = val_epoch(cfg, epoch, query_dataloader, gallery_dataloader, model, None, use_cuda, args)
            score2 = val_epoch(cfg, epoch, query_dataloader, gallery_dataloader, ema_model, writer, use_cuda, args)
            fmap_score = max(score1, score2)
         
        if fmap_score > max_fmap_score:
            for f in os.listdir(save_dir):
                os.remove(os.path.join(save_dir, f))
            save_file_path = os.path.join(save_dir, 'model_{}_{:.4f}.pth'.format(epoch, fmap_score))
            save_model = model if score1 > score2 else ema_model
            states = {
                'epoch': epoch + 1,
                'state_dict': save_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
            max_fmap_score = fmap_score
