from importlib import import_module
import os
from configuration import build_config
import torch
from torch.autograd.variable import Variable
from torch.utils.data import Dataset, DataLoader
import cv2
import random
import time
from tqdm import tqdm
import numpy as np
import cv2


def default_collate(batch):
    frames, subject_ids, labels, fnames = [], [], [], []
    for item in batch:
        if item[0] is not None and item[1] is not None and item[2] is not None and item[3] is not None:
            frames.append(item[0])
            subject_ids.append(item[1])
            labels.append(item[2])
            fnames.append(item[3])     
    frames = torch.stack(frames)
    subject_ids = torch.tensor(subject_ids)
    labels = torch.tensor(labels)
    return frames, subject_ids, labels, fnames


class MEVID(Dataset):
    def __init__(self, cfg, file, tracks, data_split, frame_size, shuffle=True):
        self.data_split = data_split
        self.resolutions = cfg.resolutions
        self.file_list = self._get_names(file)
        self.tracks = tracks
        self.frames_folder = os.path.join(cfg.frames_folder, 'bbox_' + self.data_split)
        self.subjects = sorted(os.listdir(self.frames_folder))
        self.frame_size = frame_size
        self.skip = 0 #Skip for skip frames Praveen
        if shuffle:
            random.shuffle(self.tracks)

    def __len__(self):
        return len(self.tracks)

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def build_label(self, width, height):
        pix_count = width * height
        indices = np.where(pix_count > np.array(self.resolutions))[0]
        if len(indices) == 0:
            return -1
        else:
            return indices[-1]
            

    def __getitem__(self, index):
        track = self.tracks[index]
        start_index, end_index, pid, oid, camid = track
        if end_index - start_index < 0:
            return None, None, None, None
        if start_index == end_index:
            fname = self.file_list[start_index]
        else:
            assert end_index - start_index > 0, track
            frames = self.file_list[start_index:end_index]
            assert len(frames) > 0, track
            if self.data_split == 'train':
                fname = random.choice(frames)
            else:
                fname = frames[len(frames)//2]

        fname = fname.replace('.jpg', '')
        subject_id, outfit_id, camera_id, track_id, frame_id = fname[0:4], fname[4:8], fname[8:12], fname[12:16], fname[16:22]
        f = os.path.join(self.frames_folder, subject_id, fname + '.jpg')
        assert os.path.exists(f), f
        frame = cv2.imread(f)[:, :, ::-1]
        label = self.build_label(frame.shape[0], frame.shape[1])
        if label < 0:
            return None, None, None, None
        frame = cv2.resize(frame, (self.frame_size, self.frame_size))
        frame = frame/255.0
        frame = torch.from_numpy(np.array(frame)).permute(2, 0, 1)
        return frame, self.subjects.index(subject_id), label, fname





class MEVID_Video(Dataset):
    def __init__(self, cfg, file, tracks, data_split, frame_size, batch_size, shuffle=True):
        self.data_split = data_split
        self.resolutions = cfg.resolutions
        self.file_list = self._get_names(file)
        self.tracks = tracks
        self.scale = []
        self.frames_folder = os.path.join(cfg.frames_folder, 'bbox_' + self.data_split)
        self.subjects = sorted(os.listdir(self.frames_folder))
        self.frame_size = frame_size
        self.sub_scale = {}
        #print(len(open(cfg.train_scales, 'r').readlines()))
        for i, track in enumerate(self.tracks):
            _, _, pid, _, _ = track
            if pid not in self.sub_scale:
                self.sub_scale[pid] = {} 
            
            scale = float(open(cfg.train_scales, 'r').readlines()[i])
            scale = build_scale(scale)
            if scale == -1:
                continue
            if scale not in self.sub_scale[pid].keys():
                self.sub_scale[pid][scale] = []
            self.sub_scale[pid][scale].append(track)
            #print(self.sub_scale[pid])  
                           
        if shuffle:
            random.shuffle(self.tracks)
            
    def __len__(self):
        return len(self.tracks)

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def build_label(self, width, height):
        pix_count = width * height
        indices = np.where(pix_count > np.array(self.resolutions))[0]
        if len(indices) == 0:
            return -1
        else:
            return indices[-1]


    def __getitem__(self, index):
        farr = []
        track = self.tracks[index]
        start_index, end_index, pid, oid, camid = track
        if end_index - start_index < 16:
            #print('tracklet too small', flush=True)
            #print(start_index, end_index, track)
            return None, None, None, None
        else:
            assert end_index - start_index > 0, track
            frames = self.file_list[start_index:end_index]
            length = len(frames)
            assert len(frames) > 0, track
            #if self.data_split == 'train':
            fname = random.choice(frames)
            #else:
            #    fname = frames[len(frames)//2]
        #print(frames, length, flush=True)
        fname = fname.replace('.jpg', '')
        subject_id, outfit_id, camera_id, track_id, frame_id = fname[0:4], fname[4:8], fname[8:12], fname[12:16], fname[16:22]
        print(pid, subject_id)
        start = random.randrange(0, length-(16 + (16 * self.skip)))
        end = start + 16 + (16 * self.skip)
        frame_indexer = range(start, end)[::(self.skip + 1)]
        for i, frame_number in enumerate(frame_indexer):
            f = os.path.join(self.frames_folder, subject_id, fname[:17] + str(frame_number).zfill(5) + '.jpg')
            assert os.path.exists(f), f
            frame = cv2.imread(f)[:, :, ::-1]
            if i == 0:
                label = self.build_label(frame.shape[0], frame.shape[1])
                scale = build_scale(frame.shape[1])
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            frame = frame/255.0
            frame = torch.from_numpy(np.array(frame)).permute(2, 0, 1)
            farr.append(frame)
        if label < 0:
            return None, None, None, None
        frames = torch.stack([frame2 for frame2 in farr])
        
        diff_scale = random.choice([x for x in range(6) if x != scale])
        diff_sub = int(random.choice([sub for sub in self.subjects if sub != str(pid).zfill(4)]))
        print(pid, diff_sub, scale, diff_scale)
        ssub = random.choice(self.sub_scale[pid][diff_scale]) ##CHECK SCALE LABEL
        sscale = random.choice(self.sub_scale[diff_sub][label])
        exit()
        #torch.save(frames, 'mevid_vid.pt')
        return frames, self.subjects.index(subject_id), label, fname


def build_scale(scale):
    if 32 <= scale < 48:
        return 0
    elif 48 <= scale < 64:
        return 1
    elif 64 <= scale < 96:
        return 2
    elif 96 <= scale < 128:
        return 3
    elif 128 <= scale < 192:
        return 4
    elif 192 <= scale:
        return 5
    else:
        print(f'Scale out of bounds, {scale}')
        return -1


if __name__ == '__main__':
    cfg = build_config('MEVID')
    shuffle = True
    tracks_train = np.loadtxt(cfg.tracks_train_file).astype(np.int32) 
    
    dataset = MEVID_Video(cfg, cfg.train_file, tracks_train, 'train', 224)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=shuffle, num_workers=8, collate_fn=default_collate)
    start = time.time()
    counts = []
    for i, (frame, subject_id, class_label, fname) in enumerate(tqdm(dataloader)):
        print(frame.shape)
        print(subject_id)
        print(class_label)
        #exit()
        
    
