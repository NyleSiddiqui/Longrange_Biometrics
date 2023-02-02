import random
import cv2
import os
import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt



def custom_track_scales():
    train_track = open("/home/c3-0/datasets/MEVID/mevid-v1-annotation-data/track_train_info.txt", 'r')
    names = []
    frame_names = []
    resos = []
    df_rows = []
    with open("/home/c3-0/datasets/MEVID/mevid-v1-annotation-data/train_name.txt", 'r') as f:
        for line in f:
            new_line = line.rstrip()
            names.append(new_line)
        
    for i, row in enumerate(train_track.readlines(), 1):
        start_index, end_index, pid, oid, camid = [int(float(value)) for value in row.split(' ')]
        
        if end_index - start_index < 16:
            print(f'{i} too small tracklet')
            continue
        
        frames = names[start_index:end_index]
        #print(frames)
        #print(len(frames))
        num_windows = len(frames) // 16
        #print(num_windows)
        assert len(frames) > 0, track
        
        count = 1
        for fname in frames:
            fname = fname.replace('.jpg', '')
            subject_id, outfit_id, camera_id, track_id, frame_id = fname[0:4], fname[4:8], fname[8:12], fname[12:16], fname[16:22]
            f = os.path.join('/home/c3-0/datasets/MEVID/bbox_train', subject_id, fname + '.jpg')
            assert os.path.exists(f), f
            frame = cv2.imread(f)#[:, :, ::-1]
            reso = frame.shape[0] * frame.shape[1]
            resos.append(reso)
            frame_names.append(fname)
            if len(resos) == 16:
                temp1 = copy.deepcopy(resos)
                temp2 = copy.deepcopy(frame_names)
                df_rows.append([i, count, temp2, temp1])
                count += 1
                resos.clear()
                frame_names.clear()
            if count == num_windows + 1:
                break
        print(i, flush=True)
        
    df = pd.DataFrame(df_rows, columns=['track', 'window', 'images', 'scales'])
    df.to_csv('MEVID_train_tracks_scales.csv', index=False)
    
    
def scale_dist_calc():
    train_track = open("/home/c3-0/datasets/MEVID/mevid-v1-annotation-data/track_test_info.txt", 'r')
    names = []
    frame_names = []
    resos = []
    df_rows = []
    with open("/home/c3-0/datasets/MEVID/mevid-v1-annotation-data/test_name.txt", 'r') as f:
        for line in f:
            new_line = line.rstrip()
            names.append(new_line)
        
    for i, row in enumerate(train_track.readlines(), 1):
        start_index, end_index, pid, oid, camid = [int(float(value)) for value in row.split(' ')]
        
        if end_index - start_index < 16:
            print(f'{i} too small tracklet')
            continue
        
        frames = names[start_index:end_index]
        frames_for_scaledist = round(len(frames) * 0.10)
        assert len(frames) > 0, track
        frame_indexer = np.linspace(0, len(frames)-1, frames_for_scaledist).astype(int)
        count = 1
        for index in frame_indexer:
            fname = frames[index] 
            fname = fname.replace('.jpg', '')        
            subject_id, outfit_id, camera_id, track_id, frame_id = fname[0:4], fname[4:8], fname[8:12], fname[12:16], fname[16:22]
            f = os.path.join('/home/c3-0/datasets/MEVID/bbox_test', subject_id, fname + '.jpg')
            assert os.path.exists(f), f
            frame = cv2.imread(f)#[:, :, ::-1]
            reso = frame.shape[0] * frame.shape[1]
            reso = build_scale(reso)
            resos.append(reso)
            frame_names.append(fname)
        temp1 = copy.deepcopy(frame_names)
        temp2 = copy.deepcopy(resos)
        df_rows.append([i, len(frames), temp1, temp2])
        resos.clear()
        frame_names.clear()
            
        print(i, flush=True)
        
    df = pd.DataFrame(df_rows, columns=['track', 'track length', 'images', 'scales'])
    df.to_csv('MEVIDTest_scales_dist.csv', index=False)
    
    
    # 1536, 3072, 6144, 12288, 24576, 49152
    
def build_scale(scale):
    if 1536 <= scale < 3072:
        return 0
    elif 3072 <= scale < 6144:
        return 1
    elif 6144 <= scale < 12288:
        return 2
    elif 12288 <= scale < 24576:
        return 3
    elif 24576 <= scale < 49152:
        return 4
    elif 49152 <= scale:
        return 5
    else:
        print(f'Scale out of bounds, {scale}')
        return -1
        
def build_histograms():
    histogram = []
    for i, row in enumerate(open('MEVIDTest_scales_dist.csv', 'r').readlines()[1:]):
        scales = row.split('"')[3]
        scales = [int(s) for s in scales if s.isdigit()]
        scales = set(scales)
        histogram.append(len(scales))
        print(scales, len(scales))
    #fig = plt.hist(histogram, 6)
    #plt.show()
    #plt.savefig('test.png')
        
        
        
if __name__ == '__main__':
    build_histograms()
    
    