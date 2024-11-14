import os
import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
from glob import glob

SEGMENTATION_DIR = '/data/panini/MCS_DATA/squat-segmentation-data'

class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, window_size = 64, unit_length = 4, mode = 'train'):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name

        if dataset_name == 'mcs':
            self.data_root = '/home/ubuntu/data/HumanML3D' + "/" + mode
            self.max_motion_length = 196 # Maximum length of sequence, change this for different lengths
            self.meta_dir = '/home/ubuntu/data/HumanML3D'
        
        self.file_list = glob("/home/ubuntu/data/MCS_DATA/Data/*/OpenSimData/Kinematics/*.mot")
        self.data = []
        self.lengths = []
        self.names = []
        self.mode = mode
        self.segments = []
        
        for file in tqdm(self.file_list):

            tmp_name = file.split('/')[-1]
            if "sqt" not in tmp_name and "SQT" not in tmp_name and "Sqt" not in tmp_name:
                continue

            mot_file_name = os.path.abspath(file).split('/')[-1].replace('.mot','') 
            subject_id = os.path.abspath(file).split('/')[-4]
            
            segments_data = np.load(os.path.join(SEGMENTATION_DIR, subject_id + '.npy'),allow_pickle=True).item()
            
            # Hard constrains on what will be used as test data. It must have segmentation present
            trial_details = mot_file_name.split('_')
            if len(trial_details) < 3: # Not a valid trial
                # print(f"Skipping {file}. No segmentation found")
                continue
            
            
            if trial_details[0] not in segments_data.keys(): # Not a valid trial
                print(f"Skipping {file}. Trial does not have segmentation. {trial_details}")
                continue
            
            if 'segment' != trial_details[1] and not trial_details[2].isdigit(): # Not a valid trial
                print(f"Skipping {file}. segment not middle keyword. {trial_details}")
                continue
            
            if len(segments_data[trial_details[0]]) < int(trial_details[-1]) and int(trial_details[-1]) > 0 : # Not a valid trial
                print(f"Skipping {file}. Trial does not have segmentation. {trial_details}")
                continue
            
            segment = segments_data[trial_details[0]][int(trial_details[2])-1]
                  
            with open(file,'r') as f:
                file_data = f.read().split('\n')
                # print(file_data)
                data = {'info':'', 'poses': []}
                read_header = False
                read_rows = 0
                
                for line in file_data:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    
                    if not read_header:
                        if line == 'endheader':
                            read_header = True
                            continue
                        if '=' not in line:
                            data['info'] += line + '\n'
                        else:
                            k,v = line.split('=')
                            if v.isnumeric():
                                data[k] = int(v)
                            else:
                                data[k] = v
                    else:
                        rows = line.split()
                        if read_rows == 0:
                            data['headers'] = rows
                        else:
                            rows = [float(row) for row in rows]
                            data['poses'].append(rows)
                        read_rows += 1
            data['poses'] = np.array(data['poses'])[:,1:] # Change to remove time
            

            if segment[1] - segment[0] < self.window_size:
                print(f"Skipping {file}. Segment {segment} is too small")
                continue      
            
            
            if segment[0] > data['nRows'] or segment[1] > data['nRows']:
                print(f"Segment {segment} is out of bounds for {file}")
                continue

            nRows = segment[1] - segment[0] + 1            
            poses = data['poses'][segment[0]:segment[1]+1]
            
            # Based on manual segmentation select time frames 
            # print(data['poses'].shape)
            self.data.append(poses)
            self.lengths.append(nRows)
            self.names.append(file)
            self.segments.append(segment)
        
        print("Total number of motions {}".format(len(self.data)))

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def compute_sampling_prob(self) : 
        
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.mode == 'train':
            motion = self.data[item]
            len_motion = len(motion) if len(motion) <=self.max_motion_length else self.max_motion_length
            name = self.names[item]
            
            # idx = random.randint(0, len(motion) - self.window_size)
            # motion = motion[idx:idx+self.window_size]
            
            if len(motion) >= self.max_motion_length:
                idx = random.randint(0, len(motion) - self.max_motion_length)
                motion = motion[idx:idx+self.max_motion_length]
            else:
                pad_width = self.max_motion_length - len(motion)
                motion = np.pad(motion, ((0, pad_width), (0,0)), mode='constant', constant_values=0)
            
            "Z Normalization"
            # motion = (motion - self.mean) / self.std

            return motion, len_motion, name
        
        if self.mode == 'limo':
            motion = self.data[item]
            len_motion = len(motion)
            name = self.names[item]
            
            if len_motion < self.max_motion_length:
                pad_width = self.max_motion_length - len_motion
                motion = np.pad(motion, ((0, pad_width), (0,0)), mode='constant', constant_values=0)
                return [motion], [len_motion], [name]
            
            subsequences = []
            subsequence_lengths = []
            names = []

            for start_idx in range(0, len_motion - self.max_motion_length + 1,4):
                subseq = motion[start_idx:start_idx + self.max_motion_length]
                subsequences.append(subseq)
                subsequence_lengths.append(self.max_motion_length)
                names.append(name)

            return subsequences, subsequence_lengths, names

def DATALoader(dataset_name,
               batch_size,
               num_workers = 8,
               window_size = 64,
               unit_length = 4,
               mode = 'train'):
    
    trainSet = VQMotionDataset(dataset_name, window_size=window_size, unit_length=unit_length, mode=mode)
    prob = trainSet.compute_sampling_prob()
    sampler = torch.utils.data.WeightedRandomSampler(prob, num_samples = len(trainSet) * 1000, replacement=True)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size,
                                              shuffle=True,
                                              #sampler=sampler,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True)
    
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
