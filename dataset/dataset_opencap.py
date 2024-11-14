import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
from glob import glob


class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, window_size = 64, unit_length = 4, mode = 'train'):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name

        if dataset_name == 'mcs':
            self.data_root = '/home/ubuntu/data/HumanML3D' + "/" + mode
            self.max_motion_length = 196 # Maximum length of sequence, change this for different lengths
            self.meta_dir = '/home/ubuntu/data/HumanML3D'
        
        self.file_list = glob("/home/ubuntu/data/opencap-processing/Data/*/OpenSimData/Kinematics/*.mot")
        self.data = []
        self.lengths = []
        self.names = []
        
        for file in tqdm(self.file_list):
            tmp_name = file.split('/')[-1]
            if "sqt" not in tmp_name and "SQT" not in tmp_name and "Sqt" not in tmp_name:
                continue
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
            if data['nRows'] < self.window_size:
                continue
            data['poses'] = np.array(data['poses'])[:,1:] # Change to remove time 
            # print(data['poses'].shape)
            self.data.append(data['poses'])
            self.lengths.append(data['nRows'])
            self.names.append(file)
        
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

def DATALoader(dataset_name,
               batch_size,
               num_workers = 8,
               window_size = 64,
               unit_length = 4):
    
    trainSet = VQMotionDataset(dataset_name, window_size=window_size, unit_length=unit_length)
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
