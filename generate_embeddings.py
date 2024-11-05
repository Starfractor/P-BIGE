import os
import json

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import models.vqvae as vqvae
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_TM_eval
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
import numpy as np

args = option_vq.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')


dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

args.nb_joints = 21 if args.dataname == 'kit' else 22

val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 1, w_vectorizer, unit_length=2**args.down_t, mode='train')

##### ---- Network ---- #####
net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate,
                       args.vq_act,
                       args.vq_norm)

if args.resume_pth : 
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
# net.train()
net.eval()
net.cuda()

action_to_desc = {
        "bend and pull full" : [],
        "countermovement jump" : [],
        "left countermovement jump" : [],
        "left lunge and twist" : [],
        "left lunge and twist full" : [],
        "right countermovement jump" : [],
        "right lunge and twist" : [],
        "right lunge and twist full" : [],
        "right single leg squat" : [],
        "squat" : [],
        "bend and pull" : [],
        "left single leg squat" : [],
        "push up" : []
    }

for i,batch in enumerate(val_loader):
    word_emb, pos_one_hot, caption, sent_len, motion, m_length, token, name = batch
    print(i,motion.shape, caption[0])
    motion = motion.cuda()
    m = net.vqvae.preprocess(motion)
    emb = net.vqvae.encoder(m)
    emb = torch.squeeze(emb)
    emb = torch.transpose(emb,0,1)
    emb_list = emb.tolist()
    for vec in emb_list:
        action_to_desc[str(caption[0])].append(vec)

for k,v in action_to_desc.items():
    array = np.array(v)
    print(array.shape)
    np.save(f"{k}.npy",array)
    
def load_embeddings(directory="embeddings/"):
    embedding_dict = {}
    
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            key = filename.split('.')[0]
            embedding = np.load(os.path.join(directory,filename))
            
            embedding_dict[key] = embedding
    
    return embedding_dict

def proximity_loss(sample, label, embedding_dict, net):
    sample = sample.cuda()
    m = net.vqvae.preprocess(sample)
    emb = net.vqvae.encoder(m)
    emb = torch.squeeze(emb)
    emb = torch.transpose(emb,0,1)
    emb = np.array(emb)
    embedding_map = embedding_dict[label]
    
    distances = np.linalg.norm(emb[:,np.newaxis] - embedding_map, axis = 2)
    
    min_distances = np.min(distances, axis = 1)
    
    return np.min(distances)

