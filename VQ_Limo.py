import os
import json
from tqdm import tqdm

import torch
import torch.nn.functional as F
import random
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import models.vqvae as vqvae
import options.option_limo as option_limo
import utils.utils_model as utils_model
from dataset import dataset_TM_eval
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from classifiers import get_classifier,desc_to_action
from utils.motion_process import recover_from_ric

##### ---- Exp dirs ---- #####
args = option_limo.get_args_parser()
torch.manual_seed(args.seed)

# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark=False


args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')


dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)


##### ---- Dataloader ---- #####
args.nb_joints = 21 if args.dataname == 'kit' else 22

val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer, unit_length=2**args.down_t,data_root=args.data_root)

##### ---- Device ---- #####
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

assert args.resume_pth is not None, "Cannot run the optimization without a trained VQ-VAE"
logger.info('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.to(device)


# Load Classifier 
classifier = get_classifier(os.path.join(args.data_root,'classifier.pt')).to(device)

# Test classifier: 
# for batch in val_loader:
    # word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name = batch
    # net(motion.to(device))



#### TASK-1: Similiar to LIMO given a target function. Optimize the latents to reach that target function.  
def decode_latent(net, x_d):
    x_d = x_d.permute(0, 2, 1).contiguous().float()
    x_decoder = net.vqvae.decoder(x_d)
    x_out = x_decoder.permute(0, 2, 1)

    return x_out

def classifiy_motion(classifier, motion, classifier_window = 64):
    B,T,D = motion.shape 

    # Discard timesteps that are not multiple of classifier_window
    T = (T // 64) * 64
    motion = motion[:,:T].view(B,T//classifier_window, classifier_window, D)
    motion = motion.contiguous().view(-1, classifier_window, D)

    pred_labels = classifier(motion.float())
    # pred_val,pred_labels = torch.max(pred_labels, 1)

    # pred_val = pred_val.view((B,T//classifier_window))
    # pred_labels = pred_labels.view(B,T//classifier_window)

    return pred_labels


def get_optimized_z(category=0): 
    z = np.random.choice(args.nb_code, (args.batch_size, args.seq_len))
    z = torch.from_numpy(z).to(device)
    
    z = net.vqvae.quantizer.dequantize(z)    
    z.requires_grad = True
 
    loss_fn = torch.nn.CrossEntropyLoss()


    optimizer = torch.optim.Adam([z], lr=args.lr)

    data_mean = torch.from_numpy(val_loader.dataset.mean).to(device)
    data_std = torch.from_numpy(val_loader.dataset.std).to(device)

    for epoch in tqdm(range(args.total_iter)):
        optimizer.zero_grad()
        pred_motion = decode_latent(net,z)

        # De-Normalize
        pred_motion = pred_motion * data_std.view(1,1,-1) + data_mean.view(1,1,-1)

        pred_labels = classifiy_motion(classifier,pred_motion)
        B = pred_labels.shape[0]
        category_vec = category*torch.ones(B).long().to(device)



        pred_xyz = recover_from_ric(pred_motion,22)
        loss_temporal = F.smooth_l1_loss(pred_xyz[:,1:], pred_xyz[:,:-1])
 
        loss = loss_fn(pred_labels, category_vec) 
        loss += loss_temporal * args.loss_temporal
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            logger.info('Epoch [{}/{}], CELoss: {:.4f}'.format(epoch, args.total_iter, loss.item()))

    ## SORT THE LATENTS BY THE LABELS
    loss_fn = torch.nn.CrossEntropyLoss(reduce=False)
    with torch.no_grad():
        pred_motion = decode_latent(net,z)

        # De-Normalize
        pred_motion = pred_motion * data_std.view(1,1,-1) + data_mean.view(1,1,-1)

        pred_labels = classifiy_motion(classifier,pred_motion)
        B = pred_labels.shape[0]
        category_vec = category*torch.ones(B).long().to(device)

        loss = loss_fn(pred_labels, category_vec)

        loss = loss.view(args.batch_size,-1) # Reshape to match sample x classifier window 

        loss = loss.sum(1) # Sum across classifier windows      

        sort_indices = torch.argsort(loss)

        z = z[sort_indices]
        loss = loss[sort_indices]

    return z,loss


data_mean = torch.from_numpy(val_loader.dataset.mean).to(device)
data_std = torch.from_numpy(val_loader.dataset.std).to(device)
for i in range(13):

    torch.cuda.empty_cache() # Clear cache to avoid extra memory usage

    category_name = "_".join(desc_to_action[i].split(' '))
    save_folder = os.path.join(args.out_dir,'category_'+category_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    z,score = get_optimized_z(category=i)
    decoded_z = decode_latent(net,z)
    # De-Normalize
    decoded_z = decoded_z * data_std.view(1,1,-1) + data_mean.view(1,1,-1)

    bs = decoded_z.shape[0]
    bs = min(bs,args.topk)
    for j in range(bs):
        entry = decoded_z[j]
        file_path = os.path.join(save_folder,f'entry_{j}.npy')
        np.save(file_path, entry.cpu().detach().numpy())

    np.save(os.path.join(args.out_dir,f'scores_{category_name}.npy'), score.cpu().detach().numpy())

    del z 
    del score
    del decoded_z

