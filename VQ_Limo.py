import os
import sys
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
logger.info(f"Command:{' '.join(sys.argv)}")
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')


dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

action_to_desc = {
        "bend and pull full" : 0,
        "countermovement jump" : 1,
        "left countermovement jump" : 2,
        "left lunge and twist" : 3,
        "left lunge and twist full" : 4,
        "right countermovement jump" : 5,
        "right lunge and twist" : 6,
        "right lunge and twist full" : 7,
        "right single leg squat" : 8,
        "squat" : 9,
        "bend and pull" : 10,
        "left single leg squat" : 11,
        "push up" : 12
    }

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

assert args.vq_name is not None, "Cannot run the optimization without a trained VQ-VAE"
logger.info('loading checkpoint from {}'.format(args.vq_name))
ckpt = torch.load(args.vq_name, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.to(device)

def generate_train_embeddings(mcs_name=None):
    
    print("Generating Embeddings for proximity loss at: ", os.path.join(args.out_dir,'embeddings', mcs_name))
    os.makedirs(os.path.join(args.out_dir,'embeddings', mcs_name),exist_ok=True)

    data_dict = dict([ (x,[]) for x in action_to_desc ])

    for i,batch in enumerate(val_loader):
        word_emb, pos_one_hot, caption, sent_len, motion, m_length, token, name, mcs_score = batch
        if "mcs_" in mcs_name and mcs_score != int(mcs_name.replace("mcs_","")): 
            continue
        # if mcs_score == -1:
        #     continue
        # if mcs_score != 5:
        #     continue
        print(i,motion.shape, caption[0], mcs_score)
        motion = motion.cuda()
        # print("motion shape:", motion.shape)
        m = net.vqvae.preprocess(motion)
        # print("m shape:", m.shape)
        emb = net.vqvae.encoder(m)
        # print("emb shape:", emb.shape)
        # emb_proc = net.vqvae.postprocess(emb)
        # print("emb proc shape:", emb_proc.shape)
        
        emb = torch.squeeze(emb)
        # emb = torch.transpose(emb,0,1)
        emb = emb.cpu().detach().numpy()
        print(emb.shape)
        for i in range(emb.shape[0]):
            data_dict[str(caption[i])].append(emb[i])
        
        # emb_list = emb.tolist()
        # for vec in emb_list:
        #     action_to_desc[str(caption[0])].append(vec)


    # os.makedirs(os.path.join(args.out_dir, 'embeddings'), exist_ok = True)
    for k,v in data_dict.items():
        array = np.array(v)
        print(array.shape)
        np.save(os.path.join(args.out_dir, 'embeddings', mcs_name, f"{k}.npy"),array)

def load_train_embeddings(directory='embeddings', mcs_score = None):
    # directory = os.path.join(args.out_dir, 'embeddings')
    
    if mcs_score is None: 
        mcs_score = "all_embeddings"
    elif mcs_score > 0 and mcs_score <= 5: 
        mcs_score = "mcs_" + str(mcs_score)
    else: 
        mcs_score = "all_embeddings"
        
    embedding_dict = {}
    
    if not os.path.exists(os.path.join(args.out_dir,'embeddings', mcs_score)) or len(os.listdir(os.path.join(args.out_dir,'embeddings', mcs_score))) == 0:
        generate_train_embeddings(mcs_name=mcs_score)
    
    for filename in os.listdir(os.path.join(args.out_dir,'embeddings', mcs_score,)):
        if filename.endswith(".npy"):
            key = filename.split('.')[0]
            embedding = np.load(os.path.join(args.out_dir,'embeddings', mcs_score,filename))
            if len(embedding)==0:
                continue
            
            embedding_dict[action_to_desc[key]] = embedding
    
    return embedding_dict

embedding_dict = load_train_embeddings(mcs_score = args.mcs)
print("Completeting loading training embeddings:")
for k,v in embedding_dict.items():
    print(k,desc_to_action[k],v.shape)




# Load Classifier 
classifier = get_classifier(os.path.join(args.data_root,'classifier.pt')).to(device)

# Test classifier: 
# for batch in val_loader:
    # word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name = batch
    # net(motion.to(device))



#### TASK-1: Similiar to LIMO given a target function. Optimize the latents to reach that target function.  
def decode_latent(net, x_d):
    # x_d = x_d.permute(0, 2, 1).contiguous().float()
    x_quantized, _, _ = net.vqvae.quantizer(x_d)
    x_decoder = net.vqvae.decoder(x_quantized)
    x_out = x_decoder.permute(0, 2, 1)

    return x_out


def compute_proximity_codebook(z,codebook): 
    bs, seq_len, emb_dim = z.shape
    z = z.view(bs,seq_len,1,emb_dim)
    codebook = codebook.view(1,1,codebook.shape[0],codebook.shape[1])
    dist = torch.norm(z - codebook, dim=3)
    closest_codebook_dist, closest_codebook_indices = torch.min(dist, dim=2)

    return closest_codebook_indices


def classifiy_motion(classifier, motion, classifier_window = 64,stride=3):
    B,T,D = motion.shape 

    window_size = classifier_window * stride

    # Discard timesteps that are not multiple of classifier_window
    T = (T // window_size) * window_size
    motion = motion[:,:T].view(B,T//window_size, window_size, D)
    
    motion = motion[:,:,::stride]
    
    motion = motion.contiguous().view(-1, classifier_window, D)

    pred_labels = classifier(motion.float())
    # pred_val,pred_labels = torch.max(pred_labels, 1)

    # pred_val = pred_val.view((B,T//classifier_window))
    # pred_labels = pred_labels.view(B,T//classifier_window)

    return pred_labels


def get_proximity_loss(z, embedding):
    # z = np.array(z)
    # embedding = np.array(embedding)
    
    batch_size = z.shape[0]
    num_embeddings = embedding.shape[0]

    min_distances = torch.zeros(batch_size, device=z.device)

    for i in range(batch_size):
        distances = torch.norm(z[i].unsqueeze(0) - embedding, dim=(1,2))
        min_distances[i] = distances.min()
    
    # Sum the minimum distances
    proximity_loss = min_distances.mean()
    
    return proximity_loss


def get_foot_losses(motion, y_translation=0.0,feet_threshold=0.01):
    

    # y_translation = 0.0
    min_height, idx = motion[..., 1].min(dim=-1)

    # y_translation = -min_height.median() # Change reference to median  (Other set of experiments determine this. See paper)

    # print(min_height,idx,motion[..., 1].shape)
    min_height = min_height + y_translation
    pn = -torch.minimum(min_height, torch.zeros_like(min_height))  # penetration
    pn[pn < feet_threshold] = 0.0
    fl = torch.maximum(min_height, torch.zeros_like(min_height))  # float
    fl[fl < feet_threshold] = 0.0

    bs, t = idx.shape

    I = torch.arange(bs).view(bs, 1).expand(-1, t-1).long()
    J = torch.arange(t-1).view(1, t-1).expand(bs, -1).long()
    J_next = J + 1
    feet_motion = motion[I, J, idx[:, :-1]]
    feet_motion_next = motion[I, J_next, idx[:, :-1]]
    sk = torch.norm(feet_motion - feet_motion_next, dim=-1)
    contact = fl[:, :t] < feet_threshold

    sk = sk[contact[:, :-1]]  # skating
    # action: measure the continuity between frames
    vel = motion[:, 1:] - motion[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    acc = torch.norm(acc, dim=-1)
    # all losses
    loss_pn = pn[:, :t].view(-1)
    loss_fl = fl[:, :t].view(-1)
    loss_sk = sk.view(-1)
    metr_act = acc[:, :t].view(-1)
    
    return loss_pn.sum()/bs, loss_fl.sum()/bs, loss_sk.sum()/bs


def get_optimized_z(category=0,initialization='random', device='cuda'): 
    if initialization == 'random':
        # z = np.random.rand(args.batch_size, args.nb_code, args.seq_len).astype(np.float32)
        # z = torch.from_numpy(z).to(device)
        z = torch.rand(args.batch_size, args.nb_code, args.seq_len, device=device)
    elif initialization == 'codebook-uniform':
        z = np.random.choice(args.nb_code, (args.batch_size, args.seq_len))
        z = torch.from_numpy(z).to(device)
        z = net.vqvae.quantizer.quantize(z)
        z_quantized,_,_ = net.vqvae.quantizer(z)

    # print("Z shape:",z.shape)
    # z_np = z.detach().cpu().numpy()
    proximity_embedding = torch.tensor(embedding_dict[category],device=device)
    loss_proximity = get_proximity_loss(z, proximity_embedding)
    print("Initial proximity loss:",loss_proximity)

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


        # Foot losses
        loss_penetration, loss_floating, loss_sliding = get_foot_losses(pred_xyz,y_translation=0.0,feet_threshold=args.feet_threshold) 
        
        foot_loss_type = "all"
        # foot_loss_type = "penetration"
        #foot_loss_type = "floating"
        # foot_loss_type = "sliding"
        if foot_loss_type == "all":
            loss_foot = loss_penetration +  loss_floating + loss_sliding
        elif foot_loss_type == "penetration":
            loss_foot = loss_penetration
        elif foot_loss_type == "floating": 
            loss_foot = loss_floating
        elif foot_loss_type == "sliding": 
            loss_foot = loss_sliding 
        else: 
            loss_foot = torch.Tensor([0.0])
        
        # Proximity loss 
        # closest_codebook_indices = compute_proximity_codebook(z,net.vqvae.quantizer.codebook)
        # z_np = z.detach().cpu().numpy()  # Convert z to numpy for proximity loss calculation
        # loss_proximity = get_proximity_loss(z_np, embedding_dict[category])
        # loss_proximity = torch.tensor(loss_proximity, dtype=torch.float32, device=device)
        loss_proximity = get_proximity_loss(z, proximity_embedding)
    
        score = loss_fn(pred_labels, category_vec)
    
        loss = score
        loss += loss_temporal * args.loss_temporal
        loss += loss_proximity * args.loss_proximity
        loss += loss_foot * args.loss_foot
        loss.backward()

        # print(z.grad)
        optimizer.step()

        if epoch % 10 == 0:
            logger.info('Epoch [{}/{}], score:{:.4f}, Temporal loss:{:.4f}, Proximity loss:{:4f} Foot Loss:{:.4f} Total loss:{:.4f}'.format(epoch, args.total_iter, score.item(), loss_temporal.item()* args.loss_temporal, loss_proximity.item()*args.loss_proximity, loss_foot * args.loss_foot, loss.item()))

    ## SORT THE LATENTS BY THE LABELS
    loss_fn = torch.nn.CrossEntropyLoss(reduce=False)
    with torch.no_grad():
        # z_quantized, _, _ = net.vqvae.quantizer(z)
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

    del optimizer
    del loss_fn
    del proximity_embedding
    
    return z,loss


data_mean = torch.from_numpy(val_loader.dataset.mean).to(device)
data_std = torch.from_numpy(val_loader.dataset.std).to(device)
for i in range(0,13):
    if i not in embedding_dict.keys():
        continue

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


    # np.save(os.path.join(args.out_dir,f'codebook_{category_name}.npy'), z.cpu().detach().numpy())
    np.save(os.path.join(args.out_dir,f'scores_{category_name}.npy'), score.cpu().detach().numpy())

    del z 
    del score
    del decoded_z

