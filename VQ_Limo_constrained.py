import os
import json
from tqdm import tqdm

import torch
import random
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import models.vqvae as vqvae
import options.option_limo as option_limo
import utils.utils_model as utils_model
from dataset import dataset_MOT_MCS
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from classifiers import get_classifier


import write_mot

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

val_loader = dataset_MOT_MCS.DATALoader(args.dataname,
                                        1,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t)

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


def generate_train_embeddings():
    
    print("Generating Embeddings for proximity loss at: ", 'embeddings')
    os.makedirs('embeddings',exist_ok=True)

    data_dict = dict([ (x,[]) for x in action_to_desc ])

    for i,batch in enumerate(val_loader):
        motion, m_length, name = batch
        # print(i,motion.shape, name)
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
        # for j in range(emb.shape[0]):
        #     data_dict["squat"].append(emb[i])
        data_dict["squat"].append(emb)
    


    # os.makedirs(os.path.join(args.out_dir, 'embeddings'), exist_ok = True)
    for k,v in data_dict.items():
        if len(v) == 0:
            continue
        array = np.array(v)
        print(array.shape)
        np.save("embeddings/squat.npy",array)

generate_train_embeddings()

def load_train_embeddings(directory='embeddings'):
    # directory = os.path.join(args.out_dir, 'embeddings')
        
    embedding_dict = {}
    
    if not os.path.exists('embeddings') or len(os.listdir('embeddings')) == 0:
        generate_train_embeddings()
    
    for filename in os.listdir('embeddings'):
        if filename.endswith(".npy"):
            key = filename.split('.')[0]
            embedding = np.load('embeddings/'+filename)
            if len(embedding)==0:
                continue
            
            embedding_dict[action_to_desc[key]] = embedding
    
    return embedding_dict

embedding_dict = load_train_embeddings()
print("Completing loading training embeddings:")
for k,v in embedding_dict.items():
    print(k,v.shape)

def decode_latent(net, x_d):
    # x_d = x_d.permute(0, 2, 1).contiguous().float()
    x_quantized, _, _ = net.vqvae.quantizer(x_d)
    x_decoder = net.vqvae.decoder(x_quantized)
    x_out = x_decoder.permute(0, 2, 1)

    return x_out

def get_proximity_loss(z, embedding, reduce = True):
    # z = np.array(z)
    # embedding = np.array(embedding)
    # print(z.shape)
    # print(embedding.shape)
    
    batch_size = z.shape[0]
    num_embeddings = embedding.shape[0]

    min_distances = torch.zeros(batch_size, device=z.device)
    min_indices = torch.zeros(batch_size, dtype=torch.long, device=z.device)

    for i in range(batch_size):
        distances = torch.norm(z[i].unsqueeze(0) - embedding, dim=(1, 2))
        min_distances[i], min_indices[i] = torch.min(distances, dim=0)
    
    # Sum the minimum distances
    if reduce:
        proximity_loss = min_distances.mean()
    else:
        proximity_loss = min_distances
    
    return proximity_loss, min_indices

def get_optimized_z(category=9,initialization='random', device='cuda'): 
    if initialization == 'random':
        # z = np.random.rand(args.batch_size, args.nb_code, args.seq_len).astype(np.float32)
        # z = torch.from_numpy(z).to(device)
        z_initial = torch.randn(args.batch_size, args.nb_code, args.seq_len, device=device,requires_grad=True)
        noise = torch.randn_like(z_initial) * 5
        z = (z_initial * 100 + noise).detach().requires_grad_(True)

    # print("Z shape:",z.shape)
    # z_np = z.detach().cpu().numpy()
    proximity_embedding = torch.tensor(embedding_dict[category],device=device)
    loss_proximity = get_proximity_loss(z, proximity_embedding)
    print("Initial proximity loss:",loss_proximity)

    z.requires_grad = True
    # z = torch.register
    optimizer = torch.optim.Adam([z], lr=args.lr)
    proximity = []
    constrain = []

    for epoch in tqdm(range(args.total_iter)):
        old_z = torch.from_numpy(z.detach().cpu().numpy()).to(device)
        optimizer.zero_grad()
        pred_motion = decode_latent(net,z)
        
        # print(torch.norm(old_pred-pred_motion))
        
        # tmp = pred_motion[:,:,7].max(dim=1).values
        # print(tmp)
        
        hip_flexion_l = -torch.abs(pred_motion[:,:,7].max(dim=1).values.mean())
        hip_flexion_r = -torch.abs(pred_motion[:,:,15].max(dim=1).values.mean())
        # knee angle 10, 18
        knee_angle_l = -torch.abs(pred_motion[:,:,10].max(dim=1).values.mean())
        knee_angle_r = -torch.abs(pred_motion[:,:,18].max(dim=1).values.mean())
        # ankle angle 12, 20
        ankle_angle_l = -torch.abs(pred_motion[:,:,12].max(dim=1).values.mean())
        ankle_angle_r = -torch.abs(pred_motion[:,:,20].max(dim=1).values.mean())
        
        # print(hip_flexion_l, hip_flexion_r, knee_angle_l, knee_angle_r, ankle_angle_l, ankle_angle_r)

        loss_proximity, min_indices = get_proximity_loss(z, proximity_embedding)
        
        loss_constrain = (hip_flexion_l+hip_flexion_r+knee_angle_l+knee_angle_r+ankle_angle_l+ankle_angle_r)
    
        loss = 0.002*loss_constrain + 0.01 * loss_proximity
        loss.backward()
        # print(z.grad)

        # print(pred_motion.grad)
        optimizer.step()

        if epoch % 10 == 0:
            print("Epoch:", epoch, "Proximity Loss:", 0.01*loss_proximity.item(), "Constrained Loss:", 0.002*loss_constrain.item(), "Total Loss:", loss.item(), "Difference:", torch.norm(z-old_z))
        
        if epoch % 1000 == 0:
            os.makedirs(f"{args.out_dir}/constrained",exist_ok=True)
            np.save(f"{args.out_dir}/constrained/z_"+str(epoch)+".npy",z.detach().cpu().numpy())
            np.save(f"{args.out_dir}/constrained/pred_motion_"+str(epoch)+".npy",pred_motion.detach().cpu().numpy())
            proximity.append(loss_proximity.item())
            constrain.append(loss_constrain.item())
        
        df = pd.DataFrame({'proximity': proximity, 'constrain': constrain})
        df.to_csv(f'{args.out_dir}/constrained/losses.csv', index=False)

    ## SORT THE LATENTS BY THE LABELS
    with torch.no_grad():
        # z_quantized, _, _ = net.vqvae.quantizer(z)
        pred_motion = decode_latent(net,z)
        
        loss, min_indices = get_proximity_loss(z, proximity_embedding, reduce = False)

        loss = loss.view(args.batch_size,-1) # Reshape to match sample x classifier window 

        loss = loss.sum(1) # Sum across classifier windows      

        sort_indices = torch.argsort(loss)

        z = z[sort_indices]
        loss = loss[sort_indices]
        min_idx = min_indices[sort_indices]
        print("Sorted min indices:",min_idx)

    # del optimizer
    # del loss_fn
    # del proximity_embedding
    
    return z,loss

i = 9
if i == 9:

    torch.cuda.empty_cache() # Clear cache to avoid extra memory usage

    category_name = "squat"
    save_folder = os.path.join(f"{args.out_dir}/constrained/npy",'category_'+category_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    z,score = get_optimized_z(category=i)
    decoded_z = decode_latent(net,z)

    bs = decoded_z.shape[0]
    bs = min(bs,10)
    for j in range(bs):
        entry = decoded_z[j]
        file_path = os.path.join(save_folder,f'entry_{j}.npy')
        
        entry_np = entry.cpu().detach().numpy()
        
        np.save(file_path, entry_np)
        write_mot.write_mot(f"{args.out_dir}/constrained/mot/category_{category_name}/entry_{j}.mot", entry_np)
    # np.save(os.path.join(args.out_dir,f'scores_{category_name}.npy'), score.cpu().detach().numpy())

    del z 
    del score
    del decoded_z








#############################################################################################################################################
# # Load Classifier 
# classifier = get_classifier(os.path.join(args.data_root,'classifier.pt')).to(device)

# # Test classifier: 
# # for batch in val_loader:
#     # word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name = batch
#     # net(motion.to(device))



# #### TASK-1: Similiar to LIMO given a target function. Optimize the latents to reach that target function.  
# def decode_latent(net, x_d):
#     x_d = x_d.permute(0, 2, 1).contiguous().float()
#     x_decoder = net.vqvae.decoder(x_d)
#     x_out = x_decoder.permute(0, 2, 1)

#     return x_out

# def classifiy_motion(classifier, motion, classifier_window = 64):
#     B,T,D = motion.shape 

#     # Discard timesteps that are not multiple of classifier_window
#     T = (T // 64) * 64
#     motion = motion[:,:T].view(B,T//classifier_window, classifier_window, D)
#     motion = motion.contiguous().view(-1, classifier_window, D)

#     pred_labels = classifier(motion.float())
#     # pred_val,pred_labels = torch.max(pred_labels, 1)

#     # pred_val = pred_val.view((B,T//classifier_window))
#     # pred_labels = pred_labels.view(B,T//classifier_window)

#     return pred_labels


# def get_optimized_z(category=0): 
#     z = np.random.choice(args.nb_code, (args.batch_size, args.seq_len))
#     z = torch.from_numpy(z).to(device)
    
#     z = net.vqvae.quantizer.dequantize(z)    
#     z.requires_grad = True
 
#     loss_fn = torch.nn.CrossEntropyLoss()


#     optimizer = torch.optim.Adam([z], lr=args.lr)

#     for epoch in tqdm(range(args.total_iter)):
#         optimizer.zero_grad()
#         pred_motion = decode_latent(net,z)

#         pred_labels = classifiy_motion(classifier,pred_motion)
#         B = pred_labels.shape[0]
#         category_vec = category*torch.ones(B).long().to(device)

#         loss = loss_fn(pred_labels, category_vec) 
#         loss.backward()
#         optimizer.step()

#         if epoch % 10 == 0:
#             logger.info('Epoch [{}/{}], CELoss: {:.4f}'.format(epoch, args.total_iter, loss.item()))


#     return z


# save_path = "LIMO_generations/"

# for i in range(13):
#     save_folder = save_path + 'category_'+str(i)
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
    
#     z = get_optimized_z(category=i)
#     decoded_z = decode_latent(net,z)
    
#     bs = decoded_z.shape[0]
#     for j in range(bs):
#         entry = decoded_z[j]
#         file_path = os.path.join(save_folder,f'entry_{j}.npy')
#         np.save(file_path, entry.cpu().detach().numpy())

# z = get_optimized_z()
# # z = z.detach().float()
# # decoded_z = net.vqvae.quantizer.dequantize(z)
# decoded_z = decode_latent(net,z)



