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
import nimblephysics as nimble 
from classifiers import get_classifier
from write_mot import write_mot33, write_mot35


def write_mot(path, data, framerate=60):
    header_string = f"Coordinates\nversion=1\nnRows={data.shape[0]}\nnColumns=36\ninDegrees=yes\n\nUnits are S.I. units (second, meters, Newtons, ...)\nIf the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).\n\nendheader\ntime	pelvis_tilt	pelvis_list	pelvis_rotation	pelvis_tx	pelvis_ty	pelvis_tz	hip_flexion_r	hip_adduction_r	hip_rotation_r	knee_angle_r	knee_angle_r_beta	ankle_angle_r	subtalar_angle_r	mtp_angle_r	hip_flexion_l	hip_adduction_l	hip_rotation_l	knee_angle_l	knee_angle_l_beta	ankle_angle_l	subtalar_angle_l	mtp_angle_l	lumbar_extension	lumbar_bending	lumbar_rotation	arm_flex_r	arm_add_r	arm_rot_r	elbow_flex_r	pro_sup_r	arm_flex_l	arm_add_l	arm_rot_l	elbow_flex_l	pro_sup_l\n"

    with open(path, 'w') as f:
        f.write(header_string)
        for i,d in enumerate(data):
            d = [str(i/60)] + [str(x) for x in d]
            
            # print(d)
            d = "      " +  "\t     ".join(d) + "\n"
            # print(d)
            f.write(d)

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
                                        unit_length=2**args.down_t,
                                        mode='limo')



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


############ Module to load subject info ################
from osim_sequence import load_osim, groundConstraint, GetLowestPointLayer
assert os.path.isdir(args.subject), "Location to subject info does not exist"
osim_path = os.path.join(args.subject,'OpenSimData','Model', 'LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim')
assert os.path.exists(osim_path), f"Osim file:{osim_path} does not exist"
osim_geometry_dir = os.path.join("/data/panini/MCS_DATA",'OpenCap_LaiArnoldModified2017_Geometry')
assert os.path.exists(osim_geometry_dir), f"Osim geometry path:{osim_geometry_dir} does not exist"

osim = load_osim(osim_path, osim_geometry_dir, ignore_geometry=False)



# Assert data is being loaded is compatible with nimble physics engine   
# for i,batch in tqdm(enumerate(val_loader)):
#     print("Testing ground constraint for batch:",i)
#     motions, m_lengths, names = batch
#     indices_to_keep = [i for i in range(motions[0].shape[2]) if i not in [10,18]]   
    
#     # for m_index, motion in enumerate(motions): 
#     motion = motions[0]
#     m= motion[:,:,indices_to_keep]
#     x = groundConstraint(osim, torch.tensor(m).cpu())
    
#     assert x.item() < 5, f"Person cannot fly, average lowest point on the skeleton above 5 meters.:{names[m_index]}"
        




def generate_train_embeddings():
    
    print("Generating Embeddings for proximity loss at: ", 'embeddings')
    os.makedirs('embeddings',exist_ok=True)

    data_dict = dict([ (x,[]) for x in action_to_desc ])

    for i,batch in tqdm(enumerate(val_loader)):
        ##################### OLD
        # motion, m_length, name = batch
        # # print(i,motion.shape, name)
        # motion = motion.cuda()
        # # print("motion shape:", motion.shape)
        # m = net.vqvae.preprocess(motion)
        # # print("m shape:", m.shape)
        # emb = net.vqvae.encoder(m)
        # # print("emb shape:", emb.shape)
        # # emb_proc = net.vqvae.postprocess(emb)
        # # print("emb proc shape:", emb_proc.shape)
        
        # emb = torch.squeeze(emb)
        # # emb = torch.transpose(emb,0,1)
        # emb = emb.cpu().detach().numpy()
        # # print(emb.shape)
        # # for j in range(emb.shape[0]):
        # #     data_dict["squat"].append(emb[i])
        # data_dict["squat"].append(emb)
        ###########################
        
        motions, m_lengths, names = batch
        # print(i,motion.shape, name)
        # motions = motions.cuda()
        for motion in motions:
            # print("motion shape:", motion.shape)
            motion = motion.cuda()
            m = net.vqvae.preprocess(motion)
            # print("m shape:", m.shape)
            emb = net.vqvae.encoder(m)
            # print("emb shape:", emb.shape)
            # emb_proc = net.vqvae.postprocess(emb)
            # print("emb proc shape:", emb_proc.shape)
            
            emb = torch.squeeze(emb)
            # emb = torch.transpose(emb,0,1)
            emb = emb.cpu().detach().numpy()
            # print(emb.shape)
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

# generate_train_embeddings()

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

# Generate mot reconstruction for training data
# with torch.no_grad():
#     for i,batch in enumerate(val_loader):
#         motion, m_length, name = batch
#         out,_,_ = net(motion[:,:m_length[0]].cuda())
#         pred = out
#         write_mot('train_forward_pass/mot_output/'+str(i)+".npy", pred[0,:,:].detach().cpu().numpy())
#         out = out.squeeze(0).cpu().detach().numpy()
#         np.save('train_forward_pass/model_output/'+str(i)+".npy", out)
#         print(out.shape, np.array(motion).shape)

embedding_dict = load_train_embeddings()
print("Completing loading training embeddings:")
for k,v in embedding_dict.items():
    print(k,v.shape)

# def load_train_embeddings(directory='embeddings'):
#     # directory = os.path.join(args.out_dir, 'embeddings')
        
#     embedding_dict = {}
    
#     if not os.path.exists('embeddings') or len(os.listdir('embeddings')) == 0:
#         generate_train_embeddings()
    
#     for filename in os.listdir('embeddings'):
#         if filename.endswith(".npy"):
#             key = filename.split('.')[0]
#             embedding = np.load('embeddings/'+filename)
#             if len(embedding)==0:
#                 continue
            
#             embedding_dict[action_to_desc[key]] = embedding
    
#     return embedding_dict

def decode_latent(net, x_d):
    # x_d = x_d.permute(0, 2, 1).contiguous().float()
    x_quantized, _, _ = net.vqvae.quantizer(x_d)
    x_decoder = net.vqvae.decoder(x_quantized)
    x_out = x_decoder.permute(0, 2, 1)

    return x_out

def get_proximity_loss(z, embedding, reduce = True, chunk_size = 1000):
    
    batch_size = z.shape[0]
    num_embeddings = embedding.shape[0]

    min_distances = torch.full((batch_size,), float('inf'), device=z.device)
    min_indices = torch.zeros(batch_size, dtype=torch.long, device=z.device)

    for i in range(batch_size):
        # Initialize to keep track of the minimum distance and index for this batch
        min_dist = float('inf')
        min_idx = -1

        # Process embedding in chunks to avoid memory overload
        for start in range(0, num_embeddings, chunk_size):
            end = min(start + chunk_size, num_embeddings)
            embedding_chunk = embedding[start:end]

            # Compute distances for the chunk, keep dims (1, 2) as in original
            distances = torch.norm(z[i].unsqueeze(0) - embedding_chunk, dim=(1, 2))

            # Find the minimum distance and corresponding index in the chunk
            chunk_min_dist, chunk_min_idx = torch.min(distances, dim=0)

            # Update overall minimum distance and index if chunk's min is smaller
            if chunk_min_dist < min_dist:
                min_dist = chunk_min_dist
                min_idx = start + chunk_min_idx  # Adjust index for chunk offset
        
        # Store the final minimum distance and index for this batch
        min_distances[i] = min_dist
        min_indices[i] = min_idx

    # Sum the minimum distances
    if reduce:
        proximity_loss = min_distances.mean()
    else:
        proximity_loss = min_distances

    return proximity_loss, min_indices
    
def get_optimized_z(category=9,initialization='mean', device='cuda'): 
    print("Optimizing for category:",category)
    print("Initialization:", embedding_dict)
    
    if initialization == 'random':
        # z = np.random.rand(args.batch_size, args.nb_code, args.seq_len).astype(np.float32)
        # z = torch.from_numpy(z).to(device)
        z_initial = torch.randn(args.batch_size, args.nb_code, args.seq_len, device=device,requires_grad=True)
        mult1 = np.random.uniform(50,120)
        mult2 = np.random.uniform(1,10)
        noise = torch.randn_like(z_initial) * mult2
        z = (z_initial * mult1 + noise).detach().requires_grad_(True)
    
    elif initialization == 'mean':
        mean = torch.tensor(embedding_dict[category],device=device).mean(dim=0)
        std = torch.tensor(embedding_dict[category],device=device).std(dim=0)
        z = torch.randn(args.batch_size, args.nb_code, args.seq_len, device=device)
        # z = z * std + mean
        mult1 = np.random.uniform(10,200)
        z = mean + torch.randn_like(z) * mult1 #500
        z.requires_grad_(True)

    print("Z shape:",z.shape)
    # z_np = z.detach().cpu().numpy()
    proximity_embedding = torch.tensor(embedding_dict[category],device=device)
    loss_proximity = get_proximity_loss(z, proximity_embedding)
    print("Initial proximity loss:",loss_proximity)

    z.requires_grad = True
        
    
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam([z], lr=args.lr)

    # data_mean = torch.from_numpy(val_loader.dataset.mean).to(device)
    # data_std = torch.from_numpy(val_loader.dataset.std).to(device)
    proximity = []
    constrain = []

    for epoch in tqdm(range(args.total_iter)):
        old_z = torch.from_numpy(z.detach().cpu().numpy()).to(device)
        optimizer.zero_grad()
        pred_motion = decode_latent(net,z)
        pred_motion.requires_grad_(True)

        # De-Normalize
        # pred_motion = pred_motion * data_std.view(1,1,-1) + data_mean.view(1,1,-1)
        loss_proximity,min_indices = get_proximity_loss(z, proximity_embedding)
        # print("Min indices:",min_indices)
    
        loss_temp = torch.mean((pred_motion[:,1:,:]-pred_motion[:,:-1,:])**2)

        # Reduce jitter in the translation
        loss_temp_trans = torch.mean((pred_motion[:,1:,[3,5]]-pred_motion[:,:-1,[3,5]])**2)
        

        foot_loss = torch.tensor([0.0],device=device)
        indices_to_keep = [i for i in range(pred_motion.shape[2]) if i not in [10,18]]
        motion = pred_motion[:,:,indices_to_keep]
        for i in range(pred_motion.shape[0]):
            # m = motion[i]
            # m_tensor = torch.tensor(m, dtype=torch.float32, device=device, requires_grad=True)
            m_tensor = pred_motion[i,:,indices_to_keep]
            # for j in range(1,pred_motion.shape[1],3):
            for rand_j in range(1,int(epoch*10/3000)):
                j = random.randint(1,pred_motion.shape[1]-1)
                nimble_input_motion = m_tensor[j]
                nimble_input_motion[6:] = torch.deg2rad(nimble_input_motion[6:])
                nimble_input_motion[:3] = 0 # Set pelvis to 0
                nimble_input_motion = nimble_input_motion.cpu()
                x = GetLowestPointLayer.apply(osim.skeleton, nimble_input_motion).to(device)
                foot_loss += x**2

        # loss = loss_proximity * 0.0001
        # foot_loss = foot_loss.to(device)
        # foot_loss = torch.tensor(0,device=device)
        if epoch < 500: # Early start for proximity loss 
            loss = loss_proximity * 0.001
        else:
            loss = loss_proximity * 0.001 + foot_loss * 0.01 + 0.5*loss_temp + 5*loss_temp_trans
        # loss = foot_loss*0.01
        
        loss.backward()
        # print("Z grad",z.grad)
        optimizer.step()
        if epoch % 10 == 0:
            print("Epoch:", epoch, "Loss:", loss.item(), "Penetration:", foot_loss.item()*0.01, "Temporal Loss:", 0.5*loss_temp.item(), "Proximity Loss:", 0.001*loss_proximity.item(), "Trans Temporal:", 0.5*loss_temp_trans)#"Difference:", torch.norm(z-old_z))
            
        if epoch % 1000 == 0:
            os.makedirs("save_LIMO/normal_subject",exist_ok=True)
            np.save("save_LIMO/normal_subject/z_"+str(epoch)+".npy",z.detach().cpu().numpy())
            np.save("save_LIMO/normal_subject/pred_motion_"+str(epoch)+".npy",pred_motion.detach().cpu().numpy())
    
    df = pd.DataFrame({'proximity': proximity})
    df.to_csv('losses.csv', index=False)

    ## SORT THE LATENTS BY THE LABELS
    with torch.no_grad():
        # z_quantized, _, _ = net.vqvae.quantizer(z)
        pred_motion = decode_latent(net,z)
        
        loss, min_indices = get_proximity_loss(z, proximity_embedding, reduce = False)
        print(min_indices)

        loss = loss.view(args.batch_size,-1) # Reshape to match sample x classifier window 

        loss = loss.sum(1) # Sum across classifier windows      

        sort_indices = torch.argsort(loss)

        z = z[sort_indices]
        loss = loss[sort_indices]
        min_idx = min_indices[sort_indices]
        print("Sorted min indices:",min_idx)

    del optimizer
    del loss_fn
    del proximity_embedding
    
    return z,loss

i = 9

if i == 9:
    # Added to run multiple iterations of LIMO for evaluation purposes
    for run in range(args.num_runs):

        torch.cuda.empty_cache() # Clear cache to avoid extra memory usage

        category_name = "squat"
        # save_folder = os.path.join("latents",'category_'+category_name)
        save_folder = os.path.join("latents_subject","run_"+str(run))
        save_folder_mot = os.path.join(args.out_dir, "mot_visualization", "latents_subject_" + "run_"+str(run))
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        z,score = get_optimized_z(category=i)
        
        decoded_z = decode_latent(net,z)

        bs = decoded_z.shape[0]
        bs = min(bs,args.min_samples)
        for j in range(bs):
            entry = decoded_z[j]
            file_path = os.path.join(save_folder,f'entry_{j}.npy')
            print(f"Saving results in file:{file_path}")
            np.save(file_path, entry.cpu().detach().numpy())
            
            pred_motion_saved = np.load(file_path)
            mot_file_path = os.path.join(save_folder_mot,f'entry_{j}.mot')
            write_mot35(mot_file_path, pred_motion_saved)
            print(f"Saving mot in file:{mot_file_path}")
        # np.save(os.path.join(args.out_dir,f'scores_{category_name}.npy'), score.cpu().detach().numpy())

        del z 
        del score
        del decoded_z
