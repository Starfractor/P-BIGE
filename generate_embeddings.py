import os
import json

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import models.vqvae as vqvae
import options.option_vq as option_limo
import utils.utils_model as utils_model
from dataset import dataset_TM_eval
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
import numpy as np

args = option_limo.get_args_parser()
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

val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 1, w_vectorizer, unit_length=2**args.down_t,data_root=args.data_root, mode='train')

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

# Create codebook vectors 
z = np.arange(args.nb_code).reshape((1, args.nb_code))
# z = np.arange(args.nb_code).reshape((args.nb_code,1)) ## This will give the same result as the above line
z = torch.from_numpy(z).to(device)
z = net.vqvae.quantizer.dequantize(z)    
z = z.cpu().detach().numpy()
z = z.reshape(args.nb_code, args.code_dim)
np.save(os.path.join(args.out_dir, 'codebook.npy'), z)


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

os.makedirs(os.path.join(args.out_dir, 'embeddings'), exist_ok = True)
for k,v in action_to_desc.items():
    array = np.array(v)
    print(array.shape)
    np.save(os.path.join(args.out_dir, 'embeddings',f"{k}.npy"),array)



def load_embeddings():
    directory = os.path.join(args.out_dir, 'embeddings')
    
    embedding_dict = {}
    
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            key = filename.split('.')[0]
            embedding = np.load(os.path.join(args.out_dir, 'embeddings',filename))
            
            embedding_dict[key] = embedding
    
    return embedding_dict



### Plot T-SNE

import plotly.express as px
import pandas as pd
import plotly.graph_objects as go


train_embeddings = load_embeddings()
codebook_embeddings = np.load(os.path.join(args.out_dir, 'codebook.npy'))

percentile_values_dict = {}
for k in train_embeddings:
    distances = np.linalg.norm(codebook_embeddings[:,np.newaxis] - train_embeddings[k], axis = 2) 
    closest_codebook_ind = np.argmin(distances, axis = 0)   

    # Step 1: Count the occurrences of each index
    unique, counts = np.unique(closest_codebook_ind, return_counts=True)

    # Step 2: Sort these counts by their occurrence
    sorted_indices = np.argsort(counts)[::-1]  # Sort in descending order
    sorted_counts = 100*counts[sorted_indices]/sum(counts)
    sorted_unique = unique[sorted_indices]

    # Step 3: Plot the sorted counts
    fig = go.Figure(data=go.Bar(x=sorted_unique, y=sorted_counts))
    fig.update_layout(title='Distribution of Closest Codebook Vector Indices for {}'.format(k),
                    xaxis_title='Codebook Vector Index',
                    yaxis_title='Percentage of Occurrence',
                    xaxis={'type': 'category'})
    # fig.show()
    fig.write_html(os.path.join(args.out_dir,'Distribution of Closest Codebook Vector Indices for {}'.format(k)))


    # Calculate the required percentiles and median
    closest_codebook_distance = np.min(distances, axis=0)

    closest_distance = np.min(closest_codebook_distance)
    tenth_percentile = np.percentile(closest_codebook_distance, 10)
    mean_distance = np.mean(closest_codebook_distance)
    median_distance = np.median(closest_codebook_distance)
    ninetieth_percentile = np.percentile(closest_codebook_distance, 90)
    
    # Store the values in the dictionary
    percentile_values_dict[k] = {
        'Closest': closest_distance,
        '10th Percentile': tenth_percentile,
        'Median': median_distance,
        'Mean': mean_distance,
        '90th Percentile': ninetieth_percentile
    }
    

# Plotting
categories = list(percentile_values_dict.keys())
percentiles = ['Closest', '10th Percentile', 'Median', 'Mean',  '90th Percentile']

# Create a figure
fig = go.Figure()

for percentile in percentiles:
    fig.add_trace(go.Bar(
        x=categories,
        y=[percentile_values_dict[category][percentile] for category in categories],
        name=percentile
    ))

# Update layout
fig.update_layout(
    barmode='group',
    title='Distance Percentiles for Each Category',
    xaxis_title='Category',
    yaxis_title='Distance',
    legend_title='Percentiles'
)
fig.write_html(os.path.join(args.out_dir,"Distance-Percentiles-for-Each-Category.html"))

# fig.show()

# TSNE: Did not help in visualization. Embeddings for all samples look equally distributed. 
# from sklearn.manifold import TSNE
# fit_vector = codebook_embeddings
# fit_label = ['codebook']*codebook_embeddings.shape[0]

# for k in train_embeddings:
#     if train_embeddings[k].shape[0] > 500:
#         sample_inds = np.random.choice(train_embeddings[k].shape[0], 500, replace = True)
#     else: 
#         sample_inds = np.arange(train_embeddings[k].shape[0])
#     fit_label += [k]*len(sample_inds)

#     fit_vector = np.concatenate([fit_vector, train_embeddings[k][sample_inds]], axis = 0)

# fit_vector_2D = TSNE_model = TSNE(n_components=2, random_state=0).fit_transform(fit_vector)

# df = pd.DataFrame(fit_vector_2D, columns=['Dimension 1', 'Dimension 2'])
# df['Label'] = fit_label

# # Use Plotly Express to create a scatter plot
# fig = px.scatter(df, x='Dimension 1', y='Dimension 2', color='Label',
#                  title='2D TSNE Visualization of Embeddings',
#                  labels={'Dimension 1': 'TSNE-1', 'Dimension 2': 'TSNE-2'})

# # Show the plot
# fig.show()


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

