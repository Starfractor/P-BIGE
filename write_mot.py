import os
import numpy as np
from glob import glob
# from osim_sequence import OSIMSequence



def write_mot33_simulation(path, data, framerate=60):
    # header_string = f"Coordinates\nversion=1\nnRows={data.shape[0]}\nnColumns=34\ninDegrees=yes\n\nUnits are S.I. units (second, meters, Newtons, ...)\nIf the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).\n\nendheader\ntime	pelvis_tilt	pelvis_list	pelvis_rotation	pelvis_tx	pelvis_ty	pelvis_tz	hip_flexion_r	hip_adduction_r	hip_rotation_r	knee_angle_r	ankle_angle_r	subtalar_angle_r	mtp_angle_r	hip_flexion_l	hip_adduction_l	hip_rotation_l	knee_angle_l	ankle_angle_l	subtalar_angle_l	mtp_angle_l	lumbar_extension	lumbar_bending	lumbar_rotation	arm_flex_r	arm_add_r	arm_rot_r	elbow_flex_r	pro_sup_r	arm_flex_l	arm_add_l	arm_rot_l	elbow_flex_l	pro_sup_l\n"
    
    header_string = f"Coordinates\nversion=1\nnRows={data.shape[0]}\nnColumns=34\ninDegrees=yes\n\nUnits are S.I. units (second, meters, Newtons, ...)\nIf the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).\n\nendheader\ntime	pelvis_tilt	pelvis_list	pelvis_rotation	pelvis_tx	pelvis_ty	pelvis_tz	hip_flexion_l	hip_adduction_l	hip_rotation_l	hip_flexion_r	hip_adduction_r	hip_rotation_r	knee_angle_l	knee_angle_r	ankle_angle_l	ankle_angle_r	subtalar_angle_l	subtalar_angle_r	mtp_angle_l	mtp_angle_r	lumbar_extension	lumbar_bending	lumbar_rotation	arm_flex_l	arm_add_l	arm_rot_l	arm_flex_r	arm_add_r	arm_rot_r	elbow_flex_l	elbow_flex_r	pro_sup_l	pro_sup_r\n"

    with open(path, 'w') as f:
        f.write(header_string)
        for i,d in enumerate(data):
            d = [str(i/60)] + [str(x) for x in d]
            
            # print(d)
            d = "      " +  "\t     ".join(d) + "\n"
            # print(d)
            f.write(d)

# writes data in 196x35 format
def write_mot35(path, data, framerate=60):
    os.makedirs(os.path.dirname(path),exist_ok=True)

    header_string = f"Coordinates\nversion=1\nnRows={data.shape[0]}\nnColumns=36\ninDegrees=yes\n\nUnits are S.I. units (second, meters, Newtons, ...)\nIf the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).\n\nendheader\ntime	pelvis_tilt	pelvis_list	pelvis_rotation	pelvis_tx	pelvis_ty	pelvis_tz	hip_flexion_r	hip_adduction_r	hip_rotation_r	knee_angle_r	knee_angle_r_beta	ankle_angle_r	subtalar_angle_r	mtp_angle_r	hip_flexion_l	hip_adduction_l	hip_rotation_l	knee_angle_l	knee_angle_l_beta	ankle_angle_l	subtalar_angle_l	mtp_angle_l	lumbar_extension	lumbar_bending	lumbar_rotation	arm_flex_r	arm_add_r	arm_rot_r	elbow_flex_r	pro_sup_r	arm_flex_l	arm_add_l	arm_rot_l	elbow_flex_l	pro_sup_l\n"

    if data.shape[-1] == 36:
        print(f"WARNING [write_mot.py]: Assuming first channel is time, because data should be Tx35, found:{data.shape}. Check file data for file:{path}") 
        data = data[:,1:]

    
    assert data.shape[-1] == 35, f"Data shape should be Tx35 found:{data.shape}."

    with open(path, 'w') as f:
        f.write(header_string)
        for i,d in enumerate(data):
            d = [str(i/60)] + [str(x) for x in d]
            
            # print(d)
            d = "      " +  "\t     ".join(d) + "\n"
            # print(d)
            f.write(d)

#writes data in 196x33 format, remove knee_angle_r_beta(idx 10) and knee_angle_l_beta(idx 18); idx is assuming 0-indexing          
def write_mot33(path, data, framerate=60):
    os.makedirs(os.path.dirname(path),exist_ok=True)

    header_string = f"Coordinates\nversion=1\nnRows={data.shape[0]}\nnColumns=36\ninDegrees=yes\n\nUnits are S.I. units (second, meters, Newtons, ...)\nIf the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).\n\nendheader\ntime	pelvis_tilt	pelvis_list	pelvis_rotation	pelvis_tx	pelvis_ty	pelvis_tz	hip_flexion_r	hip_adduction_r	hip_rotation_r	knee_angle_r	ankle_angle_r	subtalar_angle_r	mtp_angle_r	hip_flexion_l	hip_adduction_l	hip_rotation_l	knee_angle_l	ankle_angle_l	subtalar_angle_l	mtp_angle_l	lumbar_extension	lumbar_bending	lumbar_rotation	arm_flex_r	arm_add_r	arm_rot_r	elbow_flex_r	pro_sup_r	arm_flex_l	arm_add_l	arm_rot_l	elbow_flex_l	pro_sup_l\n"

    if data.shape[-1] == 34:
        print(f"WARNING [write_mot.py]: Assuming first channel is time, because data should be Tx33, found:{data.shape}. Check file data for file:{path}") 
        data = data[:,1:]

    indices_to_keep = [i for i in range(data.shape[1]) if i not in [10,18]]
    data = data[:,indices_to_keep]
    
    assert data.shape[-1] == 33, f"Data shape should be Tx33 found:{data.shape}."

    with open(path, 'w') as f:
        f.write(header_string)
        for i,d in enumerate(data):
            d = [str(i/60)] + [str(x) for x in d]
            
            # print(d)
            d = "      " +  "\t     ".join(d) + "\n"
            # print(d)
            f.write(d)
            
            
def write_muscle_activations(path, data, framerate=60): 
    os.makedirs(os.path.dirname(path),exist_ok=True)
    header_string = f"Coordinates\nversion=1\nnRows={data.shape[0]}\nnColumns=92\ninDegrees=yes\n\nUnits are S.I. units (second, meters, Newtons, ...)\nIf the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).\n\nendheader\ntime	addbrev_l/activation	addlong_l/activation	addmagDist_l/activation	addmagIsch_l/activation	addmagMid_l/activation	addmagProx_l/activation	bflh_l/activation	bfsh_l/activation	edl_l/activation	ehl_l/activation	fdl_l/activation	fhl_l/activation	gaslat_l/activation	gasmed_l/activation	glmax1_l/activation	glmax2_l/activation	glmax3_l/activation	glmed1_l/activation	glmed2_l/activation	glmed3_l/activation	glmin1_l/activation	glmin2_l/activation	glmin3_l/activation	grac_l/activation	iliacus_l/activation	perbrev_l/activation	perlong_l/activation	piri_l/activation	psoas_l/activation	recfem_l/activation	sart_l/activation	semimem_l/activation	semiten_l/activation	soleus_l/activation	tfl_l/activation	tibant_l/activation	tibpost_l/activation	vasint_l/activation	vaslat_l/activation	vasmed_l/activation	addbrev_r/activation	addlong_r/activation	addmagDist_r/activation	addmagIsch_r/activation	addmagMid_r/activation	addmagProx_r/activation	bflh_r/activation	bfsh_r/activation	edl_r/activation	ehl_r/activation	fdl_r/activation	fhl_r/activation	gaslat_r/activation	gasmed_r/activation	glmax1_r/activation	glmax2_r/activation	glmax3_r/activation	glmed1_r/activation	glmed2_r/activation	glmed3_r/activation	glmin1_r/activation	glmin2_r/activation	glmin3_r/activation	grac_r/activation	iliacus_r/activation	perbrev_r/activation	perlong_r/activation	piri_r/activation	psoas_r/activation	recfem_r/activation	sart_r/activation	semimem_r/activation	semiten_r/activation	soleus_r/activation	tfl_r/activation	tibant_r/activation	tibpost_r/activation	vasint_r/activation	vaslat_r/activation	vasmed_r/activation\n"
    
    assert data.shape[-1] == 80, f"Data shape should be Tx80 found:{data.shape}."
    
    with open(path, 'w') as f:
        f.write(header_string)
        for i,d in enumerate(data):
            d = [str(i/60)] + [str(x) for x in d]
            
            # print(d)
            d = "      " +  "\t     ".join(d) + "\n"
            # print(d)
            f.write(d)
            
    

# files = glob("latents/*/*.npy")
files = glob("/data/panini/T2M-GPT/latents_subject/run_1/*.npy")
# files = glob("/data/panini/T2M-GPT/train_forward_pass/model_output/*.npy")
print(len(files))

for f in files:
    name = f.split('/')[-1]
    data = np.load(f)
    write_mot33("mot_visualization/latents_subject_run1/"+ name.split('.')[0] + ".mot", data)
    # write_mot("/data/panini/T2M-GPT/mot_visualization/constrained_latents/"+ name.split('.')[0] + ".mot", data)
    # write_mot("/data/panini/T2M-GPT/train_forward_pass/mot_output/"+ name.split('.')[0] + ".mot", data)
    
# files = glob("mot_visualization/*.mot")
# for f in files:
#     print(f)
#     OSIMSequence.from_files("/home/ubuntu/data/opencap-processing/Data/0d9e84e9-57a4-4534-aee2-0d0e8d1e7c45/OpenSimData/Model/LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim", "/home/ubuntu/data/T2M-GPT/"+ f)