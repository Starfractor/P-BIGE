import os
import numpy as np
from glob import glob
# from osim_sequence import OSIMSequence

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