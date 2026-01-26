import os
import sys
import torch
# import trimesh
import glob
import json
import shutil
import argparse
import joblib
import tqdm

sys.path.insert(0, "./")

import numpy as np
import preprocess_datasets.easymocap.mytools.camera_utils as cam_utils

from scipy.spatial.transform import Rotation

from human_body_prior.body_model.body_model import BodyModel

from preprocess_datasets.easymocap.smplmodel import load_model

parser = argparse.ArgumentParser(
    description='Preprocessing for AIST.'
)
parser.add_argument('--data_dir', type=str, help='Directory that contains AIST data.')
parser.add_argument('--seqname', type=str, help='Sequence to process.')

if __name__ == '__main__':
    args = parser.parse_args()
    seq_name = args.seqname
    data_dir = os.path.join(args.data_dir, seq_name)

    pkl_file = os.path.join(data_dir, seq_name + ".pkl") 
    results = joblib.load(pkl_file)

    body_model = BodyModel(bm_path='/data/aggelina/datasets/smpl/models_nochumpy/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl', num_betas=10, batch_size=1).cuda()

    faces = np.load('/data/aggelina/datasets/smpl/misc/faces.npz')['faces']

    cam_names = ["1"]

    all_cam_params = {'all_cam_names': cam_names}
    smpl_out_dir = os.path.join(data_dir, 'models')
    if not os.path.exists(smpl_out_dir):
        os.makedirs(smpl_out_dir)
 
    focal_length = 5000
    width = 1080 #1920 #1024  # 640
    height = 1080 #1024  # 640
    up_scale = 4 #7 #4  # 2
    #focal_length = focal_length / 1920

    K = np.eye(3)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = width / 2
    K[1, 2] = height / 2
    #camera_center = np.zeros((2,))
    #K[:-1, -1] = camera_center

    R = np.eye(3)
    T = np.zeros((3,1))
    cam_params = {'K': K.tolist(), 'R': R.tolist(), 'T': T.tolist()}

    all_cam_params.update({"1": cam_params})

    for img_path, img_res in tqdm.tqdm(results.items()):
        #img_name = img_res["img_name"][0]
        img_name = os.path.basename(img_path)
        try:
            tracked_camera = img_res["camera"][0]
        except:
            print("no camera for {}".format(idx))
            continue
        smpl_params = img_res["smpl"][0]
        idx = int(img_name.replace(".jpg", "")) - 1

        t = np.array(tracked_camera)
        t[2] = t[2] / up_scale

        img_file = os.path.join(data_dir, "1", img_name)
        new_img_file = os.path.join(data_dir, "1", '{:06d}.jpg'.format(idx))
        mask_file = os.path.join(data_dir, "1", img_name.replace(".jpg", ".png"))
        new_mask_file = os.path.join(data_dir, "1", '{:06d}.png'.format(idx))
        os.system("mv {} {}".format(img_file, new_img_file))
        os.system("mv {} {}".format(mask_file, new_mask_file))

        global_orient = np.array(smpl_params['global_orient'], dtype=np.float32)
        # root_orient = Rotation.from_matrix(global_orient).as_rotvec().astype(np.float32)
        betas = np.array(smpl_params['betas'], dtype=np.float32)[None]
        poses = np.array(smpl_params['body_pose'], dtype=np.float32)
        poses = np.concatenate((global_orient, poses), axis=0)
        poses = Rotation.from_matrix(poses).as_rotvec().astype(np.float32).reshape(-1)[None]
        pose_body = poses[:, 3:66].copy()
        pose_hand = poses[:, 66:].copy()

        poses_torch = torch.from_numpy(poses).cuda()
        pose_body_torch = torch.from_numpy(pose_body).cuda()
        pose_hand_torch = torch.from_numpy(pose_hand).cuda()
        betas_torch = torch.from_numpy(betas).cuda()        
    
        root_orient = global_orient.copy()
        #root_orient = Rotation.from_rotvec(np.array(params['Rh']).reshape([-1])).as_matrix()
        #trans = np.array(params['Th']).reshape([3, 1])
        
        new_root_orient = Rotation.from_matrix(root_orient).as_rotvec().reshape([1, 3]).astype(np.float32)
        new_trans = np.zeros((1, 3), dtype=np.float32) #
        #new_trans = t.reshape([1, 3]).astype(np.float32)

        new_root_orient_torch = torch.from_numpy(new_root_orient).cuda()
        new_trans_torch = torch.from_numpy(new_trans).cuda()

        # Get shape vertices
        body = body_model(betas=betas_torch)
        minimal_shape = body.v.detach().cpu().numpy()[0]

        # Get bone transforms
        body = body_model(root_orient=new_root_orient_torch, pose_body=pose_body_torch, pose_hand=pose_hand_torch, betas=betas_torch, trans=new_trans_torch)

        #body_model_em = load_model(gender='neutral', model_type='smpl', model_path='/home/ubuntu/data/smpl/')
        #verts = body_model_em(poses=poses_torch, shapes=betas_torch, Rh=new_root_orient_torch, Th=new_trans_torch, return_verts=True)[0].detach().cpu().numpy()

        #vertices = body.v.detach().cpu().numpy()[0]
        #new_trans = new_trans + (verts - vertices).mean(0, keepdims=True)
        #new_trans_torch = torch.from_numpy(new_trans).cuda()

        #body = body_model(root_orient=new_root_orient_torch, pose_body=pose_body_torch, pose_hand=pose_hand_torch, betas=betas_torch, trans=new_trans_torch)

        ## Visualize SMPL mesh
        #import trimesh
        #import pdb; pdb.set_trace()
        #vertices = body.v.detach().cpu().numpy()[0]
        #mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        #rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        #mesh.apply_transform(rot)
        #out_filename = os.path.join(smpl_out_dir, img_name.replace(".jpg", ".ply"))
        #mesh.export(out_filename)

        bone_transforms = body.bone_transforms.detach().cpu().numpy()
        Jtr_posed = body.Jtr.detach().cpu().numpy()

        out_filename = os.path.join(smpl_out_dir, img_name.replace(".jpg", ".npz"))

        np.savez(out_filename,
                 minimal_shape=minimal_shape,
                 betas=betas,
                 Jtr_posed=Jtr_posed[0],
                 bone_transforms=bone_transforms[0],
                 t=t,
                 trans=new_trans[0],
                 root_orient=new_root_orient[0],
                 pose_body=pose_body[0],
                 pose_hand=pose_hand[0])


    with open(os.path.join(data_dir, 'cam_params.json'), 'w') as f:
        json.dump(all_cam_params, f)
