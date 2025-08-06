import os
import pickle as pkl
import numpy as np

path_to_smpl = "/path/to/datasets/smpl/"

if __name__ == '__main__':
    male_path = f'{path_to_smpl}/models_nochumpy/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
    female_path = f'{path_to_smpl}/models_nochumpy/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
    neutral_path = f'{path_to_smpl}/models_nochumpy/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'

    data_m = pkl.load(open(male_path, 'rb'), encoding='latin1')
    data_f = pkl.load(open(female_path, 'rb'), encoding='latin1')
    data_n = pkl.load(open(neutral_path, 'rb'), encoding='latin1')

    if not os.path.exists(f'{path_to_smpl}/misc'):
        os.makedirs(f'{path_to_smpl}/misc')

    np.savez(f'{path_to_smpl}/misc/faces.npz', faces=data_m['f'].astype(np.int64))
    np.savez(f'{path_to_smpl}/misc/J_regressors.npz', male=data_m['J_regressor'].toarray(), female=data_f['J_regressor'].toarray(), neutral=data_n['J_regressor'].toarray())
    np.savez(f'{path_to_smpl}/misc/posedirs_all.npz', male=data_m['posedirs'], female=data_f['posedirs'], neutral=data_n['posedirs'])
    np.savez(f'{path_to_smpl}/misc/shapedirs_all.npz', male=data_m['shapedirs'], female=data_f['shapedirs'], neutral=data_n['shapedirs'])
    np.savez(f'{path_to_smpl}/misc/skinning_weights_all.npz', male=data_m['weights'], female=data_f['weights'], neutral=data_n['weights'])
    np.savez(f'{path_to_smpl}/misc/v_templates.npz', male=data_m['v_template'], female=data_f['v_template'], neutral=data_n['v_template'])
    np.save(f'{path_to_smpl}/misc/kintree_table.npy', data_m['kintree_table'].astype(np.int32))
