import torch
import numpy as np
import os
from tqdm import tqdm, trange
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import fix_random
from scene import MultiScene, MultiGaussianModel

from utils.general_utils import Evaluator, PSEvaluator

import hydra
from omegaconf import OmegaConf
import wandb

def predict(config):
    with torch.set_grad_enabled(False):
        gaussians = MultiGaussianModel(config.model.gaussian, subjects=config.dataset.subjects)
        scene = MultiScene(config, gaussians, config.exp_dir)
        scene.eval()
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = os.path.join(scene.save_dir, "ckpt" + str(config.opt.iterations) + ".pth")
        print("Loading {}".format(load_ckpt))
        scene.load_checkpoint(load_ckpt)

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(config.exp_dir, config.suffix, 'renders')
        makedirs(render_path, exist_ok=True)

        #iter_start = torch.cuda.Event(enable_timing=True)
        #iter_end = torch.cuda.Event(enable_timing=True)
        #times = []
        for idx in trange(len(scene.test_dataset), desc="Rendering progress"):
            view = scene.test_dataset[idx]
        #    iter_start.record()

            render_pkg = render(view, config.opt.iterations, scene, config.pipeline, background,
                                compute_loss=False, return_opacity=False, multigaussian=True)
         #   iter_end.record()
         #   torch.cuda.synchronize()
         #   elapsed = iter_start.elapsed_time(iter_end)

            rendering = render_pkg["render"]

            wandb_img = [wandb.Image(rendering[None], caption='render_{}'.format(view.image_name)),]
            wandb.log({'test_images': wandb_img})

            torchvision.utils.save_image(rendering, os.path.join(render_path, f"render_{view.image_name}.png"))

            # evaluate
         #   times.append(elapsed)

        #_time = np.mean(times[1:])
        #wandb.log({'metrics/time': _time})
        np.savez(os.path.join(config.exp_dir, config.suffix, 'results.npz'))
         #        time=_time)



def test(config):
    if config.dataset.test_subject not in config.dataset.subjects:
        dancer_test = config.dataset.test_subject.split("_")[3]
        idx_dancer = [idx for idx, name in enumerate(config.dataset.subjects) if name.split("_")[3] == dancer_test][0]
        config.dataset.subjects[idx_dancer] = config.dataset.test_subject
    with torch.no_grad():
        gaussians = MultiGaussianModel(config.model.gaussian, subjects=config.dataset.subjects)
        scene = MultiScene(config, gaussians, config.exp_dir)
        scene.eval()
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = os.path.join(scene.save_dir, "ckpt" + str(config.opt.iterations) + ".pth")
        scene.load_checkpoint(load_ckpt)

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(config.exp_dir, config.suffix, 'renders')
        makedirs(render_path, exist_ok=True)

        #iter_start = torch.cuda.Event(enable_timing=True)
        #iter_end = torch.cuda.Event(enable_timing=True)

        evaluator = PSEvaluator() if config.dataset.name == 'people_snapshot' else Evaluator()

        psnrs = []
        ssims = []
        lpipss = []
        #times = []
        test_dataset = scene.test_dataset if config.mode == "test" else scene.train_dataset
        for idx in trange(len(test_dataset), desc="Rendering progress"):
            view = test_dataset[idx]
        #    iter_start.record()

            render_pkg = render(view, config.opt.iterations, scene, config.pipeline, background,
                                compute_loss=False, return_opacity=False, multigaussian=True)

        #    iter_end.record()
    #        torch.cuda.synchronize()
        #    elapsed = iter_start.elapsed_time(iter_end)

            rendering = render_pkg["render"]

            gt = view.original_image[:3, :, :]
            mask = view.original_mask[0]

            wandb_img = [wandb.Image(rendering[None], caption='render_{}'.format(view.image_name)),
                         wandb.Image(gt[None], caption='gt_{}'.format(view.image_name))]

            wandb.log({'test_images': wandb_img})

            torchvision.utils.save_image(rendering, os.path.join(render_path, f"render_{view.image_name}.png"))

            # evaluate
            if config.evaluate:
                metrics = evaluator(rendering, gt) #, mask=mask)
                psnrs.append(metrics['psnr'])
                ssims.append(metrics['ssim'])
                lpipss.append(metrics['lpips'])
            else:
                psnrs.append(torch.tensor([0.], device='cuda'))
                ssims.append(torch.tensor([0.], device='cuda'))
                lpipss.append(torch.tensor([0.], device='cuda'))
        #    times.append(elapsed)

        _psnr = torch.mean(torch.stack(psnrs))
        _ssim = torch.mean(torch.stack(ssims))
        _lpips = torch.mean(torch.stack(lpipss))
        #_time = np.mean(times[1:])
        wandb.log({'metrics/psnr': _psnr,
                   'metrics/ssim': _ssim,
                   'metrics/lpips': _lpips})
        #           'metrics/time': _time})
        np.savez(os.path.join(config.exp_dir, config.suffix, 'results.npz'),
                 psnr=_psnr.cpu().numpy(),
                 ssim=_ssim.cpu().numpy(),
                 lpips=_lpips.cpu().numpy())
        #         time=_time)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    OmegaConf.set_struct(config, False)
    config.dataset.preload = False

    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    os.makedirs(config.exp_dir, exist_ok=True)

    # set wandb logger
    if config.mode == 'test' or config.mode == 'train':
        config.suffix = config.mode + '-' + config.dataset.test_mode
    elif config.mode == 'predict':
        predict_seq = config.dataset.predict_seq
        if 'zjumocap' in config.dataset.name:
            predict_dict = {
                0: 'dance0',
                1: 'dance1',
                2: 'flipping',
                3: 'canonical',
                4: 'gJB_sFM_cAll_d08_mJB4_ch12', #'gLO_sFM_cAll_d15_mLO4_ch19', #'gJB_sFM_cAll_d08_mJB4_ch12',
                5: '377',
                6: '386',
                7: '387',
                8: '392',
                9: '393',
                10: '394',
            }
        else:
            predict_dict = {
                0: 'rotation',
                1: 'dance2',
            }
        if isinstance(predict_seq, int):
            predict_seq = predict_dict[predict_seq]
        config.suffix = config.mode + '-' + predict_seq
    else:
        raise ValueError
    if config.dataset.freeview:
        config.suffix = config.suffix + '-freeview'
    if hasattr(config.dataset, "test_subject"):
        config.suffix = config.suffix + '-' + config.dataset.test_subject

    wandb_name = config.name + '-' + config.suffix
    wandb.init(
        mode="disabled" if config.wandb_disable else None,
        name=wandb_name,
        project='gaussian-splatting-avatar-multi-test',
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method='fork'),
    )

    fix_random(config.seed)

    if config.mode == 'test' or config.mode == 'train':
        test(config)
    elif config.mode == 'predict':
        predict(config)
    else:
        raise ValueError

if __name__ == "__main__":
    main()
