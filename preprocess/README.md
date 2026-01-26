## Data Preprocessing for MIGS

### Install dependencies

Install 4D-Humans and PHALP. Follow their instructions: https://github.com/shubham-goel/4D-Humans

Install arah-release. Follow their instructions: https://github.com/taconite/arah-release

The repos will be under the same parent directory, e.g.: `/path/to/repos/migs/`, `/path/to/repos/4D-Humans/`, `/path/to/repos/PHALP/`, `/path/to/repos/arah-release/`.

### Preprocess

An example pre-processing script is given as `./run.sh`. Copy it under the 4D-Humans repo: `/path/to/repos/4D-Humans/`.

Copy the provided `./preprocess_AIST_4DHumans.py` under the arah-release repo: `/path/to/repos/arah-release/preprocess_datasets/preprocess_AIST_4DHumans.py`.

Optionally, you can change some paths under `/path/to/repos/PHALP/phalp/configs/base.py`, e.g. we used `CACHE_DIR = /path/to/datasets/` in L8, `output_dir: str = f"{CACHE_DIR}/PHALP_outputs/` in L13, `MODEL_PATH: str = f"{CACHE_DIR}/smpl/"` in L87.

Add these lines in `/path/to/repos/PHALP/phalp/visualize/visualizer.py` after L116:
```
image = np.zeros_like(image)
image[idx[0], idx[1], :] = 255.
return image.astype(np.uint8)
```
And comment out L411-414.

Assuming the AIST++ videos are under `/path/to/datasets/AIST/videos`, we created a new directory `/path/to/datasets/AIST/videos_crop` to save the 1080x1080 videos (cropped from the original videos). Crop the videos, e.g.:
```shell
ffmpeg -i /path/to/datasets/AIST/videos/gBR_sBM_c01_d04_mBR1_ch05.mp4 -vf "crop=1080:1080:420:0" /path/to/datasets/AIST/videos_crop/gBR_sBM_c01_d04_mBR1_ch05_crop1080.mp4
```

Change the paths to your paths in `run.sh`, e.g. `/path/to/datasets/`.

```shell
cd 4D-Humans
bash run.sh
```


