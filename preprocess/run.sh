for f in "gJS_sBM_c01_d02_mJS0_ch06_crop1080" "gBR_sBM_c01_d04_mBR1_ch05_crop1080" "gBR_sBM_c01_d06_mBR2_ch06_crop1080" "gJB_sBM_c01_d08_mJB0_ch05_crop1080" "gJB_sBM_c01_d08_mJB0_ch10_crop1080" "gPO_sBM_c01_d12_mPO4_ch06_crop1080" "gLO_sBM_c01_d13_mLO1_ch07_crop1080" "gLO_sBM_c01_d14_mLO0_ch01_crop1080" "gLO_sBM_c01_d15_mLO2_ch08_crop1080" "gHO_sBM_c01_d19_mHO0_ch10_crop1080"

do
    if [ ! -d "/data/aggelina/datasets/zju_mocap_arah/${f}/models" ]; then
        echo $f
        CUDA_VISIBLE_DEVICES=0 python track.py video.source="/data/aggelina/datasets/AIST/videos_crop/${f}.mp4" render.type=HUMAN_MASK render.fps=60 render.output_resolution=1080
            name=$f
        cp -r /data/aggelina/datasets/PHALP_outputs/_DEMO/${name} /data/aggelina/datasets/zju_mocap_arah/${name}
        mv /data/aggelina/datasets/zju_mocap_arah/${name}/img/  /data/aggelina/datasets/zju_mocap_arah/${name}/1/
        ffmpeg -i /data/aggelina/datasets/PHALP_outputs/PHALP_${name}.mp4 /data/aggelina/datasets/zju_mocap_arah/${name}/1/%06d.png
        cp /data/aggelina/datasets/PHALP_outputs/results/demo_${name}.pkl /data/aggelina/datasets/zju_mocap_arah/${name}/${name}.pkl
            
        cd ../arah-release
        python preprocess_datasets/preprocess_AIST_4DHumans.py --data_dir /data/aggelina/datasets/zju_mocap_arah/ --seqname ${name}
        cd ../4D-Humans
    fi
done