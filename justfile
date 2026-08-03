# Lists all available recipes
default:
    @just --list
grasp:
python demo.py eval.checkpoint=weights/m2t2.pth eval.data_dir=sample_data/real_world/00 eval.mask_thresh=0.4 eval.num_runs=5
