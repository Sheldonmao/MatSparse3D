#  Export the trained model to mesh 
exp_folder=path/to/saved/experiment # path to the experiment folder
config=$exp_folder/configs/parsed.yaml
ckpt=$exp_folder/ckpts/last.ckpt

python launch.py --config $config --export --gpu 0 \
            resume=$ckpt \
            name=export