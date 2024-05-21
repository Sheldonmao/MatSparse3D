#  test the model on by rendering test views
exp_folder=path/to/saved/experiment # path to the experiment folder
config=$exp_folder/configs/parsed.yaml
ckpt=$exp_folder/ckpts/last.ckpt

python launch.py --config $config --test --gpu 0 \
            resume=$ckpt \
            name=test
