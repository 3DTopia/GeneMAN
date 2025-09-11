
config_path="GeneMAN/outputs/geneman-texture-refine/0759_20250730/configs/parsed.yaml"
ckpt="GeneMAN/outputs/geneman-texture-refine/0759_20250730/ckpts/last.ckpt"
python launch.py --config $config_path \
             --export --gpu 0 resume=$ckpt   system.exporter_type=mesh-exporter system.exporter.fmt=obj-mtl
