img_dir=data/examples
out_dir=data/examples_processed

python preprocessing.py $img_dir \
    --output_path $out_dir \
    --recenter --enable_captioning