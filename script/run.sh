 
data_dir="data/examples_processed"
timestamp="_20250909"
exp_root_dir="./outputs/examples_processed"

 
for image_path in "$data_dir"/*_caption.txt; do   
    image_id=$(basename "$image_path" | sed 's/_caption\.txt$//')
    
    echo "Processing image: $image_id"
             
    # preprocessed results
    image_path="${data_dir}/${image_id}_fg.png"
    normal_path="${data_dir}/${image_id}_normal.png"
    keypoints_path="${data_dir}/${image_id}_landmarks.npy"

    # read prompt
    export prompt="`cat ${data_dir}/${image_id}_caption.txt| cut -d'|' -f1`"
    echo $prompt
 
    # --------- Stage 1. Geometry Initialization --------- #
    exp_name="geneman-geometry-init"
    python launch.py --config configs/geneman-geometry-init.yaml \
    --train tag="\"$image_id\"" timestamp="$timestamp" data.image_path="$image_path" \
    system.prompt_processor.prompt="$prompt" \
 
    # --------- export NeRF as mesh --------- #
    ckpt_path=$exp_root_dir/$exp_name/${image_id}${timestamp}/ckpts/last.ckpt
    python launch.py --config configs/geneman-geometry-init.yaml \
    --export tag="\"$image_id\"" timestamp=$timestamp exp_root_dir=$exp_root_dir \
    resume=$ckpt_path data.image_path="$image_path" \
    system.prompt_processor.prompt="$prompt" \
    system.exporter_type=mesh-exporter \
    system.exporter.save_texture=False system.exporter.save_uv=False \
    system.geometry.isosurface_method=mc-cpu \
    system.geometry.isosurface_resolution=256 \
 
    # --------- Stage 2. Geometry Sculpting --------- #
    mesh_path=$exp_root_dir/$exp_name/${image_id}${timestamp}/save/it5000-export/model.obj
    exp_name="geneman-geometry-sculpt"
    python launch.py --config configs/geneman-geometry-sculpt.yaml \
            --train tag="\"$image_id\"" timestamp=$timestamp data.sampling_type="full_body" exp_root_dir=$exp_root_dir \
            data.image_path="$image_path" data.normal_path="$normal_path" \
            data.keypoints_path="$keypoints_path" \
            system.prompt_processor.prompt="$prompt, black background, normal map" \
            system.prompt_processor_add.prompt="$prompt, black background, depth map" \
            system.prompt_processor.human_part_prompt=false \
            system.geometry.shape_init="mesh:$mesh_path"

    # --------- Stage 3. Coarse Texture --------- #
    ckpt_path=$exp_root_dir/$exp_name/${image_id}${timestamp}/ckpts/last.ckpt
    exp_name="geneman-texture-coarse"
    coarse_save_path=$exp_root_dir/$exp_name/${image_id}${timestamp}/rgb_cache
    python launch.py --config configs/geneman-texture-coarse.yaml \
            --train tag="\"$image_id\"" timestamp=$timestamp  exp_root_dir=$exp_root_dir\
            data.image_path="$image_path" data.normal_path="$normal_path" \
            data.keypoints_path="$keypoints_path" \
            data.random_camera.test_save_path=$coarse_save_path \
            system.prompt_processor.prompt="$prompt" \
            system.prompt_processor.human_part_prompt=false \
            system.geometry_convert_from=$ckpt_path \
 
    # --------- Stage 4. Refine Texture --------- #
    ckpt_path=$exp_root_dir/$exp_name/${image_id}${timestamp}/ckpts/last.ckpt
    exp_name="geneman-texture-refine"
    refine_save_path=$exp_root_dir/$exp_name/${image_id}${timestamp}/rgb_cache
    python launch.py --config configs/geneman-texture-refine.yaml \
            --train tag="\"$image_id\"" timestamp=$timestamp data.image_path="$image_path"  exp_root_dir=$exp_root_dir\
            data.normal_path="$normal_path" \
            data.keypoints_path="$keypoints_path" \
            data.random_camera.dataroot=$coarse_save_path \
            system.prompt_processor.prompt="$prompt" \
            system.prompt_processor.human_part_prompt=false \
            system.geometry_convert_from=$ckpt_path

done