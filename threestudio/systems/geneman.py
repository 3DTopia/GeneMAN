from dataclasses import dataclass, field

import os
import json
import torch
import torch.nn.functional as F
from threestudio.utils.perceptual import PerceptualLoss
from threestudio.utils.misc import cleanup, get_device
from torchmetrics import PearsonCorrCoef

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

import imageio
import numpy as np
import random
import shutil

@threestudio.register("geneman-system")
class GeneMAN(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        stage: str = "coarse"  # coarse, geometry, texture
        sampling_type: str = "default"
        start_sdf_loss_step: int = -1
        freq: dict = field(default_factory=dict)
        guidance_3d_type: str = ""
        guidance_3d: dict = field(default_factory=dict)
        use_mixed_camera_config: bool = False
        control_guidance_type: str = ""
        control_guidance: dict = field(default_factory=dict)
        control_prompt_processor_type: str = ""
        control_prompt_processor: dict = field(default_factory=dict)
        visualize_samples: bool = False
        guidance_normal_type: str = ""
        guidance_normal: dict = field(default_factory=dict)
        guidance_3d_norm_type: str = ""
        guidance_3d_norm: dict = field(default_factory=dict)
        latent_steps: int = 1000
    
    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
    
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        
        # 3d-aware guidance
        if self.cfg.guidance_3d_type != "":
            self.guidance_3d = threestudio.find(self.cfg.guidance_3d_type)(
                self.cfg.guidance_3d
            )
        else:
            self.guidance_3d = None
            
        # depth guidance
        if len(self.cfg.guidance_add_type) > 0:
            self.guidance_add = threestudio.find(self.cfg.guidance_add_type)(self.cfg.guidance_add)
        else:
            self.guidance_add = None
        
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()
        
        if len(self.cfg.prompt_processor_add_type) > 0:
            self.prompt_processor_add = threestudio.find(self.cfg.prompt_processor_add_type)(
                self.cfg.prompt_processor_add
            )
            self.prompt_utils_add = self.prompt_processor_add()
        else:
            self.prompt_processor_add = None

        # loss term
        self.perceptual_loss = PerceptualLoss().to(self.device)
        self.pearson = PearsonCorrCoef().to(self.device)
        
        self.frames = []
        self.transforms = {
            "camera_model": "OPENCV",
            "orientation_override": "none",
        }

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_rgb = (self.cfg.stage != "geometry")
        render_out = self.renderer(**batch, render_rgb=render_rgb)
        return {
            **render_out,
        }
            
    def on_fit_start(self) -> None:
        super().on_fit_start()

        if self.cfg.stage == "geometry":  # initialize SDF
            self.geometry.initialize_shape()
     
    def training_substep(self, batch, batch_idx, guidance: str, render_type="rgb"):
        loss = 0.0
        
        if guidance == "ref":
            batch_ref = batch
            gt_normal = batch_ref["ref_normal"]
            gt_depth = batch_ref["ref_depth"]
            gt_mask = batch_ref["mask"]
            gt_rgb = batch_ref["rgb"]
            out = self(batch_ref)

        elif guidance == "guidance":
            batch = batch["random_camera"]
            out = self(batch)
        
        if self.true_global_step == self.cfg.start_sdf_loss_step:
            mesh_v_path = '.threestudio_cache/mesh_v_pos.npy'
            mesh_t_pos_idx_path = '.threestudio_cache/mesh_t_pos_idx.npy'
            mesh_v_save_path, mesh_t_save_path = self.save_mesh_npy(
                mesh_v_path, mesh_t_pos_idx_path, out['mesh']
            )
            self.geometry.cfg.mesh_v_path = mesh_v_save_path
            self.geometry.cfg.mesh_t_pos_idx_path = mesh_t_save_path
        
        # reference stage
        if guidance == "ref":
            if render_type == "rgb":
                if self.C(self.cfg.loss.lambda_rgb) > 0:
                    gt_rgb = gt_rgb * gt_mask.float() + out["comp_rgb_bg"] * (
                        1 - gt_mask.float()
                    )
                    pred_rgb = out["comp_rgb"]  
                    
                    if self.cfg.stage == "coarse":
                        loss += F.mse_loss(gt_rgb, pred_rgb) * self.C(self.cfg.loss.lambda_rgb)
                    else:      
                        grow_mask = F.max_pool2d(1 - gt_mask.float().permute(0, 3, 1, 2), (9, 9), 1, 4)
                        grow_mask = (1 - grow_mask).permute(0, 2, 3, 1)
                        loss += F.l1_loss(gt_rgb * grow_mask, pred_rgb * grow_mask) * self.C(
                            self.cfg.loss.lambda_rgb
                        )

                if self.cfg.stage == "coarse":
                    # mask loss
                    if self.C(self.cfg.loss.lambda_mask) > 0:
                        loss += F.mse_loss(gt_mask.float(), out["opacity"], reduction='sum') * self.C(
                            self.cfg.loss.lambda_mask
                        )

                    # mask binary cross loss
                    if self.C(self.cfg.loss.lambda_mask_binary) > 0:
                        loss += (
                            F.binary_cross_entropy(
                                out["opacity"].clamp(1.0e-5, 1.0 - 1.0e-5),
                                gt_mask.float()
                            ) * self.C(self.cfg.loss.lambda_mask_binary)
                        )
                
                    # depth loss
                    if self.C(self.cfg.loss.lambda_depth) > 0:
                        valid_gt_depth = gt_depth[gt_mask.squeeze(-1)].unsqueeze(1)
                        valid_pred_depth = out["depth"][gt_mask].unsqueeze(1)
                        with torch.no_grad():
                            A = torch.cat(
                                [valid_gt_depth, torch.ones_like(valid_gt_depth)], dim=-1
                            )  # [B, 2]
                            X = torch.linalg.lstsq(A, valid_pred_depth).solution  # [2, 1]
                            valid_gt_depth = A @ X  # [B, 1]
                        loss += F.mse_loss(valid_gt_depth, valid_pred_depth) * self.C(
                            self.cfg.loss.lambda_depth
                        )
                    
                    # relative depth loss
                    if self.C(self.cfg.loss.lambda_depth_rel) > 0:
                        valid_gt_depth = gt_depth[gt_mask.squeeze(-1)]  # [B,]
                        valid_pred_depth = out["depth"][gt_mask]  # [B,]
                        loss += (
                            (1 - self.pearson(valid_pred_depth, valid_gt_depth)) * self.C(self.cfg.loss.lambda_depth_rel)
                        )
                    
            else:
                if self.C(self.cfg.loss.lambda_normal) > 0:
                    pred_normal = out["comp_normal"]
                    valid_gt_normal = gt_normal[gt_mask.squeeze(-1)]  # [B, 3]
                    valid_pred_normal = pred_normal[gt_mask.squeeze(-1)]
                    # NOTE: convert to [-1, 1] for computing cos similarity
                    loss += (1 - F.cosine_similarity(valid_pred_normal * 2 - 1, valid_gt_normal * 2 - 1).mean()) * self.C(self.cfg.loss.lambda_normal) 
                    # loss += F.mse_loss(valid_pred_normal, valid_gt_normal) * self.C(self.cfg.loss.lambda_normal) 

                # perceptual loss
                if self.C(self.cfg.loss.lambda_vgg) > 0:
                    pred_normal_BCHW = (pred_normal * gt_mask).permute(0, 3, 1, 2)
                    gt_normal_BCHW = (gt_normal * gt_mask).permute(0, 3, 1, 2)
                    loss += self.perceptual_loss(pred_normal_BCHW, gt_normal_BCHW).mean() * self.C(self.cfg.loss.lambda_vgg) 
                    
                # mask loss
                if self.C(self.cfg.loss.lambda_mask) > 0:
                    loss += F.mse_loss(gt_mask.float(), out["opacity"], reduction='sum') * self.C(self.cfg.loss.lambda_mask)

        # guidance stage
        elif guidance == "guidance":
            if self.cfg.stage == "geometry" and render_type == "normal":
                guidance_inp = out["comp_normal"]
            else:
                guidance_inp = out["comp_rgb"]
                
            if self.cfg.stage == "coarse":  # coarse stage
                guidance_out = self.guidance(
                    guidance_inp, self.prompt_utils, **batch
                )
                
            elif self.cfg.stage == "geometry":  # geometry stage 
                # normal SDS loss
                guidance_out = self.guidance(
                    guidance_inp, self.prompt_utils, **batch
                )

                if self.guidance_add is not None and self.C(self.cfg.loss.lambda_sds_add) > 0:
                    guidance_inp_add = out["comp_depth"].repeat(1, 1, 1, 3)
                    guidance_out_add = self.guidance_add(
                        guidance_inp_add, self.prompt_utils_add, **batch
                    )
                    guidance_out.update({"loss_sds_add": guidance_out_add["loss_sds"]})
                
                else:
                    guidance_out.update({"loss_sds_add": 0})
                
            else:  # texture stage
                if isinstance(
                    self.guidance,
                    (
                        threestudio.models.guidance.controlnet_guidance.ControlNetGuidance,
                        threestudio.models.guidance.controlnet_vsd_guidance.ControlNetVSDGuidance,
                        threestudio.models.guidance.sds_du_controlnet_guidance.SDSDUControlNetGuidance,
                    ),
                ):
                    cond_inp = out["comp_normal"] # conditon for controlnet
                    guidance_out = self.guidance(
                        guidance_inp, cond_inp, self.prompt_utils, **batch
                    )
                else:
                    guidance_out = self.guidance(
                        guidance_inp, self.prompt_utils, **batch,
                    )

            # guidance loss
            for name, value in guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
            
            # 3d guidance
            if self.guidance_3d is not None and self.C(self.cfg.loss.lambda_sds_3d) > 0:
                guidance_out_3d = self.guidance_3d(
                    guidance_inp, **batch, rgb_as_latents=False
                )                    
                for name, value in guidance_out_3d.items():
                    self.log(f"train/{name}_3d", value)
                    if name.startswith("loss_"):
                        loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")+"_3d"])                

            """
            if (self.true_global_step + 1) % 1000 == 0:
                self.save_image_grid(
                    f"it{self.true_global_step}-train-guid.jpg",
                    (
                        [
                            {
                                "type": "rgb",
                                "img": out["comp_normal"][0] + (1 - out["opacity"][0, :, :, :]),
                                "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                            },
                        ]
                        if "comp_normal" in out
                            else []
                    )
                    + (
                        [
                            {
                                "type": "rgb",
                                "img": out["comp_rgb"][0] + (1 - out["opacity"][0, :, :, :]),
                                "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                            },
                        ]
                        if "comp_rgb" in out
                            else []                    
                    ),
                    name="train_step",
                    step=self.true_global_step,
                )
            """

        ### regularization loss
        if self.cfg.stage == "coarse":
            # normal smoothness loss
            if self.C(self.cfg.loss.lambda_normal_smooth) > 0:
                normal = out["comp_normal"]
                loss += (
                    (normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean()
                    + (normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean()
                ) * self.C(self.cfg.loss.lambda_normal_smooth)

            # 3d normal smoothness loss
            if self.C(self.cfg.loss.lambda_3d_normal_smooth) > 0:
                normals = out["normal"]
                normals_perturb = out["normal_perturb"]
                loss += (normals - normals_perturb).abs().mean() * self.C(self.cfg.loss.lambda_3d_normal_smooth)

            # orientation loss
            if self.C(self.cfg.loss.lambda_orient) > 0:
                loss += ((
                        out["weights"].detach()
                        * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                    ).sum()
                    / (out["opacity"] > 0).sum()
                ) * self.C(self.cfg.loss.lambda_orient)
  
            # orientation loss
            if guidance != "ref" and self.C(self.cfg.loss.lambda_sparsity) > 0:
                loss += (out["opacity"] ** 2 + 0.01).sqrt().mean() * self.C(self.cfg.loss.lambda_sparsity)

            # opaque loss
            if self.C(self.cfg.loss.lambda_opaque) > 0:
                opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
                loss += (
                    binary_cross_entropy(opacity_clamped, opacity_clamped)
                ) * self.C(self.cfg.loss.lambda_opaque)

        elif self.cfg.stage == "geometry" and guidance == "guidance":
            # SDF loss
            if out['sdf_loss'] is not None and self.C(self.cfg.loss.lambda_sdf) > 0:
                self.log("train/loss_sdf", out['sdf_loss'])
                loss += out['sdf_loss'] * self.C(self.cfg.loss.lambda_sdf)
            
            # normal consistency
            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )
            
            # laplacian smoothness
            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                loss_laplacian_smoothness = out["mesh"].laplacian()
                self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
                loss += loss_laplacian_smoothness * self.C(
                    self.cfg.loss.lambda_laplacian_smoothness
                )
              
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss = 0.0
        
        if self.cfg.freq.ref_or_guidance == "accumulate":
            do_ref = True
            do_guidance = True
        
        elif self.cfg.freq.ref_or_guidance == "alternate":
            do_ref = (
                self.true_global_step < self.cfg.freq.ref_only_steps
                or self.true_global_step % self.cfg.freq.n_ref == 0
            )
            do_guidance = not do_ref
            if hasattr(self.guidance.cfg, "only_pretrain_step"):
                if (self.guidance.cfg.only_pretrain_step > 0) and (self.global_step % self.guidance.cfg.only_pretrain_step) < (self.guidance.cfg.only_pretrain_step // 5):
                    do_guidance = True
                    do_ref = False        

        if self.cfg.stage == "geometry": 
            render_type = "normal"
        else:
            render_type = "rgb"

        # locate 3d keypoints at the start of the training
        if self.cfg.sampling_type != "default" and self.true_global_step == 0:
            
            # use back-projection to locate 3d keypoints
            self.renderer.locate_keypoints(**batch)

            # if keypoint is outside image bound, update sampling ratio
            invalid_indices = np.setdiff1d([1, 2, 3], list(self.renderer.keypoints_3d.keys()))
            if len(invalid_indices) > 0:
                body_part_ratio = np.asarray(self.trainer.train_dataloader.dataset.random_pose_generator.body_part_ratio)
                body_part_ratio[invalid_indices] = 0   # set the ratio to 0
                body_part_ratio /= body_part_ratio.sum()
                self.trainer.train_dataloader.dataset.random_pose_generator.body_part_ratio = body_part_ratio
                print(f'[INFO] reset the body part sampling ratio to {body_part_ratio}')         

        if do_guidance:
            out = self.training_substep(batch, batch_idx, guidance="guidance", render_type=render_type)
            loss += out["loss"]

        if do_ref:
            out = self.training_substep(batch, batch_idx, guidance="ref", render_type=render_type)
            loss += out["loss"]
    
        self.log("train/loss", loss, prog_bar=True)
        
        return {"loss": loss}                                      
       
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-val-color/{batch['index'][0]}.jpg",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0] + (1 - out["opacity"][0, :, :, :]),
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal_viewspace"][0] + (1 - out["opacity"][0, :, :, :]),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal_viewspace" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale", 
                        "img": out["depth"][0], 
                        "kwargs": {}
                    }
                ] 
                if "depth" in out
                else [] 
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-val-color",
            f"it{self.true_global_step}-val-color",
            "(\d+)\.jpg",
            save_format="mp4",
            fps=30,
            name="val",
            step=self.true_global_step,
        )
        shutil.rmtree(
            os.path.join(self.get_save_dir(), f"it{self.true_global_step}-val-color")
        )

    def test_step(self, batch, batch_idx):
        out = self(batch)
    
        self.save_image_grid(
            f"it{self.true_global_step}-test-color/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )

        # save camera parameters and views in coarse texture stage for multi-step SDS loss in fine texture stage
        if 'focal' in batch and self.cfg.stage == "texture":
            # save camera parameters
            c2w = batch['c2w'][0].cpu().numpy()

            down_scale = float(batch['width'] / 512) # ensure the resolution is set to 512
     
            frame = {
                "fl_x": float(batch['focal'][0].cpu()) / down_scale,
                "fl_y": float(batch['focal'][0].cpu()) / down_scale ,
                "cx": float(batch['cx'][0].cpu()) / down_scale,
                "cy": float(batch['cy'][0].cpu()) / down_scale,
                "w": int(batch['width'] / down_scale),
                "h": int(batch['height'] / down_scale),
                "file_path": f"./image/{batch['index'][0]}.png",
                "transform_matrix": c2w.tolist(),
                "elevation": float(batch['elevation'][0].cpu()),
                "azimuth": float(batch['azimuth'][0].cpu()),
                "camera_distances": float(batch['camera_distances'][0].cpu()),
            }
            self.frames.append(frame)

            if batch['index'][0] == (batch['n_views'][0]-1):
                os.makedirs(f"{batch['test_save_path'][0]}", exist_ok=True)
                self.transforms["frames"] = self.frames
                with open(os.path.join(batch['test_save_path'][0], 'transforms.json'), 'w') as f:
                    f.write(json.dumps(self.transforms, indent=4))

                # init
                self.frames.clear()

            save_img = out["comp_rgb"]
            save_img = F.interpolate(save_img.permute(0,3,1,2), (512, 512), mode="bilinear", align_corners=False)
            os.makedirs(f"{batch['test_save_path'][0]}/image", exist_ok=True)
            imageio.imwrite(f"{batch['test_save_path'][0]}/image/{batch['index'][0]}.png", (save_img.permute(0, 2, 3, 1)[0].detach().cpu().numpy() * 255).astype(np.uint8))

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test-color",
            f"it{self.true_global_step}-test-color",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )