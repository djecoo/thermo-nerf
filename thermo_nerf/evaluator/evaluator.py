import json
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import torch
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from PIL import Image

from thermo_nerf.nerfacto_config.thermal_nerfacto import ThermalNerfactoModel
from thermo_nerf.rendered_image_modalities import RenderedImageModality


class Evaluator:
    """
    Evaluates a model by computing metrics on the eval data extracted from the model"""

    def __init__(
        self,
        pipeline: Pipeline,
        config: TrainerConfig,
        job_param_identifier: Optional[str] = None,
        modalities_to_save: list[RenderedImageModality] = [RenderedImageModality.RGB],
        threshold: float | None = None,
    ) -> None:
        """
        Initializes the parameters which are `output_file` to save the metrics, the
        'job_param_identifier' is an optional parameter to identify the job parameters.
        It is saved with metrics to identify job parameters in the metrics json.
        """
        self._pipeline = pipeline
        self._pipeline.datamanager.setup_eval()
        self.identifier = job_param_identifier
        self._evaluation_images: dict[RenderedImageModality, list[np.ndarray]] = {}

        self.modalities_to_save = modalities_to_save
        self._metrics = self._compute_metrics(threshold=threshold)

        self._benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "job_param_identifier": self.identifier,
            "results": self._metrics,
        }

    def _compute_metrics(self, threshold: float | None) -> dict[str, float]:
        """
        Computes metrics on eval data extracted from 'self._pipeline'

        :returns: dictionary of metrics
        """
        metrics_dict_list = []
        datamanager = self._pipeline.datamanager

        if datamanager.fixed_indices_eval_dataloader is None:
            raise RuntimeError(
                "Cannot evaluate without a fixed indices eval dataloader"
            )
        for modality in self.modalities_to_save:
            self._evaluation_images[modality] = []

        for (
            cameras,
            batch,
        ) in datamanager.fixed_indices_eval_dataloader:
            # Generate camera indices based on the number of camera_to_worlds
            camera_indices = torch.arange(cameras.camera_to_worlds.shape[0])
            camera_ray_bundle = cameras.generate_rays(camera_indices)

            assert isinstance(self._pipeline.model, ThermalNerfactoModel)

            outputs = self._pipeline.model.get_outputs_for_camera_ray_bundle(
                camera_ray_bundle
            )
            (
                metrics_dict,
                images_dict,
            ) = self._pipeline.model.get_image_metrics_and_images(
                outputs, batch, threshold=threshold
            )

            for modality in self.modalities_to_save:
                self._evaluation_images[modality].append(
                    (images_dict[modality.value] * 255).byte().cpu().numpy()
                )
            metrics_dict_list.append(metrics_dict)

        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            key_std, key_mean = torch.std_mean(
                torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
            )
            metrics_dict[f"{key}_mean"] = float(key_mean)
            metrics_dict[f"{key}_std"] = float(key_std)
            metrics_dict[key] = [
                metrics_dict[key] for metrics_dict in metrics_dict_list
            ]

        return metrics_dict
    
    def add_thermal_ref(
        self, modalities: list[RenderedImageModality], dataset_path: Path
    ) -> None:
        """
        Add the thermal reference to '_evaluation_images'
        """
        for modality in modalities:
            if modality == RenderedImageModality.THERMAL:
                warning_issued = False
                eval_img = []
                for idx, image in enumerate(self._evaluation_images[modality]):
                    base_filename = f"frame_eval_{idx + 1:05d}"
                    thermal_format = ["jpeg", "png", "PNG", "JPG"] 
                    img_eval_thermal = None
                    for ext in thermal_format:
                        potential_file = dataset_path / "thermal" / f"{base_filename}.{ext}"
                        if potential_file.is_file():
                            img_eval_thermal = potential_file
                            break

                    if img_eval_thermal is None and not warning_issued:
                        warnings.warn(
                        f"No matching image found in {dataset_path / 'thermal'}",
                        category=UserWarning
                        )
                        warning_issued = True

                    elif img_eval_thermal is not None and not warning_issued:
                        with Image.open(img_eval_thermal) as img:
                            eval_img.append(np.array(img))
                
                eval_img = np.expand_dims(eval_img, axis=-1)
                eval_img = np.repeat(eval_img, 3, axis=-1)
                
                self._evaluation_images[modality] = np.concatenate(
                    (eval_img, self._evaluation_images[modality]), axis=2
                    )

                


    def save_images(
        self, modalities: list[RenderedImageModality], output_path: Path, keep_eval: bool = True, find_thermal_eval: bool =True
    ) -> None:
        """
        Saves evaluation images to `output_path`.
        """
        # Removes the eval images reference
        if not keep_eval:
            new_list = []

            for img in self._evaluation_images[RenderedImageModality.RGB]:
                split1, split2 = np.split(img, 2, axis=1)
                new_list.append(split2)
            self._evaluation_images[RenderedImageModality.RGB] = new_list
            if find_thermal_eval:
                new_list = []
                for img in self._evaluation_images[RenderedImageModality.THERMAL]:
                    split1, split2 = np.split(img, 2, axis=1)
                    new_list.append(split2)
                self._evaluation_images[RenderedImageModality.THERMAL] = new_list
        
        for modality in modalities:
            
            for idx, image in enumerate(self._evaluation_images[modality]):
                if image.shape[-1] == 4:
                    image = image[:, :, 3]
                    Image.fromarray(image).save(
                        output_path / f"{modality.value}_{idx:05d}.jpg"
                    )
                    
                else:
                    Image.fromarray(image).save(
                        output_path / f"{modality.value}_{idx:05d}.jpg"
                    )
                    

    def save_metrics(self, output_folder: Path) -> None:
        """
        Saves the metrics in the `output_folder`
        """
        output_file = Path(output_folder, "metrics.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(self._benchmark_info, indent=2), "utf8")

        if self.identifier is None:
            return

        psnr_folder_path = output_folder.joinpath("psnr")
        psnr_folder_path.mkdir(parents=True, exist_ok=True)

        ssim_folder_path = output_folder.joinpath("ssim")
        ssim_folder_path.mkdir(parents=True, exist_ok=True)

        lpips_folder_path = output_folder.joinpath("lpips")
        lpips_folder_path.mkdir(parents=True, exist_ok=True)

        psnr_folder_path.joinpath(self.identifier + ".txt").write_text(
            json.dumps(self._metrics["psnr"], indent=2), "utf8"
        )
        ssim_folder_path.joinpath(self.identifier + ".txt").write_text(
            json.dumps(self._metrics["ssim"], indent=2), "utf8"
        )
        lpips_folder_path.joinpath(self.identifier + ".txt").write_text(
            json.dumps(self._metrics["lpips"], indent=2), "utf8"
        )
        if (
            RenderedImageModality.THERMAL
            or RenderedImageModality.THERMAL_COMBINED in self.modalities_to_save
        ):
            psnr_folder_path.joinpath(self.identifier + "_thermal.txt").write_text(
                json.dumps(self._metrics["psnr_thermal"], indent=2), "utf8"
            )
            ssim_folder_path.joinpath(self.identifier + "_thermal.txt").write_text(
                json.dumps(self._metrics["ssim_thermal"], indent=2), "utf8"
            )
            lpips_folder_path.joinpath(self.identifier + "_thermal.txt").write_text(
                json.dumps(self._metrics["lpips_thermal"], indent=2), "utf8"
            )
