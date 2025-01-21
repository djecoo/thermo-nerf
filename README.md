## Installation
Please follow the installation instruction from nerfstudio (https://docs.nerf.studio/quickstart/installation.html, install colmap extension aswell) and InstantSplat (https://github.com/NVlabs/InstantSplat, please download the mast3r model accodringly to the instructions). 

It is recommended to use separated conda environment for nerfstudio and InstantSplat. 

To install the requirements for this project, please run `pip install -e requirements.txt`

Upload your dataset within this folder, it should follow the same structure as ThermoScenes (image and thermal folders with train and eval images) and be named "ThermoScenes". Feel free to add more scenes, for example from ThermalScenes: https://drive.google.com/file/d/1DY44JH8I3vS0c8P6whO8cu_QLC3nMzU0/view?usp=sharing

## Rune the sliding window

First create your project by creating and populating different sub-scenes by running `python scripts/template_start.py`

Then run the mast3r pose estimation using `bash scripts/run_dust3r.sh`. Dont forget to update the path at the beginning of the folder.

Then run `bash scripts/dust3r_convert.sh` and `bash scripts/txt_to_bin.sh` in order to convert the ply and txt files.  Don't forget to update the path at the beginning of the folder aswell.

After that you can run `bash scripts/colmap2nerf.sh` in order to create the transforms.json for each subscene. Add the thermal images to it by running `python scripts/update_transforms.sh`

Finally you can run for each scene `python scripts/sliding_window.py --directory directory_to_the scene (eg /project/dust3r/trees)`. You can also use sliding_window_outliers.py instead to have the outlier rejection layer. 

## Contribute

We welcome contributions! Feel free to fork and submit PRs.

We format code using [ruff](https://docs.astral.sh/ruff) and follow PEP8.
The code needs to be type annotated and following our documentation style.

## How to cite

For now the paper is on arxiv:

```bibtex
@misc{hassan2024thermonerf,
      title={ThermoNeRF: Multimodal Neural Radiance Fields for Thermal Novel View Synthesis},
      author={Mariam Hassan and Florent Forest and Olga Fink and Malcolm Mielle},
      year={2024},
      eprint={2403.12154},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
