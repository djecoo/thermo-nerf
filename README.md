## Useful links
ThermalScenes: https://drive.google.com/file/d/1DY44JH8I3vS0c8P6whO8cu_QLC3nMzU0/view?usp=sharing \
Supplementary material: https://drive.google.com/file/d/1yp9EXMLNKsoLan_aEH6l3049BfzLbNo1/view?usp=sharing



## Installation
Please follow the installation instruction from nerfstudio (https://docs.nerf.studio/quickstart/installation.html, install colmap extension aswell) and InstantSplat (https://github.com/NVlabs/InstantSplat, please download the mast3r model accordingly). 

It is recommended to use separated conda environment for nerfstudio and InstantSplat. 

Upload your dataset within this folder, it should follow the same structure as ThermoScenes (image and thermal folders with train and eval images) and be named "ThermoScenes". Feel free to add more scenes, for example from ThermalScenes.

## Rune the sliding window

First create your project by creating and populating different sub-scenes by running 

```bash
python script/template_start.py
```

You can now activate your instantsplat/mast3r environment. \


Then run the mast3r pose estimation using 
```bash
bash scripts/run_dust3r.sh
```
Dont forget to update the path at the beginning of the folder.

Then run 
```bash
bash script/dust3r_convert.sh
```
Activate your nerfstudio environment then execute 
```bash 
bash script/txt_to_bin.sh
```
in order to convert the ply and txt files to bin.  Don't forget to update the path at the beginning of the folder aswell.

Please activate your nerfstudio environment from now on if that is not yet the case  \
After that you can run 
```bash
bash script/colmap2nerf.sh
```

 in order to create the transforms.json for each subscene.\
Note: You may meet some issue due to the way the ply file has been created. If that is the case, change the line 327 `(error = float(elems[7]))` to `error = 0` in nerfstudio/nerfstudio/data/utils/colmap_parsing_utils.py \

Add the thermal images to it by running 
```bash
python script/update_transforms.py
```

Finally you can run for each scene 
```bash
python script/sliding_window.py --directory directory_to_the_scene (eg project/dust3r/trees)
```
You can also use `sliding_window_outliers.py` instead to have the outlier rejection layer. It will output the transforms.json file to the current working directory. 


Congratulations, you have now your transforms.json that you can use to train and evaluate on any Nerfstudio model! Please refer to the ThermoNerf repo to train and evaluate your dataset on it.






## Contribute (to ThermoNerf)

We welcome contributions! Feel free to fork and submit PRs.

We format code using [ruff](https://docs.astral.sh/ruff) and follow PEP8.
The code needs to be type annotated and following our documentation style.

## How to cite (ThermoNerf)

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
