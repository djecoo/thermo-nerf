## Rune the files
Extract the data from https://drive.google.com/file/d/1C6wRYyCbAOEPDV1gWCOBawxEyVLEqlhx/view?usp=sharing
Run the scripts for the sliding windows using `python .\script\dust3r_sliding_window.py`. 

dust3r_sliding_window.py : Run a sliding window on dust3r poses and output the transforms.json in results

optimized_single.py : export a single set of 30 poses optimized by instantsplat to a transfom.json in results

optimized_sliding_window.py : Run a sliding window on poses optimized by instantsplat and export it to a transforms.json in results

-> used scale = 1 in every script

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
