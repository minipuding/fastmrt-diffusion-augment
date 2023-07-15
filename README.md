# fastmrt-diffusion-augment

FastMRT-Diffusion-Augment is a repository containing code for augmenting diffusion MRI data using the Diffusion Model, aimed at enhancing the performance of the FastMRT model.

We only use viallana-DDPM to generate more datasets.

The samples of generated phantom datasets are as follows. The gray images are amplitude maps and the right of them are corresponding phase maps.

![t](./docs/fastmrt_vis_phantom_T600.png)

## Installing

Prepare codes:

```
git clone https://github.com/minipuding/fastmrt-diffusion-augment.git
cd fastmrt-diffusion-augment
```

Install requirements:

```
pip install -r requirements.txt
```

Prepare datasets:

download fastmrt dataset at [here](https://fastmrt.github.io/).

## Running

run following commond at `fastmrt-diffusion-augment`

```
python train.py --stage $STAGE --subset $SUBSET --cfg $CONFIG
```

where:

* $STAGE: one of `train` or `eval`
* $SUBSET: subdataset type, one of `phantom` or `exvivo`
* $CONFIG: config file path, default is `./config.yaml`

## Acknowledgements

* [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
* [DenoisingDiffusionProbabilityModel-ddpm--github](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-)

## Reference
