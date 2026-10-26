# diffusion-transformer

Attempt at implementing a frankensteined hybrid SD3/FLUX.1-inspired MM-DiT for latent flow matching (LFM) image generation.


## Set-up

Main files in [`diffusion/`](diffusion/)

Training: [`diffusion/train.py`](diffusion/train.py), [`diffusion/train_vae.py`](diffusion/train_vae.py)

Models: [`diffusion/model.py`](diffusion/model.py), [`diffusion/vae.py`](diffusion/vae.py)

### Environments and Packages

```bash
git clone https://github.com/someonecantcode/diffusion-transformer.git
cd diffusion-transformer

conda create -name myenv python=3.12
conda activate myenv
pip install -r requirements.txt
```

### Dataset Preparation

**GPIC Image Dataset**

Please run the following:

```bash
cd data
python download_data.py
```

### Training

To begin training without SLURM:

```bash
cd diffusion
python train_vae.py
python train.py
```

With SLURM:

```bash
cd diffusion
sbatch vae_batchjob.sh
sbatch flowmatch_job.sh
```

## References

Esser, P., et al. (2024). Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206. https://doi.org/10.48550/arXiv.2403.03206

Greenberg, O. (2025). Demystifying Flux architecture. arXiv preprint arXiv:2507.09595. https://doi.org/10.48550/arXiv.2507.09595

Peebles, W., & Xie, S. (2023). Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748. https://doi.org/10.48550/arXiv.2212.09748