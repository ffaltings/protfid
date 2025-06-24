# Protein FID

The Protein FID is a measure of similarity between sets of protein structures. Formally, it is the Wassertstein-2 distance between Gaussian approximations of two distributions embedded in a latent space. See our preprint for details (https://arxiv.org/abs/2505.08041). 

The main application of the FID is to evaluate structures sampled from a generative model. In this case, the generated structures are compared against a reference distribution.

## Installation

The package was tested with python 3.10 and can be installed from source using pip. Optionally, you can first create a virtual environment, for example with conda,

```
conda create --name protfid python=3.10
conda activate protfid
```

Then clone the repository,

`git clone git@github.com:ffaltings/protfid.git`

navigate to the directory,

`cd protfid`

and pip install it,

`pip install .`

Or to install in editable mode, run

`pip install -e .`

You also need to have access to the ESM3 weights, which we use for computing embeddings. You can request access here: `https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1`

Once you have access, you need to log in to your huggingface account:

`huggingface-cli login`

You should only need to login once.

## Usage

### Command Line
We provide a command line script `protfid` for computing the fid. 

A simple example is:

`protfid --pdb_dir examples/esm_test`

You can swap out `examples/esm_test` for any directroy containing `.pdb` files.

By default, the script will use the reference set from our paper, which will be downloaded for you the first time you run it. The ESM3 weights will also be downloaded the first time you run it. If you want to use your own reference, you will need to specify a directory containing your reference structures as PDB files and a name

`protfid --pdb_dir path/to/pdb_dir --ref_pdb_dir path/to/ref_pdb_dir --reference my_reference`

Under the hood this will save the reference embeddings to your cache directory so that they do not need to be recomputed in the future. You can override the default cache directory with the `PROTFID_CACHE_DIR` environment variable.

The arguments for the command line script are:
| Argument | Description |
| --- | --- |
|`reference`        | Reference to use. Defaults to `esm_train_ref` which is the reference used in our paper. This reference will also be automatically downloaded if it is not found in the cache. |
|`ref_pdb_dir`      | Path to directory with PDB files of reference set. |
|`pdb_dir`          | Path to directory with PDB files to be evaluated. |
|`n_replicates`     | Number of replicates to run. Defaults to 1.|
|`subsample_frac`   | Fraction to subsample when computing replicates. Only applies to samples in `pdb_dir`. Defaults to 0.8.| 
|`overwrite`        | Whether to overwrite reference checkpoint. Defaults to False. |   

### API
You can also directly use the python API. For example,

```
from protfid.fid import FID

fid_evaluator = FID()
# compute reference embeddings
fid_evaluator.fit("path/to/ref_pdb_dir")
# or load from a checkpoint
fid_evaluator.load_from_checkpoint("path/to/ref_ckpt.ckpt")
# save to a checkpoint
fid_evaluator.save("path/to/ref_ckpt.ckpt")

# compute the FID
fid = fid_evaluator("path/to/pdb_dir")
```

If you want to load the default reference used in our paper you can either run the script which will download it to your system's default cache directory, or you can directly download it from here: `https://zenodo.org/records/15660186/files/esm_train_ref.ckpt`.

If you want to compute the FID of many subsets of a larger set of structures without needing to recompute embeddings each time (as is done in some of the experiments in our paper), you can accomplish this as in the following example,

```
from protfid.fid import FID

fid_evaluator = FID()
# either compute reference embeddings or load a checkpoint. See example above.
fid_evaluator.fit("path/to/ref_pdb_dir")

# compute embeddings for all structures
device = torch.device("cuda") # change to whatever device you have available
data_embeddings = self.compute_embeddings(
    "path/to/pdb_dir", desc="Computing sample embeddings", device=device
)

fids = []
for si in subset_idxs:
    fid = fid_evaluator.compute_from_embeddings(data_embeddings[si])
    fids.append(fid)
```

## Requirements

Our code uses the `esm` package to compute the embeddings (the model used is `ESM3_sm_open_v0`). Running ESM3 is best done on a GPU. We were able to run all the experiments in our paper on a single NVIDIA Tesla V100 GPU with 32GB of memory.

## Notes

The FID computed with ESM3 embeddings will take on large values. For example, the FID of a set of PDB structures to our reference set will be around 100,000 (to see this, you can run e.g. `protfid --pdb_dir examples/esm_test`). This is normal and is due to the large norms of the ESM3 embeddings. See also the range of values found in our paper for different perturbations and generative models.

If the sample size is too small (or not diverse enough), you may encounter an error when computing the FID due to a large imaginary component. The only thing you can really do is increase the sample size.

When computing embeddings with ESM3, the batch size is set adaptively based on the length of the sequences. The sequences are processed in decreasing order so the first few batches will have a small batch size and take longer.

# Citation
If you use the Protein FID, please consider citing our work.

BibTeX citation:
```
@article{faltings2025protein,
  title={Protein FID: Improved Evaluation of Protein Structure Generative Models},
  author={Faltings, Felix and Stark, Hannes and Jaakkola, Tommi and Barzilay, Regina},
  journal={arXiv preprint arXiv:2505.08041},
  year={2025}
}
```

arXiv link: https://arxiv.org/abs/2505.08041