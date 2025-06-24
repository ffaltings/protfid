import argparse
import os
from pathlib import Path
import re
from sklearn.decomposition import PCA
import torch
import warnings
import torch.nn.functional as F
import tqdm
import concurrent
import requests
import numpy as np
from appdirs import user_cache_dir
from scipy import linalg

import protfid.residue_constants as residue_constants
from esm.utils.structure.protein_chain import ProteinChain
from esm.pretrained import (
    ESM3_sm_open_v0,
    ESM3_structure_encoder_v0,
)
from esm.tokenization.sequence_tokenizer import (
    EsmSequenceTokenizer,
)


def get_cache_dir():
    custom = os.environ.get("PROTFID_CACHE_DIR")
    if custom:
        return os.path.abspath(custom)
    return user_cache_dir("protfid")


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6, atol=1e-3):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, (
        "Training and test mean vectors have different lengths"
    )
    assert sigma1.shape == sigma2.shape, (
        "Training and test covariances have different dimensions"
    )

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=atol):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def compute_fid_from_embeddings(
    embeddings1,
    embeddings2,
    n_replicates=1,
    subsample_frac1=1.0,
    subsample_frac2=1.0,
    pca_dim=None,
):
    fids = []

    assert subsample_frac1 > 0.0 and subsample_frac1 <= 1.0, (
        "subsample_frac1 must be in (0,1]"
    )
    assert subsample_frac2 > 0.0 and subsample_frac2 <= 1.0, (
        "subsample_frac2 must be in (0,1]"
    )

    if pca_dim is not None:
        all_embed = torch.concatenate((embeddings1, embeddings2), dim=0)
        pca = PCA(n_components=min(pca_dim, all_embed.shape[1]), whiten=False)
        pca.fit(all_embed)
        _embed1 = torch.from_numpy(pca.transform(embeddings1))
        _embed2 = torch.from_numpy(pca.transform(embeddings2))
    else:
        _embed1 = embeddings1
        _embed2 = embeddings2

    for rep in range(n_replicates):
        sample_size1 = int(subsample_frac1 * _embed1.shape[0])
        _embed1 = _embed1[
            torch.from_numpy(
                np.random.choice(
                    np.arange(_embed1.shape[0]), sample_size1, replace=False
                )
            )
        ]

        sample_size2 = int(subsample_frac2 * _embed2.shape[0])
        _embed2 = _embed2[
            torch.from_numpy(
                np.random.choice(
                    np.arange(_embed2.shape[0]), sample_size2, replace=False
                )
            )
        ]

        mu1, sigma1 = _embed1.mean(axis=0), torch.cov(_embed1.T)
        mu2, sigma2 = _embed2.mean(axis=0), torch.cov(_embed2.T)

        print(f"Computing fid between {_embed1.shape} and {_embed2.shape} samples.")

        try:
            fid_score = calculate_frechet_distance(
                mu1=mu1.float().cpu().numpy(),
                sigma1=sigma1.float().cpu().numpy(),
                mu2=mu2.float().cpu().numpy(),
                sigma2=sigma2.float().cpu().numpy(),
                atol=0.01,
            ).item()
        except ValueError as e:
            print(e)
            continue

        fids.append(fid_score)

    if len(fids) == 0:
        raise RuntimeError("All FID computations failed")
    else:
        return np.mean(fids), np.std(fids)


def load_chain(pdb_path):
    try:
        chain = ProteinChain.from_pdb(pdb_path)
        seq_len = len(chain.sequence)
        chain.sequence = "_" * seq_len
        # erase all coordinates except N, CA, C, O which all models produce
        atom_idxs = [residue_constants.atom_order[a] for a in ["N", "CA", "C", "O"]]
        mask_idxs = [i for i in range(37) if i not in atom_idxs]
        chain.atom37_positions[:, mask_idxs] = np.nan
        chain.atom37_mask[:, mask_idxs] = 0
    except Exception as e:
        return (False,)
    return True, len(chain), chain, pdb_path


def load_chains(pdb_dir, max_batch_tokens=800):
    pdb_dir = Path(pdb_dir)
    pdb_paths = list(pdb_dir.glob("*.pdb"))
    if len(pdb_paths) == 0:
        raise RuntimeError(f"PDB directory {pdb_dir} is empty.")

    chains = []
    load_failures = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        jobs = [executor.submit(load_chain, pdb_path) for pdb_path in pdb_paths]
        with tqdm.tqdm(total=len(jobs), desc="Loading PDB files") as pbar:
            for job in concurrent.futures.as_completed(jobs):
                succ, *ret = job.result()
                if not succ:
                    load_failures += 1
                    continue
                chain_len, chain, pdb_path = ret
                chains.append((chain_len, pdb_path, chain))
                pbar.update(1)
    print(f"Failed to load {load_failures} pdb files")
    chains = sorted(chains, reverse=True)
    chains = [
        (chain_len, pdb_path, chain)
        for chain_len, pdb_path, chain in chains
        if chain_len <= max_batch_tokens
    ]
    return chains
    # chains = [(ProteinChain.from_pdb(pdb_path), pdb_path) for pdb_path in tqdm.tqdm(pdb_paths, desc='initial chain loading')]


def merge_batch(
    batch_sizes, coords_batch, plddt_batch, structure_tokens_batch, pdb_paths_batch
):
    max_len = np.max(batch_sizes)
    paddings = [max_len - s for s in batch_sizes]
    coords_batch = torch.concatenate(
        [
            F.pad(coords, (0, 0, 0, 0, 0, pad), value=torch.inf)
            for coords, pad in zip(coords_batch, paddings)
        ],
        dim=0,
    )
    plddt_batch = torch.concatenate(
        [F.pad(plddt, (0, pad), value=0) for plddt, pad in zip(plddt_batch, paddings)],
        dim=0,
    )
    structure_tokens_batch = torch.concatenate(
        [
            F.pad(structure_tokens, (0, pad), value=0)
            for structure_tokens, pad in zip(structure_tokens_batch, paddings)
        ],
        dim=0,
    )
    return (
        batch_sizes,
        coords_batch,
        plddt_batch,
        structure_tokens_batch,
        pdb_paths_batch,
    )


class FID:
    def __init__(self, max_batch_tokens=800, pca_dim=32):
        super().__init__()

        self.tokenizer = EsmSequenceTokenizer()
        self.encoder = ESM3_structure_encoder_v0("cpu")
        self.model = ESM3_sm_open_v0("cpu")
        self.embedding_size = 32  # TODO

        self.ref_embeddings = None
        self.max_batch_tokens = max_batch_tokens
        self.pca_dim = pca_dim

    def fit(self, pdb_dir, device=torch.device("cuda")):
        self.ref_embeddings = self.compute_embeddings(
            pdb_dir, device, desc="Computing reference embeddings"
        )

    def fit_embeddings(self, embeddings):
        self.ref_embeddings = embeddings

    def save(self, ckpt_path):
        torch.save(self.ref_embeddings, ckpt_path)

    def load_from_checkpoint(self, ckpt_path):
        self.ref_embeddings = torch.load(ckpt_path)

    def __call__(self, pdb_dir, n_replicates=1, subsample_frac=1.0, device=None):
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        data_embeddings = self.compute_embeddings(
            pdb_dir, desc="Computing sample embeddings", device=device
        )
        return self.compute_from_embeddings(
            data_embeddings, n_replicates=n_replicates, subsample_frac=subsample_frac
        )

    def compute_from_embeddings(
        self,
        data_embeddings,
        n_replicates=1,
        subsample_frac=1.0,
    ):
        if data_embeddings.device != self.ref_embeddings.device:
            raise RuntimeError(
                f"Ensure ref_embeddings ({self.ref_embeddings.device}) and data_embeddings ({self.data_embeddings}) are on the same device."
            )
        if n_replicates == 1:
            fid, _ = compute_fid_from_embeddings(
                self.ref_embeddings, data_embeddings, pca_dim=self.pca_dim
            )
            return fid
        else:
            fid, std = compute_fid_from_embeddings(
                self.ref_embeddings,
                data_embeddings,
                n_replicates=n_replicates,
                subsample_frac2=subsample_frac,
                pca_dim=self.pca_dim,
            )
            return fid, std

    def compute_embeddings(self, pdb_dir, device=torch.device("cuda"), desc=""):
        self.encoder = self.encoder.to(device)
        self.model = self.model.to(device)
        chains = load_chains(pdb_dir)
        embeddings = []
        proc_pdb_paths = []
        lengths = []
        with torch.no_grad():
            for (
                batch_sizes,
                coords_batch,
                plddt_batch,
                structure_tokens_batch,
                pdb_paths_batch,
            ) in tqdm.tqdm(self.batch_generator(chains, device), desc=desc):
                output = self.model.forward(
                    structure_coords=coords_batch,
                    per_res_plddt=plddt_batch,
                    structure_tokens=structure_tokens_batch,
                )
                for i, batch_size in enumerate(batch_sizes):
                    embeddings.append(
                        output.embeddings[i][:batch_size].mean(axis=0).cpu()
                    )
                    lengths.append(batch_size)
                proc_pdb_paths.extend(pdb_paths_batch)

        embeddings = torch.stack(embeddings, axis=0)
        return embeddings

    def batch_generator(self, chains, device):
        batch_sizes = []
        coords_batch = []
        plddt_batch = []
        structure_tokens_batch = []
        pdb_paths_batch = []
        for seq_len, pdb_path, chain in tqdm.tqdm(chains):
            potential_batch_size = len(batch_sizes) + 1
            potential_batch_len = max(
                np.max(batch_sizes) if len(batch_sizes) > 0 else 0, seq_len
            )
            if potential_batch_size * potential_batch_len > self.max_batch_tokens:
                yield merge_batch(
                    batch_sizes,
                    coords_batch,
                    plddt_batch,
                    structure_tokens_batch,
                    pdb_paths_batch,
                )
                batch_sizes = []
                coords_batch = []
                plddt_batch = []
                structure_tokens_batch = []
                pdb_paths_batch = []
            batch_sizes.append(seq_len + 2)  # +2 for the padding
            coords, plddt, residue_index = chain.to_structure_encoder_inputs()
            coords = coords.to(device)
            plddt = plddt.to(device)
            residue_index = residue_index.to(device)
            _, structure_tokens = self.encoder.encode(
                coords, residue_index=residue_index
            )
            coords = F.pad(coords, (0, 0, 0, 0, 1, 1), value=torch.inf)
            plddt = F.pad(plddt, (1, 1), value=0)
            structure_tokens = F.pad(structure_tokens, (1, 1), value=0)
            structure_tokens[:, 0] = 4098
            structure_tokens[:, -1] = 4097
            coords_batch.append(coords)
            plddt_batch.append(plddt)
            structure_tokens_batch.append(structure_tokens)
            pdb_paths_batch.append(pdb_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=str,
        default="esm_train_ref",
        help="Reference to use. Default refers to the reference used in our paper.",
    )
    parser.add_argument(
        "--ref_pdb_dir",
        type=str,
        default="esm_train_ref.ckpt",
        help="Path to directory with PDB files of reference set.",
    )
    parser.add_argument(
        "--pdb_dir",
        type=str,
        help="Path to directory with PDB files to be evaluated.",
    )
    parser.add_argument(
        "--n_replicates",
        type=int,
        default=1,
        help="Number of replicates to run.",
    )
    parser.add_argument(
        "--subsample_frac",
        type=float,
        default=0.8,
        help="Fraction to subsample when computing replicates. Only applies to samples in `pdb_dir`.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Whether to overwrite reference checkpoint. Otherwise only writes if ref_ckpt_path does not exists.",
    )
    args = parser.parse_args()

    filename_pattern = re.compile(r'^[^<>:"/\\|?*\x00-\x1F]+$')
    # Use default reference
    cache_dir = get_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    default_ckpt_path = os.path.join(cache_dir, "esm_train_ref.ckpt")
    if args.reference == "esm_train_ref":
        if not os.path.exists(default_ckpt_path):
            response = (
                input(
                    "Would you like to download the default PDB reference set embeddings? (yes/no): "
                )
                .strip()
                .lower()
            )
            while not (response in ("yes", "no")):
                response = input("Please answer (yes/no): ").strip().lower()
            if response == "yes":
                url = "https://zenodo.org/records/15660186/files/esm_train_ref.ckpt"
                r = requests.get(url)
                with open(default_ckpt_path, "wb") as f:
                    f.write(r.content)
            else:
                raise ValueError(f"Please specify a different reference set to use.")
        ref_ckpt_path = default_ckpt_path
    elif filename_pattern.match(args.reference):
        ref_ckpt_path = os.path.join(cache_dir, args.reference + ".ckpt")
    else:
        raise ValueError(
            f"Invalid reference name: {args.reference}. Should be a valid file name (without any extension)."
        )

    if ref_ckpt_path is None:
        if args.ref_pdb_dir is None:
            raise ValueError(
                f"No reference specified and no reference pdb directory specified."
            )
        if not os.path.exists(args.ref_pdb_dir):
            raise ValueError(
                f"No reference specified and reference pdb directory {args.ref_pdb_dir} does not exist."
            )
    elif not os.path.exists(ref_ckpt_path):
        if args.ref_pdb_dir is None:
            raise ValueError(
                f"Reference: {args.reference} does not exist and no reference pdb directory specified."
            )
        if not os.path.exists(args.ref_pdb_dir):
            raise ValueError(
                f"Reference: {args.reference} does not exist and reference pdb directory {args.ref_pdb_dir} does not exist."
            )

    fid_evaluator = FID()
    if not os.path.exists(ref_ckpt_path):
        warnings.warn(
            f"Specified reference: {args.reference} does not exist. Computing reference embeddings from pdb directory {args.ref_pdb_dir} instead."
        )
        fid_evaluator.fit(args.ref_pdb_dir)
    else:
        fid_evaluator.load_from_checkpoint(ref_ckpt_path)

    if not args.pdb_dir is None:
        if args.n_replicates == 1:
            fid = fid_evaluator(args.pdb_dir)
            print(f"Computed FID: {fid}")
        else:
            fid, std = fid_evaluator(
                args.pdb_dir,
                n_replicates=args.n_replicates,
                subsample_frac=args.subsample_frac,
            )
            print(f"Computed FID: {fid} ({std})")

    if not os.path.exists(ref_ckpt_path) or args.overwrite:
        fid_evaluator.save(ref_ckpt_path)


if __name__ == "__main__":
    main()
