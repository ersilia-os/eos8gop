"""Monroe encoder inference: SMILES in, 720-d embedding out.

Wraps the pretrained Monroe graph transformer (Banaszewski & Fitzgibbon,
arXiv:2608.18982), vendored from https://github.com/blazejba/Monroe at commit
57238edfffea03808abe761a00cd9a75fa41bb95.

Only the encoder is served. The authors pair it with an in-context TabPFN head,
which needs a labelled support set per task and so has no place behind a
single-compound interface.

Three upstream problems are worked around here, in the wrapper only — the
vendored model source and weights are left untouched:

1. The released checkpoint is pickled from CUDA tensors and the public
   ``load_ckpt`` takes no device argument, so upstream's ``run_inference.py``
   cannot run on a CPU-only machine. Loaded here with ``map_location="cpu"``.

2. ``predict_structure`` enables ``useRandomCoords`` without setting
   ``randomSeed``, so each call builds a different ETKDGv3 conformer and returns
   a different embedding. The seed is pinned below, which makes output
   bit-identical within a process, across processes and against batching.

3. ``monroe.model.ckpt`` imports ``wandb`` at module level for a training-only
   code path, which would drag the package and its dependencies into the image.
   A stub is registered below instead.

With these applied, the authors' released per-target MoleculeACE results
reproduce to a mean absolute deviation of 0.005 RMSE, against their own
across-seed standard deviation of 0.004.
"""

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

# monroe.model.ckpt imports wandb at module level, but uses it only in
# save_training_checkpoint (to stamp a W&B run id into state.pt while
# pretraining). load_ckpt never touches it. Registering an empty stub keeps the
# vendored source byte-identical to upstream while dropping wandb and its
# transitive dependencies from the image; the single use site reads
# getattr(wandb, "run", None), which yields None on a stub.
sys.modules.setdefault("wandb", types.ModuleType("wandb"))

EMB_DIM = 720
CONFORMER_SEED = 42

CHECKPOINT_DIR = Path(
    os.environ.get(
        "MONROE_CHECKPOINT_DIR",
        Path(__file__).resolve().parents[2] / "checkpoints",
    )
)

_encoder = None


def _pin_conformer_seed():
    """Make ETKDGv3 deterministic.

    The upstream featurizer resolves ``rdDistGeom.ETKDGv3`` through ``getattr``
    at call time, so replacing the factory is enough and no vendored source
    needs editing.
    """
    from rdkit.Chem import rdDistGeom

    if getattr(rdDistGeom.ETKDGv3, "_seeded", False):
        return
    original = rdDistGeom.ETKDGv3

    def seeded():
        params = original()
        params.randomSeed = CONFORMER_SEED
        return params

    seeded._seeded = True
    rdDistGeom.ETKDGv3 = seeded


def load_encoder():
    """Build the encoder on CPU from the bundled checkpoint, once per process."""
    global _encoder
    if _encoder is not None:
        return _encoder

    _pin_conformer_seed()

    config_path = CHECKPOINT_DIR / "config.json"
    weights_path = CHECKPOINT_DIR / "weights.pt"
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            "Monroe checkpoint not found in %s (need config.json and weights.pt)"
            % CHECKPOINT_DIR
        )

    from monroe.model.ckpt import _build_encoder

    with config_path.open() as f:
        hyperparameters = json.load(f)

    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    prefix = "encoder."
    encoder_state = {
        key[len(prefix):].replace("._orig_mod.", "."): value
        for key, value in state.items()
        if key.startswith(prefix)
    }

    encoder = _build_encoder(hyperparameters)
    # strict=True by choice: upstream loads with strict=False, which would let a
    # key mismatch pass unnoticed and leave part of the encoder random. All 368
    # tensors are present in the released checkpoint, so strict costs nothing.
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.eval()
    torch.set_num_threads(1)
    _encoder = encoder
    return encoder


def embed_one(smiles):
    """Return the 720-d embedding for one SMILES, or None if it cannot be built."""
    import datamol as dm
    from torch_geometric.data import Data

    from monroe.model.featurizer import build_single_graph

    encoder = load_encoder()

    # Monroe builds its graph from InChI. to_inchi returns None for an
    # unparseable SMILES but *raises* on a non-string (an integer column, or the
    # NaN a reader yields for an empty cell), and passing a None InChI on to
    # build_single_graph raises an opaque RDKit ArgumentError. Every one of
    # those is a single bad row, so none of them may take down the batch.
    try:
        inchi = dm.to_inchi(smiles)
        if inchi is None:
            return None
        graph = build_single_graph(inchi=inchi, symmetrize=True)
    except Exception:
        return None

    positions = np.asarray(graph["pos_rdkit"], dtype=np.float32)
    data = Data(
        x=torch.tensor(graph["node_float"], dtype=torch.float32),
        node_codes=torch.tensor(graph["node_codes"], dtype=torch.long),
        edge_index=torch.tensor(graph["edge_index"], dtype=torch.long),
        edge_codes=torch.tensor(graph["edge_codes"], dtype=torch.long),
        pos_in=torch.tensor(positions, dtype=torch.float32),
    )
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)

    with torch.no_grad():
        embedding, _ = encoder(data)

    return embedding.squeeze(0).numpy()


def predict(smiles_list):
    """Embed a list of SMILES, returning an (n, 720) array.

    Molecules that cannot be featurised yield a row of NaN so the output stays
    aligned with the input, one row per molecule.
    """
    rows = []
    for smiles in smiles_list:
        vector = embed_one(smiles)
        if vector is None:
            vector = np.full(EMB_DIM, np.nan, dtype=np.float32)
        rows.append(vector)
    return np.asarray(rows, dtype=np.float32)
