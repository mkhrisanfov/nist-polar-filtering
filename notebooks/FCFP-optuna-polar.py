import json
from time import perf_counter

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from nist_polar_filtering import BASE_DIR, SEED
from nist_polar_filtering.datasets import (
    FCFP_Dataset,
)
from nist_polar_filtering.models import FCFP
from nist_polar_filtering.preprocessing import (
    load_nist_2023_dataset,
)
from nist_polar_filtering.utils import (
    generate_fingerprints,
)


def train_fn(
    model,
    optim,
    loss_fn,
    epochs,
    train_dl,
    eval_dl,
    name,
    epoch_start=0,
    trial=None,
):

    writer = SummaryWriter(log_dir=BASE_DIR / "logs" / name, flush_secs=15)

    torch.cuda.empty_cache()
    b_train = 1e5
    b_eval = 1e5

    bar = tqdm(range(epoch_start, epoch_start + epochs), leave=False, position=1)
    for epoch in bar:
        start = perf_counter()
        epoch_train_loss = model.train_fn(optim, loss_fn, train_dl)
        b_train = min(b_train, epoch_train_loss)

        epoch_eval_loss = model.eval_fn(loss_fn, eval_dl)
        b_eval = min(b_eval, epoch_eval_loss)

        bar.set_postfix_str(
            f"{epoch_train_loss:.3f}({b_train:.3f}) | {epoch_eval_loss:.3f}({b_eval:.3f})"
        )

        writer.add_scalar("loss/train", epoch_train_loss, epoch)
        writer.add_scalar("loss/eval", epoch_eval_loss, epoch)
        if trial is not None:
            trial.report(epoch_eval_loss, epoch)
            if trial.should_prune() or (perf_counter() - start) > 200:
                raise optuna.TrialPruned()
        if epoch_eval_loss <= b_eval:
            torch.save(model.state_dict(), BASE_DIR / f"models/{name}.pth")
    bar.close()
    return b_train, b_eval


if __name__ == "__main__":
    ri_df, input_molecules = load_nist_2023_dataset(
        BASE_DIR / "data/input/nist_ri_2023_0.0.1.tsv"
    )
    input_smiles = ri_df["smiles"]
    input_inchi = ri_df["inchi"]
    input_inchikeys = np.array(
        [str(x).split("-")[0] for x in ri_df["inchikey"].to_list()]
    )
    morgan_fingerprints, rdkit_fingerprints = generate_fingerprints(input_molecules)
    phases_dict = json.load((BASE_DIR / "data/output/phases-23-all.json").open("r"))
    input_phases = ri_df["active_phase"].str.lower()
    encoded_phases = np.array([phases_dict.get(x, 0) for x in input_phases])

    unique_inchikeys, unique_index, unique_inverse = np.unique(
        input_inchikeys, return_index=True, return_inverse=True
    )
    input_ri = ri_df["ri"].to_numpy()
    nonpolar_mask = encoded_phases < 43
    polar_mask = encoded_phases >= 43

    print("Full DS length", len(input_molecules))
    print("Unique DS length", len(unique_inchikeys))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)

    unique_nonpolar_inchikeys = set(np.unique(input_inchikeys[nonpolar_mask]))
    unique_polar_inchikeys = np.unique(input_inchikeys[polar_mask])
    trn_idx, tst_idx = train_test_split(
        np.arange(len(unique_polar_inchikeys)), test_size=0.4, random_state=SEED
    )
    unique_trn_polar_inchikeys = set(unique_polar_inchikeys[trn_idx])
    unique_tst_polar_inchikeys = set(unique_polar_inchikeys[tst_idx])
    (
        nonpolar_trn,
        nonpolar_tst,
        polar_trn,
        polar_tst,
    ) = (
        {"idx": [], "inchikeys": []},
        {"idx": [], "inchikeys": []},
        {"idx": [], "inchikeys": []},
        {"idx": [], "inchikeys": []},
    )
    for i, key in enumerate(input_inchikeys):
        if key in unique_tst_polar_inchikeys:
            if encoded_phases[i] < 43:
                nonpolar_tst["idx"].append(i)
                nonpolar_tst["inchikeys"].append(key)
            else:
                polar_tst["idx"].append(i)
                polar_tst["inchikeys"].append(key)
        else:
            if encoded_phases[i] < 43:
                nonpolar_trn["idx"].append(i)
                nonpolar_trn["inchikeys"].append(key)
            else:
                polar_trn["idx"].append(i)
                polar_trn["inchikeys"].append(key)

    print(f"Nonpolar TST: {len(nonpolar_tst['idx'])}")
    print(f"Nonpolar TRN: {len(nonpolar_trn['idx'])}")
    print(f"Polar TST: {len(polar_tst['idx'])}")
    print(f"Polar TRN: {len(polar_trn['idx'])}")

    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate_train", 1e-5, 1e-3, log=True)
        pretrain_epochs = trial.suggest_int("pretrain_epochs", 5, 50, step=5)
        model = FCFP(
            n_phases=max(encoded_phases) + 1,
            n_layers=trial.suggest_int("n_lin_layers", 1, 10),
            hidden_wts=2 ** trial.suggest_int("hidden_wts", 8, 12),
        ).to(device)

        eval_ds = FCFP_Dataset(
            rdkit_fingerprints=rdkit_fingerprints[nonpolar_tst["idx"]],
            morgan_fingerprints=morgan_fingerprints[nonpolar_tst["idx"]],
            phases=encoded_phases[nonpolar_tst["idx"]],
            retention_values=input_ri[nonpolar_tst["idx"]],
        )
        eval_dl = DataLoader(
            eval_ds,
            batch_size=512,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
        )
        train_ds = FCFP_Dataset(
            rdkit_fingerprints=rdkit_fingerprints[nonpolar_trn["idx"]],
            morgan_fingerprints=morgan_fingerprints[nonpolar_trn["idx"]],
            phases=encoded_phases[nonpolar_trn["idx"]],
            retention_values=input_ri[nonpolar_trn["idx"]],
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=512,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )
        b_train, b_eval = train_fn(
            model=model,
            optim=torch.optim.Adam(
                model.parameters(),
                lr=1e-4,
            ),
            loss_fn=nn.L1Loss(reduction="sum"),
            epochs=pretrain_epochs,
            train_dl=train_dl,
            eval_dl=eval_dl,
            name=(f"fcfp-optuna-pretrain-" f"{model.n_layers}-{model.hidden_wts}"),
            trial=trial,
        )
        print(f"Nonpolar results: {b_train:.3f} {b_eval:.3f}")

        eval_ds = FCFP_Dataset(
            rdkit_fingerprints=rdkit_fingerprints[polar_tst["idx"]],
            morgan_fingerprints=morgan_fingerprints[polar_tst["idx"]],
            phases=encoded_phases[polar_tst["idx"]],
            retention_values=input_ri[polar_tst["idx"]],
        )
        eval_dl = DataLoader(
            eval_ds,
            batch_size=512,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
        )
        train_ds = FCFP_Dataset(
            rdkit_fingerprints=rdkit_fingerprints[polar_trn["idx"]],
            morgan_fingerprints=morgan_fingerprints[polar_trn["idx"]],
            phases=encoded_phases[polar_trn["idx"]],
            retention_values=input_ri[polar_trn["idx"]],
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=512,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )
        b_train, b_eval = train_fn(
            model=model,
            optim=torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
            ),
            loss_fn=nn.L1Loss(reduction="sum"),
            epochs=100,
            train_dl=train_dl,
            eval_dl=eval_dl,
            name=(f"fcfp-optuna-polar-" f"{model.n_layers}-{model.hidden_wts}"),
            epoch_start=pretrain_epochs,
            trial=trial,
        )
        return b_eval

    study = optuna.create_study(
        study_name="FCFP_SiLU",
        storage=f"sqlite:///{BASE_DIR / 'logs/optuna.db'}",
        directions=["minimize"],
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=60, n_min_trials=5
        ),
    )
    study.optimize(objective, n_trials=50)
