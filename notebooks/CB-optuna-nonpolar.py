import json
from time import perf_counter

import numpy as np
import optuna
import torch
import torch.nn as nn
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from nist_polar_filtering import BASE_DIR, SEED
from nist_polar_filtering.models import FCD
from nist_polar_filtering.preprocessing import load_nist_2023_dataset
from nist_polar_filtering.utils import generate_fingerprints, generate_rdkit_descriptors


def train_fn(
    model,
    optim,
    loss_fn,
    epochs,
    train_dl,
    eval_dl,
    name,
    epoch_start=0,
    trial=None):

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
    rdkit_descriptors = generate_rdkit_descriptors(
        input_molecules, load=True, prefix="full"
    )
    morgan_fingerprints, rdkit_fingerprints = generate_fingerprints(input_molecules)

    phases_dict = json.load((BASE_DIR / "data/output/phases-23-all.json").open("r"))
    input_phases = ri_df["active_phase"].str.lower()
    encoded_phases = np.array([phases_dict.get(x, 0) for x in input_phases])
    ohe_phases = np.zeros(
        (len(input_phases), max(phases_dict.values()) + 1), dtype=np.int8
    )
    for i, x in enumerate(tqdm(input_phases)):
        ohe_phases[i, phases_dict.get(x, 0)] = 1

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

    unique_nonpolar_inchikeys = np.unique(input_inchikeys[nonpolar_mask])
    unique_polar_inchikeys = np.unique(input_inchikeys[polar_mask])
    trn_idx, tst_idx = train_test_split(
        np.arange(len(unique_nonpolar_inchikeys)), test_size=0.2, random_state=SEED
    )
    unique_trn_nonpolar_inchikeys = set(unique_nonpolar_inchikeys[trn_idx])
    unique_tst_nonpolar_inchikeys = set(unique_nonpolar_inchikeys[tst_idx])
    (
        nonpolar_trn,
        nonpolar_tst,
        polar_trn,
        polar_tst) = (
        {"idx": [], "inchikeys": []},
        {"idx": [], "inchikeys": []},
        {"idx": [], "inchikeys": []},
        {"idx": [], "inchikeys": []})
    for i, key in enumerate(input_inchikeys):
        if encoded_phases[i] < 43:
            if key in unique_tst_nonpolar_inchikeys:
                nonpolar_tst["idx"].append(i)
                nonpolar_tst["inchikeys"].append(key)
            else:
                nonpolar_trn["idx"].append(i)
                nonpolar_trn["inchikeys"].append(key)

    print(f"Nonpolar TST: {len(nonpolar_tst['idx'])}")
    print(f"Nonpolar TRN: {len(nonpolar_trn['idx'])}")

    def objective(trial):
        model = CatBoostRegressor(
            loss_function="MAE",
            task_type="GPU",
            devices="0",
            metric_period=20,
            learning_rate=trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True),
            depth=trial.suggest_int("depth", 6, 10),
            iterations=trial.suggest_int("iterations", 1000, 10000, step=500),
            use_best_model=True,
            silent=True,
            allow_writing_files=False)

        eval_ds = Pool(
            data=np.hstack(
                [
                    rdkit_descriptors[nonpolar_tst["idx"]],
                    morgan_fingerprints[nonpolar_tst["idx"]],
                    rdkit_fingerprints[nonpolar_tst["idx"]],
                    ohe_phases[nonpolar_tst["idx"]],
                ]
            ),
            label=input_ri[nonpolar_tst["idx"]])
        trn_ds = Pool(
            data=np.hstack(
                [
                    rdkit_descriptors[nonpolar_trn["idx"]],
                    morgan_fingerprints[nonpolar_trn["idx"]],
                    rdkit_fingerprints[nonpolar_trn["idx"]],
                    ohe_phases[nonpolar_trn["idx"]],
                ]
            ),
            label=input_ri[nonpolar_trn["idx"]])

        model.fit(trn_ds, eval_set=eval_ds, plot=False)
        b_train = model.get_best_score().get("learn", {"MAE": np.nan})["MAE"]
        b_eval = model.get_best_score().get("validation", {"MAE": np.nan})["MAE"]
        print(b_train, b_eval)

        return b_eval

    study = optuna.create_study(
        study_name="CB_NP2",
        storage=f"sqlite:///{BASE_DIR / 'logs/optuna.db'}",
        directions=["minimize"],
        load_if_exists=True)
    study.enqueue_trial(
        params={"learning_rate": 7e-3, "depth": 10, "iterations": 5000},
        skip_if_exists=True)
    study.optimize(objective, n_trials=100)
