import json
from time import perf_counter

import numpy as np
import optuna
import torch
from catboost import CatBoostRegressor, Pool, sum_models
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from nist_polar_filtering import BASE_DIR, SEED_END, SEED_START
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

    bar = tqdm(range(epoch_start, epoch_start + epochs), leave=False, position=2)
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

    unique_nonpolar_inchikeys = set(np.unique(input_inchikeys[nonpolar_mask]))
    unique_polar_inchikeys = np.unique(input_inchikeys[polar_mask])

    pred_df = ri_df.copy()
    for seed in tqdm(np.arange(SEED_START, SEED_END + 1), position=0):
        kf = KFold(5, shuffle=True, random_state=seed)
        pbar = tqdm(
            kf.split(np.arange(len(unique_polar_inchikeys))), position=1, total=5
        )
        predicted_values = np.zeros(len(input_inchikeys))
        for k, (trn_idx, tst_idx) in enumerate(pbar):
            unique_trn_polar_inchikeys = set(unique_polar_inchikeys[trn_idx])
            unique_tst_polar_inchikeys = set(unique_polar_inchikeys[tst_idx])
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

            pbar.write(f"----------{k}----------")
            pbar.write(f"Nonpolar TST: {len(nonpolar_tst['idx'])}")
            pbar.write(f"Nonpolar TRN: {len(nonpolar_trn['idx'])}")
            pbar.write(f"Polar TST: {len(polar_tst['idx'])}")
            pbar.write(f"Polar TRN: {len(polar_trn['idx'])}")

            np_model = CatBoostRegressor(
                loss_function="MAE",
                task_type="GPU",
                devices="0",
                metric_period=20,
                learning_rate=8.5e-3,
                depth=10,
                iterations=7000,
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
            np_model.fit(trn_ds, eval_set=eval_ds, plot=False)

            b_train = np_model.get_best_score().get("learn", {"MAE": np.nan})["MAE"]
            b_eval = np_model.get_best_score().get("validation", {"MAE": np.nan})["MAE"]
            pbar.write(f"NonPolar: {b_train} {b_eval}")

            ohe_baseline = np.zeros(
                (len(polar_tst["idx"]), max(phases_dict.values()) + 1), dtype=np.int8
            )
            ohe_baseline[:, 1] = 1

            eval_baseline = Pool(
                data=np.hstack(
                    [
                        rdkit_descriptors[polar_tst["idx"]],
                        morgan_fingerprints[polar_tst["idx"]],
                        rdkit_fingerprints[polar_tst["idx"]],
                        ohe_baseline,
                    ]
                )
            )
            eval_baseline = np_model.predict(data=eval_baseline)

            ohe_baseline = np.zeros(
                (len(polar_trn["idx"]), max(phases_dict.values()) + 1), dtype=np.int8
            )
            ohe_baseline[:, 1] = 1

            trn_baseline = Pool(
                data=np.hstack(
                    [
                        rdkit_descriptors[polar_trn["idx"]],
                        morgan_fingerprints[polar_trn["idx"]],
                        rdkit_fingerprints[polar_trn["idx"]],
                        ohe_baseline,
                    ]
                )
            )
            trn_baseline = np_model.predict(data=trn_baseline)

            model = CatBoostRegressor(
                loss_function="MAE",
                task_type="GPU",
                devices="0",
                metric_period=20,
                learning_rate=2e-3,
                depth=10,
                iterations=5000,
                use_best_model=True,
                silent=True,
                allow_writing_files=False)

            eval_ds = Pool(
                data=np.hstack(
                    [
                        rdkit_descriptors[polar_tst["idx"]],
                        morgan_fingerprints[polar_tst["idx"]],
                        rdkit_fingerprints[polar_tst["idx"]],
                        ohe_phases[polar_tst["idx"]],
                    ]
                ),
                label=input_ri[polar_tst["idx"]])
            eval_ds.set_baseline(eval_baseline)

            trn_ds = Pool(
                data=np.hstack(
                    [
                        rdkit_descriptors[polar_trn["idx"]],
                        morgan_fingerprints[polar_trn["idx"]],
                        rdkit_fingerprints[polar_trn["idx"]],
                        ohe_phases[polar_trn["idx"]],
                    ]
                ),
                label=input_ri[polar_trn["idx"]])
            trn_ds.set_baseline(trn_baseline)

            model.fit(trn_ds, eval_set=eval_ds, plot=False)

            b_train = model.get_best_score().get("learn", {"MAE": np.nan})["MAE"]
            b_eval = model.get_best_score().get("validation", {"MAE": np.nan})["MAE"]
            pbar.write(f"Polar: {b_train} {b_eval}")
            model.save_model(BASE_DIR / f"models/cb-5fold-polar-{seed}-{k}.cbm")

            summed_model = sum_models([np_model, model])
            summed_model.save_model(BASE_DIR / f"models/cb-5fold-sum-{seed}-{k}.cbm")

            predicted_values[polar_tst["idx"]] = np.round(
                model.predict(eval_ds),
                5)
            pbar.write(f"{np.nonzero(predicted_values)}")
        pred_df[f"cb-{seed}"] = predicted_values
    pred_df.to_parquet(
        BASE_DIR / f"data/output/5fold-cb-{SEED_START}-{SEED_END}.parquet"
    )
