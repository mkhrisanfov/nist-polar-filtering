import json
from time import perf_counter

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from nist_polar_filtering import BASE_DIR, SEED_END, SEED_START  # , SEED
from nist_polar_filtering.datasets import (
    CNN_Dataset,
)
from nist_polar_filtering.models import CNN
from nist_polar_filtering.preprocessing import (
    load_nist_2023_dataset,
)
from nist_polar_filtering.utils import (
    encode_smiles,
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
    smiles_sym_dict, encoded_smiles = encode_smiles(input_smiles)
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

    pred_df = ri_df.copy()
    for seed in tqdm(np.arange(SEED_START, SEED_END + 1), position=0):
        kf = KFold(5, shuffle=True, random_state=seed)
        pbar = tqdm(
            kf.split(np.arange(len(unique_polar_inchikeys))),
            position=1,
            total=5,
            leave=False,
        )
        predicted_values = np.zeros(len(input_inchikeys))
        for k, (trn_idx, tst_idx) in enumerate(pbar):
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

            pbar.write(f"----------{k}----------")
            pbar.write(f"Nonpolar TST: {len(nonpolar_tst['idx'])}")
            pbar.write(f"Nonpolar TRN: {len(nonpolar_trn['idx'])}")
            pbar.write(f"Polar TST: {len(polar_tst['idx'])}")
            pbar.write(f"Polar TRN: {len(polar_trn['idx'])}")

            learning_rate = 6e-5
            pretrain_epochs = 30
            model = CNN(
                len(smiles_sym_dict) + 1,
                max(encoded_phases) + 1,
                n_conv_layers=3,
                kernel_size=5,
                conv_channels=512,
                n_lin_layers=4,
            ).to(device)

            eval_ds = CNN_Dataset(
                encoded_smis=encoded_smiles[nonpolar_tst["idx"]],
                encoded_phases=encoded_phases[nonpolar_tst["idx"]],
                retention_values=input_ri[nonpolar_tst["idx"]],
            )
            eval_dl = DataLoader(
                eval_ds,
                batch_size=512,
                shuffle=False,
                num_workers=4,
                persistent_workers=True,
            )
            train_ds = CNN_Dataset(
                encoded_smis=encoded_smiles[nonpolar_trn["idx"]],
                encoded_phases=encoded_phases[nonpolar_trn["idx"]],
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
                name=(
                    f"cnn-5fold-pretrain-{seed}-{k}-"
                    f"{model.kernel_size}-{model.n_conv_layers}"
                    f"-{model.conv_channels}-{model.n_lin_layers}"
                ),
            )
            pbar.write(f"Nonpolar results: {b_train:.3f} {b_eval:.3f}")

            eval_ds = CNN_Dataset(
                encoded_smis=encoded_smiles[polar_tst["idx"]],
                encoded_phases=encoded_phases[polar_tst["idx"]],
                retention_values=input_ri[polar_tst["idx"]],
            )
            eval_dl = DataLoader(
                eval_ds,
                batch_size=512,
                shuffle=False,
                num_workers=4,
                persistent_workers=True,
            )
            train_ds = CNN_Dataset(
                encoded_smis=encoded_smiles[polar_trn["idx"]],
                encoded_phases=encoded_phases[polar_trn["idx"]],
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
                epochs=150,
                train_dl=train_dl,
                eval_dl=eval_dl,
                name=(
                    f"cnn-5fold-polar-{seed}-{k}-"
                    f"{model.kernel_size}-{model.n_conv_layers}"
                    f"-{model.conv_channels}-{model.n_lin_layers}"
                ),
                epoch_start=pretrain_epochs,
            )
            pbar.write(f"Polar results: {b_train:.3f} {b_eval:.3f}")

            model.load_state_dict(
                torch.load(
                    BASE_DIR
                    / "models"
                    / (
                        f"cnn-5fold-polar-{seed}-{k}-"
                        f"{model.kernel_size}-{model.n_conv_layers}"
                        f"-{model.conv_channels}-{model.n_lin_layers}"
                        ".pth"
                    )
                )
            )
            predicted_values[polar_tst["idx"]] = np.round(
                model.eval_fn(
                    loss_fn=nn.L1Loss(reduction="sum"),
                    eval_dl=eval_dl,
                    return_predictions=True,
                ),
                5,
            )
            pbar.write(f"{np.nonzero(predicted_values)}")
        pred_df[f"cnn-{seed}"] = predicted_values
    pred_df.to_parquet(
        BASE_DIR / f"data/output/5fold-cnn-{SEED_START}-{SEED_END}.parquet"
    )
