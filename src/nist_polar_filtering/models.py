"""
models.py

This module contains the definitions of different neural network architectures
used for predicting retention times based on molecular information.
Each model includes forward methods for computation,
and train_fn and eval_fn methods for training and evaluation, respectively.

Classes:
    CNN: Convolutional Neural Network for processing molecular encodings.
    FCD: Fully Connected Network for processing molecular descriptors.
    FCFP: Fully Connected Network for processing molecular fingerprints.
    GNN: Graph Neural Network for processing molecular graphs.

Functions:
    None

Attributes:
    None
"""

import torch
import torch.nn as nn
import torch_geometric.nn as gnn


class CNN(nn.Module):
    def __init__(
        self,
        n_sym,
        n_phases,
        n_conv_layers: int = 2,
        kernel_size: int = 6,
        conv_channels: int = 300,
        n_lin_layers: int = 2,
    ):
        super().__init__()
        self.n_conv_layers = int(n_conv_layers)
        self.kernel_size = int(kernel_size)
        self.conv_channels = int(conv_channels)
        self.n_lin_layers = int(n_lin_layers)
        self.n_phases = int(n_phases)
        self.embed_atoms = nn.Embedding(n_sym, n_sym, padding_idx=0).from_pretrained(
            torch.eye(n_sym), freeze=False
        )
        self.embed_phases = nn.Embedding(n_phases, n_phases).from_pretrained(
            torch.eye(n_phases), freeze=False
        )

        self.conv_in = nn.Sequential(
            nn.Conv1d(n_sym, conv_channels, kernel_size, padding="same"),
            nn.ReLU(inplace=True),
        )

        n_conv_layers = max(1, n_conv_layers)
        self.conv_layers = nn.ModuleList(
            [
                self._make_conv_layer(conv_channels, conv_channels, kernel_size)
                for _ in range(n_conv_layers - 1)
            ]
        )
        self.lin0 = nn.Sequential(
            nn.Linear(conv_channels + n_phases, 2 * conv_channels),
            nn.ReLU(inplace=True),
        )

        n_lin_layers = max(2, n_lin_layers)
        self.lin_layers = nn.ModuleList(
            [
                self._make_lin_layer(2 * conv_channels, 2 * conv_channels)
                for _ in range(n_lin_layers - 2)
            ]
        )
        self.out = nn.Sequential(
            nn.Linear(2 * conv_channels, 1),
            nn.Identity(),
        )

    def _make_conv_layer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
            nn.ReLU(inplace=True),
        )

    def _make_lin_layer(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features), nn.ReLU(inplace=True)
        )

    def forward(self, x, phases):
        x = torch.permute(self.embed_atoms(x), (0, 2, 1))
        phases = self.embed_phases(phases)
        x = self.conv_in(x)
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
        x = torch.sum(x, dim=2).squeeze()
        x = torch.cat([x, phases], dim=1)
        x = self.lin0(x)
        for i, lin in enumerate(self.lin_layers):
            x = lin(x)
        return self.out(x).squeeze()

    def train_fn(self, optim, loss_fn, train_dl):
        self.train()
        epoch_train_loss = 0
        for encoded_smiles, encoded_phases, retention_indices in train_dl:
            optim.zero_grad()
            pred_retention_indices = self.forward(
                encoded_smiles.to(next(self.parameters()).device, non_blocking=True),
                encoded_phases.to(next(self.parameters()).device, non_blocking=True),
            )
            loss = loss_fn(
                pred_retention_indices,
                retention_indices.to(next(self.parameters()).device, non_blocking=True),
            )
            loss.backward()
            optim.step()
            epoch_train_loss += loss.detach().cpu()
        return epoch_train_loss / len(train_dl.dataset)

    def eval_fn(self, loss_fn, eval_dl, return_predictions=False):
        self.eval()
        epoch_eval_loss = 0
        prediction_retention_indices = []
        with torch.no_grad():
            for encoded_smiles, encoded_phases, retention_indices in eval_dl:
                pred_retention_indices = self.forward(
                    encoded_smiles.to(
                        next(self.parameters()).device, non_blocking=True
                    ),
                    encoded_phases.to(
                        next(self.parameters()).device, non_blocking=True
                    ),
                )
                if return_predictions:
                    prediction_retention_indices.append(pred_retention_indices)
                loss = loss_fn(
                    pred_retention_indices,
                    retention_indices.to(
                        next(self.parameters()).device, non_blocking=True
                    ),
                )
                epoch_eval_loss += loss.detach().cpu()
        if return_predictions:
            return torch.cat(prediction_retention_indices, 0).cpu().numpy()
        else:
            return epoch_eval_loss / len(eval_dl.dataset)


class FCD(nn.Module):
    def __init__(
        self,
        n_phases,
        n_descriptors,
        n_layers=5,
        hidden_wts=2048,
        device=torch.device("cuda"),
    ):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_wts = hidden_wts
        self.n_phases = n_phases
        self.n_descriptors = n_descriptors
        self.embed_phases = nn.Embedding(n_phases, n_phases).from_pretrained(
            torch.eye(n_phases), freeze=True
        )
        self.device = device
        self.fc_in = nn.Sequential(
            nn.Linear(n_descriptors + n_phases, hidden_wts),
            nn.ReLU(inplace=True),
        )
        self.hidden_linear = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_wts, hidden_wts), nn.ReLU(inplace=True))
                for _ in range(n_layers)
            ]
        )

        self.fc_out = nn.Sequential(
            nn.Linear(hidden_wts, 1),
            nn.Identity(),
        )

    def forward(self, descriptors, phases):
        ephases = self.embed_phases(phases)
        x = self.fc_in(torch.cat([descriptors, ephases], dim=1))
        for i, lin in enumerate(self.hidden_linear):
            x = lin(x)
        return self.fc_out(x)

    def train_fn(self, optim, loss_fn, train_dl):
        self.train()
        epoch_train_loss = 0
        for descriptors, encoded_phases, values in train_dl:
            optim.zero_grad()
            pred_values = self.forward(
                descriptors.to(next(self.parameters()).device, non_blocking=True),
                encoded_phases.to(next(self.parameters()).device, non_blocking=True),
            )
            loss = loss_fn(pred_values, values.to(self.device, non_blocking=True))
            loss.backward()
            optim.step()
            epoch_train_loss += loss.detach().cpu()
        return epoch_train_loss / len(train_dl.dataset)

    def eval_fn(self, loss_fn, eval_dl, return_predictions=False):
        self.eval()
        epoch_eval_loss = 0
        prediction_values = []
        with torch.no_grad():
            for descriptors, encoded_phases, values in eval_dl:
                pred_values = self.forward(
                    descriptors.to(next(self.parameters()).device, non_blocking=True),
                    encoded_phases.to(
                        next(self.parameters()).device, non_blocking=True
                    ),
                )
                if return_predictions:
                    prediction_values.append(pred_values)
                loss = loss_fn(pred_values, values.to(self.device, non_blocking=True))
                epoch_eval_loss += loss.detach().cpu()
        if return_predictions:
            return torch.cat(prediction_values, 0).cpu().squeeze().numpy()
        else:
            return epoch_eval_loss / len(eval_dl.dataset)


class FCFP(nn.Module):
    def __init__(
        self, n_phases, n_layers=5, hidden_wts=4096, device=torch.device("cuda")
    ):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_wts = hidden_wts
        self.device = device
        self.embed_phases = nn.Embedding(n_phases, n_phases).from_pretrained(
            torch.eye(n_phases), freeze=True
        )
        self.fc_in = nn.Sequential(
            nn.Linear(2048 + n_phases, hidden_wts),
            nn.SiLU(inplace=True),
        )
        self.hidden_linear = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_wts, hidden_wts), nn.SiLU(inplace=True))
                for _ in range(n_layers)
            ]
        )

        self.fc_out = nn.Sequential(
            nn.Linear(hidden_wts, 1),
            nn.Identity(),
        )

    def forward(self, morgan_fingerprints, rdkit_fingerprints, phases):
        ephases = self.embed_phases(phases)
        x = torch.cat([morgan_fingerprints, rdkit_fingerprints, ephases], dim=1)
        x = self.fc_in(x)
        for i, lin in enumerate(self.hidden_linear):
            x = lin(x)
        return self.fc_out(x)

    def train_fn(self, optim, loss_fn, train_dl):
        self.train()
        epoch_train_loss = 0
        for (
            rdkit_fingerprints,
            morgan_fingerprints,
            phases,
            retention_indices,
        ) in train_dl:
            optim.zero_grad()
            pred_retention_indices = self.forward(
                rdkit_fingerprints.to(
                    next(self.parameters()).device, non_blocking=True
                ),
                morgan_fingerprints.to(
                    next(self.parameters()).device, non_blocking=True
                ),
                phases.to(next(self.parameters()).device, non_blocking=True),
            )
            loss = loss_fn(
                pred_retention_indices,
                retention_indices.to(next(self.parameters()).device, non_blocking=True),
            )
            loss.backward()
            optim.step()
            epoch_train_loss += loss.detach().cpu()
        return epoch_train_loss / len(train_dl.dataset)

    def eval_fn(self, loss_fn, eval_dl, return_predictions=False):
        self.eval()
        epoch_eval_loss = 0
        prediction_retention_indices = []
        with torch.no_grad():
            for (
                rdkit_fingerprints,
                morgan_fingerprints,
                phases,
                retention_indices,
            ) in eval_dl:
                pred_retention_indices = self.forward(
                    rdkit_fingerprints.to(
                        next(self.parameters()).device, non_blocking=True
                    ),
                    morgan_fingerprints.to(
                        next(self.parameters()).device, non_blocking=True
                    ),
                    phases.to(next(self.parameters()).device, non_blocking=True),
                )
                if return_predictions:
                    prediction_retention_indices.append(pred_retention_indices)
                loss = loss_fn(
                    pred_retention_indices,
                    retention_indices.to(
                        next(self.parameters()).device, non_blocking=True
                    ),
                )
                epoch_eval_loss += loss
        if return_predictions:
            return torch.cat(prediction_retention_indices, 0).cpu().squeeze().numpy()
        else:
            return epoch_eval_loss / len(eval_dl.dataset)


class GNN(nn.Module):
    def __init__(
        self,
        n_fingerprints,
        n_phases,
        embed_fingerprints: int = 64,
        n_conv_layers: int = 5,
        conv_channels: int = 256,
        n_lin_layers: int = 2,
    ):
        super().__init__()
        self.n_fingerprints = n_fingerprints
        self.n_phases = n_phases
        self.embed_fingerprints = embed_fingerprints
        self.n_conv_layers = n_conv_layers
        self.conv_channels = conv_channels
        self.n_lin_layers = n_lin_layers

        self.embed = nn.Embedding(n_fingerprints, embed_fingerprints)
        self.phase_embed = nn.Embedding(n_phases, n_phases).from_pretrained(
            torch.eye(n_phases), freeze=False
        )

        self.in_conv = self._make_layer(embed_fingerprints, conv_channels)
        self.conv_layers = nn.ModuleList(
            [
                self._make_layer(conv_channels, conv_channels)
                for _ in range(n_conv_layers)
            ]
        )

        self.lin0 = nn.Sequential(
            nn.Linear(conv_channels + n_phases, 2 * conv_channels),
            nn.ReLU(),
        )

        self.lin_layers = nn.ModuleList(
            [
                self._make_lin_layer(2 * conv_channels, 2 * conv_channels)
                for _ in range(n_lin_layers)
            ]
        )
        self.out = nn.Sequential(nn.Linear(2 * conv_channels, 1), nn.Identity())

    def _make_layer(self, in_channels, out_channels):
        return gnn.Sequential(
            "x, edge_index",
            [
                (gnn.GraphConv(in_channels, out_channels), "x, edge_index -> x"),
                (nn.ReLU(inplace=True), "x -> x"),
            ],
        )

    def _make_lin_layer(self, in_weights, hidden):
        return nn.Sequential(
            nn.Linear(in_weights, hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embed(x)
        ephases = self.phase_embed(data.phase)

        x = self.in_conv(x, edge_index)
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
        x = gnn.pool.global_add_pool(x, data.batch)
        x = torch.cat([x, ephases], dim=1)
        x = self.lin0(x)
        for i, lin in enumerate(self.lin_layers):
            x = lin(x)

        return self.out(x)

    def train_fn(self, optim, loss_fn, train_dl):
        self.train()
        epoch_train_loss = 0
        for data in train_dl:
            optim.zero_grad()
            pred_retention_indices = self.forward(
                data.to(next(self.parameters()).device, non_blocking=True)
            )
            loss = loss_fn(pred_retention_indices.squeeze(), data.y)
            loss.backward()
            optim.step()
            epoch_train_loss += loss.detach().cpu()
        return epoch_train_loss / len(train_dl.dataset)

    def eval_fn(self, loss_fn, eval_dl, return_predictions=False):
        self.eval()
        epoch_eval_loss = 0
        prediction_retention_indices = []
        with torch.no_grad():
            for data in eval_dl:
                pred_retention_indices = self.forward(
                    data.to(next(self.parameters()).device, non_blocking=True)
                )
                if return_predictions:
                    prediction_retention_indices.append(pred_retention_indices)
                loss = loss_fn(pred_retention_indices.squeeze(), data.y)
                epoch_eval_loss += loss.detach().cpu()
        if return_predictions:
            return torch.cat(prediction_retention_indices, 0).cpu().squeeze().numpy()
        else:
            return epoch_eval_loss / len(eval_dl.dataset)
