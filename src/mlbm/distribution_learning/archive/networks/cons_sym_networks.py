import torch
import torch.nn as nn
import random


def msre(y_true, y_pred, eps=1e-14):
    return torch.mean(torch.square((y_pred - y_true) / (y_true + eps)))


def msre_equivariance_loss(y_true, y_pred, sym_error, eps=1e-14):
    return torch.mean(torch.square((y_pred - y_true) / (y_true + eps))) + torch.mean(sym_error)


def msre_equivariance_loss_v2(y_true, y_pred, eps=1e-14):
    return torch.mean(sum(((y - x) / (x + eps)) ** 2 for x, y in zip(y_true, y_pred)) / 8.0)


ACTIVATION_FUNCS = {
    "relu": nn.ReLU,
    "softmax": nn.Softmax(dim=1),
    "none": None,
}

LOSS_FUNCS = {
    "l1": nn.L1Loss,
    "mse": nn.MSELoss,
    "msre": msre,
    "msre_equivariance_loss": msre_equivariance_loss_v2,
}


def mirror(x):
    return torch.cat(
        [x[:, 0, None], x[:, 1, None], x[:, 4, None], x[:, 3, None], x[:, 2, None], x[:, 8, None], x[:, 7, None], x[:, 6, None], x[:, 5, None]], dim=-1
    )


def rotate(x, k):
    return torch.cat([x[:, 0, None], x[:, 1:5].roll(k, 1), x[:, 5:].roll(k, 1)], dim=-1)


class ExactConsSymNN(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()

        self.hparams = hparams
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # self.num_layers = hparams["num_layers"]
        self.layers = hparams.get("layers", [50, 50])
        # self.hidden_size = hparams["hidden_size"]
        self.activation_func = ACTIVATION_FUNCS[hparams.get("activation_func", "relu")]
        self.last_activ = ACTIVATION_FUNCS[hparams.get("last_activ", "none")]

        layers = [nn.Linear(9, self.layers[0]), self.activation_func()]

        for i in range(len(self.layers) - 1):
            layers.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            layers.append(self.activation_func())

        layers.append(nn.Linear(self.layers[-1], 9))
        if self.last_activ is not None:
            layers.append(self.last_activ())

        self.model = nn.Sequential(*layers)

        self.loss_func = LOSS_FUNCS[hparams.get("loss_func", "msre")]

        A = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.5, 0.5, 0.0, -1.5, -1.0, 1.0, -1.0, -2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.5, 1.0, 0, 0, 1.0, 1.0],
        ]

        B = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, -2.0, -1.0, 0.0, -2.0, -2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, -0.5, 0.0, 1.5, 1.0, 0.0, 1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [-0.5, -0.5, 0.0, -0.5, -1.0, 0, 0, -1.0, 0.0],
        ]

        self.A = torch.tensor(A, device=self.device)
        self.B = torch.tensor(B, device=self.device)

        self.configure_optimizer()

    def rotate_data(self, x):
        return [
            x,
            rotate(x, 1),
            rotate(x, 2),
            rotate(x, 3),
            mirror(x),
            rotate(mirror(x), 1),
            rotate(mirror(x), 2),
            rotate(mirror(x), 3),
        ]

    def derotate_data(self, x):
        return [
            x[0],
            rotate(x[1], -1),
            rotate(x[2], -2),
            rotate(x[3], -3),
            mirror(x[4]),
            mirror(rotate(x[5], -1)),
            mirror(rotate(x[6], -2)),
            mirror(rotate(x[7], -3)),
        ]

    def algebraic_reconstruction(self, fpre, fpred):
        recon1 = torch.einsum("iQ,NQ->Ni", self.A, fpre)
        recon2 = torch.einsum("iQ,NQ->Ni", self.B, fpred)
        return recon1 + recon2

    def forward(self, x):
        symmetry_combinations = self.rotate_data(x)
        symmetry_results = [self.model(i) for i in symmetry_combinations]
        out = [self.algebraic_reconstruction(fpre, fpred) for fpre, fpred in zip(symmetry_combinations, symmetry_results)]
        grp_avg = sum(self.derotate_data(out)) / 8.0
        return grp_avg

    def configure_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.get("lr", 1e-3), weight_decay=self.hparams.get("weight_decay", 0))

    def training_step(self, batch):
        self.train()

        self.optimizer.zero_grad()  # reset gradients

        x, y = batch[0].to(self.device), batch[1].to(self.device)
        out = self.forward(x)
        loss = self.loss_func(y, out)
        loss.backward()
        self.optimizer.step()

        return loss

    def validation_step(self, batch):
        self.eval()

        x, y = batch[0].to(self.device), batch[1].to(self.device)
        out = self.forward(x)
        loss = self.loss_func(y, out)

        return loss

    # def save(self, path):
    #     torch.save(self, path)

    # @classmethod
    # def load(cls, path):
    #     model = torch.load(path)
    #     return model


class LossBasedConsSymNN(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()

        self.hparams = hparams
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # self.num_layers = hparams["num_layers"]
        self.layers = hparams.get("layers", [50, 50])
        # self.hidden_size = hparams["hidden_size"]
        self.activation_func = ACTIVATION_FUNCS[hparams.get("activation_func", "relu")]
        self.last_activ = ACTIVATION_FUNCS[hparams.get("last_activ", "none")]

        layers = [nn.Linear(9, self.layers[0]), self.activation_func()]

        for i in range(len(self.layers) - 1):
            layers.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            layers.append(self.activation_func())

        layers.append(nn.Linear(self.layers[-1], 9))
        if self.last_activ is not None:
            layers.append(self.last_activ())

        self.model = nn.Sequential(*layers)

        self.loss_func = LOSS_FUNCS[hparams.get("loss_func", "msre_equivariance_loss")]

        A = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.5, 0.5, 0.0, -1.5, -1.0, 1.0, -1.0, -2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.5, 1.0, 0, 0, 1.0, 1.0],
        ]

        B = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, -2.0, -1.0, 0.0, -2.0, -2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, -0.5, 0.0, 1.5, 1.0, 0.0, 1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [-0.5, -0.5, 0.0, -0.5, -1.0, 0, 0, -1.0, 0.0],
        ]

        self.A = torch.tensor(A, device=self.device)
        self.B = torch.tensor(B, device=self.device)

        self.configure_optimizer()

    def rotate_data(self, x):
        return [
            x,
            rotate(x, 1),
            rotate(x, 2),
            rotate(x, 3),
            mirror(x),
            rotate(mirror(x), 1),
            rotate(mirror(x), 2),
            rotate(mirror(x), 3),
        ]

    def algebraic_reconstruction(self, fpre, fpred):
        recon1 = torch.einsum("iQ,NQ->Ni", self.A, fpre)
        recon2 = torch.einsum("iQ,NQ->Ni", self.B, fpred)
        return recon1 + recon2

    def forward(self, x):
        pred = self.model(x)
        out = self.algebraic_reconstruction(x, pred)
        return out

    def configure_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.get("lr", 1e-3), weight_decay=self.hparams.get("weight_decay", 0))

    def training_step(self, batch):
        self.train()

        self.optimizer.zero_grad()  # reset gradients

        x, y = batch[0].to(self.device), batch[1].to(self.device)
        x_rotated = self.rotate_data(x)
        out_combinations = [self.forward(i) for i in x_rotated]
        y_rotated = self.rotate_data(y)
        # out = out_combinations[0]
        # out_rotated = self.rotate_data(out)

        # sym_error =

        # out = self.forward(x)
        loss = self.loss_func(y_rotated, out_combinations)
        # print(loss.shape)
        loss.backward()
        self.optimizer.step()

        return loss

    def validation_step(self, batch):
        self.eval()

        x, y = batch[0].to(self.device), batch[1].to(self.device)
        x_rotated = self.rotate_data(x)
        out_combinations = [self.forward(i) for i in x_rotated]
        y_rotated = self.rotate_data(y)
        # out = self.forward(x)
        loss = self.loss_func(y_rotated, out_combinations)

        return loss

    def save(self, path):
        torch.save(self, path)

    @classmethod
    def load(cls, path):
        model = torch.load(path)
        return model
