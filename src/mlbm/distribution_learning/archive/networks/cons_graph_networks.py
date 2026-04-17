import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from torchlbm.core.lattices.d2q9 import D2Q9


def msre(y_true, y_pred, eps=1e-14):
    return torch.mean(torch.square((y_pred - y_true) / (y_true + eps)))


ACTIVATION_FUNCS = {
    "relu": nn.ReLU,
    "softmax": nn.Softmax(dim=1),
    "none": None,
}

LOSS_FUNCS = {
    "l1": nn.L1Loss(),
    "mse": nn.MSELoss(),
    "msre": msre,
}

class ConsGIN(nn.Module):
    def __init__(self, hparams, input_dim) -> None:
        super().__init__()
        torch.manual_seed(12345)

        self.hparams = hparams
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.layers = hparams.get("layers", [15, 15])
        self.activation_func = ACTIVATION_FUNCS[hparams.get("activation_func", "relu")]
        self.last_activ = ACTIVATION_FUNCS[hparams.get("last_activ", "none")]

        hidden_dim = hparams.get("hidden_dim", 20)
        train_eps = hparams.get("train_eps", False)
        self.conv1 = gnn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()), train_eps=train_eps)
        self.conv2 = gnn.GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()), train_eps=train_eps)
        self.conv3 = gnn.GINConv(nn.Sequential(nn.Linear(hidden_dim, 1)), train_eps=train_eps)

        lattice = D2Q9()
        self.lattice_velocities = torch.tensor(lattice.lattice_velocities(), device=self.device)

        # self.activ = nn.Softmax(dim=1)

        self.loss_func = LOSS_FUNCS[hparams.get("loss_func", "msre")]

        self.configure_optimizer()

    def forward(self, data):
        fpre, edge_index = data.x, data.edge_index

        x = self.conv1(fpre, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        # x = self.activ(x.reshape(-1, 9))

        fpred = x.reshape(-1, 9)
        fpre = fpre.reshape(-1, 9)
        fcorr = fpred - 1/9 * torch.sum(fpred - fpre, dim=1).unsqueeze(-1) - 1/6 * torch.einsum("dQ,dN->NQ", self.lattice_velocities, torch.einsum("dQ,NQ->dN", self.lattice_velocities, fpred - fpre))
        fcorr = fcorr.reshape(-1, 1)

        return fcorr
    
    def configure_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.get("lr", 1e-3), weight_decay=self.hparams.get("weight_decay", 0))

    def training_step(self, batch):
        self.train()

        self.optimizer.zero_grad()  # reset gradients

        data = batch.to(self.device)
        out = self.forward(data)
        loss = self.loss_func(data.y, out)
        loss.backward()
        self.optimizer.step()

        return loss

    def validation_step(self, batch):
        self.eval()

        data = batch.to(self.device)
        out = self.forward(data)
        loss = self.loss_func(data.y, out)

        return loss
    
class ConsGINE(nn.Module):
    def __init__(self, hparams, input_dim) -> None:
        super().__init__()
        torch.manual_seed(12345)

        self.hparams = hparams
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.layers = hparams.get("layers", [15, 15])
        self.activation_func = ACTIVATION_FUNCS[hparams.get("activation_func", "relu")]
        self.last_activ = ACTIVATION_FUNCS[hparams.get("last_activ", "none")]

        hidden_dim = hparams.get("hidden_dim", 20)
        train_eps = hparams.get("train_eps", False)
        self.conv1 = gnn.GINEConv(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()), train_eps=train_eps, edge_dim=1)
        self.conv2 = gnn.GINEConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()), train_eps=train_eps, edge_dim=1)
        self.conv3 = gnn.GINEConv(nn.Sequential(nn.Linear(hidden_dim, 1)), train_eps=train_eps, edge_dim=1)

        lattice = D2Q9()
        self.lattice_velocities = torch.tensor(lattice.lattice_velocities(), device=self.device)

        # self.activ = nn.Softmax(dim=1)

        self.loss_func = LOSS_FUNCS[hparams.get("loss_func", "msre")]

        self.configure_optimizer()

    def forward(self, data):
        fpre, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(fpre, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.conv3(x, edge_index, edge_attr)
        # x = self.activ(x.reshape(-1, 9))

        fpred = x.reshape(-1, 9)
        fpre = fpre.reshape(-1, 9)
        fcorr = fpred - 1/9 * torch.sum(fpred - fpre, dim=1).unsqueeze(-1) - 1/6 * torch.einsum("dQ,dN->NQ", self.lattice_velocities, torch.einsum("dQ,NQ->dN", self.lattice_velocities, fpred - fpre))
        fcorr = fcorr.reshape(-1, 1)

        return fcorr
    
    def configure_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.get("lr", 1e-3), weight_decay=self.hparams.get("weight_decay", 0))

    def training_step(self, batch):
        self.train()

        self.optimizer.zero_grad()  # reset gradients

        data = batch.to(self.device)
        out = self.forward(data)
        loss = self.loss_func(data.y, out)
        loss.backward()
        self.optimizer.step()

        return loss

    def validation_step(self, batch):
        self.eval()

        data = batch.to(self.device)
        out = self.forward(data)
        loss = self.loss_func(data.y, out)

        return loss
    
class ConsGCN(nn.Module):
    def __init__(self, hparams, input_dim) -> None:
        super().__init__()
        # torch.manual_seed(12345)

        self.hparams = hparams
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.layers = hparams.get("layers", [15, 15])
        self.activation_func = ACTIVATION_FUNCS[hparams.get("activation_func", "relu")]
        self.last_activ = ACTIVATION_FUNCS[hparams.get("last_activ", "none")]

        hidden_dim = hparams.get("hidden_dim", 20)
        # encoder_dim = hparams.get("encoder_dim", 10)
        # decoder_dim = hparams.get("decoder_dim", 10)
        # train_eps = hparams.get("train_eps", False)
        # self.encoder = nn.Linear(input_dim, encoder_dim)
        # self.conv1 = gnn.GCNConv(encoder_dim, hidden_dim)
        # self.conv2 = gnn.GCNConv(hidden_dim, hidden_dim)
        # self.conv3 = gnn.GCNConv(hidden_dim, decoder_dim)
        # self.decoder = nn.Linear(decoder_dim, 1)

        self.conv1 = gnn.GCNConv(input_dim, hidden_dim, bias=True)
        self.conv2 = gnn.GCNConv(hidden_dim, hidden_dim, bias=True)
        self.conv3 = gnn.GCNConv(hidden_dim, 1, bias=True)
        # self.conv2 = gnn.GCNConv(hidden_dim, hidden_dim*2, bias=True)
        # self.conv3 = gnn.GCNConv(hidden_dim*2, hidden_dim, bias=True)
        # self.conv4 = gnn.GCNConv(hidden_dim, 1, bias=True)

        lattice = D2Q9()
        self.lattice_velocities = torch.tensor(lattice.lattice_velocities(), device=self.device)

        self.activ = nn.Softmax(dim=1)

        self.loss_func = LOSS_FUNCS[hparams.get("loss_func", "msre")]

        self.configure_optimizer()

    def forward(self, data):
        fpre, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(fpre, edge_index, edge_attr)
        x = F.relu(x)
        # x = F.gelu(x)
        # x = F.leaky_relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        # x = F.gelu(x)
        # x = F.leaky_relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        # x = F.relu(x)
        # x = self.conv4(x, edge_index, edge_attr)
        # x = self.encoder(fpre)
        # x = F.relu(x)
        # x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = self.conv3(x, edge_index)
        # x = F.relu(x)
        # x = self.decoder(x)
        # x = self.activ(x.reshape(-1, 9)).reshape(-1, 1)

        fpred = x.reshape(-1, 9)
        fpre = fpre.reshape(-1, 9)
        fcorr = fpred - 1/9 * torch.sum(fpred - fpre, dim=1).unsqueeze(-1) - 1/6 * torch.einsum("dQ,dN->NQ", self.lattice_velocities, torch.einsum("dQ,NQ->dN", self.lattice_velocities, fpred - fpre))
        fcorr = fcorr.reshape(-1, 1)

        return fcorr
    
    def configure_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.get("lr", 1e-3), weight_decay=self.hparams.get("weight_decay", 0))

    def training_step(self, batch):
        self.train()

        self.optimizer.zero_grad()  # reset gradients

        data = batch.to(self.device)
        out = self.forward(data)
        loss = self.loss_func(data.y, out)
        loss.backward()
        self.optimizer.step()

        return loss

    def validation_step(self, batch):
        self.eval()

        data = batch.to(self.device)
        out = self.forward(data)
        loss = self.loss_func(data.y, out)

        return loss