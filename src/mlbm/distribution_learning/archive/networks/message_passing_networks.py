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

class MPNNLayer(gnn.MessagePassing):
    def __init__(self, emb_dim=20, edge_dim=1) -> None:
        super().__init__(aggr="add")

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        # MLP `\psi` for computing messages `m_ij`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: (2d + d_e) -> d
        self.mlp_msg = nn.Sequential(
            nn.Linear(2*emb_dim + edge_dim, emb_dim), nn.ReLU(), #nn.BatchNorm1d(emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),#nn.BatchNorm1d(emb_dim), nn.ReLU()
          )
        
        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: 2d -> d
        self.mlp_upd = nn.Sequential(
            nn.Linear(2*emb_dim, emb_dim), nn.ReLU(), #nn.BatchNorm1d(emb_dim), nn.ReLU(), 
            nn.Linear(emb_dim, emb_dim), nn.ReLU(), #nn.BatchNorm1d(emb_dim), nn.ReLU()
          )
        
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        # x_i, x_j: [E, emb_dim]
        # edge_attr: [E, edge_dim]
        tmp = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp_msg(tmp)
    
    def update(self, aggr_out, x):
        # aggr_out: [N, emb_dim]
        # x: [N, emb_dim]
        tmp = torch.cat([x, aggr_out], dim=-1)
        return self.mlp_upd(tmp)
    
class CoordMPNNLayer(gnn.MessagePassing):
    def __init__(self, emb_dim=20, edge_dim=1, dims=2) -> None:
        super().__init__(aggr="add")

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        # MLP `\psi` for computing messages `m_ij`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: (2d + d_e) -> d
        self.mlp_msg = nn.Sequential(
            nn.Linear(2*emb_dim + 2*dims + edge_dim, emb_dim), nn.ReLU(), #nn.BatchNorm1d(emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),#nn.BatchNorm1d(emb_dim), nn.ReLU()
          )
        
        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: 2d -> d
        self.mlp_upd = nn.Sequential(
            nn.Linear(2*emb_dim, emb_dim), nn.ReLU(), #nn.BatchNorm1d(emb_dim), nn.ReLU(), 
            nn.Linear(emb_dim, emb_dim), nn.ReLU(), #nn.BatchNorm1d(emb_dim), nn.ReLU()
          )
        
    def forward(self, x, edge_index, edge_attr, pos):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, pos=pos)
    
    def message(self, x_i, x_j, pos_i, pos_j, edge_attr):
        # x_i, x_j: [E, emb_dim]
        # edge_attr: [E, edge_dim]
        tmp = torch.cat([x_i, x_j, pos_i, pos_j, edge_attr], dim=-1)
        # print(tmp.shape)
        return self.mlp_msg(tmp)
    
    def update(self, aggr_out, x):
        # aggr_out: [N, emb_dim]
        # x: [N, emb_dim]
        tmp = torch.cat([x, aggr_out], dim=-1)
        return self.mlp_upd(tmp)
    

class MPNN(nn.Module):
    def __init__(self, hparams, input_dim, edge_dim, output_dim) -> None:
        super().__init__()
        

        self.hparams = hparams
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # self.layers = hparams.get("layers", [15, 15])
        # self.activation_func = ACTIVATION_FUNCS[hparams.get("activation_func", "relu")]
        # self.last_activ = ACTIVATION_FUNCS[hparams.get("last_activ", "none")]

        hidden_dim = hparams.get("hidden_dim", 20)
        # train_eps = hparams.get("train_eps", False)

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.layer1 = MPNNLayer(emb_dim=hidden_dim, edge_dim=edge_dim)
        self.layer2 = MPNNLayer(emb_dim=hidden_dim, edge_dim=edge_dim)
        self.layer3 = MPNNLayer(emb_dim=hidden_dim, edge_dim=edge_dim)
        # self.conv2 = MPNNLayer(emb_dim=hidden_dim, edge_dim=hidden_dim)
        # self.conv3 = MPNNLayer(emb_dim=1, edge_dim=hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)

        lattice = D2Q9()
        self.lattice_velocities = torch.tensor(lattice.lattice_velocities(), device=self.device)

        # self.activ = nn.Softmax(dim=1)

        self.loss_func = LOSS_FUNCS[hparams.get("loss_func", "msre")]

        self.configure_optimizer()

    def forward(self, x, edge_index, edge_attr):
        fpre = x
        x = self.encoder(x)
        x = self.layer1(x, edge_index, edge_attr)
        # x = F.relu(x)
        x = self.layer2(x, edge_index, edge_attr)
        # x = F.relu(x)
        x = self.layer3(x, edge_index, edge_attr)
        x = self.decoder(x)

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
        out = self.forward(data.x, data.edge_index, data.edge_attr)
        loss = self.loss_func(data.y, out)
        loss.backward()
        self.optimizer.step()

        return loss
    
    def validation_step(self, batch):
        self.eval()

        data = batch.to(self.device)
        out = self.forward(data.x, data.edge_index, data.edge_attr)
        loss = self.loss_func(data.y, out)

        return loss
    
class CoordMPNN(nn.Module):
    def __init__(self, hparams, input_dim, edge_dim, output_dim) -> None:
        super().__init__()
        

        self.hparams = hparams
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # self.layers = hparams.get("layers", [15, 15])
        # self.activation_func = ACTIVATION_FUNCS[hparams.get("activation_func", "relu")]
        # self.last_activ = ACTIVATION_FUNCS[hparams.get("last_activ", "none")]

        hidden_dim = hparams.get("hidden_dim", 20)
        # train_eps = hparams.get("train_eps", False)

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.layer1 = CoordMPNNLayer(emb_dim=hidden_dim, edge_dim=edge_dim)
        self.layer2 = CoordMPNNLayer(emb_dim=hidden_dim, edge_dim=edge_dim)
        self.layer3 = CoordMPNNLayer(emb_dim=hidden_dim, edge_dim=edge_dim)
        # self.conv2 = MPNNLayer(emb_dim=hidden_dim, edge_dim=hidden_dim)
        # self.conv3 = MPNNLayer(emb_dim=1, edge_dim=hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)

        lattice = D2Q9()
        self.lattice_velocities = torch.tensor(lattice.lattice_velocities(), device=self.device)

        # self.activ = nn.Softmax(dim=1)

        self.loss_func = LOSS_FUNCS[hparams.get("loss_func", "msre")]

        self.configure_optimizer()

    def forward(self, x, edge_index, edge_attr, pos):
        fpre = x
        x = self.encoder(x)
        x = self.layer1(x, edge_index, edge_attr, pos)
        # x = F.relu(x)
        x = self.layer2(x, edge_index, edge_attr, pos)
        # x = F.relu(x)
        x = self.layer3(x, edge_index, edge_attr, pos)
        x = self.decoder(x)

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
        out = self.forward(data.x, data.edge_index, data.edge_attr, data.pos)
        loss = self.loss_func(data.y, out)
        loss.backward()
        self.optimizer.step()

        return loss
    
    def validation_step(self, batch):
        self.eval()

        data = batch.to(self.device)
        out = self.forward(data.x, data.edge_index, data.edge_attr, data.pos)
        loss = self.loss_func(data.y, out)

        return loss
    
class NaiveEMPNNLayer(gnn.MessagePassing):
    def __init__(self, emb_dim=20, edge_dim=1, dims=2) -> None:
        super().__init__(aggr="add")

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        # MLP `\psi` for computing messages `m_ij`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: (2d + d_e) -> d
        self.mlp_msg = nn.Sequential(
            nn.Linear(2*emb_dim + 1 + edge_dim, emb_dim), nn.ReLU(), #nn.BatchNorm1d(emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),#nn.BatchNorm1d(emb_dim), nn.ReLU()
          )
        
        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: 2d -> d
        self.mlp_upd = nn.Sequential(
            nn.Linear(2*emb_dim, emb_dim), nn.ReLU(), #nn.BatchNorm1d(emb_dim), nn.ReLU(), 
            nn.Linear(emb_dim, emb_dim), nn.ReLU(), #nn.BatchNorm1d(emb_dim), nn.ReLU()
          )
        
    def forward(self, x, edge_index, edge_attr, pos):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, pos=pos)
    
    def message(self, x_i, x_j, pos_i, pos_j, edge_attr):
        # x_i, x_j: [E, emb_dim]
        # edge_attr: [E, edge_dim]
        coord_diff = torch.sum((pos_i - pos_j)**2, dim=1).unsqueeze(1)
        tmp = torch.cat([x_i, x_j, coord_diff,edge_attr], dim=-1)
        return self.mlp_msg(tmp)
    
    def update(self, aggr_out, x):
        # aggr_out: [N, emb_dim]
        # x: [N, emb_dim]
        tmp = torch.cat([x, aggr_out], dim=-1)
        return self.mlp_upd(tmp)
    
class NaiveEMPNN(nn.Module):
    def __init__(self, hparams, input_dim, edge_dim, output_dim) -> None:
        super().__init__()
        

        self.hparams = hparams
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # self.layers = hparams.get("layers", [15, 15])
        # self.activation_func = ACTIVATION_FUNCS[hparams.get("activation_func", "relu")]
        # self.last_activ = ACTIVATION_FUNCS[hparams.get("last_activ", "none")]

        hidden_dim = hparams.get("hidden_dim", 20)
        # train_eps = hparams.get("train_eps", False)

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.layer1 = NaiveEMPNNLayer(emb_dim=hidden_dim, edge_dim=edge_dim)
        self.layer2 = NaiveEMPNNLayer(emb_dim=hidden_dim, edge_dim=edge_dim)
        self.layer3 = NaiveEMPNNLayer(emb_dim=hidden_dim, edge_dim=edge_dim)
        # self.conv2 = MPNNLayer(emb_dim=hidden_dim, edge_dim=hidden_dim)
        # self.conv3 = MPNNLayer(emb_dim=1, edge_dim=hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)

        lattice = D2Q9()
        self.lattice_velocities = torch.tensor(lattice.lattice_velocities(), device=self.device)

        # self.activ = nn.Softmax(dim=1)

        self.loss_func = LOSS_FUNCS[hparams.get("loss_func", "msre")]

        self.configure_optimizer()

    def forward(self, x, edge_index, edge_attr, pos):
        fpre = x
        x = self.encoder(x)
        x = self.layer1(x, edge_index, edge_attr, pos)
        # x = F.relu(x)
        x = self.layer2(x, edge_index, edge_attr, pos)
        # x = F.relu(x)
        x = self.layer3(x, edge_index, edge_attr, pos)
        x = self.decoder(x)

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
        out = self.forward(data.x, data.edge_index, data.edge_attr, data.pos)
        loss = self.loss_func(data.y, out)
        loss.backward()
        self.optimizer.step()

        return loss
    
    def validation_step(self, batch):
        self.eval()

        data = batch.to(self.device)
        out = self.forward(data.x, data.edge_index, data.edge_attr, data.pos)
        loss = self.loss_func(data.y, out)

        return loss