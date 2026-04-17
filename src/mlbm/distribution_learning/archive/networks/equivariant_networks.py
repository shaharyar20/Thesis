import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from torchlbm.core.lattices.d2q9 import D2Q9

def msre(y_true, y_pred, eps=1e-14):
    return torch.sqrt(torch.mean(torch.square((y_pred - y_true) / (y_true + eps))))

LOSS_FUNCS = {
    "l1": nn.L1Loss(),
    "mse": nn.MSELoss(),
    "msre": msre,
}

class EGNNLayer(gnn.MessagePassing):
    def __init__(self, input_dim, message_dim, output_dim, edge_dim=1, aggr='add', bias=True):
        super().__init__(aggr=aggr)

        self.input_dim = input_dim
        self.message_dim = message_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim

        self.edge_input_dim = 2*input_dim + edge_dim + 1
        self.emb_dim = self.input_dim

        self.mlp_msg = nn.Sequential(
            nn.Linear(self.edge_input_dim, 2*self.emb_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(2*self.emb_dim, self.emb_dim, bias=bias),
            nn.ReLU(),
        )

        self.mlp_upd = nn.Sequential(
            nn.Linear(2*self.emb_dim, self.emb_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim, bias=bias),
        )

    def forward(self, h, edge_index, x, edge_attr=None):
        return self.propagate(edge_index, h=h, x=x) if edge_attr is None else self.propagate(edge_index, h=h, x=x, edge_attr=edge_attr)  

    def message(self, h_i, h_j, x_i, x_j, edge_attr=None):
        coord_diff = torch.sum((x_i - x_j)**2, dim=1).unsqueeze(1)
        tmp = torch.cat([h_i, h_j, coord_diff], dim=-1) if edge_attr is None else torch.cat([h_i, h_j, coord_diff,edge_attr], dim=-1)
        return self.mlp_msg(tmp)

    def update(self, aggr_out, h):
        # print(aggr_out.shape, h.shape)
        tmp = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(tmp)
    
class EGNN(nn.Module):
    def __init__(self, hparams, input_dim, edge_dim, output_dim):
        super().__init__()

        self.hparams = hparams
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.input_dim = input_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim

        self.hidden_dim = hparams.get("hidden_dim", 20)
        self.aggr = hparams.get("aggr", "add")
        self.bias = hparams.get("bias", True)

        self.encoder = nn.Linear(self.input_dim, self.hidden_dim, bias=self.bias)
        # self.layer1 = EGNNLayer(input_dim=self.hidden_dim, message_dim=self.hidden_dim, output_dim=2*self.hidden_dim, edge_dim=self.edge_dim, aggr=self.aggr, bias=self.bias)
        # self.layer2 = EGNNLayer(input_dim=2*self.hidden_dim, message_dim=self.hidden_dim, output_dim=self.hidden_dim, edge_dim=self.edge_dim, aggr=self.aggr, bias=self.bias)
        self.layer1 = EGNNLayer(input_dim=self.hidden_dim, message_dim=self.hidden_dim, output_dim=self.hidden_dim, edge_dim=self.edge_dim, aggr=self.aggr, bias=self.bias)
        self.layer2 = EGNNLayer(input_dim=self.hidden_dim, message_dim=self.hidden_dim, output_dim=self.hidden_dim, edge_dim=self.edge_dim, aggr=self.aggr, bias=self.bias)
        # self.layer3 = EGNNLayer(input_dim=self.hidden_dim, message_dim=self.hidden_dim, output_dim=self.hidden_dim, edge_dim=self.edge_dim, aggr=self.aggr, bias=self.bias)
        self.decoder = nn.Linear(self.hidden_dim, self.output_dim, bias=self.bias)

        lattice = D2Q9()
        self.lattice_velocities = torch.tensor(lattice.lattice_velocities(), device=self.device)

        self.loss_func = LOSS_FUNCS[hparams.get("loss_func", "msre")]

        self.configure_optimizer()

    def forward(self, h, edge_index, x, edge_attr=None):
        fpre = h
        h = self.encoder(h)
        h = self.layer1(h, edge_index, x, edge_attr) # + h
        h = self.layer2(h, edge_index, x, edge_attr) # + h
        # h = self.layer3(h, edge_index, x, edge_attr)
        h = self.decoder(h)
        h += fpre # This might be cheating, but it gives the best results

        fpred = h.reshape(-1, 9)
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
        out = self.forward(data.x, data.edge_index, data.pos) if data.edge_attr is None else self.forward(data.x, data.edge_index, data.pos, edge_attr=data.edge_attr)
        loss = self.loss_func(data.y, out)
        loss.backward()
        self.optimizer.step()

        return loss
    
    def validation_step(self, batch):
        self.eval()

        data = batch.to(self.device)
        out = self.forward(data.x, data.edge_index, data.pos) if data.edge_attr is None else self.forward(data.x, data.edge_index, data.pos, edge_attr=data.edge_attr)
        loss = self.loss_func(data.y, out)

        return loss
    
class CollisionEGNNLayer(gnn.MessagePassing):
    def __init__(self, input_dim, message_dim, output_dim, edge_dim=1, aggr='add', bias=True):
        super().__init__(aggr=aggr)

        self.input_dim = input_dim
        self.message_dim = message_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim

        self.edge_input_dim = 2*input_dim + edge_dim + 1 + 1
        self.emb_dim = self.input_dim

        self.mlp_msg = nn.Sequential(
            nn.Linear(self.edge_input_dim, 2*self.emb_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(2*self.emb_dim, self.emb_dim, bias=bias),
            nn.ReLU(),
        )

        self.mlp_upd = nn.Sequential(
            nn.Linear(2*self.emb_dim + 1, self.emb_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim, bias=bias),
        )

    def forward(self, h, edge_index, x, omega, edge_attr=None):
        return self.propagate(edge_index, h=h, x=x, omega=omega) if edge_attr is None else self.propagate(edge_index, h=h, x=x, omega=omega, edge_attr=edge_attr)  

    def message(self, h_i, h_j, x_i, x_j, omega_i, edge_attr=None):
        coord_diff = torch.sum((x_i - x_j)**2, dim=1).unsqueeze(1)
        # print(coord_diff.shape)
        # print(omega_i.shape)
        # print(h_i.shape)
        # print(x_i.shape)
        tmp = torch.cat([omega_i, h_i, h_j, coord_diff], dim=-1) if edge_attr is None else torch.cat([omega_i, h_i, h_j, coord_diff,edge_attr], dim=-1)
        return self.mlp_msg(tmp)

    def update(self, aggr_out, h, omega):
        # print(aggr_out.shape, h.shape, omega.shape)
        tmp = torch.cat([omega, h, aggr_out], dim=-1)
        return h + self.mlp_upd(tmp)
    
class CollisionEGNN(nn.Module):
    def __init__(self, hparams, input_dim, edge_dim, output_dim):
        super().__init__()

        self.hparams = hparams
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.input_dim = input_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim

        self.hidden_dim = hparams.get("hidden_dim", 20)
        self.aggr = hparams.get("aggr", "add")
        self.bias = hparams.get("bias", True)

        self.encoder = nn.Linear(self.input_dim, self.hidden_dim, bias=self.bias)
        # self.layer1 = EGNNLayer(input_dim=self.hidden_dim, message_dim=self.hidden_dim, output_dim=2*self.hidden_dim, edge_dim=self.edge_dim, aggr=self.aggr, bias=self.bias)
        # self.layer2 = EGNNLayer(input_dim=2*self.hidden_dim, message_dim=self.hidden_dim, output_dim=self.hidden_dim, edge_dim=self.edge_dim, aggr=self.aggr, bias=self.bias)
        self.layer1 = CollisionEGNNLayer(input_dim=self.hidden_dim, message_dim=self.hidden_dim, output_dim=self.hidden_dim, edge_dim=self.edge_dim, aggr=self.aggr, bias=self.bias)
        self.layer2 = CollisionEGNNLayer(input_dim=self.hidden_dim, message_dim=self.hidden_dim, output_dim=self.hidden_dim, edge_dim=self.edge_dim, aggr=self.aggr, bias=self.bias)
        # self.layer3 = CollisionEGNNLayer(input_dim=self.hidden_dim, message_dim=self.hidden_dim, output_dim=self.hidden_dim, edge_dim=self.edge_dim, aggr=self.aggr, bias=self.bias)
        self.decoder = nn.Linear(self.hidden_dim, self.output_dim, bias=self.bias)

        lattice = D2Q9()
        self.lattice_velocities = torch.tensor(lattice.lattice_velocities(), device=self.device)

        self.loss_func = LOSS_FUNCS[hparams.get("loss_func", "msre")]

        self.configure_optimizer()

    def forward(self, h, edge_index, x, omega, edge_attr=None):
        fpre = h
        h = self.encoder(h)
        h = self.layer1(h, edge_index, x, omega, edge_attr)
        h = self.layer2(h, edge_index, x, omega, edge_attr)
        # h = self.layer3(h, edge_index, x, omega, edge_attr)
        h = self.decoder(h)
        h += fpre # This might be cheating, but it gives the best results

        fpred = h.reshape(-1, 9)
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
        # print(data.omega.shape) 
        # print(data.x.shape)
        # print(data.pos.shape)
        # Convert omega from (batch_size,1) to (batch_size*num_nodes, 1) by repeating the values for each node in the graph
        # data.omega = data.omega.repeat(1, data.num_nodes).reshape(-1, 1)
        # print(data.omega.shape) 
        out = self.forward(data.x, data.edge_index, data.pos, data.omega) if data.edge_attr is None else self.forward(data.x, data.edge_index, data.pos, data.omega, edge_attr=data.edge_attr)
        loss = self.loss_func(data.y, out)
        loss.backward()
        self.optimizer.step()

        return loss
    
    def validation_step(self, batch):
        self.eval()

        data = batch.to(self.device)
        out = self.forward(data.x, data.edge_index, data.pos, data.omega) if data.edge_attr is None else self.forward(data.x, data.edge_index, data.pos, data.omega, edge_attr=data.edge_attr)
        loss = self.loss_func(data.y, out)

        return loss