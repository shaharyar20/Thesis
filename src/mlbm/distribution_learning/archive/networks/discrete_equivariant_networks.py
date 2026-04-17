import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from torchlbm.core.lattices.d2q9 import D2Q9

def msre(y_true, y_pred, eps=1e-14):
    return torch.mean(torch.square((y_pred - y_true) / (y_true + eps)))

LOSS_FUNCS = {
    "l1": nn.L1Loss(),
    "mse": nn.MSELoss(),
    "msre": msre,
}

class SortedDEGNNLayer(gnn.MessagePassing):
    def __init__(self, input_dim, message_dim, output_dim, edge_dim=1, aggr='add', bias=True):
        super().__init__(aggr=aggr)

        self.input_dim = input_dim
        self.message_dim = message_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim

        self.edge_input_dim = 2*input_dim + edge_dim + 4
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

        rotate_90 = torch.FloatTensor([[0, -1], [1, 0]])
        reflect = torch.FloatTensor([[1, 0], [0, -1]])

        basic_ops = [rotate_90, reflect]

        D4h_group = [torch.eye(2)]
        for _ in range(5):
            new_elements = []
            for op in basic_ops:
                for element in D4h_group:
                    new_element = torch.mm(op, element)
                    new_elements.append(new_element)
            D4h_group.extend(new_elements)
        unique_ops = []
        for op in D4h_group:
            if not any(torch.all(torch.eq(op, unique_op)) for unique_op in unique_ops):
                unique_ops.append(op)

        self.group = []
        for op in unique_ops:
            self.group.append(op.to('cuda'))

    def coord2radial(self, x_i, x_j):
        radial = torch.cat([x_i, x_j, torch.sum((x_i - x_j)**2, dim=1).unsqueeze(1)], dim=1)
        return radial


    def forward(self, h, edge_index, x, edge_attr=None):
        return self.propagate(edge_index, h=h, x=x) if edge_attr is None else self.propagate(edge_index, h=h, x=x, edge_attr=edge_attr)  

    def message(self, h_i, h_j, x_i, x_j, edge_attr=None):
        cat_edge_features = []
        for op in self.group:
            coord1 = torch.matmul(x_i, op)
            coord2 = torch.matmul(x_j, op)
            radial = self.coord2radial(coord1, coord2)
            cat_edge_features.append(radial)
        catedge = torch.stack(cat_edge_features).permute(1, 0, 2)
        sorted_indices = torch.argsort(catedge[:, :, 0], dim=1)
        sorted_data = torch.gather(catedge, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, catedge.size(-1)))
        rank=1
        sorted_cat_edge_features=sorted_data[:,0,:].squeeze(0)
        # print(h_i.shape)
        # print(x_i.shape)
        # print(sorted_cat_edge_features.shape)



        # coord_diff = torch.sum((x_i - x_j)**2, dim=1).unsqueeze(1)
        tmp = torch.cat([h_i, h_j, sorted_cat_edge_features[:,:4]], dim=-1) if edge_attr is None else torch.cat([h_i, h_j, sorted_cat_edge_features[:,:4],edge_attr], dim=-1)
        return self.mlp_msg(tmp)

    def update(self, aggr_out, h):
        # print(aggr_out.shape, h.shape)
        tmp = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(tmp)
    
class SortedDEGNN(nn.Module):
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
        self.layer1 = SortedDEGNNLayer(input_dim=self.hidden_dim, message_dim=self.hidden_dim, output_dim=self.hidden_dim, edge_dim=self.edge_dim, aggr=self.aggr, bias=self.bias)
        self.layer2 = SortedDEGNNLayer(input_dim=self.hidden_dim, message_dim=self.hidden_dim, output_dim=self.hidden_dim, edge_dim=self.edge_dim, aggr=self.aggr, bias=self.bias)
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
    
class StupidDEGNNLayer(gnn.MessagePassing):
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

        # rotate_90 = torch.FloatTensor([[0, -1], [1, 0]])
        # reflect = torch.FloatTensor([[1, 0], [0, -1]])

        # basic_ops = [rotate_90, reflect]

        # D4h_group = [torch.eye(2)]
        # for _ in range(5):
        #     new_elements = []
        #     for op in basic_ops:
        #         for element in D4h_group:
        #             new_element = torch.mm(op, element)
        #             new_elements.append(new_element)
        #     D4h_group.extend(new_elements)
        # unique_ops = []
        # for op in D4h_group:
        #     if not any(torch.all(torch.eq(op, unique_op)) for unique_op in unique_ops):
        #         unique_ops.append(op)

        # self.group = []
        # for op in unique_ops:
        #     self.group.append(op.to('cuda'))

    def d4_invariant_function(self, x_i, x_j):
        diff = x_i - x_j
        # print(diff.shape)
        val = ((diff[:,0]**2 - diff[:,1]**2)**2).unsqueeze(-1)
        # print(val.shape)
        return val


    def forward(self, h, edge_index, x, edge_attr=None):
        return self.propagate(edge_index, h=h, x=x) if edge_attr is None else self.propagate(edge_index, h=h, x=x, edge_attr=edge_attr)  

    def message(self, h_i, h_j, x_i, x_j, edge_attr=None):
        val = self.d4_invariant_function(x_i, x_j)
        # coord_diff = torch.sum((x_i - x_j)**2, dim=1).unsqueeze(1)
        tmp = torch.cat([h_i, h_j, val], dim=-1) if edge_attr is None else torch.cat([h_i, h_j, val, edge_attr], dim=-1)
        return self.mlp_msg(tmp)

    def update(self, aggr_out, h):
        # print(aggr_out.shape, h.shape)
        tmp = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(tmp)
    
class StupidDEGNN(nn.Module):
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
        self.layer1 = StupidDEGNNLayer(input_dim=self.hidden_dim, message_dim=self.hidden_dim, output_dim=self.hidden_dim, edge_dim=self.edge_dim, aggr=self.aggr, bias=self.bias)
        self.layer2 = StupidDEGNNLayer(input_dim=self.hidden_dim, message_dim=self.hidden_dim, output_dim=self.hidden_dim, edge_dim=self.edge_dim, aggr=self.aggr, bias=self.bias)
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
    
