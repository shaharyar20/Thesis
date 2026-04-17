import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


def msre(y_true, y_pred, eps=1e-14):
    return torch.mean(torch.square((y_pred - y_true) / (y_true + eps)))


ACTIVATION_FUNCS = {
    "relu": nn.ReLU,
    "softmax": nn.Softmax(dim=1),
    "none": None,
}

LOSS_FUNCS = {
    "l1": nn.L1Loss,
    "mse": nn.MSELoss,
    "msre": msre,
}

class FirstGNN(nn.Module):
    def __init__(self, hparams, input_dim) -> None:
        super().__init__()
        torch.manual_seed(12345)

        self.hparams = hparams
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.layers = hparams.get("layers", [15, 15])
        self.activation_func = ACTIVATION_FUNCS[hparams.get("activation_func", "relu")]
        self.last_activ = ACTIVATION_FUNCS[hparams.get("last_activ", "none")]

        # layers = [
        #     gnn.GCNConv(9, self.layers[0]),
        #     # nn.Linear(9, self.layers[0]),
        #     self.activation_func()
        # ]

        # for i in range(len(self.layers) - 1):
        #     layers.append(gnn.GCNConv(self.layers[i], self.layers[i+1]))
        #     # layers.append(nn.Linear(self.layers[i], self.layers[i+1]))
        #     layers.append(self.activation_func())

        # layers.append(gnn.GCNConv(self.layers[-1], 9))
        # # layers.append(nn.Linear(self.layers[-1], 9))
        # if self.last_activ is not None:
        #     layers.append(self.last_activ())

        # self.model = nn.Sequential(*layers)
        hidden_dim = hparams.get("hidden_dim", 20)
        self.conv1 = gnn.GCNConv(input_dim, hidden_dim)
        self.conv2 = gnn.GCNConv(hidden_dim, hidden_dim)
        self.conv3 = gnn.GCNConv(hidden_dim, 1)

        self.loss_func = LOSS_FUNCS[hparams.get("loss_func", "msre")]

        # A = [
        #     [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #     [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #     [1., 0., 1., 2., 1., 0., 2., 2., 0.],
        #     [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #     [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #     [-0.5, 0.5, 0., -1.5, -1., 1., -1., -2., 0.],
        #     [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #     [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #     [0.5, 0.5, 0., 0.5, 1., 0, 0, 1., 1.]
        # ]

        # B = [
        #     [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        #     [0., 1., 0., 0., 0., 0., 0., 0., 0.],
        #     [-1., 0., 0., -2., -1., 0., -2., -2., 0.],
        #     [0., 0., 0., 1., 0., 0., 0., 0., 0.],
        #     [0., 0., 0., 0., 1., 0., 0., 0., 0.],
        #     [0.5, -0.5, 0., 1.5, 1., 0., 1., 2., 0.],
        #     [0., 0., 0., 0., 0., 0., 1., 0., 0.],
        #     [0., 0., 0., 0., 0., 0., 0., 1., 0.],
        #     [-0.5, -0.5, 0., -0.5, -1., 0, 0, -1., 0.],
        # ]

        # self.A = torch.tensor(A, device=self.device)
        # self.B = torch.tensor(B, device=self.device)

        self.configure_optimizer()

    # def algebraic_reconstruction(self, fpre, fpred):
    #     recon1 = torch.einsum("iQ,NQ->Ni", self.A, fpre.reshape(-1, 9))
    #     recon2 = torch.einsum("iQ,NQ->Ni", self.B, fpred.reshape(-1, 9))
    #     return recon1 + recon2

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        # x = self.algebraic_reconstruction(data.x, x).reshape_as(data.x) # Might need to change this reshaping
        return x

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

    # def save(self, path):
    #     torch.save(self, path)

    # @classmethod
    # def load(cls, path):
    #     model = torch.load(path)
    #     return model

class SimpleGIN(nn.Module):

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

        self.activ = nn.Softmax(dim=1)

        self.loss_func = LOSS_FUNCS[hparams.get("loss_func", "msre")]

        self.configure_optimizer()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        # x = F.relu(x)
        x = self.conv2(x, edge_index)
        # x = F.relu(x)
        x = self.conv3(x, edge_index)
        # print(x.reshape(-1,9)[3,:])

        x = self.activ(x.reshape(-1,9)).reshape(-1,1)
        # print(x[0:9])
        return x
    
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