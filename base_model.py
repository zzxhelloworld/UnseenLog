import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, GINConv
from torch.nn import Linear, Sequential, ReLU


class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, gnn_type="Transformer", heads=1):
        super(GNN, self).__init__()
        GNN_MAPPING = {
            "GCN": GCNConv,
            "GAT": GATConv,
            "Transformer": TransformerConv,
            "GIN": GINConv,
        }
        if gnn_type not in GNN_MAPPING:
            raise ValueError(f"Unknown GNN type: {gnn_type}. Supported types: {list(GNN_MAPPING.keys())}")
        ConvLayer = GNN_MAPPING[gnn_type]
        self.convs = nn.ModuleList()

        if gnn_type == "Transformer":
            self.convs.append(ConvLayer(in_channels, hidden_channels // heads, heads=heads))
        elif gnn_type == "GIN":
            self.convs.append(ConvLayer(
                Sequential(Linear(in_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels)),
                train_eps=True))
        else:
            self.convs.append(ConvLayer(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            if gnn_type == "Transformer":
                self.convs.append(ConvLayer(hidden_channels, hidden_channels // heads, heads=heads))
            elif gnn_type == "GIN":
                self.convs.append(ConvLayer(
                    Sequential(Linear(hidden_channels, hidden_channels), ReLU(),
                               Linear(hidden_channels, hidden_channels)),
                    train_eps=True))
            else:
                self.convs.append(ConvLayer(hidden_channels, hidden_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        return x


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_layers=3,
                 hidden_units=128,
                 activation='relu',
                 use_dropout=False,
                 dropout_rate=0.0):
        super().__init__()

        if num_layers < 0:
            raise ValueError("num_layers must be a non-negative integer")

        if num_layers == 0:
            self.network = nn.Linear(input_dim, output_dim)
            return

        if isinstance(hidden_units, int):
            hidden_dims = [hidden_units] * num_layers
        elif isinstance(hidden_units, list):
            if len(hidden_units) != num_layers:
                raise ValueError(f"hidden_units list length ({len(hidden_units)}) must match num_layers ({num_layers})")
            hidden_dims = hidden_units
        else:
            raise TypeError("hidden_units must be int or list[int]")

        activation_layer = self._get_activation(activation)

        layers = []
        current_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))

            if activation_layer:
                layers.append(activation_layer)

            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))

            current_dim = h_dim

        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def _get_activation(self, activation):
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leakyrelu': nn.LeakyReLU(),
            'none': None
        }
        if activation.lower() not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        return activations[activation.lower()]

    def forward(self, x):
        return self.network(x)


class MetaLearner(nn.Module):
    def __init__(self, args):
        super(MetaLearner, self).__init__()
        self.gnn = GNN(args.feature_dim, args.gnn_hidden_dim, args.gnn_num_layers, args.gnn_type, args.gnn_head_num)
        self.classifier = MLP(
            input_dim=args.gnn_hidden_dim,
            output_dim=1,
            num_layers=len(args.mlp_hidden_units),
            hidden_units=args.mlp_hidden_units,
            activation='relu'
        )

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        x = self.gnn(x, edge_index)
        x = self.classifier(x).squeeze(-1)  # logits
        return x
