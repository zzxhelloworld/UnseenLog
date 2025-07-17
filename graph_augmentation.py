import torch


class FeatureNoise:
    """add feature noise"""

    def __init__(self, noise_std=0.15):
        self.noise_std = noise_std

    def __call__(self, original_data):
        data = original_data.clone()  # deep copy
        device = data.x.device
        noise = torch.randn_like(data.x, device=device) * self.noise_std  # add noise
        data.x = data.x + noise
        return data


class EdgeDropout:
    """randomly remove specific ratio of edges"""

    def __init__(self, drop_ratio=0.2):
        self.drop_ratio = drop_ratio

    def __call__(self, original_data):
        data = original_data.clone()  # deep copy
        num_edges = data.edge_index.size(1)
        mask = torch.rand(num_edges) > self.drop_ratio
        data.edge_index = data.edge_index[:, mask]
        return data


class EdgeAddition:
    """randomly add specific ratio of edges (no duplicate or self-loop edges)"""

    def __init__(self, add_ratio=0.2):
        self.add_ratio = add_ratio

    def __call__(self, original_data):
        data = original_data.clone()  # deep copy
        device = data.edge_index.device
        num_nodes = data.num_nodes
        num_edges = data.edge_index.size(1)
        num_new_edges = int(num_edges * self.add_ratio)

        new_edges = torch.randint(0, num_nodes, (2, num_new_edges), device=device)

        mask = new_edges[0] != new_edges[1]
        new_edges = new_edges[:, mask]  # removing self-loop

        all_edges = torch.cat([data.edge_index, new_edges], dim=1)
        unique_edges = torch.unique(all_edges, dim=1)  # remove redundant edges

        # update edge index
        data.edge_index = unique_edges

        return data
