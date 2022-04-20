import torch

from pointtransformerpytorch.point_transformer_pytorch import PointTransformerLayer


class PointTransformer(torch.nn.Module) :


    def __init__(self,
                 depth,
                 hidden_dim=32) :
        super().__init__()
        self.layers = torch.nn.ModuleList([])

        self.encode = torch.nn.Linear(4, hidden_dim)

        for i in range(depth) :
            self.layers.append(PointTransformerLayer(dim=hidden_dim))

        self.to_logits = torch.nn.Linear(hidden_dim, 2)

        self.res_fixed = False

        self.input_format={'xy' : [0, 1],
                            't' : [2],
                            'raw' : [3]}


    def forward(self, data, query, res, dt) :
        out = []
        for i, d in enumerate(data) :
            d = d.unsqueeze(0).clone()
            d[..., self.input_format['xy'][0]] = d[..., self.input_format['xy'][0]] / res[i][0] * 2 - 1
            d[..., self.input_format['xy'][1]] = d[..., self.input_format['xy'][1]] / res[i][1] * 2 - 1
            d[..., self.input_format['t']] = d[..., self.input_format['t']] / dt[i] * 2 - 1
            d_emb = self.encode(d)
            pos = d[..., :3]

            x = d_emb
            for layer in self.layers :
                x = layer(x, pos) + x
            x = x.max(dim=1, keepdim=True)
            out.append(self.to_logits(x).squeeze(0))
        return out