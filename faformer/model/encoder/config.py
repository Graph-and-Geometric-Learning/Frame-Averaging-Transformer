
class FAFormerConfig():
    model_type = "faformer"

    def __init__(
        self,
        dim=3,  # dimension of the coordinates
        d_input=64, 
        d_model=64, 
        d_edge_model=64,
        n_layers=3,
        n_pos=1_000,  # maximum number of positions
        proj_drop=0.1,
        attn_drop=0.1,
        n_neighbors=16,
        valid_radius=1e6,
        embedding_grad_frac=1.0,
        n_heads=4,
        norm="layer",
        activation="silu",
        **kwargs,
    ):

        self.dim = dim
        self.d_input = d_input
        self.d_model = d_model
        self.d_edge_model = d_edge_model
        self.n_layers = n_layers
        self.n_pos = n_pos
        self.proj_drop = proj_drop
        self.attn_drop = attn_drop
        self.n_neighbors = n_neighbors
        self.valid_radius = valid_radius
        self.embedding_grad_frac = embedding_grad_frac
        self.n_heads = n_heads
        self.norm = norm
        self.activation = activation

        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self._hidden_dim_check()

    def _hidden_dim_check(self):
        # check if d_model/d_edge_model can be divided by n_heads
        assert self.d_model % self.n_heads == 0, f"d_model should be divisible by n_heads"
        assert self.d_edge_model % self.n_heads == 0, f"d_edge_model should be divisible by n_heads"