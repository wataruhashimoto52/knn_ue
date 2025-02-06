# from gpn.utils.config import ModelConfiguration
import torch
import torch.nn as nn

# from .model import Model
from .multihead_att import ProVal_MultiHeadAttention, ReMultiHeadAttention
from .evidence import Evidence, Density

class TokenLevelAlpha(nn.Module):
    """TokenLevelAlpha model"""

    def __init__(
        self,
        dim_hidden=768,
        dim_latent=50,
        num_classes=17,
        radial_layers=10,
        maf_layers=0,
        gaussian_layers=0,
        use_batched_flow=True,
        alpha_evidence_scale="latent-new",  # ['latent-old', 'latent-new', 'latent-new-plus-classes', None]
        neighbor_mode=None,
        normalize_dis=1,
        self_att_dk_ratio=8,
        self_att_droput=0.05,
        use_seq_training=False,
        hidden_dims=None,
        k_lipschitz=None,
        use_stable=False,
    ):
        super().__init__()

        self.dim_hidden = dim_hidden
        self.dim_latent = dim_latent
        self.num_classes = num_classes
        self.radial_layers = radial_layers
        self.maf_layers = maf_layers
        self.gaussian_layers = gaussian_layers
        self.use_batched_flow = use_batched_flow
        self.alpha_evidence_scale = alpha_evidence_scale
        self.neighbor_mode = neighbor_mode
        self.normalize_dis = normalize_dis
        self.self_att_dk_ratio = self_att_dk_ratio
        self.self_att_droput = self_att_droput
        self.use_seq_training = use_seq_training
        self.hidden_dims = hidden_dims
        self.k_lipschitz = k_lipschitz
        self.use_stable = use_stable

        if self.use_stable:
            self.batch_norm = torch.nn.BatchNorm1d(self.dim_hidden)

        self.latent_encoder = nn.Linear(self.dim_hidden, self.dim_latent)

        use_batched = True if self.use_batched_flow else False
        self.flow = Density(
            dim_latent=self.dim_latent,
            num_mixture_elements=self.num_classes,
            radial_layers=self.radial_layers,
            maf_layers=self.maf_layers,
            gaussian_layers=self.gaussian_layers,
            use_batched_flow=use_batched,
        )

        self.evidence = Evidence(scale=self.alpha_evidence_scale)

        if self.neighbor_mode == "self_att":
            # self.self_att = MultiHeadAttention( # unused
            #     n_head=6, d_model=self.num_classes, d_k=self.num_classes//4, d_v=self.num_classes//4, dropout=0.1
            # )

            self.self_att = ReMultiHeadAttention(  # default
                n_head=6,
                d_model=self.num_classes,
                d_k=self.num_classes // self.self_att_dk_ratio,
                dropout=self.self_att_droput,
            )

        elif self.neighbor_mode == "self_att_proval":
            self.self_att = ProVal_MultiHeadAttention(
                n_head=6,
                d_model=self.num_classes,
                d_k=self.num_classes // self.self_att_dk_ratio,
                d_v=self.num_classes,
                dropout=self.self_att_droput,
            )

        elif self.neighbor_mode == "simple_project":
            self.simple_preject = nn.Linear(self.num_classes, self.num_classes)

        # self.propagation = APPNPPropagation(
        #     K=self.K,
        #     alpha=self.alpha_teleport,
        #     add_self_loops=self.add_self_loops,
        #     cached=False,
        #     normalization='sym')

        # assert self.pre_train_mode in ('encoder', 'flow', None)
        # assert self.likelihood_type in ('UCE', 'nll_train', 'nll_train_and_val', 'nll_consistency', None)

    # def forward(self, h):
    #     return self.forward_impl(h)

    def forward_use_ori_PN_frame(self, h, global_n_c):
        # edge_index = data.edge_index if data.edge_index is not None else data.adj_t
        # h = self.input_encoder(data.x)
        global_n_c = torch.LongTensor(global_n_c).to(h.device)
        # p_c = global_n_c/global_n_c.sum() # if add this code as GPN used, the results might be wrong.
        z = self.latent_encoder(h)

        batch_size, seq_len, latent_dim = z.shape
        # print(z.shape)
        z = torch.reshape(z, (batch_size * seq_len, latent_dim))

        flow_res = self.flow(z)
        if seq_len == 1:
            if flow_res.shape[0] != 1:
                flow_res = flow_res[0].unsqueeze(0)

        """if "-plus-classes" in self.alpha_evidence_scale:
            further_scale = self.num_classes
        else:
            further_scale = 1.0"""

        beta_ft = global_n_c * torch.exp(flow_res)

        _, beta_ft_dim = beta_ft.shape

        beta_ft = torch.reshape(beta_ft, (batch_size, seq_len, beta_ft_dim))

        alpha_features = (
            1.0 + beta_ft
        )  # This is alpha^{post} the alpha used in the posterior net, without using PPR yet
        return alpha_features, beta_ft

    def cal_self_att_mask(self, batch_size, seq_len, lengths):
        res = torch.zeros(batch_size, seq_len)
        for i in range(batch_size):
            end_index = lengths[i].item()
            res[i, 0:end_index] = 1
        return res

    def forward(self, h, global_n_c, global_p_c):

        global_n_c = torch.LongTensor(global_n_c).to(h.device)
        global_p_c = torch.FloatTensor(global_p_c).to(h.device)
        if self.normalize_dis == 1:
            # p_c = global_n_c/global_n_c.sum() # if add this code as GPN used, the results might be wrong.
            p_c = global_p_c
        elif self.normalize_dis == 0:
            p_c = global_n_c
        else:
            raise ValueError("normalize_dis is set wrongly.")

        # added to stable the training
        if self.use_stable:
            h = h.transpose(2, 1)
            # print(f'start use bn h.shape={h.shape}')
            h = self.batch_norm(h)
            h = h.transpose(2, 1)
            # print(f'finished bn h.shape={h.shape}')

        ori_z = self.latent_encoder(h)

        batch_size, seq_len, latent_dim = ori_z.shape  # b=1, seq_l, latent_dim
        # print(z.shape)
        z = torch.reshape(ori_z, (batch_size * seq_len, latent_dim))

        flow_res = self.flow(z)  # if batch size > 1, need to revise here
        if seq_len == 1:  # need to be revised
            if flow_res.shape[0] != 1:
                flow_res = flow_res[0].unsqueeze(0)

        log_q_ft_per_class = (
            flow_res + p_c.view(1, -1).log()
        )  # self.flow(z) is log p(z | c), and in size of [sample_num, class_num]

        if "-plus-classes" in self.alpha_evidence_scale:
            further_scale = self.num_classes
        else:
            further_scale = 1.0

        beta_ft = self.evidence(
            log_q_ft_per_class, dim=self.dim_latent, further_scale=further_scale
        ).exp()

        _, beta_ft_dim = beta_ft.shape

        beta_ft = torch.reshape(beta_ft, (batch_size, seq_len, beta_ft_dim))

        # add self-attention below
        """if self.neighbor_mode in ["self_att", "self_att_proval"]:
            if batch_size == 1:
                used_mask=None
            else:
                used_mask = self.cal_self_att_mask(batch_size, seq_len, lengths)
                used_mask = used_mask.to(h.device)

            beta_ft, attn = self.self_att(beta_ft, beta_ft, beta_ft, mask=used_mask)
        elif self.neighbor_mode == "closest":
            fixed_mat = torch.zeros((seq_len, seq_len)).to(h.device)
            for i in range(seq_len):
                if i == 0:
                    fixed_mat[0,i]= 0.5
                    try:
                        fixed_mat[1,i] = 0.5
                    except:
                        pass
                elif i == seq_len-1:
                    fixed_mat[seq_len-1, i] = 0.5
                    try:
                        fixed_mat[seq_len-2,i] = 0.5
                    except:
                        pass
                else:
                    fixed_mat[i - 1, i] = 0.5
                    fixed_mat[i + 1, i] = 0.5

            fixed_mat = fixed_mat.unsqueeze(0).repeat(batch_size, 1, 1)
            beta_ft = torch.bmm(beta_ft.permute(0, 2, 1), fixed_mat).permute(0, 2, 1)
        elif self.neighbor_mode == "simple_project":
            beta_ft = F.relu(self.simple_preject(beta_ft))"""

        alpha_features = 1.0 + beta_ft

        return alpha_features, beta_ft, ori_z

    def forward_copy_global_priormode_only(self, h, p_c):

        p_c = torch.LongTensor(p_c).to(h.device)

        z = self.latent_encoder(h)

        batch_size, seq_len, latent_dim = z.shape
        # print(z.shape)
        z = torch.reshape(z, (batch_size * seq_len, latent_dim))

        flow_res = self.flow(z)
        if seq_len == 1:
            if flow_res.shape[0] != 1:
                flow_res = flow_res[0].unsqueeze(0)
        log_q_ft_per_class = (
            flow_res + p_c.view(1, -1).log()
        )  # self.flow(z) is log p(z | c), and in size of [sample_num, class_num]

        if "-plus-classes" in self.alpha_evidence_scale:
            further_scale = self.num_classes
        else:
            further_scale = 1.0

        beta_ft = self.evidence(
            log_q_ft_per_class, dim=self.dim_latent, further_scale=further_scale
        ).exp()

        _, beta_ft_dim = beta_ft.shape

        beta_ft = torch.reshape(beta_ft, (batch_size, seq_len, beta_ft_dim))

        alpha_features = (
            1.0 + beta_ft
        )  # This is alpha^{post} the alpha used in the posterior net, without using PPR yet

        return alpha_features, beta_ft
