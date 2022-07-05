# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class SEED(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, tencoder, sencoder, tpretrained, dim=128, K=65536, TT=0.07, ST=0.1, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        T: softmax temperature (default: 0.07)
        """
        super(SEED, self).__init__()

        self.K = K
        self.TT = TT
        self.ST = ST

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = sencoder(num_classes=dim)
        self.encoder_k = tencoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_smlp = self.encoder_q.fc.weight.shape[1]
            dim_tmlp = self.encoder_k.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_smlp, dim_smlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_tmlp, dim_tmlp), nn.ReLU(), self.encoder_k.fc)

        checkpoint = torch.load(tpretrained, map_location="cpu")
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = self.encoder_k.load_state_dict(state_dict, strict=False)
        print('Loaded teacher model')
        print(msg.missing_keys)

        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        keys = concat_all_gather(k)   # bxd
        cur_queue = self.queue.clone().detach()   # dxk
        cur_queue_enqueue = torch.cat((keys.T, cur_queue), 1)  # dx(b+k)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        s_dist = torch.einsum('nc,ck->nk', [q, cur_queue_enqueue.detach()])
        # negative logits: NxK
        t_dist = torch.einsum('nc,ck->nk', [k, cur_queue_enqueue.detach()])

        s_dist /= self.ST
        t_dist /= self.TT

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return s_dist, t_dist


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
