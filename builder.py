import paddle
from functools import partial


class UCDIR(paddle.nn.Layer):

    def __init__(self, base_encoder, dim=128, K_A=65536, K_B=65536, m=0.999,
        T=0.1, mlp=False, selfentro_temp=0.2, num_cluster=None,
        cwcon_filterthresh=0.2):
        super(UCDIR, self).__init__()
        self.K_A = K_A
        self.K_B = K_B
        self.m = m
        self.T = T
        self.selfentro_temp = selfentro_temp
        self.num_cluster = num_cluster
        self.cwcon_filterthresh = cwcon_filterthresh
        norm_layer = partial(SplitBatchNorm, num_splits=2)
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim, norm_layer=norm_layer)
        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[0]
            self.encoder_q.fc = paddle.nn.Sequential(paddle.nn.Linear(
                in_features=dim_mlp, out_features=dim_mlp), paddle.nn.ReLU(
                ), self.encoder_q.fc)
            self.encoder_k.fc = paddle.nn.Sequential(paddle.nn.Linear(
                in_features=dim_mlp, out_features=dim_mlp), paddle.nn.ReLU(
                ), self.encoder_k.fc)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            paddle.assign(param_q, param_k)
            """Tensor Attribute: torch.Tensor.requires_grad, not convert, please check whether it is torch.Tensor.* and convert manually"""
            param_k.stop_gradient = True
        self.register_buffer('queue_A', paddle.randn(shape=[dim, K_A]))
        # self.register_buffer('queue_A', paddle.ones(shape=[dim, K_A]))
        self.queue_A = paddle.nn.functional.normalize(x=self.queue_A, axis=0)
        self.register_buffer('queue_A_ptr', paddle.zeros(shape=[1], dtype=
            'int64'))
        self.register_buffer('queue_B', paddle.randn(shape=[dim, K_B]))
        # self.register_buffer('queue_B', paddle.ones(shape=[dim, K_B]))
        self.queue_B = paddle.nn.functional.normalize(x=self.queue_B, axis=0)
        self.register_buffer('queue_B_ptr', paddle.zeros(shape=[1], dtype=
            'int64'))
        self.cos_sim = paddle.nn.CosineSimilarity(axis=1, eps=1e-08)

    @paddle.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.
            encoder_k.parameters()):
            paddle.assign(param_k * self.m + param_q * (1.0 - self.m), param_k)

    @paddle.no_grad()
    def _dequeue_and_enqueue_singlegpu(self, keys, key_ids, domain_id):
        import torch
        print(self.queue_A.shape, key_ids.shape, keys.shape)
        torch.save([key_ids, keys],'assign.pth')
        import sys
        sys.exit(0)
        if domain_id == 'A':
            """Tensor Method: torch.Tensor.index_copy_, not convert, please check whether it is torch.Tensor.* and convert manually"""
            for i, idx in enumerate(key_ids):
                self.queue_A[:, idx] = keys.T[:, i]
        elif domain_id == 'B':
            """Tensor Method: torch.Tensor.index_copy_, not convert, please check whether it is torch.Tensor.* and convert manually"""
            for i, idx in enumerate(key_ids):
                self.queue_B[:, idx] = keys.T[:, i]

    @paddle.no_grad()
    def _batch_shuffle_singlegpu(self, x):
        """Tensor Method: torch.Tensor.cuda, not convert, please check whether it is torch.Tensor.* and convert manually"""
        idx_shuffle = paddle.randperm(n=x.shape[0])
        # idx_shuffle = paddle.arange(end = x.shape[0])[::-1]
        idx_unshuffle = paddle.argsort(idx_shuffle)
        return x[idx_shuffle], idx_unshuffle

    @paddle.no_grad()
    def _batch_unshuffle_singlegpu(self, x, idx_unshuffle):
        return x[idx_unshuffle]

    def forward(self, im_q_A, im_q_B, im_k_A=None, im_id_A=None, im_k_B=
        None, im_id_B=None, is_eval=False, cluster_result=None, criterion=None
        ):
        im_q = paddle.concat(x=[im_q_A, im_q_B], axis=0)
        if is_eval:
            k = self.encoder_k(im_q)
            k = paddle.nn.functional.normalize(k)
            k_A, k_B = paddle.split(x=k, num_or_sections=k.shape[0] // im_q_A.shape[0])
            return k_A, k_B
        q = self.encoder_q(im_q)
        q = paddle.nn.functional.normalize(x=q, axis=1)
        q_A, q_B = paddle.split(x=q, num_or_sections=q.shape[0] //im_q_A.shape[0])
        im_k = paddle.concat(x=[im_k_A, im_k_B], axis=0)

        with paddle.no_grad():
            self._momentum_update_key_encoder() # 0.025s vs torch 0.013s
            im_k, idx_unshuffle = self._batch_shuffle_singlegpu(im_k)
            k = self.encoder_k(im_k)
            k = paddle.nn.functional.normalize(x=k, axis=1)
            k = self._batch_unshuffle_singlegpu(k, idx_unshuffle)
            k_A, k_B = paddle.split(x=k, num_or_sections=k.shape[0] //im_k_A.shape[0])  
        self._dequeue_and_enqueue_singlegpu(k_A, im_id_A, 'A') # 0.0047s vs torch 0.00035s
        self._dequeue_and_enqueue_singlegpu(k_B, im_id_B, 'B')

        loss_instcon_A, loss_instcon_B = self.instance_contrastive_loss(q_A,
            k_A, im_id_A, q_B, k_B, im_id_B, criterion)
        losses_instcon = {'domain_A': loss_instcon_A, 'domain_B':
            loss_instcon_B}
        
        if cluster_result is not None:
            loss_cwcon_A, loss_cwcon_B = self.cluster_contrastive_loss(q_A,
                k_A, im_id_A, q_B, k_B, im_id_B, cluster_result)
            losses_cwcon = {'domain_A': loss_cwcon_A, 'domain_B': loss_cwcon_B}
            losses_selfentro = self.self_entropy_loss(q_A, q_B, cluster_result)
            losses_distlogit = self.dist_of_logit_loss(q_A, q_B,
                cluster_result, self.num_cluster)
            return losses_instcon, q_A, q_B, losses_selfentro, losses_distlogit, losses_cwcon
        else:
            return losses_instcon, None, None, None, None, None

    def instance_contrastive_loss(self, q_A, k_A, im_id_A, q_B, k_B,
        im_id_B, criterion):
        l_pos_A = paddle.einsum('nc,nc->n', q_A, k_A).unsqueeze(axis=-1)
        l_pos_B = paddle.einsum('nc,nc->n', q_B, k_B).unsqueeze(axis=-1)
        l_all_A = paddle.matmul(x=q_A, y=self.queue_A.clone().detach())
        l_all_B = paddle.matmul(x=q_B, y=self.queue_B.clone().detach())
        """Tensor Method: torch.Tensor.cuda, not convert, please check whether it is torch.Tensor.* and convert manually"""
        mask_A = paddle.arange(start=self.queue_A.shape[1]) != im_id_A[:, (None)]
        """Tensor Method: torch.Tensor.reshape, not convert, please check whether it is torch.Tensor.* and convert manually"""
        l_neg_A = paddle.masked_select(x=l_all_A, mask=mask_A).reshape([q_A.
            shape[0], -1])
        """Tensor Method: torch.Tensor.cuda, not convert, please check whether it is torch.Tensor.* and convert manually"""
        mask_B = paddle.arange(start=self.queue_B.shape[1]) != im_id_B[:
            , (None)]
        """Tensor Method: torch.Tensor.reshape, not convert, please check whether it is torch.Tensor.* and convert manually"""
        l_neg_B = paddle.masked_select(x=l_all_B, mask=mask_B).reshape([q_B.
            shape[0], -1])
        logits_A = paddle.concat(x=[l_pos_A, l_neg_A], axis=1)
        logits_B = paddle.concat(x=[l_pos_B, l_neg_B], axis=1)
        logits_A /= self.T
        logits_B /= self.T
        """Tensor Method: torch.Tensor.cuda, not convert, please check whether it is torch.Tensor.* and convert manually"""
        labels_A = paddle.zeros(shape=[logits_A.shape[0]], dtype='int64')
        """Tensor Method: torch.Tensor.cuda, not convert, please check whether it is torch.Tensor.* and convert manually"""
        labels_B = paddle.zeros(shape=[logits_B.shape[0]], dtype='int64')
        loss_A = criterion(logits_A, labels_A)
        loss_B = criterion(logits_B, labels_B)
        return loss_A, loss_B

    def cluster_contrastive_loss(self, q_A, k_A, im_id_A, q_B, k_B, im_id_B,
        cluster_result):
        all_losses = {'domain_A': [], 'domain_B': []}
        for domain_id in ['A', 'B']:
            if domain_id == 'A':
                im_id = im_id_A
                q_feat = q_A
                k_feat = k_A
                queue = self.queue_A.clone().detach()
            else:
                im_id = im_id_B
                q_feat = q_B
                k_feat = k_B
                queue = self.queue_B.clone().detach()
            mask = 1.0
            for n, (im2cluster, prototypes) in enumerate(zip(cluster_result
                ['im2cluster_' + domain_id], cluster_result['centroids_' +
                domain_id])):
                cor_cluster_id = im2cluster[im_id]
                mask *= paddle.equal(x=cor_cluster_id.reshape(shape=[-1, 1]), y=im2cluster.reshape(shape=[1, -1])).astype(dtype='float32')

                all_score = paddle.divide(paddle.matmul(x=q_feat, y=queue), paddle.to_tensor(self.T))
                exp_all_score = paddle.exp(x=all_score)
                log_prob = all_score - paddle.log(x=exp_all_score.sum(axis=
                    1, keepdim=True))
                mean_log_prob_pos = (mask * log_prob).sum(axis=1) / (mask.sum(axis=1) + 1e-08)
                cor_proto = prototypes[cor_cluster_id.astype(paddle.int32)]
                # cor_proto = paddle.unsqueeze(cor_proto, axis=0)
                # paddle.save([k_feat, cor_proto],'x.pth')
                # print(k_feat.shape,cor_proto.shape )
                inst_pos_value = paddle.exp(x=paddle.divide(paddle.einsum('nc,nc->n', k_feat, cor_proto), paddle.to_tensor(self.T)))
                inst_all_value = paddle.exp(x=paddle.divide(paddle.einsum('nc,ck->nk', k_feat, prototypes.T), paddle.to_tensor(self.T)))
                filters = (inst_pos_value / paddle.sum(x=inst_all_value, axis=1) > self.cwcon_filterthresh).astype(dtype='float32')
                filters_sum = filters.sum()
                loss = -(filters * mean_log_prob_pos).sum() / (filters_sum + 1e-08)
                all_losses['domain_' + domain_id].append(loss)

        return paddle.mean(x=paddle.stack(x=all_losses['domain_A'])
            ), paddle.mean(x=paddle.stack(x=all_losses['domain_B']))

    def self_entropy_loss(self, q_A, q_B, cluster_result):
        losses_selfentro = {}
        for feat_domain in ['A', 'B']:
            if feat_domain == 'A':
                feat = q_A
            else:
                feat = q_B
            cross_proto_domains = ['A', 'B']
            for cross_proto_domain in cross_proto_domains:
                for n, (im2cluster, self_proto, cross_proto) in enumerate(zip
                    (cluster_result['im2cluster_' + feat_domain],
                    cluster_result['centroids_' + feat_domain],
                    cluster_result['centroids_' + cross_proto_domain])):
                    if str(self_proto.shape[0]) in self.num_cluster:
                        key_selfentro = ('feat_domain_' + feat_domain +
                            '-proto_domain_' + cross_proto_domain +
                            '-cluster_' + str(cross_proto.shape[0]))
                        if key_selfentro in losses_selfentro.keys():
                            losses_selfentro[key_selfentro].append(self.
                                self_entropy_loss_onepair(feat, cross_proto))
                        else:
                            losses_selfentro[key_selfentro] = [self.
                                self_entropy_loss_onepair(feat, cross_proto)]
        return losses_selfentro

    def self_entropy_loss_onepair(self, feat, prototype):
        logits = paddle.divide(paddle.matmul(x=feat, y=prototype.T), paddle.to_tensor(self.
            selfentro_temp))
        self_entropy = -paddle.mean(x=paddle.sum(x=paddle.nn.functional.
            log_softmax(x=logits, axis=1) * paddle.nn.functional.softmax(x=
            logits, axis=1), axis=1))
        return self_entropy

    def dist_of_logit_loss(self, q_A, q_B, cluster_result, num_cluster):
        all_losses = {}
        for n, (proto_A, proto_B) in enumerate(zip(cluster_result[
            'centroids_A'], cluster_result['centroids_B'])):
            if str(proto_A.shape[0]) in num_cluster:
                domain_ids = ['A', 'B']
                for domain_id in domain_ids:
                    if domain_id == 'A':
                        feat = q_A
                    elif domain_id == 'B':
                        feat = q_B
                    else:
                        feat = paddle.concat(x=[q_A, q_B], axis=0)
                    loss_A_B = self.dist_of_dist_loss_onepair(feat, proto_A,
                        proto_B)
                    key_A_B = ('feat_domain_' + domain_id + '_A_B' +
                        '-cluster_' + str(proto_A.shape[0]))
                    if key_A_B in all_losses.keys():
                        all_losses[key_A_B].append(loss_A_B.mean())
                    else:
                        all_losses[key_A_B] = [loss_A_B.mean()]
        return all_losses

    def dist_of_dist_loss_onepair(self, feat, proto_1, proto_2):
        proto1_distlogits = self.dist_cal(feat, proto_1)
        proto2_distlogits = self.dist_cal(feat, proto_2)
        loss_A_B = paddle.nn.functional.pairwise_distance(proto1_distlogits,
            proto2_distlogits, p=2) ** 2
        return loss_A_B

    def dist_cal(self, feat, proto, temp=0.01):
        proto_logits = paddle.nn.functional.softmax(x=paddle.matmul(x=feat,
            y=proto.T) / temp, axis=1)
        proto_distlogits = 1.0 - paddle.matmul(x=paddle.nn.functional.
            normalize(x=proto_logits, axis=1), y=paddle.nn.functional.
            normalize(x=proto_logits.T, axis=0))
        return proto_distlogits


class SplitBatchNorm(paddle.nn.BatchNorm2D):

    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.momentum = 0.1
        self.eps = 1e-05

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training:
            running_mean_split = self._mean.tile(repeat_times=[self.
                num_splits])
            running_var_split = self._variance.tile(repeat_times=[self.
                num_splits])
            
            outcome = paddle.nn.functional.batch_norm(input.reshape(shape=[-
                1, C * self.num_splits, H, W]), running_mean_split,
                running_var_split, self.weight.tile(repeat_times=[self.
                num_splits]), self.bias.tile(repeat_times=[self.num_splits]
                ), True, self._momentum, self.eps).reshape(shape=[N, C, H, W])
            
            paddle.assign(running_mean_split.reshape(shape=[self.num_splits, C]).mean(axis=0), self._mean)
            paddle.assign(running_var_split.reshape(shape=[self.num_splits, C]).mean(axis=0), self._variance)

            return outcome
        else:
            return paddle.nn.functional.batch_norm(input, self._mean,
                self._variance, self.weight, self.bias, False, self.
                momentum, self.eps)
