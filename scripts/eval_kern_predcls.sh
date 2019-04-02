#!/usr/bin/env bash
python models/eval_rels.py -m predcls -p 100 -clip 5 \
-ckpt checkpoints/kern_sgcls_predcls.tar \
-test \
-b 1 \
-use_ggnn_rel \
-ggnn_rel_time_step_num 3 \
-ggnn_rel_hidden_dim 512 \
-ggnn_rel_output_dim 512 \
-use_rel_knowledge \
-rel_knowledge prior_matrices/rel_matrix.npy \
-cache caches/kern_predcls.pkl \
-save_rel_recall results/kern_rel_recall_predcls.pkl