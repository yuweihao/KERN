#!/usr/bin/env bash
python models/train_rels.py -m predcls -p 100 -clip 5 \
-tb_log_dir summaries/kern_predcls \
-save_dir checkpoints/kern_predcls \
-ckpt checkpoints/vgdet/vg-faster-rcnn.tar \
-val_size 5000 \
-adam \
-b 2 \
-lr 1e-5 \
-use_ggnn_rel \
-ggnn_rel_time_step_num 3 \
-ggnn_rel_hidden_dim 512 \
-ggnn_rel_output_dim 512 \
-use_rel_knowledge \
-rel_knowledge prior_matrices/rel_matrix.npy
 