#!/usr/bin/env bash
python models/train_detector.py -b 6 -lr 1e-3 -save_dir checkpoints/vgdet -nepoch 50 -ngpu 3 -nwork 3 -p 100 -clip 5
