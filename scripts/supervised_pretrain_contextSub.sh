#!/bin/bash

export CUDA_VISIBLE_DEVICES=2


python -m contextSub.pretrain_supervised \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.00001 \
    --lr_scale 1 \
    --decay 0.0001 \
    --num_layer 5 \
    --emb_dim 300 \
    --dropout_ratio 0.4 \
    --graph_pooling mean \
    --JK last \
    --gnn_type gin \
    --input_model_file contextSub/trained_models/contextSub_chembl_partialCharge_noNorm_filteredPattern_epoch300.pth \
    --filename contextSub_chembl_supervised_pretrained \
    --partial_charge \
    --sub_input \
    --sub_level \
    --context \
    --pooling_indicator \
    --separate_output \
    --save_model contextSub/trained_models/finetuned/contextSub/ > nohup_logs/supervised_pretrain.out 2>&1
