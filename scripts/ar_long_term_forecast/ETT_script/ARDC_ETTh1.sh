export CUDA_VISIBLE_DEVICES=0

model_name=ARDC

python -u run.py \
  --task_name ar_long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --hnet_arch_layout '["m1", ["T1"], "m1"]' \
  --hnet_d_model 256 512 \
  --hnet_d_intermediate 0 1024\
  --hnet_ssm_chunk_size 32 \
  --hnet_ssm_d_conv 2 \
  --hnet_ssm_d_state 64 \
  --hnet_ssm_expand 2 \
  --hnet_attn_num_heads 2 2 \
  --hnet_attn_rotary_emb_dim 8 8 \
  --hnet_attn_window_size -1 -1 \
  --hnet_num_experts 8 \
  --hnet_moe_loss_weight 0.01 \