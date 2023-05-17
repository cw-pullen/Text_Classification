#export CUDA_VISIBLE_DEVICES=1

python3 -u new_data_classifier.py \
--test_dataset='test.txt' \
--gpt2_xl_gltr_ckpt='./ckpts/GLTR_gpt2xl_TRAIN_k40_temp07_mix_512.sav' \
--gpt2_model='gpt2-xl' \
