#!/bin/bash

for j in   'am' #'dz' 'ha' 'ig' 'kr' 'ma' 'pcm' 'pt' 'sw' 'ts' 'twi' 'yo'
do
 
    for i in "castorini/afriteva_small"  #"masakhane/afri-mt5-base" "masakhane/afri-byt5-base"
    do

        train_data_path="SubtaskA/${j}_train_new.tsv"
        eval_data_path="SubtaskA/${j}_dev_new.tsv"
        test_data_path="SubtaskA/${j}_dev_new.tsv"

        model_name_or_path=$i
        tokenizer_name_or_path=$i
        output_dir=$i

        max_seq_length="128"
        learning_rate="3e-4"
        train_batch_size="4"
        eval_batch_size="4"
        num_train_epochs="5"
        gradient_accumulation_steps="16"
        class_labels="positive,negative,neutral"
        data_column="tweet"
        target_column="label"
        prompt="classify: "

        python classification_trainer.py --train_data_path=$train_data_path \
                --eval_data_path=$eval_data_path \
                --test_data_path=$test_data_path \
                --model_name_or_path=$model_name_or_path \
                --tokenizer_name_or_path=$tokenizer_name_or_path \
                --output_dir=$output_dir \
                --max_seq_length=$max_seq_length \
                --train_batch_size=$train_batch_size \
                --eval_batch_size=$eval_batch_size \
                --num_train_epochs=$num_train_epochs \
                --gradient_accumulation_steps=$gradient_accumulation_steps \
                --class_labels=$class_labels \
                --target_column=$target_column \
                --data_column=$data_column \
                --prompt=$prompt \
                --learning_rate="3e-4" \
                --weight_decay="0.0" \
                --adam_epsilon="1e-8" \
                --warmup_steps="0" \
                --n_gpu="1" \
                --fp_16="false" \
                --max_grad_norm="1.0" \
                --opt_level="O1" \
                --seed="42"
                
                


#             python classification_trainer.py --train_data_path=$train_data_path \
#                 --eval_data_path=$eval_data_path \
#                 --test_data_path=$test_data_path \
#                 --model_name_or_path=$model_name_or_path \
#                 --tokenizer_name_or_path=$tokenizer_name_or_path \
#                 --output_dir=$output_dir \
#                 --max_seq_length=$max_seq_length \
#                 --train_batch_size=$train_batch_size \
#                 --eval_batch_size=$eval_batch_size \
#                 --num_train_epochs=$num_train_epochs \
#                 --gradient_accumulation_steps=$gradient_accumulation_steps \
#                 --class_labels=$class_labels \
#                 --target_column=$target_column \
#                 --data_column=$data_column \
#                 --prompt=$prompt \
#                 --learning_rate="3e-4" \
#                 --weight_decay="0.0" \
#                 --adam_epsilon="1e-8" \
#                 --warmup_steps="0" \
#                 --n_gpu="1" \
#                 --fp_16="false" \
#                 --max_grad_norm="1.0" \
#                 --opt_level="O1" \
#                 --seed="42"
                
#             python classification_trainer.py --train_data_path=$train_data_path \
#                 --eval_data_path=$eval_data_path \
#                 --test_data_path=$test_data_path \
#                 --model_name_or_path=$model_name_or_path \
#                 --tokenizer_name_or_path=$tokenizer_name_or_path \
#                 --output_dir=$output_dir \
#                 --max_seq_length=$max_seq_length \
#                 --train_batch_size=$train_batch_size \
#                 --eval_batch_size=$eval_batch_size \
#                 --num_train_epochs=$num_train_epochs \
#                 --gradient_accumulation_steps=$gradient_accumulation_steps \
#                 --class_labels=$class_labels \
#                 --target_column=$target_column \
#                 --data_column=$data_column \
#                 --prompt=$prompt \
#                 --learning_rate="3e-4" \
#                 --weight_decay="0.0" \
#                 --adam_epsilon="1e-8" \
#                 --warmup_steps="0" \
#                 --n_gpu="1" \
#                 --fp_16="false" \
#                 --max_grad_norm="1.0" \
#                 --opt_level="O1" \
#                 --seed="42"

    done
done