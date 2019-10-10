commandOutput="$(python make_logs.py --model "custom_faster_rcnn_inceptionv2")"
echo "Output was $commandOutput"
PIPELINE_CONFIG_PATH='PATH TO BE CONFIGURES'
MODEL_DIR="$commandOutput" 
NUM_TRAIN_STEPS=2000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
    
