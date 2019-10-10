python3 export_inference_graph.py \
	 --input_type image_tensor \
	 --pipeline_config_path ../../../config/custom_faster_rcnn_inceptionv2.config \
	 --trained_checkpoint_prefix ../../../logs/custom_faster_rcnn_inceptionv2/logs_2019-08-28_16:11:29.305430/model.ckpt-696 \
	 --output_directory ../../../inference/frozen_inference_graph