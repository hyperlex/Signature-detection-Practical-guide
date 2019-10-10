import time
import os
import datetime
import argparse

def make_logs(root_dir, model):
	now = str(datetime.datetime.now()).replace(' ', '_')
	model_dir = os.path.join(root_dir, model)
	#print(model_dir)
	logdir_name = 'logs_'+ now
	logdir_path = os.path.join(model_dir, logdir_name)
	print(logdir_path)
	if not os.path.exists(model_dir):
		os.mkdir(model_dir)
	#os.chdir(model_dir)
	os.mkdir(logdir_path)
	print(logdir_path)
	#print('save training logs in {} directory'.format(logdir_path))
	return logdir_path
	
		
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='custom_faster_rcnn_inceptionv2',
	                    choices=['custom_faster_rcnn_inceptionv2','custom_ssdlite_mobilenet_v2_coco','lasaygues', 'arkea', 'arkea_checkbox', 'arkea_signature_poc'],
	                    help='model for object detection')
	parser.add_argument('--save_logs', type=bool, default=True, help='set to True to save the logs')

	#Parameters
	args = parser.parse_args()
	model = args.model
	save_logs = args.save_logs
	
	root_dir = '../../logs'
	if not os.path.exists(root_dir):
	    os.mkdir(root_dir)
	if save_logs:
		logdir = make_logs(root_dir, model)
		print('save training logs in {} directory'.format(logdir))



 
