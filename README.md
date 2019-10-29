<h1 align="center">
    <p>Signature Detection for Legal Documents</p>
</h1>

![image](image_blog.jpg =650x400)

**The aim of this repository is to explain explain how to train a signature detector in few steps, and use it for inference.**
In our [BLOG post](https://hyperlex.ai/blog/tech/handwriting-detection-contract-management.html), we provide complementary information on the motivations and the background of this project. We also explain how you can annotate your documents in few steps. Make sure to check it!

## Quick Setup (OS/Linux)

### Clone repository using repo URL :  
```
git clone https://github.com/hyperlex/Signature-detection-Practical-guide.git
```

### Install required packages and dependancies :
If your are running on GPUs, make sure to replace ```tensorflow``` with ```tensorflow-gpu``` in the ```requirements.txt``` file.
Also make sure to download a version older than the latest 2.0 **(we use 1.15)** :

```
pip install -r requirements.txt
```

### Setup the COCO API, used to compute object detection evaluation metrics.
We use the COCO API to compute our evaluation metrics. the COCO evaluation is more strict, enforcing various metrics with various IOUs and object sizes. For more information , see [COCO evaluation metrics](http://cocodataset.org/#detection-eval) for more information.

```
cd Signature-detection-Practical-guide/cocoapi/PythonAPI
make
cp -r pycocotools ../../models/research/
```

### Protocol Buffer
Protocol Buffers [Protobuf](https://github.com/protocolbuffers/protobuf)  are Google’s language-neutral, platform-neutral, extensible mechanism for serializing structured data, – think of it like XML, but smaller, faster, and simpler. The Protobuf libraries must be compiled. This should be done by running the following commands from the ```models/research/``` directory:

**On LINUX**
```
# From models/research/
unzip protobuf.zip
./bin/protoc object_detection/protos/*.proto --python_out=.
```

**On MacOS**
```
# From models/research/
unzip protobuf.zip
protoc object_detection/protos/*.proto --python_out=.
```
If you have **errors** while compiling, **manually install the protobuf compiler**.
If you have **homebrew**, download and install the protobuf with ```brew install protobuf```, otherwise run:
```
rm -f protobuf.zip
curl -OL https://github.com/google/protobuf/releases/download/v3.3.0/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
rm -f $PROTOC_ZIP
```
Run the compilation again 
```
# Run the compilation process again from tensorflow/models/research/ 
protoc object_detection/protos/*.proto --python_out=.
```

### Adding libraries to the PYTHONPATH

The ```PYTHONPATH``` is a list of directories for your computer to check whenever you type import library into the interpreter. When running locally, the ```Signature-detection-Practical-guide/models/research/``` and ```Signature-detection-Practical-guide/models/research/slim``` directories should be appended to ```PYTHONPATH```.

To do so go to your **bash rc** file with the ```cd ~/.bashrc file``` command and append ```export PYTHONPATH=$PYTHONPATH:PATH_TO_BE_CONFIGURED:PATH_TO_BE_CONFIGURED/slim``` with the absolute path to the ```Signature-detection-Practical-guide/models/research/``` directory instead of **```PATH_TO_BE_CONFIGURED```**. This will modify the ```PYTHONPATH``` everytime you open a new terminal. 

 If you aren’t familiar with modifying your .bashrc file, you will need to navigate every new terminal console to the ```Signature-detection-Practical-guide/models/research/``` directory and enter the command ```export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim ```


### Testing installation
```
python3 object_detection/builders/model_builder_test.py
```
If you pass the tests, you should see a message similar to the following

```
----------------------------------------------------------------------
Ran 16 tests in 0.049s

OK (skipped=1)
```

## Creating dataset
If you wish to create your own custom dataset, we explain the process of **image annotation** in the section of our [blog](https://hyperlex.ai/blog/tech/handwriting-detection-contract-management.html) post.
We also provide sample images of signed contracts in this repository. to create a dataset for signature detection with the provided images:

```
# From the ```Signature-detection-Practical-guide/dataset_utils```
python3 make_official_dataset.py --labels "signature" --img_dir ../data/dataset
```

The ```make_official_dataset.py``` imports  ```../data/dataset/signature_detection_blog-export.json ``` where all the bbox annotations are stored, and creates a TFRrecords train and test set (the TFRecord format is a simple format for storing a sequence of binary records). It serializes the data (encoded image, annotations, source path, class) ans stores it in a set of small files. 
The script will also create a ```data/tfrecords``` directory where the train and test set are saved in ```.tfrecords``` format. 
If you import your own images, you can use the parser to add labels to your dataset

If you wish to detect signatures and other handwritings, you can create a dataset with multiple labels.
```
# Multi-label example
# From the ```Signature-detection-Practical-guide/dataset_utils```
python3 make_official_dataset.py --labels "signature" "paraphe" "date"  --img_dir ../data/dataset
```


## Training a signature detector in 4 steps

### **1. Decide which model to use and download it**

[Here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) you'll find a list downloadable models. In this tutotrial, we are using the faster_rcnn_inception_v2_coco model.

```
#From Signature-detection-Practical-guide/
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
tar -zxvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
rm -r faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```

### **2. Define your label map**

In the ```config/``` directory you will find a ```label_map.pbtxt``` file. In this file, each class is mapped to an id (starting at 1 and not 0!). If you wish to train a model on more classes, you will need to append items to this file.
```
item {
  id: 1
  name: "signature"
}
```

### **3. Customize a config file for your model**

In this file you can configure parameters and your training. we will use ```faster_rcnn_inception_v2.config``` and customize it to match our needs. If you wish to use another model, refer to tensorflow's official [Github repository](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

***Change the number of classes***
```
model {
  faster_rcnn {
    num_classes: 1 # change here
    image_resizer {
      keep_aspect_ratio_resizer {
	min_dimension: 600
	max_dimension: 1024
      }
    }
```

If you are dealing with a multi-class problem, change ```num_classes```. You can also change the resizer's parameters 	    (can be useful when dealing with smaller objects). For instance, with the above parameters, if your input is a ```1200 x 512``` image, it will be resized to the size ```1024 x 600```.


***Configure the training***
```
train_config {
  batch_size: 8
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  optimizer {
    momentum_optimizer {
      learning_rate {
        manual_step_learning_rate {
          initial_learning_rate: 0.002
          schedule {
          
            step: 100
            learning_rate: 0.0002          }
          schedule {
            step: 250
            learning_rate: 0.00002          }
        }
      }
```
If your GPU is giving you memory erros, decrease the ```batch_size```. You can also modify the training schedule and update your learning rate value at specific steps


***add the path to the model downloaded in the section 1.***
```
##### change path
fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"
```

***add the absolute path to your train and test sets.***
```
train_input_reader {
  ##### change path
  label_map_path:"PATH_TO_BE_CONFIGURED/Signature-detection-Practical-guide/config/label_map.pbtxt"
  tf_record_input_reader {
  ##### change path
    input_path:"PATH_TO_BE_CONFIGURED/Signature-detection-Practical-guide/data/tfrecords/officialtrainset.tfrecords"
  }
```
```
eval_input_reader {
  ##### change path here ... (abs path)
  label_map_path:"PATH_TO_BE_CONFIGURED/Signature-detection-Practical-guide/config/label_map.pbtxt"
  shuffle: false
  num_readers: 1
  tf_record_input_reader {
  ##### change path here ... (abs path)
    input_path:"PATH_TO_BE_CONFIGURED/Signature-detection-Practical-guide/data/tfrecords/officialtestset.tfrecords"
  }
```

**4. Run training**

In the ```Signature-detection-Practical-guide/config/models/research/``` directory, go to the ```train_sign_detect_frcnn.sh``` file and replace the variable ```PIPELINE_CONFIG_PATH``` with the path to the config file of your customized model. You can also change the variable ```NUM_TRAIN_STEPS``` to choose the number of training steps.

You can also notice that the script is calling a ```make_logs.py``` script which will save the model checkpoints to a ```Signature-detection-Practical-guide/logs/NAME_OF_MODEL``` directory.

Run the bash file to start training.
```
## Run training from Signature-detection-Practical-guide/config/models/research/
bash train_sign_detect_frcnn.sh
```

## Inference

### Exporting the inference graph 
```
# from `Signature-detection-Practical-guide/models/research/object_detection/
bash export_model.sh
```
In ```export_model.sh``` you will need to specifiy the path to your config file ```pipeline_config_path```, the path to your trained model (in **```logs```**) and the directory to save the tensorflow frozen inference graph.

here's an example 
```
python3 export_inference_graph.py \
	 --input_type image_tensor \
	 --pipeline_config_path ../../../config/custom_faster_rcnn_inceptionv2.config \
	 --trained_checkpoint_prefix ../../../logs/custom_faster_rcnn_inceptionv2/logs_2019-08-28_16:11:29.305430/model.ckpt-696 \
	 --output_directory ../../../inference/frozen_inference_graph
```


### Running the inference

Once the frozen graph is exported, you are ready to run the inference on new samples.
```
# from `Signature-detection-Practical-guide/inference/
bash run_detector_inference.sh
```
In the ```run_detector_inference.sh```, you will first need to specify the path to the *inference config file* ```config_path```. **Not to be confused** with config file of the model where training parameters are configured, this one is used to configure the inference, you will find an example in ```Signature-detection-Practical guide/inference/utils/signature_detect_config.json```.

Next, add the path to the saved inference graph ```model_path``` and the directory of images to run the inference on ```img_dir```.

Here's an example
```
python3 HandwritingDetector.py \
        --config_path utils/signature_detect_config.json \
        --model_path frozen_inference_graph/frozen_inference_graph.pb \
        --img_dir ../data/dataset
```
The main function of the script will output a dictionary (```output_dict``` ) with the  predicted bounding boxes coordinates 
and a matplotlib visualization of each image with the plotted predicted bounding boxes.

Run the bash file to start the inference
```
## Run inference from Signature-detection-Practical guide/inference/
bash run_detector_inference.sh
```
