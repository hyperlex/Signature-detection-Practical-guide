import json
import numpy as np
import tensorflow as tf
import time
import os
from distutils.version import StrictVersion
import PIL
import argparse
import sys

sys.path.append("../utils/")
from utils.visualization_utils import visualize_boxes_and_labels_on_image_array


class HandwritingDetector:
    def __init__(self, config, model_path):

        #if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
            #raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

        # path to frozen graph for inference
        self.model_path = model_path
        self.label_map = config["LabelMap"]
        self.image_size = config["config"]["ImageSize"]
        self.timer = 0
        self.graph_global = None
        self.session = None
        self.num_iter = config["config"]["MaxIterVisualization"]

    def loadModel(self):
        """
        :return: no return value. sets the tensorflow graph as a global variable
        """

        # load protobuf file from the disk and parse it
        # to retrieve the unserialized graph
        with tf.gfile.GFile(self.model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # import the graph_def into a new graph and returns it
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='prefix')

        # assign loaded default graph to the global graph
        self.graph_global = graph
        self.session = tf.Session(graph=self.graph_global)

    def train(self):
        pass

    def png_to_jpg(self, img, image, is_path):
        """
        :param: img: a path to a png image or None
        :param image: a PNG image loaded with PIL
        :param is_path: whether img is a path
        :return: a PIL image object
        """
        if is_path:
            tokenized_path = img.split("/")
            if tokenized_path[0] == '':
                dir_img = "/" + os.path.join(*tokenized_path[:-1]) + "_jpg"
            else:
                dir_img = os.path.join(*tokenized_path[:-1]) + "_jpg"
            print("JPEG images will be saved in {}".format(dir_img))
            if not os.path.exists(dir_img):
                os.makedirs(dir_img)
            jpg_img = os.path.join(dir_img, tokenized_path[-1].replace(".png", ".jpg"))
            if not os.path.exists(jpg_img):
                if image.mode != "RGB":
                    image.convert("RGB").save(jpg_img)
                else:
                    image.save(jpg_img)
            image = PIL.Image.open(jpg_img)
            return image
        else:
            if image.mode != "RGB":
                image.convert("RGB").save('tmp.jpg')
            else:
                image.save('tmp.jpg')
            image = PIL.Image.open('tmp.jpg')
            return image

    def run_inference_single_image(self, img, tensors):
        """

        :param image: absolute image path/or loaded image (PIL object)
        :param tensors: tensorflow tensors
        :return: bounding boxes normalized coordinates,prediction scores,
                 predicted class and the total number of detections
                 for a single image
        """

        # Load a unique image
        if type(img) == str:  # param img is a path
            image = PIL.Image.open(img)
            if image.format == "PNG":
                image = self.png_to_jpg(img, image, is_path=True)
            else:
                if image.mode != "RGB":
                    image = image.convert("RGB")
        else:  # param img is a PIL object
            if img.format == "JPEG":
                if img.mode != "RGB":
                    image = img.convert("RGB")
                pass
            elif img.format == "PNG":
                image = self.png_to_jpg(None, img, is_path=False)

        assert type(image) == PIL.JpegImagePlugin.JpegImageFile or type(
            image) == PIL.Image.Image, "Loaded image is not a PIL object"

        # Reshape and transform image in np array
        (im_width, im_height) = image.size
        image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

        # Run the inference and calculate the time
        start = time.time()

        outputs = self.session.run([tensors['y0'],
                                    tensors['y1'],
                                    tensors['y2'],
                                    tensors['y3']],
                                   feed_dict={tensors['x']: np.expand_dims(image_np, 0)})

        end = time.time()
        print('inference time for a single image is {}'.format(end - start))

        self.timer += end - start

        return outputs, image_np

    def infer(self, inference_samples, display_vis=True):
        """

        :param inference_samples:list of abs paths of images/ loaded images (PIL objects) to run inference on.
        :param display_vis: set to True to plot images with predicted bounding boxes.
        :return:(dictionary) image absolute path is mapped to another dictionnary
        containing for each predicted bounding box, the normalized coordinates,
        the prediction scores, the predicted class and the total number of detections;
        """
        assert type(inference_samples) == list, "inference_samples must be a list"

        # dict of tensors to pass to the 'run_inference_single_image' function
        tensors = {}
        tensors['x'] = self.graph_global.get_tensor_by_name('prefix/image_tensor:0')
        tensors['y0'] = self.graph_global.get_tensor_by_name('prefix/detection_boxes:0')
        tensors['y1'] = self.graph_global.get_tensor_by_name('prefix/detection_scores:0')
        tensors['y2'] = self.graph_global.get_tensor_by_name('prefix/detection_classes:0')
        tensors['y3'] = self.graph_global.get_tensor_by_name('prefix/num_detections:0')

        # dict to return with bbox coordinates and scores
        output_dict = {}

        print('Inference ...>>>')

        cpt = 0
        n = len(inference_samples)  # number of samples
        for idx, image in zip(range(n), inference_samples):
            tmp = {}

            outputs, image_np = self.run_inference_single_image(image, tensors)
            w, h = image_np.shape[1], image_np.shape[0]

            N, D = outputs[0].shape[1], outputs[0].shape[2]

            tmp['detection_boxes'] = outputs[0].reshape(N, D)
            tmp['detection_scores'] = outputs[1].reshape(300)
            tmp['detection_classes'] = outputs[2].reshape(300).astype(int)
            tmp['num_detections'] = outputs[3]
            tmp['size'] = {'width': w, 'height': h}
            if type(image) == str:
                tmp['image_id'] = image.split('/')[-1].split('.')[0]
                output_dict[image] = tmp
            else:
                output_dict[str(idx)] = tmp

            print(">>..Bbox visualization")
            if display_vis and cpt < self.num_iter:
                if type(image) == str:
                    self.visualize_bbox(image_np, output_dict[image])
                    cpt += 1
                else:
                    self.visualize_bbox(image_np, output_dict[str(idx)])
                    cpt += 1

        return output_dict

    def visualize_bbox(self, image_np, img_output_dict):
        """
        :param image_np: numpy array image (RGB)
        :param img_output_dict: output of run_inference_single_image
        :return: no return value. display_vis=True, plots images
                 with plotted predicted bounding boxes.
        """

        import matplotlib;
        matplotlib.use('TkAgg')
        from matplotlib import pyplot as plt

        category_index_binary = self.label_map

        # Visualization of the results of a detection.
        visualize_boxes_and_labels_on_image_array(image=image_np,
                                                  boxes=img_output_dict['detection_boxes'],
                                                  classes=img_output_dict['detection_classes'],
                                                  scores=img_output_dict['detection_scores'],
                                                  category_index=category_index_binary,
                                                  use_normalized_coordinates=True,
                                                  line_thickness=4)

        plt.figure(figsize=self.image_size)
        plt.imshow(image_np)
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="abs path of the config file (json)")
    parser.add_argument("--model_path", type=str, help="abs path of the TF frozen graph")
    parser.add_argument("--img_dir", type=str, help="directory of images")
    args = parser.parse_args()
    print(args)

    with open(args.config_path, 'r') as f:
        config = json.load(f)

    detector = HandwritingDetector(config=config, model_path=args.model_path)

    inference_samples = [os.path.join(args.img_dir, f) for f in os.listdir(args.img_dir) if
                         (".png" in f or ".jpg" in f)]

    ##### If user wants to load images in memory
    ##### inference_samples = [PIL.Image.open(f) for f in inference_samples]

    # Loading model
    detector.loadModel()

    # Infered Bounding Boxes
    output_dict = detector.infer(inference_samples)

    return output_dict


if __name__ == "__main__":
    main()
