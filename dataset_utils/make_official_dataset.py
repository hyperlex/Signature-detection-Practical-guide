import random
import tensorflow as tf
import argparse
import sys
import copy
import os
import json

# change path here
sys.path.append('../')
sys.path.append('../models/research/')

from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('train_output_path', '../data/tfrecords/officialtrainset.tfrecords',
                    'Path to output train set TFRecord')
flags.DEFINE_string('test_output_path', '../data/tfrecords/officialtestset.tfrecords',
                    'Path to output train set TFRecord')
FLAGS = flags.FLAGS


def create_tf_example(filename, label0, labels, signature_vs_others):
    """
    Creates a tf.Example proto from sample image.
    Args:
    encoded_cat_image_data: The jpg encoded data of the cat image.
    Returns:ii
    example: The created tf.Example.
    """
    image_format = b'jpg'

    with open(filename, 'rb') as image:
        f = image.read()
        encoded_image_data = bytes(f)

    width, height = label0['asset']['size']['width'], label0['asset']['size']['height']
    regions = label0['regions']

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for bbox in regions:
        if bbox['tags'][0] not in labels:
            continue

        # checking if bbox coordinates are correct:

        w, h = bbox['boundingBox']['width'], bbox['boundingBox']['height']
        assert (bbox['boundingBox']['left'] == bbox['points'][0]['x'])
        assert (bbox['boundingBox']['top'] == bbox['points'][0]['y'])
        assert (bbox['boundingBox']['left'] + w - bbox['points'][2]['x'] <= 0.0001)
        assert (bbox['boundingBox']['top'] + h - bbox['points'][2]['y'] <= 0.0001)

        xmins.append(bbox['points'][0]['x'] / width)
        xmaxs.append(bbox['points'][2]['x'] / width)
        ymins.append(bbox['points'][0]['y'] / height)
        ymaxs.append(bbox['points'][2]['y'] / height)

        if signature_vs_others:
            if bbox['tags'][0] == 'signature':
                classes_text.append('signature'.encode('utf-8'))
                classes.append(1)
            else:
                classes_text.append('others'.encode('utf-8'))
                classes.append(2)
        else:
            if bbox['tags'][0] in labels:
                idx = labels.index(bbox['tags'][0]) + 1
                classes_text.append(bbox['tags'][0].encode('utf-8'))
                classes.append(idx)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        choices=["signature", "paraphe", "date", "stamp", "autre"],
        help="labels to be considered in the dataset",
        default="signature"
    )
    parser.add_argument("--signature_vs_others", action="store_true")
    parser.add_argument("--img_dir", help="directory containing all images", required=True)
    parser.add_argument(
        "--json_path",
        default="../data/dataset/signature_detection_blog-export.json",
        help="json with all the labels"
    )
    args = parser.parse_args()
    print(args)

    if not os.path.exists("../data/tfrecords/"):
        os.mkdir("../data/tfrecords/")

    # Label map
    with open(args.json_path) as json_file:
        label_map = json.load(json_file)
        keys = list(label_map["assets"].keys())
        # train and test split
        keys_train = random.sample(keys, 30)
        keys_test = [k for k in keys if k not in keys_train]
    random.shuffle(keys_train), random.shuffle(keys_test)
    print('{} images in the train folder and {} images in the test folder'.format(len(keys_train), len(keys_test)))

    writer_train = tf.python_io.TFRecordWriter(FLAGS.train_output_path)
    writer_test = tf.python_io.TFRecordWriter(FLAGS.test_output_path)

    for writer, keys in zip([writer_train, writer_test], [keys_train, keys_test]):
        count = 0
        for key in keys:
            label0 = label_map["assets"][key]
            filename = copy.deepcopy(label0["asset"]["name"])
            filepath = os.path.join(args.img_dir, filename)
            if os.path.exists(filepath):
                count += 1
            else:
                continue
            encoded_filepath = filepath.encode('utf-8')
            tf_example = create_tf_example(encoded_filepath, label0, args.labels, args.signature_vs_others)
            writer.write(tf_example.SerializeToString())
        writer.close()
        print(f'{count} images')


if __name__ == '__main__':
    tf.app.run()
