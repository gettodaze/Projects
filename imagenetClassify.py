# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs the top 5 predictions along with their probabilities.
It has been editted by John to fit modularly into other programs.
Calls to the flag and app tensorflow methods have been removed.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/

---------------------------------------------------------------------------------
This is Inception v3. I dont know if a better neural network can be found on the internet.
It also uses the 2012 imagenet data. I tried searching for newer data but was unsuccessful. However,
I do know that the network only maps to 1000 categories, and not all of those categories are very applicable to our
services. Perhaps we can take the imagenet data and use pictures and labels from different categories that contain more
of what we would like to do.
TODO: Train the network with newer data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'


class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,
                 label_lookup_path=None,
                 uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(
                model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(
                model_dir, 'imagenet_synset_to_human_label_map.txt')
            self.node_lookup_to_name, self.node_lookup_to_uid = \
                self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.

        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.

        Returns:
          node_id_to_name: dict from integer node ID to human-readable string
          node_id_to_uid: dict from integer node ID to ID in
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name, node_id_to_uid

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup_to_name:
            return ''
        return self.node_lookup_to_name[node_id]

    def id_to_noun_offset(self, node_id):

        def get_offset_from_uid(uid):
            offset = re.sub("\An", "", uid)
            offset = int(offset)
            return offset

        if node_id not in self.node_lookup_to_uid:
            return None

        uid = self.node_lookup_to_uid[node_id]
        offset = get_offset_from_uid(uid)
        return offset


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
            model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image, tensorboard):
    """Runs inference on an image.

    Args:
      image: String. Image filepath string.

    Returns:
      Nothing
    """
    results = []

    def write_event():
        """
        Writes the Tensor Flow Session to ..\TenosrboardFiles to it can be
        visualized from the Anaconda command line by running Tensorboard.
        This can be done with the command "Tensorboard --logdir
        C:\My\Project\Directory\ImageRec\TensorboardFiles". It will output
        text containing: TensorBoard 1.9.0 at http://---------- (Press
        CTRL+C to quit)" - Go to the address in any browser (Chrome and
        Firefox Recommended) to visualize the network.

        No return value.
        """
        dest_directory = (os.path.join(os.getcwd(), "TensorboardFiles"))
        print(os.path.abspath(dest_directory))
        print(os.getcwd())
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        writer = tf.summary.FileWriter(dest_directory)
        writer.add_graph(tf.get_default_graph())
        writer.flush()
        writer.close()

    # Will log that the file does not exist if the logger has been told to
    # output critical messages... I am not sure how the logger gets this
    # information. I deleted a lot of flagging methods for convenience.
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)

    # Gets data from image.
    image_data = tf.gfile.FastGFile(image, 'rb').read()

    # Creates graph from saved GraphDef (should be in
    # "..\tmp\imagenet\ 'classify_image_graph_def.pb'".)
    create_graph()

    # Script to run while the Tensorflow Session is open
    with tf.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.

        # Gets the tensor named softmax:0 (The final neuron)
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

        # Runs the graph for the output of the softmax_tensor (the final
        # neuron) (aka gets the output from the network from the image data
        # input)
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})

        # Writes the event to Tensorboard
        if tensorboard:
            write_event()

        # Formats the output into a 1D numpy array
        predictions = np.squeeze(predictions)



        # Creates node ID --> English string and node ID --> noun offset
        # lookup.
        node_lookup = NodeLookup()

        argsort = np.argsort(predictions)
        top_k = argsort[-num_top_predictions:][::-1]
        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            score = predictions[node_id]
            offset = node_lookup.id_to_noun_offset(node_id)
            # print('%s (score = %.5f)' % (human_string, score))
            results.append((human_string, score, offset))


    return results


def maybe_download_and_extract():
    """Download and extract model tar file from DATA_URL aka
    'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'.

    Downloads files if they do not exist.
    Should download and unzip inception-2015-12-05.tgz to:
     ..\tmp\imagenet
    containing:
        classify_image_graph_def.pb
        cropped_panda.jpg (sample image)
        imagenet_2012_challenge_label_map_proto.pbtxt
        imagenet_synset_to_human_label_map.txt
        LICENSE
    """

    # Makes destination directory if it does not exist
    dest_directory = model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    # gets the download zip file name
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)

    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            """
            This is the reporthook argument for the
            urllib.request.urlretrieve method. From the urlretrieve method
            doc: "The reporthook argument should be a callable that accepts
            a block number, a read size, and the total file size of the URL
            target."
            :param count: block number
            :param block_size: read size
            :param total_size: total file size
            """
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        # Writes DATA_URL to filepath, outputting the filepath. (The
        # underscore accepts additional returns that will not be used.) See
        # method doc for more details.
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)

        # Successful message printout
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    # Extracts the tar file.
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    print(os.getcwd())


def classify_image(impath, tensorboard):
    """
    Main method.
    Sets global variables, downloads the data and models if necessary, and then runs the inference_on_image method.
    GLOBALS:
    model_dir: Path to classify_image_graph_def.pb, imagenet_synset_to_human_label_map.txt, and
    imagenet_2012_challenge_label_map_proto.pbtxt. Defaults to  "tmp/imagenet"
    image_file: String. Global variable name for impath. An absolute path is recommended.
    num_top_predictions: The number of prediction outputs will be in the final list.
    :param impath: String. An absolute path is recommended.
    :return:classification_list: List<Tuple(name,score,id)>. classification output from the
    run_inference_on_image(image) method aka a list of tuplets.
    [(name1, score1, id1), (name2, score2, id2)...]
    """
    # Initialize globals
    global image_file
    global model_dir
    global num_top_predictions
    image_file = impath
    model_dir = (os.path.join(os.getcwd(), "tmp/imagenet"))
    num_top_predictions = 5

    # Downloads the pretrained inception V3 model if it has not been
    # downloaded already
    maybe_download_and_extract()
    image = image_file

    return run_inference_on_image(image, tensorboard)


# This will run if this method is called as the main method.
if __name__ == '__main__':
    classify_image()
