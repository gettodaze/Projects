# -*- coding: utf-8 -*-
from pathlib import Path
from matplotlib.font_manager import FontProperties
from matplotlib import pyplot as plt
import math
from modules import imagenetClassify
from nltk.corpus import wordnet as wn
import os

#see description of constants under the show_images_and_synonyms method.
FONT_SIZE = 10
FONT_PATH = r'C:\WINDOWS\Fonts\yugothm.ttc'
PLOT_MAX_LENGTH = 6
PLOT_SPACING = 2


def get_jap_synonyms(synsets, min_lemmas=2):
    """
    :rtype: List<String>.
    :param synsets: List<Synset>. List of synset objects
    :param min_lemmas: int, The minimum number of lemmas to output. The program will progress through hypernyms until it
    reaches the minimum number of lemmas to output
    :return: List<String>. a string list of all the lemma names in the synset objects in synsets. This may include lemma names from
    hypernyms of the synset objects if the min_lemmas cannot be reached with the lemmas from just the synset objects in
    synsets.
    """
    def synset_info(synset):
        """
        Shows some information about the given synset including hypernyms and holonyms.
        Please see "Relations" at wordnet.princeton.edu for more information about what is contained in a synset and
        its relations.
        :rtype: none
        :param synset: Synset. Synset object
        """
        print("""
        """)
        base_set_offset = synset.offset()
        print("Debug help: {} ({:08})".format(synset, base_set_offset))
        print("-----------------")
        set = synset
        print("Base set: " + repr(set))
        hypernyms = set.hypernyms()
        print("Hypernyms: " + repr(hypernyms))
        hyponyms = set.hyponyms()
        print("Hyponyms: " + repr(hyponyms))
        holonyms = set.member_holonyms()
        print("Holonyms: " + repr(holonyms))
        root_hypernyms = set.root_hypernyms()
        print("Root hypernyms: " + repr(root_hypernyms))
        print("""----------------------
        """)

    def append_item(insert_items, destination_list):
        """
        Appends the items of one list onto another list and returns the combined list.
        :param insert_items: List. Items to be inserted (type: list)
        :param destination_list: List. List to insert into.
        :return: Combined list
        """
        if insert_items:
            for i in insert_items:  destination_list.append(i)
        return destination_list

    def get_lemma_names(lemma_list, return_list=[]):
        """
        Returns a list of names (strings) of lemmas from a list of lemma objects.
        :param lemma_list: List. List of lemma objects
        :param return_list: List. List of string lemma names that to add the lemmas to (empty by default)
        :return: List of string lemma names.
        """
        if lemma_list:
            for l in lemma_list:
                return_list.append(l.name())
        return return_list

    def add_japanese_lemmas_from_synsets(synsets, return_list=[]):
        """
        Returns all the lemma objects in any of the synsets in a list of synset objects
        :param synsets: List<Synset>. List of synset objects to get the lemmas from
        :param return_list: List<Lemma>. List to add the lemmas to (empty by default
        :return: List<Lemma>. List of  all the lemma objects in any of the synsets in a list of synset objects
        """
        # assert (len(synsets) == 0, "No synsets in add japanese lemmas from synsets: {}".format(synsets))
        for syn in synsets:
            # try:
            #     lemmas = syn.lemmas("jpn")
            # except AttributeError:
            #     lemmas = []
            lemmas = syn.lemmas("jpn")
            if not lemmas:
                print("No Japanese entry for {} -- did not add.".format(syn))
                synset_info(syn)
            else:
                return_list = append_item(lemmas, return_list)
        return return_list

    lemmas = add_japanese_lemmas_from_synsets(synsets, [])
    while len(lemmas) < min_lemmas:
        hypernyms = []
        for synset in synsets:
            append_item(synset.hypernyms(), hypernyms)
        synsets = hypernyms
        print("adding from {}".format(synsets))
        lemmas = add_japanese_lemmas_from_synsets(synsets, lemmas)
    synonyms = get_lemma_names(lemmas, return_list=[])
    synonyms = set(synonyms)
    return synonyms


def get_probable_synsets(classification_list, thresh=.2):
    """

    :param classification_list: List<Tuple(name,score,offset)>.
    classification output from the image classify method aka a list of tuplets.
    [(name1, score1, offset1), (name2, score2, offset2)...]
    :param thresh: double. The minimum value of the score for the score to
    be called probable.
    :return: List<Synset>. List of synset objects.
    """

    probable_synsets = []
    for classification in classification_list:
        if classification[1] > thresh:
            synset = wn.synset_from_pos_and_offset(wn.NOUN, classification[2])
            probable_synsets.append(synset)

    return probable_synsets


def file_exists(path):
    """

    :param path: String. path (relative or absolute)
    :return: boolean. indicator of existance or non existance
    """
    my_file = Path(path)
    if my_file.is_file():
        return True
    else:
        return False


def classify_img(impath, tensorboard):
    """
    Calls the imagenetClassify.classify_image method in the modules folder. This will input the image through a neural
    network and attempt to identify the top five nouns in wordnet (a database of a network of words connected via
    a "is-a" word hierarchy. ) that are deemed to be contained in the image. Please see the homepage at
    wordnet.princeton.edu for more information.
    :param impath: String. path to an image (I believe many different types of images are readable, such as png,
    JPG and jpg)
    :return: List<Tuple(name,score,offset)>. a list of 5 tuplets.
    [(name1, score1, offset1), (name2, score2, offset2)...]
    Name is the name of the synset in WordNet
    Score is a measure of likelihood that the given synset is in the picture
    Offset is the offset of the noun synset (a form of identification for the
    synset).
    """
    if file_exists(impath):
        return imagenetClassify.classify_image(impath, tensorboard)
    else:
        raise FileNotFoundError("Path did not exist %s" % impath)


def get_synoyms_verbose(impath, thresh, tensorboard):

    """
    Utilized with the verbose tag in the main method. Recommended way to find synonyms from an image.
    Will output details of the program as its running through the three main methods:
            category_data = classify_img(impath)
            probable_synsets = get_probable_synsets(category_data)
            if probable_synsets:
                synonyms = get_jap_synonyms(probable_synsets)
    :param impath: string. path of an image
    :param thresh: int. threshold for a probable synset
    :return: List<String> List of string of words that are thought to be in the picture or categorically related to
    things in the picture.
    """
    synonyms = []
    print("impath: {}.".format(impath))
    category_data = classify_img(impath, tensorboard)
    # category data format: name, score, offset
    print(category_data)
    names = [cd[0] for cd in category_data]
    print("Image Output Categories: {}.".format(names))  # cd[0] for cd in category_data))
    probable_synsets = get_probable_synsets(category_data, thresh=thresh)
    print("Probable synset IDs: {}".format(probable_synsets))
    if not probable_synsets:
        print("No probable categories.")
    else:
        synonyms = get_jap_synonyms(probable_synsets)
    print("Synonym list: {}".format(synonyms))
    return synonyms


def show_images_and_synonyms(impaths, thresh=.2, verbose=True, tensorboard=False):
    """
        Shows images in a square pyplot grid.
    Constants (initialized at the top of the file):
    FONT_SIZE = 10 (font size on the plot)
    FONT_PATH = r'C:\WINDOWS\Fonts\yugothm.ttc' (location of the sinograph-containing font to use)
    PLOT_MAX_LENGTH = 6 (the max size of rows and columns in the plot)
    ---note: only the first length^2 images from impaths will be shown. For more images, it may be best to write
    to a file or to run iterations of the program.
    PLOT_SPACING = 2 (interval at which to show images in the plot. I find 1 is too crowded.)
    :param impaths: List<String> list of filepath locations of the images to be read.
    Paths must avoid accidental escape sequences from \ (backslash)
    characters combined with escape characters. Otherwise r can be put
    before strings to indicate a raw string, free of escapes.
    Relative paths must be relative to the location of this program--an image folder is recommended to be put in the
    same folder as "modules".
    :param thresh: double. The threshold score for a meaningful categorization in the program. This parameter should be tested
    with different values to see if perhaps a higher or lower value gives better results.
    :param verbose: boolean. Indicates whether the program will output text during the program or not. Unless it is a
    nuisance, this should always be True so that users can see scores and categories not listed in the plot.
    """
    num_images = len(impaths)
    plot_length = math.ceil(math.sqrt(num_images))
    if plot_length > PLOT_MAX_LENGTH:
        plot_length = PLOT_MAX_LENGTH
    if plot_length * plot_length < num_images:
        num_images = plot_length * plot_length
    fp = FontProperties(fname=FONT_PATH, size=FONT_SIZE)
    plt.figure(figsize=(plot_length * PLOT_SPACING, plot_length * PLOT_SPACING))
    plt.suptitle("Thresh = {}".format(thresh))

    for i in range(num_images):
            impath = impaths[i]

            synonyms = []
            if verbose:
                synonyms = get_synoyms_verbose(impath, thresh, tensorboard)
            else:
                category_data = classify_img(impath, tensorboard)
                probable_synsets = get_probable_synsets(category_data, thresh=thresh)
                if probable_synsets:
                    synonyms = get_jap_synonyms(probable_synsets, lemmas=[])

            plt.subplot(plot_length, plot_length, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            I = plt.imread(impath)
            plt.imshow(I)
            label = repr(synonyms)
            if not synonyms:
                label = "無し"
            plt.xlabel(label, fontproperties=fp, wrap=True)
            print("""
            -------------------
            """)

    plt.show()


if __name__ == '__main__':
    # be careful because \U can sometimes mean "Character with 32-bit hex
    # value xxxxxxxx" - C:\\ is always acceptable or r"..\.."
    # the links need to be in an array
    image_filepaths_init_file = "exampleimg/path_list.txt"
    print(os.getcwd())
    image_filepaths_init_file = os.path.join(os.getcwd(),image_filepaths_init_file)
    print(image_filepaths_init_file)
    example_images = []
    with open(image_filepaths_init_file, 'r') as f:
        example_images = f.read().splitlines()
    print(example_images)
    show_images_and_synonyms(example_images, thresh=0.2)
