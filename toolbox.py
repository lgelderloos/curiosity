import os

def mkdir(path):
    if not os.path.exists(path):
        print('creating dir {}'.format(path))
        os.mkdir(path)

def make_index_table(_vocabulary: list) -> dict:
    """
    For all the words in the vocabulary list assigns an index number respectively
    Structure
    ----
    :param _vocabulary: word list
    :type _vocabulary: list
    :return: indexed dictionary of vocabulary
    :rtype: dict
    """

    _word_to_index = {}
    index = 0
    for word in _vocabulary:  # for all the words in the vocabulary list assigns an index number respectively
        _word_to_index[word] = index
        index += 1  # increment the index
    return _word_to_index
