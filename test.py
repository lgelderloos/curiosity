import sys
import random

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# general imports
import json
import pickle
import time
import numpy as np

# custom imports
import agent

###############################################################################
# SETTING CONSTANTS & INITIALIZATION
###############################################################################
seed = int(sys.argv[4])

random.seed(a=seed)

# setting torch seeds
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

np.random.seed(seed)

###############################################################################
# evaluation function
###############################################################################

def evaluate(epoch, split='val'):
    listener.eval()
    speaker.eval()
    if split == 'val':
        batchlist = val_batchlist
    elif split == 'test':
        batchlist = test_batchlist
    n_batches = len(batchlist)
    start_time = time.time()
    eval_loss = np.empty(n_batches)
    li_eval_acc = np.empty(n_batches)
    sp_eval_acc = np.empty(n_batches)
    batch_size = np.empty(n_batches)

    batch = 0

    while batch < n_batches:
        language_input, visual_input, targets = load_val_batch(dict_words_boxes,
                                                               batchlist[batch],
                                                               word_to_ix,
                                                               device)
        optimizer.zero_grad()
        obj_guesses = listener(language_input, visual_input)
        word_guesses = speaker(visual_input, obj_guesses)

        # calculate listener accuracy
        li_eval_acc[batch], batch_size[batch] = calc_accuracy(obj_guesses, targets, average=False)

        # calculate speaker accuracy
        sp_eval_acc[batch], _ = calc_accuracy(word_guesses, language_input, average=False)

        # calculate loss
        loss = criterion(word_guesses, language_input)
        eval_loss[batch] = loss.item() * batch_size[batch]

        batch += 1
        if batch % printerval == 0:
            print('| epoch {:2d} | batch {:3d}/{:3d} | t {:6.2f} | L {:6.4f} | l.A {:5.4f} | s.A {:6.4f} |'.format(
                    epoch, batch, n_batches, (time.time() - start_time),
                    np.sum(eval_loss[batch - printerval:batch]) / np.sum(batch_size[batch - printerval:batch]),
                    np.sum(li_eval_acc[batch - printerval:batch]) / np.sum(batch_size[batch - printerval:batch]),
                    np.sum(sp_eval_acc[batch - printerval:batch]) / np.sum(batch_size[batch - printerval:batch])))

    avg_eval_loss = np.sum(eval_loss) / np.sum(batch_size)
    avg_li_eval_acc = np.sum(li_eval_acc) / np.sum(batch_size)
    avg_sp_eval_acc = np.sum(sp_eval_acc) / np.sum(batch_size)

    if split == 'val':
        print('-' * 89)
        print("overall performance on validation set:")
        print('| Loss {:8.4f} | Li.acc. {:8.4f} | Sp.acc. {:8.4f}'.format(avg_eval_loss, avg_li_eval_acc, avg_sp_eval_acc))
        print('-' * 89)
    elif split == 'test':
        print('-' * 89)
        print("overall performance on test set:")
        print('| Loss {:8.4f} | Li.acc. {:8.4f} | Sp.acc. {:8.4f}'.format(avg_eval_loss, avg_li_eval_acc, avg_sp_eval_acc))
        print('-' * 89)
    return avg_eval_loss, avg_li_eval_acc, avg_sp_eval_acc

def load_val_batch(dict_words_boxes, batch, word_to_ix, device):
    # Loads the batches for the validation and test splits of the data
    language_input = []
    visual_input = []
    targets = []

    for img in batch:
        vggs = torch.load("./data/ha_bbox_vggs/" + img + ".pt").to(device)
        for obj in dict_words_boxes[img]:
            language_input.append(get_word_ix(word_to_ix, dict_words_boxes[img][obj]["word"]))

            bbox_indices = []
            n = 0

            for obj_id in dict_words_boxes[img]:
                bbox_indices.append(ha_vggs_indices[img][obj_id][0])
                if obj_id == obj:
                    targets.append(n)
                n += 1
            visual_input.append(vggs[bbox_indices, :])

    lang_batch = torch.tensor(language_input, dtype=torch.long, device=device)
    vis_batch = torch.stack(visual_input)
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    return lang_batch, vis_batch, targets

def calc_accuracy(guesses, targets, average=True):
    """
    in: log probabilities for C classes (i.e. candidate nrs), target 'class'
    indices (from 0 up-to-and-icluding C-1) (object position in your case)
    """
    score = 0
    guess = torch.argmax(guesses.data, 1)

    for i in range(targets.data.size()[0]):
        if guess.data[i] == targets.data[i]:
            score += 1

    if average:
        return score / targets.data.size()[0], targets.data.size()[0]
    else:
        return score, targets.data.size()[0]

#################################################################################
# Interpret command line arguments
#################################################################################

print("\nSys.argv:", sys.argv)
batchsize = int(sys.argv[1])
lr = float(sys.argv[2])
setting = sys.argv[3]

#################################################################################
# DATA PREPROCESSING
#################################################################################

def no_of_objs(data_dict, data_split):
    """
    Returns a dictionary with the
    number of objects per image
    for a train/validation/test
    split of the data
    """

    no_of_objs = {}

    for file in data_split:
        if len(data_dict[file]) not in no_of_objs:
            no_of_objs[len(data_dict[file])] = []
            no_of_objs[len(data_dict[file])].append(file)
        else:
            no_of_objs[len(data_dict[file])].append(file)

    return no_of_objs


def get_word_ix(word_to_ix, word):
    if word in word_to_ix:
        return word_to_ix[word]
    else:
        return word_to_ix["<UNK>"]


def dict_to_batches(no_objs_split, bsz):
    """
    Returns a list of batches. A batch is a
    batch-size lists of file/img ids, of imgs
    containing the same amount of objects.
    The batches are shuffled so that batches
    of different amounts of objects follow
    each other.
    """
    batch_list = []

    for num in no_objs_split.keys():
        batch_list.extend([no_objs_split[num][x:x + bsz] for x in range(0, len(no_objs_split[num]), bsz)])

    np.random.shuffle(batch_list, )

    return batch_list


###############################################################################
# ACTUAL CODE (NO MORE DEFINITIONS)
###############################################################################

# Object vgg indices (object information)
with open("./data/ha_vgg_indices.json", "rb") as input_file:
    ha_vggs_indices = json.load(input_file)

# Regular data (dictionary with all images and their object ids, corresponding words)
with open("./data/dict_words_boxes.json", "rb") as input_file:
    dict_words_boxes = json.load(input_file)

# Train split, image ids
with open("./data/train_data.txt", "rb") as fp:
    train_data = pickle.load(fp)

# Validation split, image ids
with open("./data/validation_data.txt", "rb") as fp:
    validation_data = pickle.load(fp)

# Test split, image ids
with open("./data/test_data.txt", "rb") as fp:
    test_data = pickle.load(fp)

###############################################################################
# PREPROCESSING
###############################################################################

# load the embedding index dictionary associated with this model
with open('word_to_ix/word_to_ix_{}_{}_{}.json'.format(setting, seed, str(lr)), 'r') as wtx:
    word_to_ix = json.load(wtx)

no_objs_test = no_of_objs(dict_words_boxes, test_data)
test_batchlist = dict_to_batches(no_objs_test, batchsize)

ntokens = len(word_to_ix.keys())
print("ntokens:", ntokens)

###############################################################################
# SPECIFY MODEL
###############################################################################

object_size = 4096  # Length vgg vector?
att_hidden_size = 256  # Number of hidden nodes
wordemb_size = 256  # Length word embedding
nonlin = "sigmoid"
print("hidden layer size:", att_hidden_size)
epochs = 40

device = torch.device('cuda')  # Device = GPU

# Makes the listener part of the model:
listener = agent.Listener(object_size, ntokens, wordemb_size,
                          att_hidden_size, nonlinearity=nonlin).to(device)
# Makes the speaker part of the model:
speaker = agent.Speaker(object_size, ntokens, att_hidden_size, nonlinearity=nonlin).to(device)

# Loss function: binary cross entropy
criterion = nn.CrossEntropyLoss(size_average=True)

###############################################################################
# TRAIN LOOP
###############################################################################
# Print after this many batches:
printerval = 100

print("parameters of listener agent:")
for param in listener.parameters():
    print(type(param.data), param.size())
print("parameters of speaker agent:")
for param in speaker.parameters():
    print(type(param.data), param.size())

optimizer = optim.Adam(list(listener.parameters()) + list(speaker.parameters()), lr=lr)

# Creating numpy arrays to store loss and accuracy
test_loss = np.empty(epochs)
listener_test_acc = np.empty(epochs)
speaker_test_acc = np.empty(epochs)

# At any point you can hit Ctrl + C to break out of testing early.
try:
    for epoch in range(1, epochs+1):
        # Load model
        listener.load_state_dict(torch.load('models/{}/liModel_{}_{}_{}_ep{}.pth'.format(setting, setting, str(lr), seed, epoch)))
        speaker.load_state_dict(torch.load('models/{}/spModel_{}_{}_{}_ep{}.pth'.format(setting, setting, str(lr), seed, epoch)))
        epoch_test_loss, epoch_listener_test_acc, epoch_speaker_test_acc = evaluate(epoch, split = 'test')
        test_loss[epoch-1] = epoch_test_loss
        listener_test_acc[epoch-1] = epoch_listener_test_acc
        speaker_test_acc[epoch-1] = epoch_speaker_test_acc

# To enable to hit Ctrl + C and break out of testing:
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from testing early')

# Saving the loss and accuracy numpy arrays:
np.save('loss_acc/test_loss_{}_{}_{}'.format(
    str(lr), setting, seed), test_loss)
np.save('loss_acc/li_test_acc_{}_{}_{}'.format(
    str(lr), setting, seed), listener_test_acc)
np.save('loss_acc/sp_test_acc_{}_{}_{}'.format(
    str(lr), setting, seed), speaker_test_acc)
