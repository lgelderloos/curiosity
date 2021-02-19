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
import toolbox


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
        eval_loss[batch] = loss.item() * batch_size[batch] # weighting for different batch sizes

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
        vggs = torch.load("data/ha_bbox_vggs/" + img + ".pt").to(device)
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


def curiosity(i, o, condition="normal"):
    # Calculates the curiosity values
    # i and o are torch tensors of shape [batch, ntokens]
    subjective_novelty = abs(i - o)
    plasticity = abs(o - o ** 2)
    try:
        if condition == "plasticity":
            return torch.mean(plasticity, dim=1)
        elif condition == "sn":
            return torch.mean(subjective_novelty, dim=1)
        else:
            return torch.mean((subjective_novelty * plasticity), dim=1)
    except:
        print("problem with CURIOSITY function!")


def load_img(dict_words_boxes, ha_vggs_indices, img):
    vggs = torch.load("data/ha_bbox_vggs/" + img + ".pt").to(device)  # Edit path
    # dict met obj ids als keys en een dictionary met words : '', bboxes :
    n = 0
    bbox_indices = []
    words = []
    for obj in dict_words_boxes[img]:  # For every object in this image
        words.append(get_word_ix(word_to_ix, dict_words_boxes[img][obj]["word"]))
        bbox_indices.append(ha_vggs_indices[img][obj][0])
    visual_input = vggs[bbox_indices, :]
    language_input = torch.tensor(words, dtype=torch.long, device=device)
    return language_input, visual_input


def curious_look_at_img(dict_words_boxes, ha_vggs_indices, img, _setting):
    language_input, scene = load_img(dict_words_boxes, ha_vggs_indices, img)
    # repeat scene n_objects times as input to listener
    visual_input = scene.expand(scene.size()[0], scene.size()[0], scene.size()[1])
    curiosity_targets = torch.eye(visual_input.size()[0], dtype=torch.float, device=device)
    # targets is simply 0, 1, ...., n because they are in order of appearance
    targets = torch.tensor([i for i in range(len(language_input))], dtype=torch.long, device=device)
    # word guesses by child - use as attention over word embeddings
    word_guesses = speaker(visual_input, curiosity_targets, apply_softmax=False)
    # only keep most likely words
    words = torch.argmax(word_guesses, dim=1)
    # give these as input to listener
    object_guesses = listener(words, visual_input)
    curiosity_values = curiosity(curiosity_targets, object_guesses, _setting)
    max_curious = torch.argmax(curiosity_values)
    return language_input[max_curious], scene, targets[max_curious]


def random_look_at_img(dict_words_boxes, ha_vggs_indices, img):
    language_input, scene = load_img(dict_words_boxes, ha_vggs_indices, img)
    # targets is simply 0, 1, ...., n because they are in order of appearance
    targets = torch.tensor([i for i in range(len(language_input))], dtype=torch.long, device=device)
    i = np.random.randint(len(targets))
    return language_input[i], scene, targets[i]


def load_select_obj(dict_words_boxes, ha_vggs_indices, img, _setting):
    if setting == "random":
        return random_look_at_img(dict_words_boxes, ha_vggs_indices, img)
    elif (setting == "curious") | (setting == "plasticity") | (setting == "sn"):
        return curious_look_at_img(dict_words_boxes, ha_vggs_indices, img, _setting)


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


###############################################################################
# Training function
###############################################################################

def train():
    listener.train()
    speaker.train()
    start_time = time.time()
    n_batches = len(batches)
    train_loss = np.empty(n_batches)
    li_train_accuracy = np.empty(n_batches)
    sp_train_accuracy = np.empty(n_batches)
    batch_size = np.empty(n_batches)

    batch = 0

    # batches shuffled during training
    while batch < n_batches:
        language_batch = []  # All word indices in the batch
        visual_batch = []  # All vgg vector indices in the batch
        target_batch = []  # All target word indices in the batch

        for img in batches[batch]:
            language_input, visual_input, target = load_select_obj(dict_words_boxes, ha_vggs_indices, img,
                                                                   setting)
            language_batch.append(language_input)
            visual_batch.append(visual_input)
            target_batch.append(target)
        language_input = torch.stack(language_batch)
        visual_input = torch.stack(visual_batch)
        targets = torch.stack(target_batch)

        optimizer.zero_grad()
        obj_guesses = listener(language_input, visual_input)

        # Saves the batch length for weighted mean accuracy:
        batch_size[batch] = len(batches[batch])
        # calculate listener accuracy
        li_train_accuracy[batch], _ = calc_accuracy(obj_guesses, targets, average=False)

        word_guesses = speaker(visual_input, obj_guesses)

        loss = criterion(word_guesses, language_input)
        loss.backward()
        optimizer.step()

        # Loss/accuracy times batch size for weighted average over epoch:
        train_loss[batch] = loss.item() * batch_size[batch]
        sp_train_accuracy[batch], _ = calc_accuracy(word_guesses, language_input, average=False)

        batch += 1
        if batch % printerval == 0:
            print(
                '| epoch {:2d} | batch {:3d}/{:3d} | t {:6.2f} | L {:6.4f} | l.A {:5.4f} | s.A {:6.4f} |'.format(
                    epoch, batch, n_batches, (time.time() - start_time),
                    np.sum(train_loss[batch - printerval:batch]) / np.sum(batch_size[batch - printerval:batch]),
                    np.sum(li_train_accuracy[batch - printerval:batch]) / np.sum(batch_size[batch - printerval:batch]),
                    np.sum(sp_train_accuracy[batch - printerval:batch]) / np.sum(batch_size[batch - printerval:batch])))

    avg_train_loss = np.sum(train_loss) / np.sum(batch_size)
    avg_li_train_acc = np.sum(li_train_accuracy) / np.sum(batch_size)
    avg_sp_train_acc = np.sum(sp_train_accuracy) / np.sum(batch_size)

    print('-' * 89)
    print("overall performance on training set:")
    print('| Loss {:8.4f} | Listener acc. {:8.4f} | Speaker acc. {:8.4f}'.format(avg_train_loss, avg_li_train_acc, avg_sp_train_acc))
    print('-' * 89)
    return avg_train_loss, avg_li_train_acc, avg_sp_train_acc


#################################################################################
# Interpret command line arguments
#################################################################################

print("\nSys.argv:", sys.argv)
batchsize = int(sys.argv[1])
lr = float(sys.argv[2])
setting = sys.argv[3]

# curiosity_condition = sys.argv[5] todo: to remove this comment

#################################################################################
# DATA PREPROCESSING
#################################################################################


def make_vocab(data_dict):
    '''
    Returns a vocabulary and a vocabulary with frequencies
    '''

    freq = {}
    for file in data_dict.keys():
        for obj in data_dict[file].keys():
            word = data_dict[file][obj]["word"]
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1

    vocab = list(freq.keys())

    return vocab, freq

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
# ACTUAL CODE
###############################################################################

# Object vgg indices (object information)
with open("data/ha_vgg_indices.json", "rb") as input_file:
    ha_vggs_indices = json.load(input_file)

# Regular data (dictionary with all images and their object ids, corresponding words)
with open("data/dict_words_boxes.json", "rb") as input_file:
    dict_words_boxes = json.load(input_file)

# Train split, image ids
with open("data/train_data.txt", "rb") as fp:
    train_data = pickle.load(fp)

# Validation split, image ids
with open("data/validation_data.txt", "rb") as fp:
    validation_data = pickle.load(fp)

# Test split, image ids
with open("data/test_data.txt", "rb") as fp:
    test_data = pickle.load(fp)

###############################################################################
# PREPROCESSING
###############################################################################

# Makes a vocabulary of the entire set of objects:
vocab, freq = make_vocab(dict_words_boxes)

# Gives an index number to every word in the vocabulary:
word_to_ix = toolbox.make_index_table(vocab)
path_word_to_ix = 'word_to_ix/'
toolbox.mkdir(path_word_to_ix)
with open(path_word_to_ix+'word_to_ix_{}_{}_{}.json'.format(setting, seed, str(lr)), 'w') as wtx:
    json.dump(word_to_ix, wtx)

# Returns a dictionary with the number of objects per image:
no_objs = no_of_objs(dict_words_boxes, train_data)

# Returns a list of batch-size batches:
# A batch contains images with the same no. of objs
batches = dict_to_batches(no_objs, batchsize)

no_objs_val = no_of_objs(dict_words_boxes, validation_data)
val_batchlist = dict_to_batches(no_objs_val, batchsize)

ntokens = len(word_to_ix.keys())
print("ntokens:", ntokens)

###############################################################################
# SPECIFY MODEL
###############################################################################
object_size = 4096  # Length vgg vector
att_hidden_size = 256  # Number of hidden nodes
wordemb_size = 256  # Length word embedding
nonlin = "sigmoid"
print("hidden layer size:", att_hidden_size)

epochs = 40

device = torch.device('cuda')  # Device = GPU

# makes folder to store models
toolbox.mkdir('models/{}'.format(setting))

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
# for train, validation, and test splits

train_loss = np.empty(epochs)
listener_train_acc = np.empty(epochs)
speaker_train_acc = np.empty(epochs)
val_loss = np.empty(epochs)
listener_val_acc = np.empty(epochs)
speaker_val_acc = np.empty(epochs)

# first run evaluate to get baseline (scores are printed to user but not stored)
val_loss_0, li_val_acc, sp_val_acc = evaluate(0)

# save model at epoch 0
torch.save(listener.state_dict(),
          'models/{}/liModel_{}_{}_{}_ep{}.pth'.format(setting, setting, str(lr), seed, 0))
torch.save(speaker.state_dict(),
          'models/{}/spModel_{}_{}_{}_ep{}.pth'.format(setting, setting, str(lr), seed, 0))

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()  # todo: should we do anything with this time?
        epoch_train_loss, epoch_listener_train_acc, epoch_speaker_train_acc = train()
        train_loss[epoch - 1] = epoch_train_loss
        listener_train_acc[epoch - 1] = epoch_listener_train_acc
        speaker_train_acc[epoch - 1] = epoch_speaker_train_acc
        torch.save(listener.state_dict(),
                   ('models/{}/liModel_{}_{}_{}_ep{}.pth'.format(setting, setting, str(lr), seed, epoch)))
        torch.save(speaker.state_dict(),
                   ('models/{}/spModel_{}_{}_{}_ep{}.pth'.format(setting, setting, str(lr), seed, epoch)))
        epoch_val_loss, epoch_listener_val_acc, epoch_speaker_val_acc = evaluate(epoch)
        val_loss[epoch - 1] = epoch_val_loss
        listener_val_acc[epoch - 1] = epoch_listener_val_acc
        speaker_val_acc[epoch - 1] = epoch_speaker_val_acc

# To enable to hit Ctrl + C and break out of training:
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

###############################################################################
# Saving scores
###############################################################################

# Saving the loss and accuracy numpy arrays
np.save('loss_acc/train_loss_{}_{}_{}'.format(
    str(lr), setting, seed), train_loss)
np.save('loss_acc/li_train_acc_{}_{}_{}'.format(
    str(lr), setting, seed), listener_train_acc)
np.save('loss_acc/sp_train_acc_{}_{}_{}'.format(
    str(lr), setting, seed), speaker_train_acc)
np.save('loss_acc/val_loss_{}_{}_{}'.format(
    str(lr), setting, seed), val_loss)
np.save('loss_acc/li_val_acc_{}_{}_{}'.format(
    str(lr), setting, seed), listener_val_acc)
np.save('loss_acc/sp_val_acc_{}_{}_{}'.format(
    str(lr), setting, seed), speaker_val_acc)
