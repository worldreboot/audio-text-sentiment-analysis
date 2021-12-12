import nlpaug.augmenter.word as naw
import nlpaug
from nlpaug.util import Action


def back_translate(data, to_lang):
    back_translation_aug = None
    if to_lang == 'de':
        back_translation_aug = naw.BackTranslationAug(
            from_model_name='facebook/wmt19-en-' + to_lang,
            to_model_name='facebook/wmt19-' + to_lang + '-en'
        )
    else:
        back_translation_aug = naw.BackTranslationAug(
            from_model_name='Helsinki-NLP/opus-mt-en-' + to_lang,
            to_model_name='Helsinki-NLP/opus-mt-' + to_lang + '-en'
        )
    #translator = Translator()
    translated = []
    for sentence in data:
        # t = translator.translate(sentence, src='en', dest=to_lang)
        # sentence = translator.translate(t.text, src=to_lang, dest='en')
        translated.append(back_translation_aug.augment(sentence))
        print('translated')
    return translated


def embedding_augmentor(data, action):
    aug = naw.WordEmbsAug(model_type='glove', model_path='../../glove.6B.100d.txt' , action=action)
    augmented = []
    for sentence in data:
        augmented.append(aug.augment(sentence))
        print('auged')
    return augmented


def wordnet_syn(data):
    aug = naw.SynonymAug(aug_src='wordnet')
    augmented = []
    for sentence in data:
        augmented.append(aug.augment(sentence))
        print('syn coverted')
    return augmented


def ppdb_syn(data):
    aug = naw.SynonymAug(aug_src='ppdb', model_path='../ppdb-2.0-tldr')
    augmented = []
    for sentence in data:
        augmented.append(aug.augment(sentence))
        print('syn coverted')
    return augmented

def context_sub(data, action):
    aug = naw.ContextualWordEmbsAug(
        model_path='bert-base-uncased', action=action)
    augmented = []
    for sentence in data:
        text = aug.augment(sentence)
        try:
            text.encode('ascii')
            augmented.append(text)
        except UnicodeError:
            augmented.append(sentence)


        print('context subbed')
    return augmented


def swap(data, max_swap):
    aug = naw.RandomWordAug(action="swap", aug_max=max_swap)
    augmented = []
    for sentence in data:
        augmented.append(aug.augment(sentence))
        print('swapped')
    return augmented


def delete(data, max_del):
    aug = naw.RandomWordAug(aug_max=max_del)
    augmented = []
    for sentence in data:
        augmented.append(aug.augment(sentence))
        print('deleted')
    return augmented

def proper_simple_augment(data: list, labels: list):
    """
    same as simple_augment but I remember how lists work 😃
    :param data:
    :param lables:
    :return:
    """
    return_data = []
    return_labels = []

    return_data.extend(back_translate(data, 'de'))  # german
    print('done')
    return_labels.extend(labels)
    return_data.extend(back_translate(data, 'fr'))  # french
    print('done')
    return_labels.extend(labels)

    return_data.extend(embedding_augmentor(data, 'insert'))
    print('done')
    return_labels.extend(labels)
    return_data.extend(embedding_augmentor(data, 'substitute'))
    print('done')
    return_labels.extend(labels)


def simple_augment(data: list, labels: list):
    temp = data
    temp_labels = labels

    data.extend(back_translate(data, 'de'))  # german
    print('done')
    labels.extend(temp_labels)
    data.extend(back_translate(temp, 'fr'))  # french
    print('done')
    labels.extend(temp_labels)

    data.extend(embedding_augmentor(temp, 'insert'))
    print('done')
    labels.extend(temp_labels)
    data.extend(embedding_augmentor(temp, 'substitute'))
    print('done')
    labels.extend(temp_labels)

    return data, labels

def simple_with_swap(data: list, labels: list):
    temp = data
    temp_labels = labels

    #data, labels = simple_augment(data, labels)

    data.extend(swap(data, max_swap=2))
    print('done')
    labels.extend(temp_labels)
    data.extend(delete(temp, max_del=1))
    print('done')
    labels.extend(temp_labels)
    data.extend(swap(temp, max_swap=4))
    print('done')
    labels.extend(temp_labels)
    data.extend(delete(temp, max_del=2))
    print('done')
    labels.extend(temp_labels)

    return data, labels

def simple_with_context_sub(data, labels):
    temp = data
    temp_labels = labels

    #data, labels = simple_augment(data, labels)
    data.extend(context_sub(temp, 'substitute'))
    print('done')
    labels.extend(temp_labels)
    data.extend(context_sub(temp, 'insert'))
    print('done')
    labels.extend(temp_labels)

    return data, labels


def nested_augment(data: list, labels: list):
    temp = data
    temp_labels = labels
    data.extend(context_sub(temp, 'substitute'))
    print('done')
    labels.extend(temp_labels)
    data.extend(context_sub(temp, 'insert'))
    print('done')
    labels.extend(temp_labels)
    temp2 = data
    data.extend(back_translate(data, 'de'))  # german
    print('done')
    labels.extend(temp_labels)
    data.extend(back_translate(temp2, 'fr'))  # french
    print('done')
    labels.extend(temp_labels)
    temp3 = data
    data.extend(ppdb_syn(data))
    print('done')
    labels.extend(temp_labels)
    data.extend(embedding_augmentor(temp, 'insert'))
    print('done')
    labels.extend(temp_labels)
    data.extend(embedding_augmentor(temp, 'substitute'))
    print('done')
    labels.extend(temp_labels)
    temp4 = data
    data.extend(swap(data, max_swap=2))
    print('done')
    labels.extend(temp_labels)
    data.extend(delete(temp4, max_del=1))
    print('done')
    labels.extend(temp_labels)
    data.extend(swap(temp4, max_swap=4))
    print('done')
    labels.extend(temp_labels)
    data.extend(delete(temp4, max_del=2))
    print('done')
    labels.extend(temp_labels)
    return data, labels


def remove_duplicates(data, labels):
    """
    slow but necessary
    :param data:
    :param labels:
    :return: nothing
    """
    new_labels = []
    i = 0
    if len(data) != len(labels):
        #print(len(data), len(labels))
        while len(data) > len(new_labels):
           # print(len(data), len(new_labels))
            new_labels.extend(labels)
        labels = new_labels[:len(data)]

    assert len(data) == len(labels)
    while i + 1 < len(data):
        item = data[i]
        if item in data[i + 1:]:
            del(data[i])
            del(labels[i])
            i -= 1
        i += 1

    return data, labels

def clean_test_train_data(data: list, test: list, labels: list):
    for item in test:
        i = 0
        while i < len(data):
            if item == data[i]:
                del(data[i])
                del(labels[i])
                i -= 1
            i += 1

