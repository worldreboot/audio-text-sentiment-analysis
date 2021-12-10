import numpy as np
from mmsdk import mmdatasdk

def load_data():
    mydictLabels = {
        'myfeaturesLabels': '../../cmumosi/CMU_MOSI_Opinion_Labels.csd'}
    # cmumosi_highlevel=mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel['glove_vectors'],'cmumosi/')
    # mydictText = {'myfeaturesText':'cmumosi/CMU_MOSI_TimestampedWordVectors.csd'}
    mydatasetLabels = mmdatasdk.mmdataset(mydictLabels)
    # mydatasetText = mmdatasdk.mmdataset(mydictText)
    # print(mydataset.computational_sequences['myfeatures'].data)

    # Get text with labels
    totalSegments = 0
    for key in mydatasetLabels.computational_sequences[
        'myfeaturesLabels'].data.keys():
        totalSegments += len(
            mydatasetLabels.computational_sequences['myfeaturesLabels'].data[
                key][
                'features'])

    textInput = np.zeros(totalSegments, dtype=object)
    labelInput = np.zeros(totalSegments)
    segmentCounter = 0
    for key in mydatasetLabels.computational_sequences[
        'myfeaturesLabels'].data.keys():
        textPath = '../../raw/Raw/Transcript/Segmented/%s.annotprocessed' % (
            key)
        with open(textPath) as file:  # Use file to refer to the file object
            text = file.read()
            text = text.replace("_DELIM_", "")
            text = text.split("\n")
            for segment in range(len(mydatasetLabels.computational_sequences[
                                         'myfeaturesLabels'].data[key][
                                         'features'])):
                labelInput[segmentCounter] = \
                    mydatasetLabels.computational_sequences[
                        'myfeaturesLabels'].data[
                        key]['features'][segment]
                text[segment] = ''.join(
                    [i for i in text[segment] if not i.isdigit()])
                textInput[segmentCounter] = text[segment]
                segmentCounter += 1


    newInput = [sentence.lower() for sentence in textInput]
    labels = [1 if labelInput[i] > 0 else 0 for i in range(len(labelInput))]

    return newInput, labels
