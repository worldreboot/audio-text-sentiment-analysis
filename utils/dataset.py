import os
import pandas as pd

def load_ravdess_dataset(path_to_ravdess):
    # CREATE DIRECTORY OF AUDIO FILES 
    audio = path_to_ravdess
    actor_folders = os.listdir(audio) #list files in audio directory
    actor_folders.sort()

    # CREATE FUNCTION TO EXTRACT EMOTION NUMBER, ACTOR AND GENDER LABEL
    emotion = []
    gender = []
    actor = []
    file_path = []
    for i in actor_folders:
        filename = os.listdir(audio + i) #iterate over Actor folders
        for f in filename: # go through files in Actor folder
            part = f.split('.')[0].split('-')
            emotion.append(int(part[2]))
            actor.append(int(part[6]))
            bg = int(part[6])
            if bg % 2 == 0:
                bg = "female"
            else:
                bg = "male"
            gender.append(bg)
            file_path.append(audio + i + '/' + f)

    # PUT EXTRACTED LABELS WITH FILEPATH INTO DATAFRAME
    audio_df = pd.DataFrame(emotion)
    audio_df = audio_df.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'})
    audio_df = pd.concat([pd.DataFrame(gender),audio_df,pd.DataFrame(actor)],axis=1)
    audio_df.columns = ['gender','emotion','actor']
    audio_df = pd.concat([audio_df,pd.DataFrame(file_path, columns = ['path'])],axis=1)

    # EXPORT TO CSV
    audio_df.to_csv('parsed/ravdess.csv')