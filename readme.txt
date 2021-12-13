# Prerequisites
You need to download the CMU MOSI dataset in order to run our models. 
You can download it here: http://immortal.multicomp.cs.cmu.edu/raw_datasets/ (CMU_MOSI.zip).
Extract the contents in datasets/ in a folder called CMU_MOSI

# Audio-based part of the project
In order to train and run our CNN model for audio analysis, you can run

python testing.py

This loads and preprocesses all of the data from the CMU MOSI dataset and outputs plots of the
accuracy and loss.

# Text-based part of the project
You can run the text-based models mentioned in our report by doing 

python "binary model.py"
python "closer categories.py"
python "divide and conquer model.py"

which are located under "testing models"/text