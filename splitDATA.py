import os
import random
import shutil
from itertools import islice
from sympy import sequence


#######################################

outputFolderPath = "Dataset/SplitData"
inputFolderPath = "Dataset/All"
splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ["Fake", "Real"]

#####################################


try:
    shutil.rmtree(outputFolderPath)
    print("Removed Directory")
except OSError as e:
    os.mkdir(outputFolderPath)


# ----------------- Directories To Create ----------------------

os.makedirs(f"{outputFolderPath}/train/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels", exist_ok=True)


# -------------------------- Getting Names ------------------------------

listNames = os.listdir(inputFolderPath)
uniqueNames = []
for name in listNames:
    uniqueNames.append(name.split('.')[0])
uniqueNames = list(set(uniqueNames))


# -------------------------- Shuffle ---------------------------

random.shuffle(uniqueNames)


# ----------------------- Find Number of Images for Each Folder ------------------------

lenData = len(uniqueNames)
lenTrain = int(lenData * splitRatio["train"])
lenVal = int(lenData * splitRatio["val"])
lenTest = int(lenData * splitRatio["test"])


# ----------------------- Remaining Images in Training ----------------------

if lenData != lenTrain+lenVal+lenTest:
    remaining = lenData - (lenTrain+lenVal+lenTest)
    lenTrain += remaining


# ----------------------- Split the Data ---------------------------------

lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]
print(f'Total Images : {lenData} \nSplit : {len(Output[0])} {len(Output[1])} {len(Output[2])}')


# -------------------------- Copy the Data ----------------------------------

sequence = ['train', 'val', 'test']
for i, out in enumerate(Output):
    for fileName in out:
        shutil.copy(f"{inputFolderPath}/{fileName}.jpg", f"{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg")
        shutil.copy(f"{inputFolderPath}/{fileName}.txt", f"{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt")


print("Split Done....")

# --------------------- Creating data.yaml File ---------------------------

dataYaml = f'path: ../Data\n\
train: ../train/images\n\
val: ..val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'


f = open(f'{outputFolderPath}/data.yaml', "a")
f.write(dataYaml)
f.close()

print("Data.yaml Created")