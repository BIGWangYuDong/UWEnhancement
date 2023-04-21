import os
import random
import glob
from collections.abc import Iterable
import argparse
import shutil
from pathlib import Path


def findCaseInsensitiveList(dir: str, ext) -> list[str]:
    if isinstance(ext, str):
        ext = [].append(ext)
    elif isinstance(ext, Iterable):
        print(ext)
        if not all(isinstance(item, str) for item in ext):
            raise Exception(f"Wrong data type for defining extensions.")
    else:
        raise Exception(f"Wrong data type ({ext.__class__}) for defining extensions.")
    ext.extend([e.upper() for e in ext])
    fileList = []
    for e in ext:
        fileList.extend(list(glob.glob(os.path.join(dir, f"*{e}"))))
    return sorted(fileList)


def findCaseInsensitive(dir: str, ext):
    if isinstance(ext, str):
        ext = [].append(ext)
    elif isinstance(ext, Iterable):
        if not all(isinstance(item, str) for item in ext):
            raise Exception(f"Wrong data type for defining extensions.")
    else:
        raise Exception(f"Wrong data type ({ext.__class__}) for defining extensions.")
    #ext.extend([e.upper() for e in ext])
    yield from glob.glob(os.path.join(dir, f"*"))


def checkFilenames(l1: Iterable, l2: Iterable):
    for a,b in zip(l1,l2):
        if os.path.splitext(os.path.split(a)[-1])[0] != os.path.splitext(os.path.split(b)[-1])[0]:
            raise Exception(f"Filenames are not identical!\n\
                            a : {a}\n\
                            b : {b}")


def writeFiles(path: str, l: Iterable):
    with open(path, 'w') as f:
        for item in l:
            f.write(f"{item}\n")
        f.close()


def prepareDataset(args: argparse.Namespace):
    dataFiles = sorted(findCaseInsensitive(args.dataDir, args.ext))
    gtFiles = sorted(findCaseInsensitive(args.gtDir, args.ext))
    checkFilenames(dataFiles, gtFiles)

    trainIdx = random.sample(range(len(dataFiles)), k=int(len(dataFiles)*0.8))

    train_data = Path(os.path.join(args.outDir, "Train/train"))
    train_gt = Path(os.path.join(args.outDir, "Train/gt"))
    test_data = Path(os.path.join(args.outDir, "Test/test"))
    test_gt = Path(os.path.join(args.outDir, "Test/gt"))
    
    for item in [train_data, train_gt, test_data, test_gt]:
        item.mkdir(parents=True, exist_ok=True)

    trainList = []
    testList = []

    for idx, item in enumerate(zip(dataFiles, gtFiles)):
        data, gt = item
        if idx in trainIdx:
            trainList.append(os.path.split(data)[-1])
            shutil.copy(data, train_data)
            shutil.copy(gt, train_gt)
        else:
            testList.append(os.path.split(data)[-1])
            shutil.copy(data, test_data)
            shutil.copy(gt, test_gt)

    writeFiles(os.path.join(args.outDir, "Train/train.txt"), trainList)
    writeFiles(os.path.join(args.outDir, "Test/test.txt"), testList)


    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataDir", type=str, default="../../../Data/EUVP/Paired/underwater_dark/trainA")
    parser.add_argument("--gtDir", default="../../../Data/EUVP/Paired/underwater_dark/trainB")
    parser.add_argument("--ext", default=[".jpg", ".jpeg", ".png"])
    parser.add_argument("--outDir", default="./DATA")

    args = parser.parse_args()

    prepareDataset(args)