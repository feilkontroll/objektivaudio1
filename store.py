
# USAGE
# python search.py --db books.csv --covers covers --query queries/query01.png

# import the necessary packages
from __future__ import print_function
from pyimagesearch.coverdescriptor import CoverDescriptor
from pyimagesearch.covermatcher import CoverMatcher
import argparse
import glob
import numpy
import csv
import cv2
import sqlite3
import io

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    numpy.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return numpy.load(out)

# Converts np.array to TEXT when inserting
sqlite3.register_adapter(numpy.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

ap = argparse.ArgumentParser()
args = vars(ap.parse_args())

useSIFT = False
useHamming = True

# initialize the cover descriptor and cover matcher
cd = CoverDescriptor(useSIFT = useSIFT)
conn = sqlite3.connect('db', detect_types=sqlite3.PARSE_DECLTYPES)
cur = conn.cursor()

c = conn.cursor()


for imgPath in glob.glob("newimages/*.png"):

    # load the query image, convert it to grayscale, and
    # extract keypoints and descriptors
    cover = cv2.imread(imgPath)
    gray = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
    (kps, descs) = CoverDescriptor.describe(cd, gray)

    c.execute('INSERT INTO images(filename,keypoints,descriptors) VALUES(?,?,?)',
             (imgPath, kps, descs))

#    print(kps)
#    print(descs)
conn.commit()
conn.close()






