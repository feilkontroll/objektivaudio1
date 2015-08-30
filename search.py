# USAGE
# python search.py --db books.csv --covers covers --query queries/query01.png

# import the necessary packages
from __future__ import print_function
from pyimagesearch.coverdescriptor import CoverDescriptor
from pyimagesearch.covermatcher import CoverMatcher
import argparse
import glob
import csv
import cv2


# initialize the default parameters using BRISK is being used
useSIFT = False
useHamming = True
ratio = 0.7
minMatches = 40

# if SIFT is to be used, then update the parameters
if useSIFT:
	minMatches = 50

# initialize the cover descriptor and cover matcher
cd = CoverDescriptor(useSIFT = useSIFT)
cv = CoverMatcher(cd, ratio = ratio, minMatches = minMatches, useHamming = useHamming)

# load the query image, convert it to grayscale, and extract
# keypoints and descriptors
queryImage = cv2.imread("queries/query02.png")
gray = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)
(queryKps, queryDescs) = cd.describe(gray)

# try to match the book cover to a known database of images
results = cv.search(queryKps, queryDescs)

# show the query cover
cv2.imshow("Query", queryImage)

# check to see if no results were found
if len(results) == 0:
	print("I could not find a match for that cover!")
	cv2.waitKey(0)

# otherwise, matches were found
else:
	# loop over the results
	for (i, (score, coverPath)) in enumerate(results):
		# grab the book information

		# load the result image and show it
		result = cv2.imread(coverPath)
		cv2.imshow("Result", result)
		cv2.waitKey(0)