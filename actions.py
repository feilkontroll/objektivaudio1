# import the necessary packages
import numpy as np
import cv2
from pyimagesearch.coverdescriptor import CoverDescriptor
from pyimagesearch.covermatcher import CoverMatcher

class Actions:


	def __init__(self, useSIFT = False, useHamming = True, ratio = 0.7, minMatches = 40):
		# store whether or not SIFT should be used as the feature
		# detector and extractor
		self.useSIFT = useSIFT
		self.useHamming = useHamming
		self.ratio = ratio
		self.minMatches = minMatches
		# if SIFT is to be used, then update the parameters
		if useSIFT:
			self.minMatches = 50

		self.cd = CoverDescriptor(useSIFT = useSIFT)
		self.cv = CoverMatcher(self.cd, ratio = ratio, minMatches = minMatches, useHamming = useHamming)


    def store(self, image):
        return 0


    def search(self, queryImage):
        gray = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)
        (queryKps, queryDescs) = self.cd.describe(gray)
        results = self.cv.search(queryKps, queryDescs)

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