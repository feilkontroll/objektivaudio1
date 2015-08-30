# import the necessary packages
import numpy as np
import cv2
import sqlite3
import io


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


class CoverMatcher:
    def __init__(self, descriptor, ratio = 0.7, minMatches = 40, useHamming = True):
        # store the descriptor, book cover paths, ratio and minimum
        # number of matches for the homography calculation, then
        # initialize the distance metric to be used when computing
        # the distance between features
        self.descriptor = descriptor
        self.ratio = ratio
        self.minMatches = minMatches
        self.distanceMethod = "BruteForce"

        # if the Hamming distance should be used, then update the
        # distance method
        if useHamming:
            self.distanceMethod += "-Hamming"


    def search(self, queryKps, queryDescs):
        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("array", convert_array)

        # initialize the dictionary of results
        results = {}
        conn = sqlite3.connect('db', detect_types=sqlite3.PARSE_DECLTYPES)
        c = conn.cursor()
        res = c.execute("SELECT * FROM images").fetchall()
        c.fetchall()


        # loop over the images stored in database
        for image in res:
            # load the query image, convert it to grayscale, and
            # extract keypoints and descriptors

            #cover = cv2.imread(image[0])
            #gray = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
            #(kps, descs) = self.descriptor.describe(gray)

            # determine the number of matched, inlier keypoints,
            # then update the results

            kps = image[3]
            descs = image[4]

            score = self.match(queryKps, queryDescs, kps, descs)
            results[image[0]] = score
            print(image[0] + ": " + str(score))

        # if matches were found, sort them
        if len(results) > 0:
            results = sorted([(v, k) for (k, v) in results.items() if v > 0],
                reverse = True)

        # return the results
        return results


    def match(self, kpsA, featuresA, kpsB, featuresB):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create(self.distanceMethod)
        rawMatches = matcher.knnMatch(featuresB, featuresA, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other
            if len(m) == 2 and m[0].distance < m[1].distance * self.ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # check to see if there are enough matches to process
        if len(matches) > self.minMatches:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (i, _) in matches])
            ptsB = np.float32([kpsB[j] for (_, j) in matches])

            # compute the homography between the two sets of points
            # and compute the ratio of matched points
            (_, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)

            # return the ratio of the number of matched keypoints
            # to the total number of keypoints
            return float(status.sum()) / status.size

        # no matches were found
        return -1.0