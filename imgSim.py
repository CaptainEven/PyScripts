#!/usr/bin/env python

###### Import Modules
import sys
import os

# import json
# import pprint
# import cPickle as pickle

###### Document Decription
'''  '''

###### Version and Date
prog_version = '0.1.1'
prog_date = '2017-10-26'


######## Global Variable


#######################################################################
############################  BEGIN Class  ############################
#######################################################################
class CDF(object):
    """This class create a instance to calculate the Cumulative Distribution Function
       of Normal Standard (Mean=0, SD=1).
            Distribution = exp{- (x-mn)^2 / 2 sigma^2} / [ sigma * SQRT(2 pi) ]
       Math module was used to create CDF for any accuracy. The distribution will be
       saved into a dict for quick accessment.
            Initialize(floor): A int of precision. 1 means 0.1, 2 means 0.01.
            ip(x): Return the interpolation of the cumulative value for position x.
            toList(): Return a list of distribution dict.
       This might be useful in normalizing the DNB's to have unity peak intensity.
    """

    def __init__(self, floor):
        # super(CDF, self).__init__()
        self.floor = floor
        self.cdfDist = self.createCDF(self.floor)

    def createCDF(self, floor=2, start=-8., end=8.):
        from math import erf, sqrt
        cumulateDict = {}
        step = 10 ** (-floor)
        while start <= end:
            value = round(start, floor)
            cumulateDict[value] = (1.0 + erf(value / sqrt(2.0))) / 2.0
            start = value + step
        return cumulateDict

    def toList(self):
        newList = []
        for i in sorted(self.cdfDist.keys()):
            newList.append([i, self.cdfDist[i]])
        return newList

    def ip(self, x):
        return self.cdfDist[round(x, self.floor)]

    def __str__(self):
        lines = ["%f\t%f" % tuple(x) for x in self.toList()]
        return "\n".join(lines)


class AirySpot(object):
    """This class create a instance which can generate intensity distribution
       as a single gaussian spot (Airy Spot) in given 2D array.
       The gaussian spot is scaled to best match the central blob of an airy disk.
       The gaussian approximation fits the central core of Airy spot pretty well,
       but does have higher energy out at the edges.  This difference should be
       negligable for small spots.
       r,c will be the spot center (which need not be integers), 0,0 refers
       to the upper left corner of the image.
       *** NOTE ***
       Here we use the row and column instead X,Y to adapt 2D list (numpy.narry) order.
       The r grows from top to bottom.
       The c grows from left to right.
       Integer values of r,c put the spot in the center of the pixel.

       Initialization(floor=3): The precision of CDF function.
       createSpot(img, r, c, radius, energy): Create a airy spot at (r,c) of the img buffer.
       img: 2D array like object, can be list or numpy.narry. [2D list]
       r: Row coordinate of the center of spot. [float]
       c: Column coordinate of the center of spot. [float]
       airyRadius: Airy disk radius of the spot by pixel. [int]
       energyInit: Total energy deposited in the spot. [int]
    """

    def __init__(self, floor=3):
        # super(PSF, self).__init__()
        import math
        self.exp = math.exp
        self.cdf = CDF(floor)

    def createSpot(self, img, r, c, airyRadius, energyInit):
        """
        # This empirically dervied kludge matches the gaussian spot
        # simulated profile to an airy disk with 1st zero radius of
        # airyRadius [pixel]. The radius of gaussian spot is gaussianR [pixel].
        :param img:
        :param r:
        :param c:
        :param airyRadius:
        :param energyInit:
        :return:
        """
        gaussianR = airyRadius * 2 / 3

        # Here's another emprical kludge found that adjusts for deposited energy.
        # Only important for big spots.
        spotEngery = energyInit * (1.1 - 0.13 * self.exp(-airyRadius / 7.0))

        # The peak of spot would be spotEngery/(1.73 * pow(gaussianR,2))
        # this amplitude calc is FYI, not used in spot drawing

        # Used in normalizing CDF argument
        normFactor = 0.525 * gaussianR

        # Normally there will be the threshold of 0.99 to estimate the how many engery
        # got deposited around the central spot.  The idea was to make sure 99% of
        # total energy really got deposited. However, we don't need to calculate this,
        # we can just calculate the appropriate ring number from the spot width
        # (84% of energy is in circle of radius 2).
        # Normally gaussianR will be close to 0.5 and we want to have rmax = 1 to do 3x3 block.
        # If gaussianR goes to near 1, then we want to be go out to 2 to go to 5x5. For big gaussianR it doesn't
        # matter if we lose some of the energy.
        rmax = int(round(gaussianR)) + 1

        # Central pixel has integer coords that are rounded input args.
        # rowCent,colCent will be the closest pixel of r,c
        rowCent = int(round(r))
        colCent = int(round(c))

        # deposite engery for each pixel, just ignore the pix out of range
        topPx = rowCent - rmax if rowCent - rmax >= 0 else 0
        botPx = rowCent + rmax if rowCent + rmax < len(img) else len(img) - 1
        leftPx = colCent - rmax if colCent - rmax >= 0 else 0
        rightPx = colCent + rmax if colCent + rmax < len(img[0]) else len(img[0]) - 1

        # interpolation function
        cdf = self.cdf.ip

        # accumalate real engery for all pixels
        totalEnergy = 0

        for rowPx in range(topPx, botPx + 1):
            for colPx in range(leftPx, rightPx + 1):
                # Now calculate spot energy of each pixel
                pixEnergy = int(spotEngery \
                                * (cdf((rowPx + 0.5 - r) / normFactor) - cdf((rowPx - 0.5 - r) / normFactor)) \
                                * (cdf((colPx + 0.5 - c) / normFactor) - cdf((colPx - 0.5 - c) / normFactor)))
                totalEnergy += pixEnergy
                img[rowPx][colPx] += pixEnergy
        # return the real sum of energy
        return totalEnergy


class SimulateImg(object):
    """This class create a instance to generate zebra image."""

    def __init__(self, cfg_file_path):
        # super(SimulateImg, self).__init__()
        ## import lib
        import numpy as np
        from origin.tifffile import imsave
        import datetime

        self.np = np
        self.imsave = imsave
        self.dt = datetime
        # initialize instance to create airy spot
        self.ap = AirySpot(3)

        # load config from JSON file
        self.cfg = self.loadConfig(cfg_file_path)

        # load optical parameters
        self.calOptics()

        # init proc parameters
        self.cfg["proc"]["totalSpot"] = 0
        self.cfg["proc"]["specSpot"] = []

        # if a spot information from *info.tsv file
        if "specificSpot" in self.cfg["layout"]:
            self.loadInfoTsv(self.cfg["layout"]["specificSpot"])

        # load block layout from TSV file
        if "template" in self.cfg["layout"]:
            self.loadTemplate(self.cfg["layout"]["template"])

    def _logger(self, msg):
        timeNow = self.dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("[%s] %s" % (timeNow, msg), file=sys.stderr)
        sys.stdout.flush()

    def calOptics(self):
        optset = self.cfg["optics"]["setting"][self.cfg["optics"]["solution"]]
        self.cfg["proc"]["opt"] = optset

        # calculate pixel per DNB
        self.cfg["proc"]["dtX"] = self.cfg["layout"]["pitchX"] / optset["magnif"]
        self.cfg["proc"]["dtY"] = self.cfg["layout"]["pitchY"] / optset["magnif"]

        # calculate track width
        self.cfg["proc"]["trackWidth"] = self.cfg["layout"]["trackWidth"] / optset["magnif"]

        # reformat crosstalk
        baseList = ["A", "C", "G", "T"]
        self.cfg["proc"]["crosstalk"] = {}
        for c in optset["crosstalk"]:
            tmpDict = dict(zip(baseList, optset["crosstalk"][c]))
            # Crosstalk for unloaded spot should be zero
            tmpDict["U"] = 0.0
            self.cfg["proc"]["crosstalk"][c] = tmpDict

        # create a ref to the parameters of specific optical solution
        self.cfg["proc"]["radius"] = {}
        for channel in optset["channel"]:
            pixelSize = optset["magnif"]
            lam = optset["waveLength"][channel]
            # Airy spot radius
            self.cfg["proc"]["radius"][channel] = 1.22 * lam * optset["aberrFac"] / (2.0 * optset["NA"] * pixelSize)

        # calculate rotation factor
        # rotation is clockwise
        import math
        self.rotSin = math.sin(-self.cfg["img"]["rotation"])
        self.rotCos = math.cos(-self.cfg["img"]["rotation"])
        self.rotCntX = (self.rotCos - 1) * self.cfg["img"]["width"] / 2.0 + self.rotSin * self.cfg["img"][
            "height"] / 2.0
        self.rotCntY = -self.rotSin * self.cfg["img"]["width"] / 2.0 + (self.rotCos - 1) * self.cfg["img"][
            "height"] / 2.0
        return self.cfg["proc"]["radius"]

    def loadConfig(self, cfg_f_path):
        """
        :param cfg_f_path:
        :return:
        """
        import json

        cfg_f_path = os.path.abspath(cfg_f_path)
        if not os.path.isfile(cfg_f_path):
            print('[Err]: invalid config file path {:s}.'.format(os.path.abspath(cfg_f_path)))
            return None

        with open(cfg_f_path, 'r') as fh:
            jsCfg = json.load(fh)
            if 'proc' not in jsCfg:
                jsCfg["proc"] = {}

        return jsCfg

    def loadTemplate(self, tmplFile):
        """Load a tsv template with DNB layout
            A template file contain 5 columns,
            and all values shoule be integer:
              [0]     [1]     [2]     [3]     [4]
            blockId  fovCol  fovRow  dnbCol  dnbRow

            The line started with '#' will be ignored.
            fovCol, fovRow: start position of specific block, start from 1.
            dnbCol, dnbRow: DNB number in this block.
            All information will be saved to a dict: cfg.proc.blocks
            format like this:
            blocks: [[block(0,0), block(1,0), ...],
                     [block(0,1), block(1,1), ...],
                     ...
                    ]
            block(0,0): {"blockId": a,
                         "fovCol":  b,
                         "fovRow":  c,
                         "dnbCol":  d,
                         "dnbRow":  e}
        """
        tmpDict = {}
        totalSpot = 0
        keyList = ["blockId", "fovCol", "fovRow", "dnbCol", "dnbRow"]
        with open(tmplFile, 'r') as fh:
            for l in fh:
                if l.startswith('#'):
                    continue
                info = [int(x) for x in l.split()]
                # create Row dict
                if info[2] not in tmpDict:
                    tmpDict[info[2]] = {}
                tmpDict[info[2]][info[1]] = dict(zip(keyList, info))
                totalSpot += info[3] * info[4]

        # calculate absolute position
        blocks = []
        for r in sorted(tmpDict):
            blocks.append([])
            for c in sorted(tmpDict[r]):
                blocks[r - 1].append(tmpDict[r][c])

        self.cfg["proc"]["blocks"] = blocks
        self.cfg["proc"]["totalSpot"] = totalSpot
        return self.cfg["proc"]["blocks"]

    def loadInfoTsv(self, infoFile):
        """ Load coordinate and engery information from TSV file.
            the file format is same as the file created by function self.saveInfo()
            file format:
            blockId  base  dnbCol  dnbRow  channel_1_X channel_1_Y  channel_1_Energy [channel_2_X...]

            Column 4-6 will be treat as channel 1.
            Column 7-9 will be treat as channel 2.
            ...
        """
        self.cfg["proc"]["specSpot"] = []
        with open(infoFile, 'r') as fh:
            for l in fh:
                if l.startswith('#'):
                    continue
                # info = [int(x) for x in l.split()]
                self.cfg["proc"]["specSpot"].append(l.split())
        return self.cfg["proc"]["specSpot"]

    def absPos(self, x, y):
        """ Aapply the offset, rotation and distortion to
            the given coordinate.
        """
        # apply offset
        newX = x + self.cfg["img"]["offsetX"]
        newY = y + self.cfg["img"]["offsetY"]
        if self.rotSin:
            # rotX = self.rotCos*(newX - self.midX) - self.rotSin*(newY - self.midY) + newX
            # rotY = self.rotCos*(newY - self.midY) + self.rotSin*(newX - self.midX) + newY
            rotX = self.rotCos * newX + self.rotSin * newY - self.rotCntX
            rotY = -self.rotSin * newX + self.rotCos * newY - self.rotCntY
            return (rotX, rotY)
        return (newX, newY)

    def addNoise(self, img=None):
        noiseMean = self.cfg["img"]["noise"]
        # return raw img with no noise
        # calculate width and height of image
        if img:
            height, width = img.shape()
        else:
            height = self.cfg["img"]["height"]
            width = self.cfg["img"]["width"]

        # create image with noise by gaussian distribution
        if noiseMean > 0:
            noiseLay = self.np.random.normal(noiseMean, self.cfg["img"]["noiseSD"], (height, width))
            if img:
                img += noiseLay
            else:
                img = noiseLay
        else:
            # if no noise config, create a pure drak image
            if not img:
                img = self.np.zeros((height, width))
        return img

    def addBlocks(self, img, engy, base, record):
        # draw blocks
        blockInitX = 0
        blockInitY = 0
        baseSt = 0
        for rowBlocks in self.cfg["proc"]["blocks"]:
            for blk in rowBlocks:
                self._logger("Block %s with DNB %d" % (blk["blockId"], blk["dnbCol"] * blk["dnbRow"]))
                baseEd = baseSt + blk["dnbCol"] * blk["dnbRow"]
                # Add Spots
                self.addDNBs(img, blk, engy[baseSt:baseEd], base[baseSt:baseEd], record, blockInitX, blockInitY)
                blockInitX += blk["dnbCol"] * self.cfg["proc"]["dtX"] + self.cfg["proc"]["trackWidth"]
                baseSt = baseEd
            blockInitX = 0
            blockInitY += rowBlocks[0]["dnbRow"] * self.cfg["proc"]["dtY"] + self.cfg["proc"]["trackWidth"]
        return img

    def addDNBs(self, img, blk, engy, base, record, stX=0, stY=0):
        # inistialize energy of DNBs
        # now only allow gaussian distribution of intensity
        blockId = blk["blockId"]
        channel = record["channel"]
        crosstalk = self.cfg["proc"]["crosstalk"][channel]
        radius = self.cfg["proc"]["radius"][channel]
        dtX = self.cfg["proc"]["dtX"]
        dtY = self.cfg["proc"]["dtY"]
        dnbRecord = record["dnb"]

        initX = stX
        initY = stY
        ptr = 0
        for r in range(blk["dnbRow"]):
            for c in range(blk["dnbCol"]):
                curBase = base[ptr]
                curEngy = engy[ptr]

                # real energy = initialEnergy * crosstalk factor for this image
                curEngy = crosstalk[curBase] * curEngy

                # apply offset, rotation and distortion for current spot
                realX, realY = self.absPos(initX, initY)
                depositedEngy = self.ap.createSpot(img, realY, realX, radius, curEngy)
                dnbRecord.append([blockId, curBase, c, r, realX, realY, depositedEngy])
                ptr += 1
                initX += dtX
            initX = stX
            initY += dtY
        return img

    def addSpecSpot(self, imgBuffer, record):
        firstCol = self.cfg["layout"]["specificChannel"][record["channel"]]
        radius = self.cfg["proc"]["radius"][record["channel"]]
        for d in self.cfg["proc"]["specSpot"]:
            self.ap.createSpot(imgBuffer, float(d[firstCol + 1]), float(d[firstCol]), radius, float(d[firstCol + 2]))
        return len(self.cfg["proc"]["specSpot"])

    def createImg(self, engy, base, record):
        imgWidth = self.cfg["img"]["width"]
        imgHeight = self.cfg["img"]["height"]
        self._logger("Creating channel %s." % record["channel"])
        self._logger("Adding noise...")
        # initialize the image, dtype is float64
        imgBuffer = self.addNoise()

        # add blocks
        if self.cfg["proc"]["totalSpot"] > 0:
            self._logger("Adding Blocks...")
            self.addBlocks(imgBuffer, engy, base, record)

        # add specific spots
        if len(self.cfg["proc"]["specSpot"]) > 0:
            self._logger("Adding Specific Spots...")
            self.addSpecSpot(imgBuffer, record)

        # print(imgBuffer)
        return imgBuffer

    def saveImg(self, imgBuffer, fileName):
        """ Save the image buffer to a tif file and return the file name.
            Force to convert intensity from float32 to uint16.
        """
        # TODO: Since the DNB intensity and noise are generated by a random
        # function of normal distribution. There exists a very very small
        # probability that the number will be <0, or >65535.
        self.imsave(fileName, imgBuffer.astype('uint16'))

        return fileName

    def saveInfo(self, fileName, info):
        """ Save the Spot information to a tsv file
            A header start with '#' to declare metrics for each column
            File format:
            blockId  base  dnbCol  dnbRow  channel_1_X channel_1_Y  channel_1_Energy [channel_2_X...]
        """
        channels = [x["channel"] for x in info]
        headerList = ["%s_X\t%s_Y\t%s_Engy" % (x, x, x) for x in channels]
        # check channels
        if channels != self.cfg["proc"]["opt"]["channel"]:
            raise RuntimeError("Missing channels for image: %s\nExpect: %s\nGot: %s" % (
                fileName, str(self.cfg["proc"]["opt"]["channels"], channels)))
        # check dnb number
        if len(set([len(x["dnb"]) for x in info])) != 1:
            strList = ["%s:\t%d" % (x["channel"], len(x["dnb"])) for x in info]
            raise RuntimeError("Unequal spot number betweens channels.\n" + "\n".join(strList))

        with open(fileName, 'w') as fh:
            # dnb level information
            # format:
            # blockId  base  dnbCol  dnbRow  channel_1_X channel_1_Y  channel_1_Energy [channel_2_X...]

            # header
            fh.write("#blockId\tbase\tdnbCol\tdnbRow\t" + "\t".join(headerList) + "\n")
            # each Spot
            channelNum = len(info)
            for i, v in enumerate(info[0]["dnb"]):
                outStr = "%d\t%s\t%d\t%d\t%.2f\t%.2f\t%.2f" % tuple(v)
                if channelNum > 1:
                    outStr += "\t" + "\t".join(["%.2f\t%.2f\t%.2f" % tuple(j["dnb"][i][4:]) for j in info[1:]])
                fh.write(outStr + "\n")
        return fileName

    def simulateSinge(self, pathPrefix, saveInfo=True):
        info = []

        # initialize base of DNBs
        baseList = sorted(self.cfg["spot"]["bases"])
        probability = [self.cfg["spot"]["bases"][x] for x in baseList]
        spotBase = self.np.random.choice(baseList, self.cfg["proc"]["totalSpot"], p=probability)

        # inistialize energy of DNBs
        if self.cfg["spot"]["random"] == "gaussian":
            if self.cfg["spot"]["intVar"] == 0:
                # All energy should be constant if variant is 0.
                spotEnergy = self.np.empty(self.cfg["proc"]["totalSpot"])
                spotEnergy.fill(self.cfg["spot"]["meanInt"])
            else:
                # create a gaussian distribution of all spots
                spotEnergy = self.np.random.normal(self.cfg["spot"]["meanInt"], self.cfg["spot"]["intVar"],
                                                   self.cfg["proc"]["totalSpot"])
                spotEnergy[spotEnergy < 0] = 0
        else:
            raise KeyError("Not exist intensity distribution method: %s" % self.cfg["spot"]["dist"])

        if "name" not in self.cfg["proc"]["opt"]:
            self.cfg["proc"]["opt"]["name"] = self.cfg["proc"]["opt"]["channel"]
        for i, channel in enumerate(self.cfg["proc"]["opt"]["channel"]):
            record = {"channel": channel, "dnb": []}
            buff = self.createImg(spotEnergy, spotBase, record)
            imgName = pathPrefix + "_%s.tif" % self.cfg["proc"]["opt"]["name"][i]
            self.saveImg(buff, imgName)
            info.append(record)
        if saveInfo == True:
            infoFile = pathPrefix + "_info.tsv"
            self.saveInfo(infoFile, info)
        return 0

    def simulateImgSet(self):
        """
        :return:
        """
        import os

        outDir = self.cfg["img"]["outdir"]
        if not os.path.exists(outDir):
            os.makedirs(outDir)

        curRow = 1
        curCol = 1
        for n in range(self.cfg["img"]["fov"]):
            if curRow > self.cfg["img"]["rowMax"]:
                print("Image number out of range:\nShould less than rowMax * colMax.", file=sys.stderr)
                break
            for cycle in range(1, self.cfg["img"]["cycle"] + 1):
                outDir = os.path.join(self.cfg["img"]["outdir"], "S%03d" % cycle)
                if not os.path.exists(outDir):
                    os.makedirs(outDir)
                filePrefix = os.path.join(outDir, "C%03dR%03d" % (curCol, curRow))
                self._logger("Simulating image: cycle %d, C%03dR%03d." % (cycle, curCol, curRow))
                self.simulateSinge(filePrefix, self.cfg["img"]["saveInfo"])
            curCol += 1
            if curCol > self.cfg["img"]["colMax"]:
                curCol = 1
                curRow += 1
        self._logger("All finished.")
        return


##########################################################################
############################  BEGIN Function  ############################
##########################################################################


######################################################################
############################  BEGIN Main  ############################
######################################################################
#################################
##
##   Main function of program.
##
#################################
def main():
    import os

    ###### Usage
    usage = '''

    Version %s  by Vincent Li  %s

    Usage: %s <config> >STDOUT
    ''' % (prog_version, prog_date, os.path.basename(sys.argv[0]))

    ######################### Phrase parameters #########################
    import argparse
    ArgParser = argparse.ArgumentParser(usage=usage)
    ArgParser.add_argument("--version",
                           action="version",
                           version=prog_version)
    ArgParser.add_argument('--cfg',
                           type=str,
                           default='./tests/testCfg.json',
                           help='The json config file.')

    (para, args) = ArgParser.parse_known_args()

    # if len(args) != 1:
    #     ArgParser.print_help()
    #     print("\nERROR: The parameters number is not correct!", file=sys.stderr)
    #     sys.exit(1)
    # else:
    #     (config,) = args
    config = para

    ############################# Main Body #############################
    creator = SimulateImg(config.cfg)
    creator.simulateImgSet()

    # newCDF = CDF(int(xCoord))
    # print(-3, newCDF.ip(-3))
    # print(-3, newCDF.ip(-3))
    # print(0, newCDF.ip(0))
    # print(3, newCDF.ip(3))
    # print("")
    # print(newCDF)


#################################
##
##   Start the main program.
##
#################################
if __name__ == '__main__':
    main()

################## God's in his heaven, All's right with the world. ##################
