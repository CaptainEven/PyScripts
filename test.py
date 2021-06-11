
from origin.tifffile import imread, imsave, TiffFile
imgFile = 'FS001.L001.S002.C003R050.P000.B.2016-07-19_13-16-20-150.tif'

img = imread(imgFile)
print(img)
print(img.dtype)

img[2][1] = 65535
print(img)

outImg = 'new.tif'
testTag = (43210, 'I', 1, 4678501)
imsave(outImg, img, extratags=[testTag])

with TiffFile(outImg) as tif:
    print(str(tif))
    # print(tif.series)
    for page in tif:
        for tag in page.tags.values():
            print("%s\t%s\t%s\t%s" % (tag.code, tag.name, tag.dtype, tag.value))
    print(len(tif))
