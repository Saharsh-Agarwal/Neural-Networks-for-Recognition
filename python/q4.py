import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(img):
    bb = []
# Estimate the average noise standard deviation across color channels.
    sigma_est = skimage.restoration.estimate_sigma(img, average_sigmas=True, multichannel= True, channel_axis=-1) #single sigma as avg.
    print(sigma_est)
#plt.imshow(img)
    denoised_img = skimage.restoration.denoise_bilateral(img, win_size=5, sigma_color=sigma_est, multichannel=True, channel_axis=-1)
#denoise 4 other options 
#plt.imshow(denoised_img)
    grey_img = skimage.color.rgb2gray(denoised_img)
#plt.imshow(grey_img)
    thresh = skimage.filters.threshold_otsu(grey_img) #try_all_thrshold
#print(thresh)
    binary = grey_img<thresh
#plt.imshow(binary)
    im = skimage.morphology.closing(binary, skimage.morphology.square(5))
#plt.imshow(im)
#plt.show()
    im1 = skimage.morphology.dilation(im, skimage.morphology.square(9)) #mota karte hue
    #plt.imshow(im1, cmap="gray")
    #plt.show()
    label_img = skimage.measure.label(im1,connectivity=2, background=0)
    reg = skimage.measure.regionprops(label_image=label_img)
#print(len(reg))

    a = 0 
    for r in reg:
        a += r.area
    mean_a = a/len(reg)
    print("Avg Area =",mean_a)

    for r in reg:
        if r.area >= mean_a/5:
            t,l,b,r = r.bbox
            bb.append([t,l,b,r]) #row,col,row,col

    im1 = 1-im1 #letters and background interchanged
    return bb, im1