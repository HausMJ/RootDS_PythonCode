# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:09:48 2020

@author: Logan Pierz
"""

#importing the necessary libraries
from plantcv import plantcv as pcv
import os
import argparse
import sys
from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import Image
import glob
from time import localtime, strftime 



### Parse command-line varriables
#Defines the necessary variables for the program to function
def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument("-i", "--image", help="Input image file.", required=True)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=True)
    parser.add_argument("-r","--result", help="result file.", required= True )
    parser.add_argument("-w","--writeimg", help="write out images.", default=False, action="store_true")
    parser.add_argument("-D", "--debug", help="can be set to 'print' or None (or 'plot' if in jupyter) prints intermediate images.", default=None)
    parser.add_argument("-n", "--Naive_Bayes", help="Naive Bayes File to be Used for Image Processing", required= True)
    args = parser.parse_args()
    return args
args = options()

dirs = os.listdir(args.image)
#Defining whether a contour is not part of the main stem
def is_contour_bad(c):
    peri = cv2.contourArea(c)
    return peri > 5000
#defining whether a contour is part of the main stem
def is_contour_good(c):
    peri = cv2.contourArea(c)
    return peri < 5000


#Defining the main Program
def main():
    #Creating a Loop to run the program over an entire directory of images
    #deciding whether the necessary output paths exist or need to be created before the program begins creating images
    binpath = os.path.join(args.outdir,"Binary_Images")
    if not os.path.exists(binpath):
        os.mkdir(binpath)
        
    dispath = os.path.join(args.outdir,"Diseased_Images")
    if not os.path.exists(dispath):
        os.mkdir(dispath)
    helpath = os.path.join(args.outdir,"Healthy_Images")
    if not os.path.exists(helpath):
        os.mkdir(helpath)
        #begins running the core program over the images in the output directory
    for item in dirs:
        if os.path.isfile(args.image+item):

            #prints the time to the console to monitor speed of the program over a large batch of images
            print(strftime("%H:%M:%S", localtime()))
            
            #imports the image
            img, path, filename = pcv.readimage(args.image+item)
            #creates the filename variable by copying the image name and removing the .jpg extension
            filename2 = (filename.replace('.jpg',''))
            filename3 = (filename2.replace('.JPG',''))
            #uses the created filename to create a folder in the output directory named after the image being processed
            path = os.path.join(args.outdir, filename3)
            if not os.path.exists(path):
                os.mkdir(path)
            
            pcv.params.debug=args.debug #set debug mode
            
            pcv.params.debug_outdir=(path) #set output directory
            
            
            #reprints the original image into the new output directory and reopens it into a new format so size can be accessed
            pcv.print_image(img = img, filename = (path + '/' + filename))
            image = Image.open(path + '/' + filename)

            #gets the width and height of the image
            w, h = image.size
            total_pixels = w * h
            
            #creates the masks for the background, shoot, diseased root and non-diseased root for the import image
            masks = pcv.naive_bayes_classifier(img, args.Naive_Bayes)
            
            
            #processes the contours of the shoot mask to help seperate stem and root tissue that the original naive bayes program wasnt able to distinguish
            shoots_1 = masks['Shoot']
            shoots_1 = cv2.morphologyEx(shoots_1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
            shoots_2 = cv2.morphologyEx(shoots_1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
            _, cnts, _ = cv2.findContours(shoots_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            _, cnts2, _ = cv2.findContours(shoots_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


            root_shoot = shoots_1
            shoot = shoots_2
            
            #draws the contours for the shoot mask and seperates the parts into two masks
            for c in cnts:
                if is_contour_bad(c):
                    cv2.drawContours(root_shoot, [c], -1, 0, -1)
                    
            for c in cnts2:
                if is_contour_good(c):
                    cv2.drawContours(shoot, [c], -1, 0, -1)
                    

            #combines masks together to simplify image analysis
            diseased_roots = pcv.logical_or(bin_img1 = masks['Diseased Root'], bin_img2 = root_shoot)
            full_roots = pcv.logical_or(bin_img1 = diseased_roots, bin_img2 = masks['Non-Diseased Root'])
            full_roots_filled = pcv.fill(bin_img = full_roots, size = 200)
            
            #prints out the binary root and diseased root images to the output directory
            pcv.print_image(img = full_roots_filled, filename = (path + '/' + filename + '_' + 'Binary_roots.jpg'))
            pcv.print_image(img = full_roots_filled, filename = (binpath + '/' + filename + '_' + 'Binary_roots.jpg'))
            
            pcv.print_image(img = diseased_roots, filename = (path + '/' + filename + '_' + 'Diseased_roots.jpg'))
            pcv.print_image(img = diseased_roots, filename = (dispath + '/' + filename + '_' + 'Diseased_roots.jpg'))
            
            pcv.print_image(img = masks['Non-Diseased Root'], filename = (path + '/' + filename + '_' + 'Healthy_roots.jpg'))
            pcv.print_image(img = masks['Non-Diseased Root'], filename = (helpath + '/' + filename + '_' + 'Healthy_roots.jpg'))
            
            full_plant = pcv.logical_or(bin_img1 = full_roots, bin_img2 = shoot)
            
            #creates a custom background based off of the plant masks created in the original mask creation
            background_mask = 255 - full_plant
            #colors the masks created by the naive_bayes_classifier
            colored_img = pcv.visualize.colorize_masks(masks=[background_mask, shoot, masks['Non-Diseased Root'], diseased_roots], colors=['white', 'dark green', 'blue', 'red'])

            #prints out the colored image to the designated folder as a jpg
            pcv.print_image(img=colored_img, filename=(path + '/' + filename + '_' + "colored_img.jpg"))

            #creates the output data file
            outfile=True
            if args.writeimg==True:
                outfile=args.outdir+"/"+filename

            #total pixel count is displayed
            print('Total Pixels =', total_pixels)

            #percentage of the image that each pixel type takes up is calculated
            background_perc = (masks['Background']>0).mean()
            stem_perc = (shoot>0).mean()
            diseased_root_perc = (diseased_roots>0).mean()
            healthy_root_perc = (masks['Non-Diseased Root']>0).mean()
            #percentages that are calculated above are multiplied by 100 and displayed
            print('Percent of Image as Background = ', (background_perc*100))
            print('Percent of Image as Shoot = ', (stem_perc*100))
            print('Percent of Image as Diseased Root = ', (diseased_root_perc*100))
            print('Percent of Iamge as Healthy Root = ', (healthy_root_perc*100))
            #total number of root pixels are calculated and displayed by subracting the total background and shoot pixels from the total pixel count
            root_pixels = (total_pixels - (background_perc*total_pixels) - (stem_perc*total_pixels))
            root_perc_total = (root_pixels/total_pixels)*100

            print('Total Number of Pixels in the Roots = ', root_pixels)
            #number of healthy and diseased pixels are calculated from the percentage multiplied by the total pixels
            diseased_pixelcount = (diseased_root_perc*total_pixels)

            #the ratio of diseased root to healthy root is calculated and displayed as a percentage
            print('percent of Total Roots that are Diseased = ', ((diseased_pixelcount/root_pixels)*100))
            diseased_percent = ((diseased_pixelcount/root_pixels)*100)
    
            
            #pcv.print_results(args.result)
            #write the processed data to a CSV file that can then be opened in excel for further analysis
            writefile = open(args.outdir + 'Data.csv', 'a+')
            writefile.write(filename2 + ', ' + '%.2f' % diseased_percent + ', ' + '%.2f' % root_perc_total + '\n')
            writefile.close()
#executes the above program
if __name__ == '__main__':
    main()


