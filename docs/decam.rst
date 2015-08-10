*****
DECam
*****

DECam Tools is a python package designed to aid observers in processing and analyzing their
DECam images that have already been processed by the Community Pipeline. Most of the scripts
work with the InstCal files although users can really use files generated at any stage in the
Community pipeline including raw images, resampled images, and stacks.

In addition to allowing users to customize the resampling and stacking of images, DECam Tools
also provides a number of tools for post-pipeline analysis including object detection, 
calibration with SDSS fields, photometry, and the creation of a sqlite database to keep
track of images by cataloging them by the most important fields in the DECam headers.


Acknowledgement
===============

I am not affiliated with the Dark Energy Survey, the Dark Energy Camera, CTIO, or NOAO, 
but I am a graduate student in astronomy who has been reducing my own DECam images and
thought that many of the tools I developed would be useful. Please acknowledge the use
of this code if you use it in your research.