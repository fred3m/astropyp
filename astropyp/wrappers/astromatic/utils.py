from collections import OrderedDict

sex_internal_flags = OrderedDict([
    (128, 'A memory overflow occurred during extraction'),
    (64, 'A memory overflow occurred during deblending'),
    (32, 'Objects isophotal data are incomplete or corrupted'),
    (16, 'Objects aperture data are incomplete or corrupted'),
    (8, 'The object is truncated (too close to an image boundary'),
    (4, 'At least one pixel of the object is saturated'),
    (2, 'The object was originally blended with another one'),
    (1, 'The object has neighbours, bright and close enough to significantly'
        ' bias the MAG AUTO photometry, or bad pixels (more than 10%% of'
        ' the integrated area affected')
])