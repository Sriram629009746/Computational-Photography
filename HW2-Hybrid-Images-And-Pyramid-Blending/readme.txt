hybrid_images.py : This has the implementation of hybrid images technique. The new function added to starter code is gaussian2D() which is the implementation of the gaussian filter.
pyramid_blending.py : This has the implementation of pyramid blending technique. The new functions added to starter code are :
					- pyrDown() : used to filter and downsample the image in creation of gaussian pyramid.
					- pyrUp() : used to filter and upsample the image in creation of laplacian pyramid.
					- create_gaussian_pyramid() : computes gaussian pyramid given an image and number of layers
					- create_laplacian_pyramid(): computes laplacian pyramid given an image and number of layers
					- add_borders() : adds border to image to make the dimensions a power of 2
					- remove_borders(): removes the added borders
					- blend() : combines two laplacian pyramids 
					- PyramidBlend(): creates the blended image pyramid