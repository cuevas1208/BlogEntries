########################################################################
#
# Functions for visualize images and data.
#
# Implemented in Python 3.5
#
########################################################################
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Manuel Cuevas
#
########################################################################
import matplotlib.pyplot as plt

class visualize: 
	def plot_images(class_names, images, cls_true, cls_pred=None, smooth=True):
		'''
		Function used to plot 9 images in a 3x3 grid, and writing the true 
		and predicted classes below each image.
		The images might be a bit easier for the human eye to recognize
		if smoothen is True.
		'''

		assert len(images) == len(cls_true) == 9

		# Create figure with sub-plots.
		fig, axes = plt.subplots(3, 3)

		# Adjust vertical spacing if we need to print ensemble and best-net.
		if cls_pred is None:
		    hspace = 0.3
		else:
		    hspace = 0.6
		fig.subplots_adjust(hspace=hspace, wspace=0.3)

		for i, ax in enumerate(axes.flat):
		    # Interpolation type.
		    if smooth:
		        interpolation = 'spline16'
		    else:
		        interpolation = 'nearest'

		    # Plot image.
		    ax.imshow(images[i, :, :, :],
		              interpolation=interpolation)
		        
		    # Name of the true class.
		    cls_true_name = class_names[cls_true[i]]

		    # Show true and predicted classes.
		    if cls_pred is None:
		        xlabel = "True: {0}".format(cls_true_name)
		    else:
		        # Name of the predicted class.
		        cls_pred_name = class_names[cls_pred[i]]

		        xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

		    # Show the classes as the label on the x-axis.
		    ax.set_xlabel(xlabel)
		    
		    # Remove ticks from the plot.
		    ax.set_xticks([])
		    ax.set_yticks([])

		# Ensure the plot is shown correctly with multiple plots
		# in a single Notebook cell.
		plt.show()