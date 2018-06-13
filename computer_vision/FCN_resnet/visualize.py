from matplotlib import pyplot as plt
import numpy as np
import cv2


def discrete_matshow(data, labels_names=[], title=""):
    """
    Function to nicely print segmentation results with colorbar showing class names
    :param data:
    :param labels_names:
    :param title:
    :return:
    """
    fig_size = [7, 6]
    plt.rcParams["figure.figsize"] = fig_size

    # get discrete colormap
    cmap = plt.get_cmap('Paired', np.max(data) - np.min(data) + 1)

    # set limits .5 outside true range
    mat = plt.matshow(data, cmap=cmap, vmin=np.min(data) - .5, vmax=np.max(data) + .5)

    # tell the colorbar to tick at integers
    cax = plt.colorbar(mat, ticks=np.arange(np.min(data), np.max(data) + 1))

    # The names to be printed aside the colorbar
    if labels_names:
        cax.ax.set_yticklabels(labels_names)

    if title:
        plt.suptitle(title, fontsize=15, fontweight='bold')


def visualize_image_labels(np_image, segmentation, class_names, image_segmentation):
    num_classes = len(class_names)

    # unique predicted CLASSES add label name
    segmentation = np.squeeze(segmentation)
    unique_classes, relabeled_image = np.unique(segmentation, return_inverse=True)
    segmentation_size = segmentation.shape
    relabeled_image = relabeled_image.reshape(segmentation_size)

    # INPUT IMAGE
    np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("Input Image", np_image.astype(np.uint8))
    cv2.waitKey(0)

    # output weights
    labels_names = []
    for index, current_class_number in enumerate(unique_classes):
        labels_names.append(str(index) + ' ' + class_names[current_class_number-3])

    print(class_names)
    print(unique_classes, relabeled_image)
    print(labels_names)

    discrete_matshow(data=relabeled_image, labels_names=labels_names, title="Segmentation")
    plt.show()

    # Blend output/inout
    relabeled_image = relabeled_image * 255 / (num_classes+1)
    relabeled_image = cv2.resize(relabeled_image, (np_image.shape[1], np_image.shape[0]))
    relabeled_image = cv2.cvtColor(relabeled_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    result = cv2.addWeighted(np_image, .3, relabeled_image, 0.9, 0)
    cv2.imshow("relabeled_image", result)
    cv2.waitKey(0)