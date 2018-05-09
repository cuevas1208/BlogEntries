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

# ==============================================================================

def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:,:,:3] # Grab the BRG planes
    overlay_mask = overlay_t_img[:,:,3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

# ==============================================================================


def visualize_image_labels(np_image, segmentation, class_names, image_segmentation):
    num_classes = len(class_names)

    # unique predicted CLASSES add label name
    segmentation = np.squeeze(segmentation)
    unique_classes, relabeled_image = np.unique(segmentation, return_inverse=True)

    segmentation_size = segmentation.shape

    relabeled_image = relabeled_image.reshape(segmentation_size)

    labels_names = []
    for index, current_class_number in enumerate(unique_classes):
        labels_names.append(str(index) + ' ' + class_names[current_class_number])

    discrete_matshow(data=relabeled_image, labels_names=labels_names, title="Segmentation")
    plt.show()

    # smooth lines
    '''
    segmentation = np.squeeze(image_segmentation)
    if len(unique_classes) > 0:
        discrete_matshow(data=segmentation, labels_names=labels_names, title="Segmentation") '''

    # INPUT IMAGE
    np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("Input Image", np_image.astype(np.uint8))
    cv2.waitKey(0)

    # one
    relabeled_image = cv2.cvtColor(relabeled_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    for classes in unique_classes:
        print(classes)
        mask = relabeled_image
        mask[relabeled_image == classes]
        color_mask = (255 // num_classes) * (np.random.randint(num_classes))
        print(mask//2)
        mask = color_mask * (mask//2)
        print(mask)
        mask = cv2.resize(mask, (np_image.shape[1], np_image.shape[0]))
        cv2.imshow("relabeled_image", mask)
        cv2.waitKey(0)

    # output
    relabeled_image = relabeled_image * 255 / (num_classes+1)
    relabeled_image = cv2.resize(relabeled_image, (np_image.shape[1], np_image.shape[0]))
    cv2.imshow("relabeled_image", relabeled_image.astype(np.uint8))
    cv2.waitKey(0)

    # relabeled_image = cv2.cvtColor(relabeled_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    # relabeled_image[::2] = relabeled_image / (num_classes/2)
    # relabeled_image[::3] = relabeled_image / (num_classes/4)

    # apply the overlay
    result = cv2.addWeighted(relabeled_image, 1, np_image, 0.2, 0)
    cv2.imshow("relabeled_image", result)
    cv2.waitKey(0)

    for i, classes in enumerate(unique_classes):
        mask = image_segmentation

        # replace all the values in array
        mask[image_segmentation == classes]
        mask = np.resize(mask, image_segmentation.shape)
        print("mask", mask.shape)

        color_mask = i*(255/num_classes)*np.random.randn(num_classes)
        mask = int(color_mask[0]) + mask
        print("mask", np.array(mask.shape))
        print("np.image", np_image.shape)
        np_image[:, :, (classes%3)] = np_image[:, :, (classes%3)] + mask

    print("\n", np_image.shape)

    # Show the downloaded image
    plt.figure()
    plt.imshow(np_image.astype(np.uint8))
    plt.suptitle("Input Image", fontsize=14, fontweight='bold')
    plt.axis('off')

    plt.show()