
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

def resize_and_crop_image(image,
                    desired_size,
                    padded_size,
                    aug_scale_min=1.0,
                    aug_scale_max=1.0,
                    seed=1,
                    method=tf.image.ResizeMethod.BILINEAR,
                    logarithmic_sampling=False):
    """Resizes the input image to output size (RetinaNet style).

    Resize and pad images given the desired output size of the image and
    stride size.

    Here are the preprocessing steps.
    1. For a given image, keep its aspect ratio and rescale the image to make it
         the largest rectangle to be bounded by the rectangle specified by the
         `desired_size`.
    2. Pad the rescaled image to the padded_size.
    """
    with tf.name_scope('resize_and_crop_image'):
        image_size = tf.cast(tf.shape(image)[0:2], tf.float32)

        random_jittering = (aug_scale_min != 1.0 or aug_scale_max != 1.0)

        if random_jittering:
            if logarithmic_sampling:
                random_scale = tf.exp(
                        tf.random_uniform([],
							np.log(aug_scale_min),
							np.log(aug_scale_max),
							seed=seed))
            else:
                random_scale = tf.random_uniform([],
									aug_scale_min,
									aug_scale_max,
									seed=seed)
            scaled_size = tf.round(random_scale * desired_size)
        else:
            scaled_size = desired_size

        scale = tf.minimum(scaled_size[0] / image_size[0],
							scaled_size[1] / image_size[1])
        scaled_size = tf.round(image_size * scale)

        # Computes 2D image_scale.
        image_scale = scaled_size / image_size

        # Selects non-zero random offset (x, y) if scaled image is larger than
        # desired_size.
        if random_jittering:
            max_offset = scaled_size - desired_size
            max_offset = tf.where(
                    tf.less(max_offset, 0), tf.zeros_like(max_offset), max_offset)
            offset = max_offset * tf.random_uniform([
                    2,
            ], 0, 1, seed=seed)
            offset = tf.cast(offset, tf.int32)
        else:
            offset = tf.zeros((2,), tf.int32)

        scaled_image = tf.image.resize_images(
                image, tf.cast(scaled_size, tf.int32), method=method)

        if random_jittering:
            scaled_image = scaled_image[offset[0]:offset[0] + desired_size[0],
										offset[1]:offset[1] + desired_size[1], :]

        output_image = tf.image.pad_to_bounding_box(scaled_image, 0, 0,
												padded_size[0], padded_size[1])

        image_info = tf.stack([
                image_size,
                tf.constant(desired_size, dtype=tf.float32), image_scale,
                tf.cast(offset, tf.float32)
        ])
        return output_image, image_info


def process_images(images, output_size=[640, 640]):
    def preprocess(im):
        im = tf.image.convert_image_dtype(im, tf.float32)
        im -= (128.0 / 255.0)
        im /= (128.0 / 255.0)
        return im

    def resize(image):
        image, image_info = resize_and_crop_image(
                image, output_size, output_size, aug_scale_min=1, aug_scale_max=1)
        return image, image_info

    # with tf.Graph().as_default():
    images = [preprocess(im) for im in images]
    images_resized = []
    images_info = []
    for im in images:
        im_resized, im_info = resize(im)
        images_resized.append(im_resized)
        images_info.append(im_info)

    images_resized = tf.stack(images_resized)
    return images_resized, images_info
    # with tf.Session() as sess:
        # return sess.run([images_resized, images_info, raw_images])


"""Ref: https://github.com/facebookresearch/detectron2/blob/22e04d1432363be727797a081e3e9d48981f5189/detectron2/utils/colormap.py
"""
_COLORS = np.array([
    0.000, 0.000, 0.000, 0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694,
    0.125, 0.494, 0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635,
    0.078, 0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000,
    1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000,
    1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000, 0.333, 0.667, 0.000, 0.333,
    1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000, 0.667, 1.000, 0.000,
    1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000, 1.000, 0.000, 0.000, 0.333,
    0.500, 0.000, 0.667, 0.500, 0.000, 1.000, 0.500, 0.333, 0.000, 0.500, 0.333,
    0.333, 0.500, 0.333, 0.667, 0.500, 0.333, 1.000, 0.500, 0.667, 0.000, 0.500,
    0.667, 0.333, 0.500, 0.667, 0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000,
    0.500, 1.000, 0.333, 0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000,
    0.333, 1.000, 0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000,
    0.333, 0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000,
    1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000, 1.000,
    0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.333, 0.000, 0.000,
    0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000,
    0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000,
    0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167,
    0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000,
    0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.857,
    0.857, 0.857, 1.000, 1.000, 1.000
]).astype(np.float32).reshape(-1, 3)


def create_label_colormap(max_labels_in_colormap, include_black=False):
    if include_black:
        return _COLORS[:max_labels_in_colormap]
    else:
        return _COLORS[1:max_labels_in_colormap + 1]


def label2rgb(labels, class_names, include_black=False):
    colormap = create_label_colormap(len(class_names), include_black)
    colormapped_label = colormap[labels]
    return colormapped_label


def parse_user_images(contents, output_size=[640, 640]):
    """Convert jpg/png encodings to numpy images."""

    def decode_content(content):
        return tf.io.decode_jpeg(content, channels=3)

    def preprocess(im):
        im = tf.image.convert_image_dtype(im, tf.float32)
        im -= (128.0 / 255.0)
        im /= (128.0 / 255.0)
        return im

    def resize(image):
        image, image_info = resize_and_crop_image(
                image, output_size, output_size, aug_scale_min=1, aug_scale_max=1)
        return image, image_info

    with tf.Graph().as_default():
        raw_images = [decode_content(c) for c in contents]
        images = [preprocess(im) for im in raw_images]
        images_resized = []
        images_info = []
        for im in images:
            im_resized, im_info = resize(im)
            images_resized.append(im_resized)
            images_info.append(im_info)

        images_resized = tf.stack(images_resized)

        raw_images = tf.stack(raw_images)
        with tf.Session() as sess:
            return sess.run([images_resized, images_info, raw_images])


def add_space_between_figs():
    f, ax = plt.subplots()
    f.set_visible(False)
    f.set_figheight(1)


def visualize_res(image, results,
                categories_names, crop_sz = [618, 640],
                min_confidence_score=0.04,
                show_seg_proposals=True,
                fontsize=25,
                fig_size=9):
    segm_proposal = results['segm_proposal'][
            0, :crop_sz[0], :crop_sz[1], :].numpy()
    segm_prediction = results['segm_prediction'][
            0, :crop_sz[0], :crop_sz[1]].numpy()
    segm_prediction_rw = results['segm_prediction_rw'][
            0, :crop_sz[0], :crop_sz[1]].numpy()

    pixel_prediction = results['pixel_prediction'][
            0, :crop_sz[0], :crop_sz[1]].numpy().astype(int)
    pixel_pred_confidence = results['pixel_pred_confidence'][
            0, :crop_sz[0], :crop_sz[1]].numpy()
    segm_confidence_rw = results['segm_confidence_rw'][
            0, :crop_sz[0], :crop_sz[1]].numpy()
    segm_confidence = results['segm_confidence'][
            0, :crop_sz[0], :crop_sz[1]].numpy()

    categories_names_ = categories_names

    if min_confidence_score > 0:
        categories_names_ = ['unknown']
        categories_names_.extend(categories_names)
        segm_prediction += 1
        segm_prediction[segm_confidence < min_confidence_score] = 0
        segm_prediction_rw += 1
        segm_prediction_rw[segm_confidence_rw < min_confidence_score] = 0
        pixel_prediction += 1
        pixel_prediction[pixel_pred_confidence < min_confidence_score] = 0

    if show_seg_proposals:
        fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
        plt.title('segmentation proposals', fontsize=25)
        ax.imshow(np.argmax(segm_proposal, axis=-1), cmap='jet')
        ax.axis('off')
        plt.show()
        add_space_between_figs()

    def vis_seg_(image, segm_prediction, title=''):
        predicted_classes, unique_indices, unique_inverse = np.unique(
                segm_prediction, return_index=True, return_inverse=True)
        unique_inverse = unique_inverse.reshape(segm_prediction.shape)
        pred_class_names = [categories_names_[i] for i in predicted_classes]

        include_black = min_confidence_score > 0 and np.sum(
                segm_prediction == 0) > 0

        fig, ax = plt.subplots(1, 2, figsize=(fig_size * 2, fig_size))
        ax[0].imshow(image)
        ax[0].axis('off')
        ax[1].imshow(
                label2rgb(
                        unique_inverse, pred_class_names, include_black=include_black))
        ax[1].axis('off')

        # Adds legend.
        cmap = create_label_colormap(
                len(pred_class_names), include_black=include_black)
        patches = [
                mpatches.Patch(color=cmap[i], label=pred_class_names[i])
                for i in range(len(pred_class_names))
        ]
        ax[1].legend(
                handles=patches,
                bbox_to_anchor=(1.01, 1.0),
                loc='upper left',
                borderaxespad=0.,
                fontsize=fontsize)
        plt.title(title)
        plt.show()

    if vis_segm_pred_with_orig_ranking:
        vis_seg_(image,
                segm_prediction=segm_prediction,
                title='proposal ranked based semantic score.')
    # proposal scores are re-weighted to reduce the rank of proposals that cover
    # multiple different classes.
    vis_seg_(segm_prediction=segm_prediction_rw, title='')
    if vis_per_pixel_segm_pred.value:
        vis_seg_(segm_prediction=pixel_prediction, title='per pixel prediction')

    return segm_proposal


def plot_top_k(image, k, region_probs, segm_proposal, categories_names, proposals_min_area=2000):
    segm_area = np.sum(segm_proposal, axis=(0, 1))
    for i, name in enumerate(categories_names):
        add_space_between_figs()

        indices = np.argsort(-region_probs[:, i])
        fig, ax = plt.subplots(1, k, figsize=(k * 7, 7))
        fig.suptitle(name + ' (top-{} regions)'.format(k), fontsize=50, y=1.2)
        fig.tight_layout()
        ind = 0
        for j in range(segm_proposal.shape[-1]):
            if ind >= k:
                break
            if segm_area[indices[j]] < proposals_min_area:
                continue
            segmented_region = segm_proposal[:, :, indices[j]:indices[j] + 1] * image
            segmented_region[segmented_region == 0] = 0.6
            ax[ind].imshow(segmented_region)
            ax[ind].set_title(
                    'score:{:.2f}'.format(region_probs[indices[j], i]), fontsize=40)
            ax[ind].axis('off')
            ind += 1
        plt.show()