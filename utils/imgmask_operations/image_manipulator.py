'''
API for image operations

by Stephane Vujasinovic
'''
# - IMPORTS ---
import numpy as np
from PIL import Image
import skimage
import cv2
from icecream import ic

from utils.imgmask_operations.mask_manipulator import include_the_background, get_number_of_objects


# - CLASS ---
class ImageManipulator:
    """Class for dealing with imgmask_operations processing, numpy arrays related to 'images', etc"""
    def __init__(self) -> None:
        self.objects_in_gt = set()
        self.objects_in_mask = set()  # prediction

    @staticmethod
    def load_image_with_scikit(path_to_file: str) -> np.ndarray:
        """Load imgmask_operations data with scikit"""
        assert isinstance(path_to_file, str)
        return skimage.io.imread(path_to_file)

    @staticmethod
    def load_image_with_PIL(path_to_file: str) -> np.ndarray:
        """Load imgmask_operations data with PIL"""
        assert isinstance(path_to_file, str)
        return np.array(Image.open(path_to_file))

    @staticmethod
    def load_image_with_OpenCV(path_to_file: str) -> np.ndarray:
        """Load imgmask_operations data with OpenCV"""
        assert isinstance(path_to_file, str)
        return cv2.imread(path_to_file)

    @staticmethod
    def convert_img_from_int_to_float(img: np.ndarray) -> np.ndarray:
        """Convert imgmask_operations from UINT8 to FLOAT32"""
        assert img.dtype == np.uint8
        return skimage.img_as_float32(img)

    @staticmethod
    def convert_img_from_float_to_uint8(img: np.ndarray) -> np.ndarray:
        """Convert imgmask_operations from FLOAT32 to UINT8"""
        if img.dtype == np.float64:
            return skimage.img_as_ubyte(img)
        return img

    @staticmethod
    def convert_img_to_rgb(img: np.ndarray) -> np.ndarray:
        """Convert imgmask_operations from RGBalpha to rgb and also gray scale to rgb"""
        assert isinstance(img, np.ndarray)
        if 2 == len(img.shape):
            return skimage.color.gray2rgb(img)
        elif 3 == len(img.shape):
            return img
        elif 4 == img.shape[-1]:
            return skimage.color.rgba2rgb(img)
        else:
            assert "What king of imgmask_operations is this?"

    @staticmethod
    def combine_img_with_softmask(img: np.ndarray, softmask: np.ndarray) -> np.ndarray:
        """
        TODO: Check if matrix multiplication is faster
        Combine imgmask_operations with a softmask (acts as alpha) where values are between 0 and 1.
        Image values are also expected to be between 0 and 1.
        """
        assert img.dtype == np.float32
        assert softmask.dtype == np.float32
        assert len(softmask.shape) == 2

        img_w_mask = img.copy()  # No mask for the moment
        for jdx in (0, img_w_mask.shape[-1] - 1):
            img_w_mask[:, :, jdx] = (1 - softmask) * img[:, :, jdx] + softmask

        return img_w_mask

    @staticmethod
    def get_contours(mask: np.ndarray, width=3):
        """
        Convert mask [R,G,B] mask (DAVIS style) to a boolean mask with only the contours of the mask.
        """
        if len(mask.shape)==3: # Much slower
            unique_colors = np.unique(np.vstack(mask), axis=0)[:, :-1]
            nbr_of_objs = len(unique_colors)
            zero_mask = np.zeros([nbr_of_objs, *mask.shape[:-1]], dtype=bool)

            for obj_idx, (mask_contour, color) in enumerate(zip(zero_mask, unique_colors)):
                if obj_idx == 0:  # skip the background for the contours because not available and needn't
                    continue
                mask = mask[:, :, :-1] == color
                mask = mask.min(axis=-1)
                dilated_mask = skimage.morphology.binary_dilation(mask, footprint=np.ones((width, width)))
                erosion_mask = skimage.morphology.binary_erosion(mask, footprint=np.ones((width, width)))
                contour = dilated_mask ^ erosion_mask   # XOR operation
                zero_mask[obj_idx] = contour
        elif len(mask.shape) == 2:
            nbr_of_objs = 1
            mask = mask.astype(bool)
            zero_mask = np.zeros([*mask.shape], dtype=bool)
            dilated_mask = skimage.morphology.binary_dilation(mask, footprint=np.ones((width, width)))
            erosion_mask = skimage.morphology.binary_erosion(mask, footprint=np.ones((width, width)))
            contour = dilated_mask ^ erosion_mask   # XOR operation
            zero_mask = np.expand_dims(contour, axis=0)
            
        else:
            assert "Input is not correct, check mask"

        return zero_mask  # Not filled with zeros anymore

    @staticmethod
    def color_image_w_ground_truths(
        img: np.ndarray,
        gt_contour_for_cls_x: np.ndarray,
        color: np.ndarray
    ) -> np.ndarray:
        """Color the img on the GT contour location"""
        assert color.dtype == np.float32
        img_v_stack = np.vstack(img)
        gt_contour = np.array([gt_contour_for_cls_x,
                               gt_contour_for_cls_x,
                               gt_contour_for_cls_x]).transpose(1, 2, 0)
        gt_contour_v_stack = np.vstack(gt_contour)[:, 0]
        for channel in range(0, img_v_stack.shape[-1]):
            img_v_stack[gt_contour_v_stack, channel] = color[channel]
        return img_v_stack.reshape(*img.shape)

    @staticmethod
    def color_image_with_predictions(img: np.ndarray, gt_contour_for_cls_x: np.ndarray,
                                     color: np.ndarray) -> np.ndarray:
        """Color the img on the GT contour location"""
        assert color.dtype == np.float32
        img_v_stack = np.vstack(img)
        gt_contour = np.array([gt_contour_for_cls_x, gt_contour_for_cls_x, gt_contour_for_cls_x]).transpose(1, 2, 0)
        gt_contour_v_stack = np.vstack(gt_contour)[:, 0]
        for channel in range(0, img_v_stack.shape[-1]):
            img_v_stack[gt_contour_v_stack, channel] = color[channel]
        return img_v_stack.reshape(*img.shape)


    @staticmethod
    def generate_indices_from_mask(mask:np.ndarray) -> list[list, np.ndarray, np.ndarray]:
        """Generate the indices for the maks. E.g. from [0,0,0] to 0, [0,0,128] to 1, etc."""  
        assert mask.dtype == np.uint8
        assert len(mask.shape) == 3
        v_stacked_mask = np.vstack(np.copy(mask))
        
        unique_combinations = np.unique(v_stacked_mask, axis=0)
        #unique_combinations = np.array([[0,0,0],[0,128,0],[128,0,0]]) # List all unique values for each dataset

        # Create a stack of masks
        stack_of_masks = np.zeros([*mask.shape[:2], len(unique_combinations)], dtype=bool)
        for idx, mask_x in enumerate([np.ones_like(mask)*elem for elem in unique_combinations]):
             stack_of_masks[:,:,idx] = (mask_x == mask).min(axis=2)

        concatenated_indices = np.zeros([*mask.shape[:2]], dtype=np.uint8)
        for idx, mask_x in enumerate(stack_of_masks.transpose(2,0,1)):
            concatenated_indices += (mask_x*idx).astype(np.uint8)

        return unique_combinations, stack_of_masks, concatenated_indices
    
    
    def generate_masks_from_GT_palette(self, gt:np.ndarray) -> np.ndarray: 
        # get all objects in the ground-truth
        gt_objects = np.unique(gt)
        self.objects_in_gt.update(set(gt_objects))
        
        seperated_gt = np.zeros([*gt.shape,len(gt_objects)], dtype=bool) 
        for idx, obj_idx in enumerate(gt_objects): seperated_gt[:,:,idx] = (gt == obj_idx)
            
        return seperated_gt

    @staticmethod
    def generate_masks_from_palette(
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Extract the masks from the palette .png file
        Move this method in the mask manipulator

        Args:
            mask (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        # get all objects in the ground-truth
        mask_objects = np.unique(mask)

        extracted_mask = np.zeros(
            [*mask.shape, len(mask_objects)],
            dtype=bool)
        for idx, obj_idx in enumerate(mask_objects):
            extracted_mask[:, :, idx] = (mask == obj_idx)

        return mask_objects, extracted_mask
    
    @staticmethod
    def compute_confusion():
        '''Compute the TP,TN,FP anf FN regions.'''

    def color_img_w_multiple_gt_masks(
        self,
        img: np.array,
        gt_contours: np.array
    ) -> np.array:
        """
        Combine the imgmask_operations with multiple ground truth contours
        """
        GT_COLOR_YELLOW = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        ic(get_number_of_objects(gt_contours))
        for cls_x in range(include_the_background(False),
                           get_number_of_objects(gt_contours)):
            if 0 == cls_x:  # don't want the GT for background because none
                continue
            gt_contour_for_cls_x = gt_contours[cls_x]
            img = self.color_image_w_ground_truths(img, gt_contour_for_cls_x,
                                                   GT_COLOR_YELLOW)
        return img

    def combine_img_w_multiple_softmask(
        self,
        img: np.array,
        prediction: np.array
    ) -> np.array:
        """
        Combine the imgmask_operations with multiple softmasks

        Args:
            img (np.array): _description_
            prediction (np.array): _description_

        Returns:
            np.array: _description_
        """
        for cls_x in range(
            include_the_background(False),
            get_number_of_objects(prediction)
        ):
            img = self.combine_img_with_softmask(img, prediction[cls_x])
        return img
