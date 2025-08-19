"""
Simply read the images in a folder.

Author: Stephane Vujasinovic
"""
import os
import numpy as np
import argparse
import cv2
import glob
import pyautogui

# Functions
# -----------------------------------------------------------------------------------------------------
def add_text_to_image(image: np.ndarray, text: str):
    """Put text on imgmask_operations"""
    position = (20, 150)  # (x, y) coordinates
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1
    font_color = (107, 75, 0)  # BGR color (green in this case)
    font_thickness = 2
    cv2.putText(image, text, position, font, font_scale, font_color, font_thickness)


def visualization_loop(path_to_data: str, file: str):
    """
    Loop over the folders and files to display the images.
    Keyboard Commands:
    - n: Next Image
    - b: Previous Image
    - s: Skip to Next Folder
    - a: Skip to Previous Folder
    - q: Quit Visualization
    """
    fdx = 0  # Controls the flows of the dataset
    breaking_algo = False
    while fdx < len(path_to_data):
        name = path_to_data[fdx].split("/")[-1]
        print(f"\n - Sequence: {name} -")

        list_of_image_for_folder = sorted(
            glob.glob(os.path.join(path_to_data[fdx], file, "*"))
        )
        window_width, window_height = cv2.imread(list_of_image_for_folder[0]).shape[:2][
            ::-1
        ]
        cv2.namedWindow(f"{name}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"{name}", window_width, window_height)

        idx = 0  # Controls the flows of the sequence
        while idx < len(list_of_image_for_folder):
            image = cv2.imread(list_of_image_for_folder[idx])
            add_text_to_image(image, f"{idx+1}/{len(list_of_image_for_folder)}")
            cv2.imshow(f"{name}", image)

            key = cv2.waitKey(0)
            if ord("n") == key:
                idx = idx + 1
                continue
            elif ord("b") == key:
                idx = idx - 1
                continue
            elif ord("y") == key:
                break
            elif ord("z") == key:
                fdx = fdx - 2
                break
            elif ord("q") == key:
                breaking_algo = True
                break

        if breaking_algo:
            break

        fdx += 1
        cv2.destroyAllWindows()


# Main
# -----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    message = """
    Commands:
        - N : forwards
        - B : backwards
        - Y : next sequence
        - Z : previous sequnce
        - Q : quit program
    """
    # Display the message in a pop-up window
    pyautogui.alert(message, title="Popup Message")


    parser = argparse.ArgumentParser(description="Visualize the output scores(softmax)")
    parser.add_argument(
        "--data",
        type=str,
        default='D17',
        help="""Specify the dataset. Available options are : 'D17', 'LV1'""",
    )
    parser.add_argument(
        "--model",
        type=str,
        default='XMem',
        help="""Specify the model. Available options are : 'XMem', 'MiVOS', 'STCN', 'QDMN.'""",
    )
    args = parser.parse_args()

    base_path = os.path.join("../..", "Visual_Output", args.model, args.data)
    path_to_data = sorted(glob.glob(os.path.join(base_path, "*")))


    visualization_loop(path_to_data, "global")
