# Default DAVIS 2017 Attributes

Source: [CVPR Paper 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Perazzi_A_Benchmark_Dataset_CVPR_2016_paper.pdf)

Check the Original file for the attributes:  
    - [YAML file](Official_DAVIS_attributes.yaml) / [GitHub repo](https://github.com/fperazzi/davis-2017/blob/main/data/db_info.yaml)
    - [Adapted csv file](Original_DAVIS_attributes.csv)


| ID        | Description                                                                                                                         |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| ***BC***  | **Background Clutter**: The back- and foreground regions around the object boundaries have similar colors (χ2 over histograms).     |
| ***DEF*** | **Deformation**: Object undergoes complex, non-rigid deformations.                                                                  |
| ***MB***  | **Motion Blur**: Object has fuzzy boundaries due to fast motion.                                                                    |
| ***FM***  | **Fast-Motion**: The average, per-frame object motion, computed as centroids Euclidean distance, is larger than τf m = 20 pixels.   |
| ***LR***  | **Low Resolution**: The ratio between the average object bounding-box area and the image area is smaller than tlr = 0.1. Is equivalent to small objects art to image size            |
| ***OCC*** | **Occlusion**: Object becomes partially or fully occluded.                                                                          |
| ***OV***  | **Out-of-view**: Object is partially clipped by the image boundaries.                                                               |
| ***SV***  | **Scale-Variation**: The area ratio among any pair of bounding-boxes enclosing the target object is smaller than τsv = 0.5.         |
| ***AC***  | **Appearance Change**: Noticeable appearance variation, due to illumination changes and relative camera-object rotation.            |
| ***EA***  | **Edge Ambiguity**: Unreliable edge detection. The average ground-truth edge probability (using [11]) is smaller than τe = 0.5.     |
| ***CS***  | **Camera-Shake**: Footage displays non-negligible vibrations.                                                                       |
| ***HO***  | **Heterogeneous Object**: Object regions have distinct colors.                                                                      |
| ***IO***  | **Interacting Objects**: The target object is an ensemble of multiple, spatially-connected objects (e.g. mother with stroller).     |
| ***DB***  | **Dynamic Background**: Background regions move or deform.                                                                          |
| ***SC***  | **Shape Complexity**: The object has complex boundaries such as thin parts and holes.                                               |

# Additional Attribute
*Why?*:
- Would like to introduce additional attributes to better evaluate the entropy related attributes.
- Also, the  attributes are per sequence, which is not descriptive enough for each object individually.
- Hence some attributes are either missing in the original file or don't belong to all the objects.
- Also just noted that the attributes are only listed for object that originally belonged to the DAVIS 2016, and none are given fo the DAVIS 2017 additional sequences.
- But since I'm evaluating on the DAVIS 2017 dataset, I won't have all attributes...

Check the new attributes:
    - [YAML file](Altered_d17-val_attributes.yaml)

Here the same table applies as above, but with additional attributes:

| New ID     | Description                                                                                                      |
| ---------- | ---------------------------------------------------------------------------------------------------------------- |
| ***POCC*** | Partial Occlusion                                                                                                |
| ***FOCC*** | Full Occlusion (if more than about 70% of the object is occluded)                                                |
| ***COCC*** | Cross-Occlusion: One of the OoI occludes another OoI (Leading to Entropy contamination)                          |
| ***VLR***  | Very Low Resolution:                                                                                             |
| ***DIS***  | Distractors, object of the same class and very similar to the humain eye.                                        |
| ***ROT***  | In plane Rotation: Was actually in the original attributes but missing from the paper report.                    |
| ***CAM***  | Camouflage: When an object blends with the background, but is not occluded by the background.                    |
| ***PCAM*** | Partial Camouflage: When an object blends with the background, but is also partially occluded by it.             |
|  ***TS***  | Time-Skip: Time-Skip.                                                                                            |
