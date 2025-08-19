# Is there a correlation with the Performance (IoU) and the Entropy ($H(X)$)?

## For DAVIS-2017 (validation)

### Graphics, IoU wrt to frame idx and $H$ wrt to frame idx
- *blue curve* is the average IoU - without background.  
- *red curve* is the total $H$ for the frame - simply the total sum.  
- We stack the TP, TN, FP anf FN on the second plot, again the sum. But as there might be more than one objects, we are also stacking the corresponding $H$.
- Do an analysis per object?

Observations/Conclusions:
    - Some *very* interesting sequences are: drift-chicane, drift-straight, india, lab-coat and mbike-trick. - Mostly chaos ?
    - Some that show a clear negative correlation :
    - Some that show a positive correlation :
  
Note that a High entropy for TP and TN, is not a good sign, as this mean that the tracker was unsure whetever to classify it corretly or not... 
---

- bike-packing
  ![](../temps/bike-packing.png)
- blackswan
  ![](../temps/blackswan.png)
- bmx-trees
![](../temps/bmx-trees.png)
- breakdance
  ![](../temps/breakdance.png)
- camel
  ![](../temps/camel.png)
- car-roundabout
  ![](../temps/car-roundabout.png)
- car-shadow
  ![](../temps/car-shadow.png)
- cows
  ![](../temps/cows.png)
- dance-twirl
  ![](../temps/dance-twirl.png)
- dog
  ![](../temps/dog.png)
- dogs-jump
  ![](../temps/dogs-jump.png)
- drift-chicane
  ![](../temps/drift-chicane.png)
- drift-straight
  ![](../temps/drift-straight.png)
- goat
  ![](../temps/goat.png)
- gold-fish
  ![](../temps/gold-fish.png)
- horsejump-high
  ![](../temps/horsejump-high.png)
- india
  ![](../temps/india.png)
- judo
  ![](../temps/judo.png)
- kite-surf
  ![](../temps/kite-surf.png)
- lab-coat
  ![](../temps/lab-coat.png)
- libby
  ![](../temps/libby.png)
- loading
  ![](../temps/loading.png)
- mbike-trick
  ![](../temps/mbike-trick.png)
- motocross-jump
  ![](../temps/motocross-jump.png)
- paragliding-launch
  ![](../temps/paragliding-launch.png)
- parkour
  ![](../temps/parkour.png)
- pigs
  ![](../temps/pigs.png)
- scooter-black
  ![](../temps/scooter-black.png)
- shooting
  ![](../temps/shooting.png)
- soapbox
  ![](../temps/soapbox.png)

---

### Correlation graphics
We Display the Spearman's correlation coefficient function between the IoU and $H$.  
**5 different graphics are displayed:** 
    - The first one is by only looking at the total $H$ and the average IoU of the sequence.
    - The other 4 graphcics are specifics to either to only capture the $H$ for TP, TN, FP and FN regions.

A simple observation with the global graphics, shows no clear correlation...
But looking at the False Positive graphic, there is clearly a negative correlation between IoU and $H$.

Observations/Conclusions:
    - Is the entropy reliable to predict the failure of the trackerß Kinda, looking at the FP and the FN.
    - Taking the complete $H$ can be missleading, with sometimes strong positive correlation between the IoU and $H$.
      - This mostly due to the sum, where the information is drowned by the noise.
      - Analyse with an alternative approach (ignore values under a threshold?) might the most logical
      - Or focus on the borders of the objects, but this would exlude most of the epistemic information...
      - Or perhaps process the information differently
    - Need further testing with the cases A,B,C,D,E to examine how the entropy fluctuates unders this cases.

During this evaluation we discard the IoU and $H$ of the frame 0, but also of the last frame
to be consistent with the evaluation proposed in DAVIS. 0 makes sense, but why the last frame :hushed: ? 

Some statistics:

| Collected $H$ | Mean | Std | Median |
| ---- | ---- | ---- | ---- |
| Global | -0.0622 | 0.5191 | -0.1194 |
| TP | 0.0765 | 0.5082 | -0.0185 |
| TN | -0.0463 | 0.5046 | -0.1085 |
| FP | -0.2805 | 0.4542 | -0.3526 |
| FN | -0.2416 | 0.5017 | -0.2167 |


Spearman's correlation with respect to the sequence.  



- Global 
  ![](../temps/spearman_Correlation_Global.png)
- True Positive
  ![](../temps/spearman_Correlation_TP.png)
- True Negative
  ![](../temps/spearman_Correlation_TN.png)
- False Postitive
  ![](../temps/spearman_Correlation_FP.png)
- False Negative
  ![](../temps/spearman_Correlation_FN.png)