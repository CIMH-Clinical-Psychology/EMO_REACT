## Emo React

Sleep plays an important role in memory. More specifically sleep is closely linked to memory reprocessing, and specificities such as the emotional content of the memories can have an impact on the reprocessing during sleep. Literature showed that the emotional events tend to be remembered better than neutral events. M. Schönauer, S. Alizadeh, H. Jamalabadi, A. Abraham, A. Pawlizki & S. Gais (2017) showed that memory reprocessing occurs in different stages of sleep (REM and NREM) and that each one of these specific sleep stages pertains to different aspects of the consolidation process. We will us a similar approach to answer questions related to memory emotional reprocessing. More specifically we are interested on the impact of emotional images on the memory reprocessing during sleep and will contrast them with neutral images.

## Project sketch
<img width="1163" alt="Screenshot 2023-02-10 at 10 42 17" src="https://user-images.githubusercontent.com/15683682/218058562-ba3639c7-0b71-4f0a-8e59-9d4794a2d0a1.png">

The experiment has the following setup

**Experiment part 1**:

    - Adaptation night, dummy recording to get participant used to the sleep lab

**Experiment part 2 (one week later)**:

1. participant sees 144 different images. Each image is placed in a specific corner of the screen. The task for the participant is to learn the mapping of object to corner. Depending on the randomization, all images of this night are either emotional highly arousing, or emotionally low arousing. The images are taken from the OASIS dataset which includes ratings of 100 people for each image according to their valence and arousal intensity.
2. part 1 is repeated 8 times
3. The participants memory is tested. 144 images are shown, of which 96 are images learned in 1) and 48 novel images are shown. The participant has to enter whether they have seen the image before or not, and if yes, which quadrant/corner the image was shown
4. The participant sleeps in the sleep lab for the full night
5. In the morning, the same test as in 3) is performed, again with 96 images learned in 1) and 48 novel images
6. A localizer is shown in which the participant sees 96 images, of which 48 are emotionally neutral, 24 are of negative valence and 24 are of positive valence. For each image, the participant is asked for a valence rating (positiv <-> negative on a skale 1-5) and arousal intensity rating (high<>lowon a scale 1-5).

**Experiment part 3 (another week later)**:

1. same procedure as in Experiment part 2, but instead of neutral images, the participant learns emotionally highly arousing images, if they have learned low arousing images in part 2 and vice versa.

**Diagram of Image Partition in the  Experiment
![emo_react_pics-Page-2 drawio](https://user-images.githubusercontent.com/15683682/218058901-c53c8475-27a6-42f9-86a0-2519a07eed21.png)



## Goals of this project are:

* classify emotional categories based on 64-channel EEG data during the localizer
  * can we predict arousal rating based on EEG?
    * as targets either subjective ratings, or 'ground truth' ratings from OASIS
  * can we predict arousal rating based on EEG?
  * as targets either subjective ratings, or 'ground truth' ratings from OASIS
* Using a classifier trained on the localizer (similar approach to Schreiner et al. 2020)
  * can we predict memory reprocessing during sleep?
    
    
* Using a classifier trained on the EEG of the night
  * can we separate high emotionally arousal nights from low emotionally arousal nights

### Classification approaches

The following approaches seem worthy of trying out:

- [ ] xxxx
- [ ] [Sensors | Free Full-Text | Predicting Exact Valence and Arousal Values from EEG](https://www.mdpi.com/1424-8220/21/10/3414)

Of help could be the following Python projects:

- pyeeg
- coffeine
- pyRiemann
- 
