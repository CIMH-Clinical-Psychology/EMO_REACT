======================================================================
            IMPORTANT SETTINGS FOR RECORDER WORKSPACE
======================================================================


If you use a BrainCap MR with a BrainAmp MR, BrainAmp MR plus
or a BrainAmp ExG MR amplifier for simultaneous EEG & fMRI recordings, 
make the following settings in the BrainVision Recorder workspace:


----------------------------------------------------------------------
A. OPEN YOUR WORKSPACE AND SCAN FOR AMPLIFIERS



----------------------------------------------------------------------
B. SPECIFY NUMBER OF CHANNELS

   .....Number of Channels: set value according to BrainCap MR specifications
                            (e.g. 22, 32...)


   
----------------------------------------------------------------------
C. IMPORT ELECTRODE POSITIONS


  In the Workspace click on "Use Electrode Position File". 
  Select the appropriate BVEF-file provided with your BrainCap MR.

  (For more details refer to "BrainVision Recorder User Manual".)



----------------------------------------------------------------------
D. AMPLIFIER SETTINGS

   After importing the electrode position file, make the following settings 
   related to the amplifier: 

   .....Sampling Rate [Hz]: 5000
   .....Resolution [µV]:    0.5
   .....Range [+/- mv]:     use default
   .....Low Cutoff [s]:     10 OR DC
   .....High Cutoff [Hz]:   250


   .....Low Impedance [10 MOHm] for DC/MRplus

   --> DO NOT TICK, IF YOU ARE USING ONLY ONE AMPLIFIER MODEL

   --> DO TICK, IF YOU ARE USING DIFFERENT AMPLIFIER MODELS 
       WITH DIFFERENT INPUT IMPEDANCES (e.g. BrainAmp MR and BrainAmp DC/MR)



----------------------------------------------------------------------
E. CAP SETTINGS

   Please adhere to the BrainCap MR Datasheet (included as PDF)
   for the specific values.

   DO TICK
   .....Ground Series Resistor [kOhm]:    value see Datasheet
   .....Reference Series Resistor [kOhm]: value see Datasheet



----------------------------------------------------------------------
F. INDIVIDUAL SETTINGS

   Settings in the channel table. For BrainCap MR you must consider 
   the resistor of EEG and drop-down electrodes (if applicable):
	
   .....Series Resist. [kOhms]:	value see Datasheet
	
   ***NOTE***: Drop-down electrodes (ECG, EOG, EMG) may have different values.
               Details see datasheet of BrainCap MR.



======================================================================
FURTHER READING

     For more information and safe operation, please make sure to adhere to the 
     instructions in the following documents:

	- BrainCap MR Datasheet (included as PDF)

	- BrainAmp MR Operating Instructions for use in an MR environment 
	  (http://brainproducts.com/downloads.php?kid=5#dlukat_84 -> Manuals Hardware)

	- BrainVision Recorder User Manual 
	  (http://brainproducts.com/downloads.php?kid=5#dlukat_84 -> Manuals Software)
	
	- http://www.brainproducts.com/references2.php (EEG/fMRI publications)

	- http://www.brainproducts.com/productdetails.php?id=6 (articles from newsletters and brochures)



======================================================================
				END
======================================================================