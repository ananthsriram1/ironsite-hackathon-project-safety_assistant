# ironsite-hackathon-project-safety_assistant
A Unity-Based Demo for Ironsite's UMD Hackathon. We're working on a real-time safety assistant using object detection and VLM Softwares

**_Problem Statement_**:
  - In 2023, U.S. construction workers experienced approximately 1,075 fatal injuries on the job.
  - The BLS (Bureau of Labor Statistics) measured a fatal work injury rate of 9.6 deaths per 100,000 full-time equivalent workers, the same rate as a year ago.
  - The majority of those who died at work in the U.S. were men (91.5%), and women accounted for 8.5% of all workplace fatalities.
  - Construction had the most fatalities among all industry sectors in 2023.

**_Approach_**: 
* Unity scene will be a reconstruction of a construction site
* The Unity demo will have potential problems, such as a dangerous object in the blind spot, a dangerous object moving towards you
* The model should be able to identify dangerous items and map that to a coordinate
* Model should be able to alert users if it identifies potential threats
* Model should log safety violations 
