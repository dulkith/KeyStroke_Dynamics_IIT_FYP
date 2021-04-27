# KeyStroke_Dynamics_IIT_FYP

Remote authentication service using keystroke dynamics @IIT_FYP_2021
Research Project (IIT Sri Lanka)

Protype Application
Platforms : GNS3

Abstract:
Cybercrimes are on the rise today because there is a growing need to increase the security of user accounts. Nowadays, many systems authenticate users only when they log in or enter the system, and most accounts are used single-factor authentication, making the account more vulnerable to attacks. This thesis is focused on how keystroke dynamics are used as a security method for authentication as a measure to reduce cyberattacks. By extracting data on a user’s typing biometrics on a keyboard, the authentication system is able to recognize them as the legitimate user and therefore authorize the Access to adevice, a website or an application. Respectively, it blocks the authorization if the authentication system judges the user to be an intruder. While currently being an authentication system used on local machines, this tool has not yet been commonly researched or implemented over the network. Also this paper discusses the observation of the impact of network simulations on different machine learning keystroke authentication models. It covers both a reimplementation of key dynamics authentication on a local host machine with a stable network, and an evaluation of this authentication tool over a network under different unstable conditions.
  
Key Words:
Remote Authentication, Keystroke Dynamics, Remote Authentication, Typing Biometrics

Installation:
Install python3 (or venv)
Install the packages with 
  pip install -r requirements.txt
Run the main python file 
  python main.py

Folders & Files Structure:
In src/
  In datasets/ is the used datasets, including the open dataset from Killourhy and Maxion
  In exports/ are exported graph from scripts and notebooks
  In network_gns3/ are the scripts to send and receive data, datasets that we generated from our simulations
  stats_jitter.py & stats_latency.py are the statistics about results
  Nearest_Neighbor.ipynb & Neural_Network.ipynb are the jupiter/colab notebooks
  main.py is the main script calling Subject.py and DetectorNearestNeighbor.py & DetectorNeuralNetwork.py