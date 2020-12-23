For the course project, we implemented four experiments here.
1. Caltech dataset  python caltech_fed.py
2. Domain net dataset python domain_fed.py
3. celebA dataset python face_fed.py
4. Digit dataset python main_fed.py


Utils folder: contains script for each experiment setting, e.g., the batch_size, noise_scale (larger noise, stronger privacy guarantee).

Network_model folder: the library of network backbone, e.g., alexnet, resnet50m ...


Data and Dataset folder: the setup of federated learning environment (instantiations of local agents and server using the original dataset. details can be found in the report)


metric: privacy and accuracy, runing ``python main_fed.py'' will output privacy and accuracy results.

For a typtical differentially private federated algorithm:
suppose we have n agents, samplong probability is q, noise scale is sigma
 
For i in communication rounds:
    sample agents with probability q (roughly nq agents in each communication round).
    Each agent performs E iterations of local updates (SGD).
    Each agent upload her updates (combined with noise sigma) to the server 
    the server aggregates all updates and distribute the server model to each agent

The program ouputs the privacy guarantee  and model accuracy after |communication rounds|
