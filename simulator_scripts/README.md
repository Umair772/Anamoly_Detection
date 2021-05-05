How to run inference:
- ./sim_run graph.vnnx PATH_TO_DATA
- PATH_TO_DATA could be something like data/normal/<image>.jpg where <image> is the filename of an image

How to run inference over the test set:
- python3 check_all.py graph.vnnx data/normal/ data/abnormal/
