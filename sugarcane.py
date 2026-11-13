
from roboflow import Roboflow
# the rest of your code...
from roboflow import Roboflow
rf = Roboflow(api_key="MaWsteSDUfQKP0LeaF5z")
project = rf.workspace("css1812arasaka").project("sugarcane-leaf-disease-dataset-p6fnb")
version = project.version(4)
dataset = version.download("yolov11")
                