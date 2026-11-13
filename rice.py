from roboflow import Roboflow
# the rest of your code...

from roboflow import Roboflow
rf = Roboflow(api_key="MaWsteSDUfQKP0LeaF5z")
project = rf.workspace("saad-sahriar-bomj8").project("rice-leaf-disease-a9v6f")
version = project.version(2)
dataset = version.download("yolov11")
                