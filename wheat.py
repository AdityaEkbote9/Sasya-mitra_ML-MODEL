from roboflow import Roboflow
# the rest of your code...

from roboflow import Roboflow
rf = Roboflow(api_key="MaWsteSDUfQKP0LeaF5z")
project = rf.workspace("mibrahim").project("wheat-disease-gt3yi")
version = project.version(3)
dataset = version.download("yolov11")
                