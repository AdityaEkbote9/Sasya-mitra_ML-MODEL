
from roboflow import Roboflow
# the rest of your code...
from roboflow import Roboflow
rf = Roboflow(api_key="MaWsteSDUfQKP0LeaF5z")
project = rf.workspace("work-l7gxf").project("potato-disease-ivsbp")
version = project.version(1)
dataset = version.download("yolov11")

                