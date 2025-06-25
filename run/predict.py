from ultralytics import YOLO
from PIL import Image

# Load a pretrained YOLO model (change the path to your model)
model = YOLO(r"E:\runs\detect\train\weights\best.pt")

results = model.predict(r"E:\Project\Object_Detection_Biensoxe\a.png", imgsz=416)

# Print the results
for r in results:
    print(r.boxes)  # Print the bounding boxes
    im_array = r.plot()  # PLot a BRG numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1]) #RGB PIL Image
    im.show()
    im.save("results_image.png")
