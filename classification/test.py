from transformers import BitImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

config = {
  "crop_size": {
    "height": 224,
    "width": 224
  },
  "do_center_crop": True,
  "do_convert_rgb": True,
  "do_normalize": True,
  "do_rescale": True,
  "do_resize": True,
  "image_mean": [
    0.485,
    0.456,
    0.406
  ],
  "image_processor_type": "BitImageProcessor",
  "image_std": [
    0.229,
    0.224,
    0.225
  ],
  "resample": 3,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "shortest_edge": 256
  }
}

processor = BitImageProcessor(**config)
model = AutoModelForImageClassification.from_pretrained('facebook/dinov2-large-imagenet1k-1-layer')

inputs = processor(images=image, return_tensors="pt")
print(inputs['pixel_values'].shape, inputs['pixel_values'].mean(), inputs['pixel_values'].std())
outputs = model(**inputs)
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
print('logits', logits.shape)
