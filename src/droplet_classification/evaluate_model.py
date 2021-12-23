import tensorflow as tf
import numpy as np
import cv2
import math

def evaluate_model(image_path, droplets, model_path):
    if not hasattr(evaluate_model, "cached_model_path") or evaluate_model.cached_model_path != model_path:
        evaluate_model.cached_model_path = model_path
        evaluate_model.cached_model = tf.keras.models.load_model(model_path)

    test_img = cv2.imread(image_path)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    per_droplet_images = []

    for droplet in droplets:
        x, y = droplet.get_position()
        r = droplet.get_radius()

        height, width = test_img.shape[:2]

        if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
            continue

        x_min = math.floor(max(x - r, 0))
        x_max = math.ceil(min(x + r, width))
        y_min = math.floor(max(y - r, 0))
        y_max = math.ceil(min(y + r, height))

        droplet = test_img[y_min:y_max, x_min:x_max]
        droplet = cv2.resize(droplet, (224,224))
        
        mask = np.zeros_like(droplet)
        mask = cv2.circle(mask, (112, 112), 112, (255,255,255), -1)
        masked_droplet = cv2.bitwise_and(droplet, mask)
    
        masked_droplet = np.array(masked_droplet, dtype=np.float32)

        # Normalise the images
        masked_droplet /= 255

        masked_droplet_3 = np.zeros((1, 224, 224, 3))
        masked_droplet_3[0, :, :, 0] = masked_droplet
        masked_droplet_3[0, :, :, 1] = masked_droplet
        masked_droplet_3[0, :, :, 2] = masked_droplet

        per_droplet_images.append(masked_droplet_3)

    predictions = evaluate_model.cached_model.predict(np.vstack(per_droplet_images))

    return predictions
