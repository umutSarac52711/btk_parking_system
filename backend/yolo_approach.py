# backend/yolo_approach.py
from ultralytics import YOLO
import cv2
import torch
import numpy as np
from PIL import Image
import traceback
import sys
import os

# --- LPRNET INTEGRATION: Add LPRNet to the Python Path (Robust Version) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
lprnet_path = os.path.join(script_dir, 'LPRNet_Pytorch')

if not os.path.isdir(lprnet_path):
    raise FileNotFoundError(
        f"LPRNet directory not found. Please ensure the 'LPRNet_Pytorch' repository is cloned here: {lprnet_path}"
    )

if lprnet_path not in sys.path:
    sys.path.append(lprnet_path)

# --- LPRNET FIX: Use the correct 68-character list the model was trained on ---
# This is the primary fix for the RuntimeError. The number of characters
# must match the output layer size of the pre-trained weights file.
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O',
         '-'  # CTC blank character
         ]

def lprnet_decode(preds, char_list):
    """Decodes the output of LPRNet using a greedy approach."""
    preds = preds.cpu().detach().numpy()
    pred_labels = []
    
    for i in range(preds.shape[0]):
        pred = preds[i, :, :]
        p = np.argmax(pred, axis=1)
        
        pre_c = p[0]
        if pre_c != len(char_list) - 1:
             pred_labels.append(char_list[pre_c])
        
        for c in p:
            if pre_c == c or c == len(char_list) - 1:
                if c == len(char_list) - 1:
                    pre_c = c
                continue
            pred_labels.append(char_list[c])
            pre_c = c
            
    return ["".join(pred_labels)], [1.0]

# --- Model Imports ---
from fast_plate_ocr import LicensePlateRecognizer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from model.LPRNet import LPRNet

# --- Configuration ---
YOLO_MODEL_PATH = 'license_plate_detector.pt'
LPRNET_WEIGHTS_PATH = os.path.join(lprnet_path, 'LPRNet_Pytorch_weight.pth')

def load_all_models():
    """Loads and initializes all the models for the pipeline."""
    try:
        print("Loading all models...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        yolo_model = YOLO(YOLO_MODEL_PATH)
        ocr_model_fast = LicensePlateRecognizer('cct-xs-v1-global-model')
        
        trocr_model_id = 'DunnBC22/trocr-base-printed_license_plates_ocr'
        trocr_processor_id = 'microsoft/trocr-base-printed'
        # Added use_fast=True to silence the warning and use the faster tokenizer
        trocr_processor = TrOCRProcessor.from_pretrained(trocr_processor_id, use_fast=True)
        trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_model_id)
        trocr_model.to(device).eval()

        # LPRNet model initialization now uses len(CHARS) which is 68, matching the weights
        lprnet_model = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
        lprnet_model.to(device)
        # This line will now succeed
        lprnet_model.load_state_dict(torch.load(LPRNET_WEIGHTS_PATH, map_location=torch.device(device)))
        lprnet_model.eval()
        print(f"All models loaded successfully. Running on device '{device}'.")

        return {
            "yolo": yolo_model, "fast_ocr": ocr_model_fast,
            "trocr_processor": trocr_processor, "trocr_model": trocr_model,
            "lprnet_model": lprnet_model, "device": device
        }
        
    except Exception:
        print(f"--- FATAL ERROR LOADING MODELS ---")
        traceback.print_exc()
        return None

models = load_all_models()

def preprocess_for_lprnet(image: np.ndarray):
    """Preprocesses a cropped plate image for LPRNet."""
    image = cv2.resize(image, (94, 24))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = (np.transpose(np.float32(image), (2, 0, 1)) - 127.5) * 0.0078125
    return torch.from_numpy(image).unsqueeze(0)

def recognize_plate(image_path: str, display_windows: bool = False):
    """Performs end-to-end license plate recognition using a model ensemble."""
    if not models:
        print("Models are not loaded. Cannot process image.")
        return cv2.imread(image_path), None

    image = cv2.imread(image_path)
    if image is None: 
        print(f"Error: Could not read image at {image_path}")
        return None, None

    yolo_results = models["yolo"].predict(image, verbose=False)
    all_plate_detections = []

    for box in yolo_results[0].boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
        cropped_plate = image[y1:y2, x1:x2]
        if cropped_plate.size == 0: continue

        candidates = []
        
        # Expert 1: fast-plate-ocr
        try:
            preprocessed_fast = cv2.resize(cropped_plate, (128, 64))
            fast_results = models["fast_ocr"].run(preprocessed_fast)
            if fast_results:
                text, conf = fast_results[0]
                cleaned_text = "".join(c for c in text if c.isalnum()).upper()
                if cleaned_text:
                    candidates.append({"text": cleaned_text, "confidence": conf, "source": "FastOCR"})
        except Exception as e:
            print(f"FastOCR failed: {e}")

        # Expert 2: TrOCR
        try:
            pil_image = Image.fromarray(cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB))
            pixel_values = models["trocr_processor"](images=pil_image, return_tensors="pt").pixel_values.to(models["device"])
            generated_ids = models["trocr_model"].generate(pixel_values, output_scores=True, return_dict_in_generate=True)
            generated_text = models["trocr_processor"].batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]
            probs = [torch.softmax(s, dim=-1).max().item() for s in generated_ids.scores]
            confidence = np.mean(probs) if probs else 0.0
            cleaned_text = "".join(c for c in generated_text if c.isalnum()).upper()
            if cleaned_text:
                candidates.append({"text": cleaned_text, "confidence": float(confidence), "source": "TrOCR"})
        except Exception as e:
            print(f"TrOCR failed: {e}")

        # Expert 3: LPRNet
        try:
            lprnet_input = preprocess_for_lprnet(cropped_plate).to(models["device"])
            with torch.no_grad():
                preds = models["lprnet_model"](lprnet_input)
            
            probs = torch.softmax(preds, dim=2)
            max_probs, _ = torch.max(probs, dim=2)
            decoded_text, _ = lprnet_decode(preds, CHARS)
            
            if decoded_text:
                cleaned_text = decoded_text[0].replace('-', '')
                confidence = max_probs[0, :len(cleaned_text)].mean().item() if cleaned_text else 0.0
                if cleaned_text:
                     candidates.append({"text": cleaned_text, "confidence": confidence, "source": "LPRNet"})
        except Exception as e:
            print(f"LPRNet failed: {e}")

        if candidates:
            best_candidate = max(candidates, key=lambda c: c['confidence'])
            print(f"  -> Ensemble results for a detected plate:")
            for c in candidates:
                print(f"    - {c['source']:<8}: '{c['text']}' (Conf: {c['confidence']:.3f})")
            
            all_plate_detections.append({
                "text": best_candidate['text'], "confidence": best_candidate['confidence'],
                "bbox": (x1, y1, x2, y2)
            })

    if not all_plate_detections:
        print("No license plates were successfully recognized.")
        return image, None

    best_overall_detection = max(all_plate_detections, key=lambda d: d['confidence'])
    final_image = image.copy()
    b = best_overall_detection['bbox']
    text, conf_str = best_overall_detection['text'], f"{best_overall_detection['confidence']:.2f}"
    cv2.rectangle(final_image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
    cv2.putText(final_image, f"{text} ({conf_str})", (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    if display_windows:
        cv2.namedWindow("Final Result (Ensemble)", cv2.WINDOW_NORMAL)
        cv2.imshow("Final Result (Ensemble)", final_image)
        print("\nPress any key to close the image window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return final_image, best_overall_detection


if __name__ == "__main__":
    test_images = ['car_image_california.jpg', 'dataset/car license plate close up/image_2.jpg']
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"\n--- SKIPPING: Image not found at '{img_path}' ---")
            continue
        
        print(f"\n--- PROCESSING IMAGE: {img_path} ---")
        final_image, result = recognize_plate(img_path, display_windows=True)
        
        if result:
            print(f"\n--- SUCCESS ---")
            print(f"Best Recognized Plate Text: {result['text']}")
            print(f"Confidence: {result['confidence']:.3f}")
        else:
            print("\n--- FAILURE: No plate could be read. ---")