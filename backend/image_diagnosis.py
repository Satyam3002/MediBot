import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import os
from typing import Dict, Tuple, Optional
import logging

class ImageDiagnosisSystem:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load X-ray model
        self.xray_model_path = os.path.join(os.path.dirname(__file__), "saved_xray_model")
        self.xray_processor = AutoImageProcessor.from_pretrained(self.xray_model_path)
        self.xray_model = AutoModelForImageClassification.from_pretrained(self.xray_model_path).to(self.device)

        # Load Skin model
        self.skin_model_path = os.path.join(os.path.dirname(__file__), "saved_skin_model")
        self.skin_processor = AutoImageProcessor.from_pretrained(self.skin_model_path)
        self.skin_model = AutoModelForImageClassification.from_pretrained(self.skin_model_path).to(self.device)

        # Condition mappings (expand as needed)
        self.xray_condition_mapping = {
            "normal": {
                "name": "Normal Chest X-ray",
                "details": "No significant abnormalities detected in the chest X-ray.",
                "recommendations": "• Continue regular check-ups\n• Maintain healthy lifestyle",
                "when_to_see_doctor": "If you experience any symptoms or have concerns."
            },
            # Add more mappings as needed
        }
        self.skin_condition_mapping = {
            "melanoma": {
                "name": "Melanoma (Skin Cancer)",
                "details": "Melanoma is a serious form of skin cancer that requires prompt medical attention.",
                "recommendations": "• Consult a dermatologist immediately\n• Avoid sun exposure\n• Follow prescribed treatment",
                "when_to_see_doctor": "Immediately, if you notice new or changing skin lesions."
            },
            "nevus": {
                "name": "Nevus (Mole)",
                "details": "A nevus is a common mole, usually benign but should be monitored for changes.",
                "recommendations": "• Monitor for changes in size, color, or shape\n• Regular skin checks",
                "when_to_see_doctor": "If you notice changes or have concerns."
            },
            # Add more mappings as needed
        }

    def preprocess_image(self, image_path: str, model_type: str) -> Optional[Dict]:
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if model_type == "xray":
                return self.xray_processor(images=image, return_tensors="pt").to(self.device)
            elif model_type == "skin":
                return self.skin_processor(images=image, return_tensors="pt").to(self.device)
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            return None

    def get_diagnosis(self, image_path: str, model_type: str) -> Tuple[str, float, Dict]:
        """
        model_type: 'xray' or 'skin'
        """
        try:
            inputs = self.preprocess_image(image_path, model_type)
            if inputs is None:
                return "Error", 0.0, {"error": "Failed to process image"}

            if model_type == "xray":
                model = self.xray_model
                mapping = self.xray_condition_mapping
            elif model_type == "skin":
                model = self.skin_model
                mapping = self.skin_condition_mapping
            else:
                return "Error", 0.0, {"error": "Invalid model type"}

            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidence, predicted_class = torch.max(probabilities, dim=-1)
                confidence = confidence.item()
                predicted_class = predicted_class.item()
                condition_name = model.config.id2label[predicted_class]
                details = mapping.get(
                    condition_name.lower(),
                    {
                        "name": condition_name,
                        "details": f"Condition detected: {condition_name}",
                        "recommendations": "Please consult a healthcare professional for proper diagnosis.",
                        "when_to_see_doctor": "Schedule an appointment with your doctor."
                    }
                )
                return condition_name, confidence, details
        except Exception as e:
            self.logger.error(f"Error getting diagnosis: {str(e)}")
            return "Error", 0.0, {"error": str(e)}

    def validate_image(self, image_path: str) -> Tuple[bool, str]:
        try:
            if not os.path.exists(image_path):
                return False, "Image file not found"
            if os.path.getsize(image_path) > 10 * 1024 * 1024:
                return False, "Image file too large (max 10MB)"
            image = Image.open(image_path)
            if image.size[0] < 100 or image.size[1] < 100:
                return False, "Image resolution too low"
            return True, "Image valid"
        except Exception as e:
            return False, f"Invalid image: {str(e)}"

# Singleton instance
image_diagnosis_system = ImageDiagnosisSystem() 