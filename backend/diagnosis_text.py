from datasets import load_dataset, concatenate_datasets
import os
from collections import defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import numpy as np

class DiagnosisSystem:
    def __init__(self):
        # Load and prepare datasets
        self.dataset1 = load_dataset("dux-tecblic/symptom-disease-dataset")
        self.dataset2 = load_dataset("prognosis/symptoms_disease_v1")
        self.dataset3 = load_dataset("QuyenAnhDE/Diseases_Symptoms")
        self.dataset4 = load_dataset("gretelai/symptom_to_diagnosis")
        self.dataset5 = load_dataset("fhai50032/Symptoms_to_disease_7k")
        self.dataset6 = load_dataset("Dhananjay-25/symptoms-disease_dataset_for_LLM")
        self.dataset7 = load_dataset("celikmus/mayo_clinic_symptoms_and_diseases_v1")

        # Clean datasets
        self.cleaned1 = self.dataset1["train"].map(self._clean_dataset1, remove_columns=self.dataset1["train"].column_names)
        self.cleaned2 = self.dataset2["train"].map(self._clean_dataset2, remove_columns=self.dataset2["train"].column_names)
        self.cleaned3 = self.dataset3["train"].map(self._clean_dataset3, remove_columns=self.dataset3["train"].column_names)
        self.cleaned4 = self.dataset4["train"].map(self._clean_dataset4, remove_columns=self.dataset4["train"].column_names)
        self.cleaned5 = self.dataset5["train"].map(self._clean_dataset5, remove_columns=self.dataset5["train"].column_names)
        self.cleaned6 = self.dataset6["train"].map(self._clean_dataset6, remove_columns=self.dataset6["train"].column_names)
        self.cleaned7 = self.dataset7["train"].map(self._clean_dataset7, remove_columns=self.dataset7["train"].column_names)

        # Combine all datasets
        self.all_data = concatenate_datasets([
            self.cleaned1, self.cleaned2, self.cleaned3, 
            self.cleaned4, self.cleaned5, self.cleaned6, self.cleaned7
        ])
        
        # Convert to list for faster access
        self.all_data_list = self.all_data.to_list()

        # Create a symptom-disease mapping from the datasets
        self.symptom_disease_map = defaultdict(list)
        self._build_symptom_disease_map()

        # Initialize ML model
        self.ml_model = None
        self.vectorizer = None
        self._train_ml_model()

        # Common symptom patterns and their corresponding conditions
        self.symptom_patterns = {
            r'fever|temperature|hot': {
                'condition': 'fever',
                'details': 'Fever is your body\'s natural response to infection or illness.',
                'recommendations': '• Rest and stay hydrated\n• Use fever reducers if necessary\n• Monitor temperature regularly\n• Keep the room temperature comfortable',
                'prevention': '• Practice good hygiene\n• Get adequate rest\n• Maintain a healthy diet\n• Stay hydrated\n• Avoid exposure to sick people',
                'when_to_see_doctor': 'If fever persists for more than 3 days or exceeds 103°F (39.4°C).'
            },
            r'headache|migraine': {
                'condition': 'headache',
                'details': 'Headaches can be caused by various factors including stress, tension, or underlying conditions.',
                'recommendations': '• Rest in a quiet, dark room\n• Stay hydrated\n• Consider over-the-counter pain relievers\n• Apply cold or warm compress',
                'prevention': '• Maintain regular sleep patterns\n• Stay hydrated\n• Manage stress\n• Avoid triggers\n• Practice good posture',
                'when_to_see_doctor': 'If headaches are severe, frequent, or accompanied by other symptoms like fever or vision changes.'
            },
            r'back pain|lower back': {
                'condition': 'back pain',
                'details': 'Lower back pain can be caused by muscle strain, poor posture, or underlying conditions.',
                'recommendations': '• Rest and avoid heavy lifting\n• Apply ice or heat\n• Consider over-the-counter pain relievers\n• Practice gentle stretching',
                'prevention': '• Maintain good posture\n• Exercise regularly\n• Use proper lifting techniques\n• Stay active\n• Maintain a healthy weight',
                'when_to_see_doctor': 'If pain is severe, persistent, or accompanied by other symptoms like numbness or weakness.'
            },
            r'knee|knee pain': {
                'condition': 'knee pain',
                'details': 'Knee pain can result from injuries, arthritis, or overuse.',
                'recommendations': '• Rest the knee\n• Apply ice\n• Use compression\n• Elevate the leg\n• Consider over-the-counter pain relievers',
                'prevention': '• Maintain a healthy weight\n• Exercise regularly\n• Use proper form during activities\n• Wear appropriate footwear\n• Warm up before exercise',
                'when_to_see_doctor': 'If pain is severe, persists for more than a few days, or is accompanied by swelling or difficulty walking.'
            },
            r'fatigue|tired|exhausted': {
                'condition': 'fatigue',
                'details': 'Fatigue can be caused by various factors including lack of sleep, stress, or underlying medical conditions.',
                'recommendations': '• Get adequate rest\n• Stay hydrated\n• Maintain a balanced diet\n• Exercise regularly\n• Manage stress',
                'prevention': '• Maintain regular sleep schedule\n• Eat a balanced diet\n• Exercise regularly\n• Manage stress\n• Stay hydrated',
                'when_to_see_doctor': 'If fatigue persists for more than two weeks or is accompanied by other symptoms.'
            }
        }

        # Add comprehensive condition mapping
        self.condition_mapping = {
            # Dataset 1 mappings (numeric IDs)
            "151": "Jaundice",
            "308": "Migraine",
            "149": "Viral Upper Respiratory Infection",
            "768": "Angina",
            "101": "Fever",
            "212": "Allergic Rhinitis",
            "305": "Gastroenteritis",
            "872": "Cardiovascular Condition",
            "404": "Viral Infection",
            "509": "Diabetes",
            "612": "Anxiety Disorder",
            "123": "Dehydration",
            "321": "Respiratory Tract Infection",
            "111": "Common Cold",
            "777": "Unknown Condition",
            "513": "Asthma",
            "200": "Food Poisoning",
            "266": "Urinary Tract Infection",
            "314": "Migraine",
            "401": "Anemia",
            "688": "Hormonal Imbalance",
            "809": "Allergic Reaction",
            "902": "Bronchitis",
            "999": "Severe Condition",

            # Dataset 2-7 mappings (text-based)
            "hypertensive disease": "Hypertension",
            "panic disorder": "Panic Disorder",
            "cervical spondylosis": "Cervical Spondylosis",
            "jaundice": "Jaundice",
            "fungal infection": "Fungal Infection",
            "sun-allergy": "Sun Allergy"
        }

        # Add detailed condition information
        self.condition_details = {
            "Jaundice": {
                'details': 'Jaundice is a condition where the skin and whites of the eyes turn yellow due to high bilirubin levels.',
                'recommendations': '• Seek immediate medical attention\n• Stay hydrated\n• Avoid alcohol\n• Get liver function tests\n• Follow a balanced diet',
                'prevention': '• Maintain a healthy diet\n• Avoid excessive alcohol\n• Get vaccinated for hepatitis\n• Practice safe food handling\n• Regular health check-ups',
                'when_to_see_doctor': 'Seek immediate medical attention as jaundice can indicate serious liver problems.'
            },
            "Hypertension": {
                'details': 'Hypertension (high blood pressure) is a condition where the force of blood against artery walls is too high.',
                'recommendations': '• Monitor blood pressure regularly\n• Take prescribed medications\n• Reduce salt intake\n• Exercise regularly\n• Maintain a healthy weight',
                'prevention': '• Regular exercise\n• Healthy diet (DASH diet)\n• Limit alcohol\n• Quit smoking\n• Manage stress',
                'when_to_see_doctor': 'If blood pressure readings are consistently high or if you experience symptoms like severe headache or chest pain.'
            },
            "Panic Disorder": {
                'details': 'Panic disorder is characterized by recurrent panic attacks and persistent worry about having more attacks.',
                'recommendations': '• Practice deep breathing exercises\n• Consider therapy\n• Take prescribed medications\n• Regular exercise\n• Maintain a regular sleep schedule',
                'prevention': '• Regular exercise\n• Stress management\n• Avoid caffeine and alcohol\n• Regular sleep schedule\n• Therapy or counseling',
                'when_to_see_doctor': 'If panic attacks are frequent, severe, or interfering with daily life.'
            },
            "Cervical Spondylosis": {
                'details': 'Cervical spondylosis is a general term for age-related wear and tear affecting the spinal disks in your neck.',
                'recommendations': '• Physical therapy\n• Pain medication\n• Neck exercises\n• Proper posture\n• Use of neck support',
                'prevention': '• Maintain good posture\n• Regular exercise\n• Ergonomic workspace setup\n• Avoid neck strain\n• Regular stretching',
                'when_to_see_doctor': 'If you experience severe pain, numbness, or weakness in arms or legs.'
            },
            "Fungal Infection": {
                'details': 'Fungal infections are common skin conditions caused by various types of fungi.',
                'recommendations': '• Use antifungal medications\n• Keep affected areas clean and dry\n• Avoid sharing personal items\n• Wear breathable clothing\n• Change clothes regularly',
                'prevention': '• Keep skin clean and dry\n• Avoid sharing personal items\n• Wear appropriate footwear\n• Use antifungal powder\n• Regular hygiene practices',
                'when_to_see_doctor': 'If the infection spreads, doesn\'t improve with treatment, or if you have a weakened immune system.'
            },
            "Sun Allergy": {
                'details': 'Sun allergy (polymorphic light eruption) is a condition where the skin reacts to sunlight exposure.',
                'recommendations': '• Use sunscreen regularly\n• Wear protective clothing\n• Avoid peak sun hours\n• Use antihistamines if prescribed\n• Stay in shade when possible',
                'prevention': '• Regular sunscreen use\n• Protective clothing\n• Gradual sun exposure\n• Avoid peak sun hours\n• Stay hydrated',
                'when_to_see_doctor': 'If symptoms are severe, persistent, or if you develop blisters or fever.'
            }
        }

        # Emergency conditions that require immediate attention
        self.emergency_conditions = {
            r'blood.*vomit|vomit.*blood|hematemesis': {
                'condition': 'Gastrointestinal Bleeding',
                'details': 'Vomiting blood (hematemesis) is a serious medical emergency that requires immediate attention.',
                'recommendations': '⚠️ EMERGENCY: Seek immediate medical attention. Do not wait.\n• Call emergency services or go to the nearest emergency room\n• Do not eat or drink anything\n• Stay calm and try to rest while waiting for help',
                'prevention': '• Regular check-ups with your doctor\n• Avoid alcohol and NSAIDs if you have a history of ulcers\n• Maintain a healthy diet\n• Manage stress',
                'when_to_see_doctor': 'EMERGENCY: Seek immediate medical attention. This is a life-threatening condition.'
            },
            r'chest.*pain|heart.*pain|angina': {
                'condition': 'Chest Pain',
                'details': 'Chest pain can be a sign of a serious heart condition and requires immediate medical attention.',
                'recommendations': '⚠️ EMERGENCY: Seek immediate medical attention.\n• Call emergency services\n• Sit down and try to stay calm\n• Take any prescribed heart medication if available',
                'prevention': '• Regular heart check-ups\n• Maintain a healthy lifestyle\n• Exercise regularly\n• Manage stress',
                'when_to_see_doctor': 'EMERGENCY: Seek immediate medical attention. This could be a heart attack.'
            },
            r'severe.*headache|worst.*headache|sudden.*headache': {
                'condition': 'Severe Headache',
                'details': 'A sudden, severe headache could indicate a serious condition like a stroke or aneurysm.',
                'recommendations': '⚠️ EMERGENCY: Seek immediate medical attention.\n• Call emergency services\n• Note the time when symptoms started\n• Stay in a safe position',
                'prevention': '• Regular check-ups\n• Manage blood pressure\n• Avoid triggers\n• Stay hydrated',
                'when_to_see_doctor': 'EMERGENCY: Seek immediate medical attention. This could be a stroke or other serious condition.'
            }
        }

    def _train_ml_model(self):
        """Train the ML model on the dataset"""
        try:
            # Prepare training data
            symptoms = [entry["symptoms"] for entry in self.all_data_list]
            diagnoses = [entry["diagnosis"] for entry in self.all_data_list]

            # Create and train the model
            self.ml_model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('clf', MultinomialNB())
            ])
            
            self.ml_model.fit(symptoms, diagnoses)
            
            # Save the model
            joblib.dump(self.ml_model, 'diagnosis_model.joblib')
            print("ML model trained and saved successfully!")
            
        except Exception as e:
            print(f"Error training ML model: {str(e)}")
            self.ml_model = None

    def _predict_with_ml(self, user_input):
        """Get prediction using the ML model"""
        if self.ml_model is None:
            return None
            
        try:
            # Get prediction
            prediction = self.ml_model.predict([user_input])[0]
            # Get prediction probability
            proba = self.ml_model.predict_proba([user_input])[0]
            confidence = np.max(proba)
            
            # Special handling for jaundice symptoms
            jaundice_keywords = ['yellow', 'jaundice', 'bilirubin', 'liver', 'hepatitis']
            if any(keyword in user_input.lower() for keyword in jaundice_keywords):
                return "151", confidence  # Map to jaundice ID
            
            if confidence > 0.2:  # Lowered threshold to 20% for more predictions
                return prediction, confidence
         
        except Exception as e:
            print(f"Error in ML prediction: {str(e)}")
            return None

    def _build_symptom_disease_map(self):
        """Build a mapping of symptoms to possible diseases from the datasets"""
        for entry in self.all_data_list:
            symptoms = entry["symptoms"].lower()
            diagnosis = entry["diagnosis"]
            
            # Extract individual symptoms
            symptom_words = set(symptoms.split())
            for symptom in symptom_words:
                if len(symptom) > 3:  # Only consider words longer than 3 characters
                    self.symptom_disease_map[symptom].append(diagnosis)

    def _get_matching_pattern(self, user_input):
        """Find the best matching symptom pattern"""
        user_input = user_input.lower()
        best_match = None
        max_matches = 0

        for pattern, info in self.symptom_patterns.items():
            matches = len(re.findall(pattern, user_input))
            if matches > max_matches:
                max_matches = matches
                best_match = info

        return best_match

    def _get_condition_details(self, diagnosis):
        """Get detailed information for a specific condition"""
        # First try to map numeric IDs to condition names
        condition_name = self.condition_mapping.get(str(diagnosis), diagnosis)
        
        # Then get details for the condition
        if condition_name in self.condition_details:
            return self.condition_details[condition_name]
        
        # If no specific details, try symptom patterns
        for pattern, details in self.symptom_patterns.items():
            if re.search(pattern, condition_name.lower()):
                return details
        
        # Default to general medical advice if no specific details found
        return {
            'details': f'You may be experiencing {condition_name}. This is a preliminary assessment.',
            'recommendations': '• Rest and stay hydrated\n• Monitor your symptoms\n• Consider over-the-counter medications if appropriate\n• Keep a symptom diary',
            'prevention': '• Maintain a healthy lifestyle\n• Regular exercise\n• Balanced diet\n• Adequate sleep\n• Regular check-ups',
            'when_to_see_doctor': 'If symptoms persist, worsen, or if you have concerns about your health.'
        }

    def _clean_dataset1(self, example):
        return {
            "symptoms": example["text"],
            "diagnosis": str(example["label"])
        }

    def _clean_dataset2(self, example):
        return {
            "symptoms": example["instruction"],
            "diagnosis": example["output"]
        }

    def _clean_dataset3(self, example):
        return {
            "symptoms": example["Symptoms"],
            "diagnosis": example["Name"]
        }

    def _clean_dataset4(self, example):
        return {
            "symptoms": example["input_text"],
            "diagnosis": example["output_text"]
        }

    def _clean_dataset5(self, example):
        return {
            "symptoms": example["query"],
            "diagnosis": example["response"]
        }

    def _clean_dataset6(self, example):
        return {
            "symptoms": example["instruction"],
            "diagnosis": example["output"]
        }

    def _clean_dataset7(self, example):
        return {
            "symptoms": example["text"],
            "diagnosis": example["label"]
        }

    def _check_emergency(self, user_input):
        """Check if the symptoms indicate an emergency condition"""
        user_input = user_input.lower()
        for pattern, details in self.emergency_conditions.items():
            if re.search(pattern, user_input):
                return details
        return None

    def get_diagnosis(self, user_input):
        """
        Get detailed diagnosis based on user input symptoms using both ML and pattern matching
        """
        # First check for emergency conditions
        emergency = self._check_emergency(user_input)
        if emergency:
            return (
                f"🔍 Diagnosis: {emergency['condition']}\n\n"
                f"📝 Details: {emergency['details']}\n\n"
                f"💡 Recommendations:\n{emergency['recommendations']}\n\n"
                f"🛡️ Prevention Tips:\n{emergency['prevention']}\n\n"
                f"⚠️ When to See a Doctor:\n{emergency['when_to_see_doctor']}"
            )

        # First try ML prediction
        ml_prediction = self._predict_with_ml(user_input)
        if ml_prediction:
            diagnosis, confidence = ml_prediction
            details = self._get_condition_details(diagnosis)
            condition_name = self.condition_mapping.get(str(diagnosis), diagnosis)
            
            # Clean up the condition name if it contains unnecessary text
            if "The symptoms listed indicates that the patient is dealing with" in condition_name:
                condition_name = condition_name.replace("The symptoms listed indicates that the patient is dealing with", "").strip()
            
            return (
                f"🔍 Diagnosis: Based on your symptoms, you may be experiencing {condition_name}.\n"
                f"(ML Model Confidence: {confidence:.2%})\n\n"
                f"📝 Details: {details['details']}\n\n"
                f"💡 Recommendations:\n{details['recommendations']}\n\n"
                f"🛡️ Prevention Tips:\n{details['prevention']}\n\n"
                f"⚠️ When to See a Doctor:\n{details['when_to_see_doctor']}"
            )

        # If ML prediction fails, try pattern matching
        pattern_match = self._get_matching_pattern(user_input)
        if pattern_match:
            return (
                f"🔍 Diagnosis: Based on your symptoms, you may be experiencing {pattern_match['condition']}.\n\n"
                f"📝 Details: {pattern_match['details']}\n\n"
                f"💡 Recommendations:\n{pattern_match['recommendations']}\n\n"
                f"🛡️ Prevention Tips:\n{pattern_match['prevention']}\n\n"
                f"⚠️ When to See a Doctor:\n{pattern_match['when_to_see_doctor']}"
            )

        # If all else fails, try dataset matching
        for entry in self.all_data_list:
            if any(word in entry["symptoms"].lower() for word in user_input.lower().split()):
                details = self._get_condition_details(entry["diagnosis"])
                return (
                    f"🔍 Diagnosis: Based on your symptoms, you may be experiencing {entry['diagnosis']}.\n\n"
                    f"📝 Details: {details['details']}\n\n"
                    f"💡 Recommendations:\n{details['recommendations']}\n\n"
                    f"🛡️ Prevention Tips:\n{details['prevention']}\n\n"
                    f"⚠️ When to See a Doctor:\n{details['when_to_see_doctor']}"
                )

        return "I'm sorry, I couldn't find a specific diagnosis for your symptoms. Please consult a healthcare professional for proper medical advice."

# Create a singleton instance
diagnosis_system = DiagnosisSystem()
