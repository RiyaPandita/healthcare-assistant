import unittest
import pandas as pd
import os
from agents.therapy_agent import TherapyAgent
from utils.config import Config

class TestTherapyAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create test data files
        cls.meds_data = pd.DataFrame({
            'drug_name': ['Paracetamol', 'Ibuprofen', 'Aspirin'],
            'sku': ['SKU001', 'SKU002', 'SKU003'],
            'recommended_dose': ['500mg', '200mg', '300mg'],
            'frequency': ['4-6 hours', '6-8 hours', '4-6 hours'],
            'warnings': ['liver disease;alcohol use', 'stomach ulcers;bleeding disorders', 'children under 16']
        })
        cls.interactions_data = pd.DataFrame({
            'drug_a': ['Ibuprofen', 'Aspirin', 'Paracetamol'],
            'drug_b': ['Aspirin', 'Warfarin', 'Alcohol'],
            'level': ['Moderate', 'Severe', 'Moderate'],
            'note': ['Increased bleeding risk', 'Life-threatening bleeding risk', 'Liver damage risk']
        })

        # Create test medical rules
        cls.test_rules = {
            "symptoms": {
                "fever": {
                    "otc_options": ["Paracetamol", "Ibuprofen", "Aspirin"],
                    "contraindications": []
                },
                "headache": {
                    "otc_options": ["Paracetamol", "Ibuprofen"],
                    "contraindications": []
                }
            },
            "age_groups": {
                "child": {
                    "range": "0-11",
                    "restrictions": ["Aspirin"]
                },
                "teen": {
                    "range": "12-17",
                    "restrictions": ["Aspirin"]
                },
                "adult": {
                    "range": "18-64",
                    "restrictions": []
                },
                "elderly": {
                    "range": "65-150",
                    "restrictions": ["High-dose NSAIDs"]
                }
            },
            "allergy_keywords": {
                "paracetamol": ["paracetamol", "acetaminophen"],
                "nsaids": ["ibuprofen", "aspirin", "nsaid"],
                "antihistamines": ["diphenhydramine", "chlorpheniramine"]
            },
            "covid_primary_symptoms": ["fever", "cough"],
            "covid_secondary_symptoms": ["headache", "body_ache"],
            "non_covid_symptoms": [],
            "symptom_validation": {
                "min_primary_symptoms": 1,
                "min_total_symptoms": 1
            }
        }
        
        # Save test data to temporary CSV files
        test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        os.makedirs(test_data_dir, exist_ok=True)
        
        cls.meds_path = os.path.join(test_data_dir, 'test_meds.csv')
        cls.interactions_path = os.path.join(test_data_dir, 'test_interactions.csv')
        
        cls.meds_data.to_csv(cls.meds_path, index=False)
        cls.interactions_data.to_csv(cls.interactions_path, index=False)

    def setUp(self):
        self.agent = TherapyAgent(
            meds_path=self.meds_path,
            interactions_path=self.interactions_path,
            api_key="test_api_key"
        )
        # Mock the medical rules directly
        self.agent.medical_rules = self.test_rules

    def test_age_group_determination(self):
        """Test age group determination logic"""
        self.assertEqual(self.agent._determine_age_group(5), "child")
        self.assertEqual(self.agent._determine_age_group(15), "teen")
        self.assertEqual(self.agent._determine_age_group(35), "adult")
        self.assertEqual(self.agent._determine_age_group(70), "elderly")

    def test_drug_interaction_check(self):
        """Test drug interaction checking"""
        # Test known interaction
        interaction = self.agent._check_interaction("Ibuprofen", "Aspirin")
        self.assertIsNotNone(interaction)
        self.assertIn("Increased bleeding risk", interaction)

        # Test no interaction
        interaction = self.agent._check_interaction("Paracetamol", "Ibuprofen")
        self.assertIsNone(interaction)

    def test_otc_filtering(self):
        """Test OTC medication filtering"""
        symptoms = ["fever", "headache"]
        age = 25
        allergies = ["nsaids"]
        medical_data = {"measurements": {"temperature": "38.5"}}

        recommendations, warnings = self.agent._filter_otc(symptoms, age, allergies, medical_data)
        
        # Check that recommendations exclude NSAIDs for someone with NSAID allergy
        drug_names = [rec["drug_name"] for rec in recommendations]
        self.assertNotIn("Ibuprofen", drug_names)
        self.assertNotIn("Aspirin", drug_names)
        self.assertIn("Paracetamol", drug_names)

    def test_age_restrictions(self):
        """Test age-based medication restrictions"""
        symptoms = ["fever"]
        allergies = []
        medical_data = {"measurements": {"temperature": "38.0"}}

        # Test child restrictions
        recommendations, warnings = self.agent._filter_otc(symptoms, 10, allergies, medical_data)
        drug_names = [rec["drug_name"] for rec in recommendations]
        self.assertNotIn("Aspirin", drug_names)  # Aspirin should be restricted for children

        # Test adult - no restrictions
        recommendations, warnings = self.agent._filter_otc(symptoms, 35, allergies, medical_data)
        drug_names = [rec["drug_name"] for rec in recommendations]
        self.assertIn("Paracetamol", drug_names)

    def test_allergy_handling(self):
        """Test allergy handling logic"""
        symptoms = ["fever", "headache"]
        age = 35
        allergies = ["paracetamol allergy"]
        medical_data = {"measurements": {"temperature": "38.0"}}

        recommendations, warnings = self.agent._filter_otc(symptoms, age, allergies, medical_data)
        drug_names = [rec["drug_name"] for rec in recommendations]
        self.assertNotIn("Paracetamol", drug_names)

    @classmethod
    def tearDownClass(cls):
        # Clean up test data files
        try:
            os.remove(cls.meds_path)
            os.remove(cls.interactions_path)
            os.rmdir(os.path.dirname(cls.meds_path))
        except Exception as e:
            print(f"Error cleaning up test files: {e}")

if __name__ == '__main__':
    unittest.main()