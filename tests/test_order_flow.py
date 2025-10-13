import unittest
import pandas as pd
import json
from agents.pharmacy_agent import PharmacyAgent
from agents.therapy_agent import TherapyAgent
import os

class TestOrderFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize test data paths
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.pharmacies_path = os.path.join(base_path, "data", "pharmacies.json")
        cls.inventory_path = os.path.join(base_path, "data", "inventory.csv")
        cls.zipcodes_path = os.path.join(base_path, "data", "zipcodes.csv")
        cls.interactions_path = os.path.join(base_path, "data", "interactions.csv")
        cls.meds_path = os.path.join(base_path, "data", "meds.csv")

        # Load test data
        cls.inventory_df = pd.read_csv(cls.inventory_path)
        with open(cls.pharmacies_path, 'r') as f:
            cls.pharmacies = json.load(f)
        
        # Initialize agents
        cls.pharmacy_agent = PharmacyAgent(
            pharmacies_path=cls.pharmacies_path,
            inventory_path=cls.inventory_path,
            zipcodes_path=cls.zipcodes_path
        )

    def test_andheri_order_flow(self):
        """Test complete order flow for Andheri location"""
        # Test case data
        test_payload = {
            "pincode": "400053",  # Andheri area (Western Suburbs)
            "items": [
                {"sku": "OTC001", "qty": 1},  # Paracetamol - common item
                {"sku": "OTC009", "qty": 1}   # Ibuprofen - MedQuick exclusive
            ]
        }

        # Get pharmacy recommendations
        result = self.pharmacy_agent.run(test_payload)

        # Validate pharmacy selection
        self.assertFalse(hasattr(result.output, "error"), "Should not have any errors")
        candidates = result.output.get("candidates", [])
        self.assertTrue(len(candidates) > 0, "Should find at least one pharmacy")
        
        # Validate MedQuick Andheri is first choice
        first_pharmacy = candidates[0]
        self.assertEqual(first_pharmacy["pharmacy_id"], "ph001", "MedQuick Andheri should be selected")
        self.assertLess(first_pharmacy["distance_km"], 12, "Should be within delivery radius")
        
        # Validate inventory and pricing
        items = first_pharmacy.get("items", [])
        self.assertEqual(len(items), 2, "Should have both requested items")
        self.assertEqual(
            sum(item["price"] * item["qty"] for item in items),
            40.50,  # Paracetamol (10.50) + Ibuprofen (30.00)
            "Total should match expected price"
        )
        
        # Validate delivery details
        self.assertEqual(first_pharmacy["delivery_zone"], "Western Suburbs")
        self.assertIn("delivery_time", first_pharmacy)
        self.assertIn("delivery_fees", first_pharmacy)        # Validate inventory availability
        items = first_pharmacy.get("items", [])
        self.assertEqual(len(items), 2, "Should have both requested items")
        
        # Validate delivery calculation
        delivery_fees = first_pharmacy.get("delivery_fees", {})
        self.assertIsNotNone(delivery_fees.get("total"), "Should calculate delivery fee")
        self.assertGreater(delivery_fees.get("total", 0), 0, "Delivery fee should be positive")

        # Validate delivery time calculation
        delivery_time = first_pharmacy.get("delivery_time", {})
        self.assertIsNotNone(delivery_time.get("total_time"), "Should calculate delivery time")
        self.assertGreater(delivery_time.get("total_time", {}).get("min", 0), 0, "Should have minimum delivery time")

    def test_medicine_availability(self):
        """Test medicine availability and inventory management"""
        # Test inventory for different pharmacies
        test_cases = [
            {
                "pharmacy_id": "ph001",
                "medicines": {
                    "OTC001": {"name": "Paracetamol", "price": 10.50, "min_qty": 1},
                    "OTC002": {"name": "Cough Syrup", "price": 85.75, "min_qty": 1},
                    "OTC009": {"name": "Ibuprofen", "price": 30.00, "min_qty": 1}
                }
            },
            {
                "pharmacy_id": "ph002",
                "medicines": {
                    "OTC001": {"name": "Paracetamol", "price": 11.00, "min_qty": 1},
                    "OTC004": {"name": "Throat Lozenges", "price": 25.00, "min_qty": 1}
                }
            },
            {
                "pharmacy_id": "ph003",
                "medicines": {
                    "OTC001": {"name": "Paracetamol", "price": 10.00, "min_qty": 1},
                    "OTC006": {"name": "Vitamin C", "price": 120.00, "min_qty": 1}
                }
            }
        ]

        for test_case in test_cases:
            pharmacy_id = test_case["pharmacy_id"]
            for sku, details in test_case["medicines"].items():
                # Check if medicine exists in inventory
                med_inventory = self.inventory_df[
                    (self.inventory_df["pharmacy_id"] == pharmacy_id) & 
                    (self.inventory_df["sku"] == sku)
                ]
                
                # Verify basic inventory data
                self.assertFalse(med_inventory.empty, 
                    f"{details['name']} should be in {pharmacy_id} inventory")
                self.assertGreater(med_inventory["qty"].iloc[0], 0, 
                    f"{details['name']} should have stock in {pharmacy_id}")
                
                # Verify price and quantity limits
                self.assertEqual(float(med_inventory["price"].iloc[0]), details["price"],
                    f"Price mismatch for {details['name']} in {pharmacy_id}")
                self.assertEqual(int(med_inventory["min_qty"].iloc[0]), details["min_qty"],
                    f"Minimum quantity mismatch for {details['name']} in {pharmacy_id}")
                
                # Verify additional fields
                self.assertIn("form", med_inventory.columns, "Should have dosage form")
                self.assertIn("strength", med_inventory.columns, "Should have strength info")
                self.assertIn("expiry_date", med_inventory.columns, "Should have expiry date")

    def test_delivery_radius(self):
        """Test delivery radius and zone-based calculations"""
        test_locations = [
            {
                "pincode": "400053",  # Andheri West (Western Suburbs)
                "should_deliver": True,
                "expected_pharmacies": ["ph001"],  # MedQuick Andheri within 12km
                "zone": "Western Suburbs"
            },
            {
                "pincode": "400070",  # Kurla (Eastern Suburbs)
                "should_deliver": True,
                "expected_pharmacies": ["ph003"],  # WellCare Powai within 10km
                "zone": "Eastern Suburbs"
            },
            {
                "pincode": "400001",  # Fort (South Mumbai)
                "should_deliver": False,
                "expected_pharmacies": [],  # Out of all delivery radii
                "zone": "South Mumbai"
            }
        ]

        for location in test_locations:
            # Verify location data
            result = self.pharmacy_agent.get_location_from_pincode(location["pincode"])
            self.assertIsNotNone(result, f"Should get location for {location['pincode']}")
            self.assertEqual(result["delivery_zone"], location["zone"], 
                           f"Zone mismatch for {location['pincode']}")

            # Test delivery possibility
            test_payload = {
                "pincode": location["pincode"],
                "items": [
                    {"sku": "OTC001", "qty": 1},  # Common item (Paracetamol)
                    # Use OTC006 (Vitamin C) for Eastern Suburbs test so ph003 (WellCare Powai)
                    # — which serves Eastern Suburbs — can fulfill the request. For other
                    # zones we test OTC002 to exercise different inventory paths.
                    {"sku": ("OTC006" if location["pincode"] == "400070" else "OTC002"), "qty": 1}
                ]
            }
            result = self.pharmacy_agent.run(test_payload)
            candidates = result.output.get("candidates", [])

            if location["should_deliver"]:
                self.assertTrue(len(candidates) > 0,
                    f"Should find delivery options for {location['pincode']}")
                
                # Verify expected pharmacies
                found_pharmacies = [c["pharmacy_id"] for c in candidates]
                for expected in location["expected_pharmacies"]:
                    self.assertIn(expected, found_pharmacies,
                        f"Should find pharmacy {expected} for {location['pincode']}")
                    
                # Verify delivery calculations
                for candidate in candidates:
                    self.assertIn("delivery_time", candidate)
                    self.assertIn("delivery_fees", candidate)
                    self.assertEqual(candidate["delivery_zone"], location["zone"])
            else:
                self.assertEqual(len(candidates), 0,
                    f"Should not find delivery options for {location['pincode']}")

if __name__ == '__main__':
    unittest.main()