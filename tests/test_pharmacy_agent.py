import pytest
from agents.pharmacy_agent import PharmacyAgent
import os
import json
import pandas as pd

@pytest.fixture
def test_data_dir(tmpdir):
    # Create test pharmacy data
    pharmacies = [
        {
            "id": "PH001",
            "name": "Test Pharmacy 1",
            "latitude": 19.0760,
            "longitude": 72.8777,
            "delivery_zones": ["Western Suburbs"],
            "services": ["delivery"],
            "delivery_km": 15.0,
            "min_delivery_amount": 100
        }
    ]
    
    # Create test inventory data
    inventory_data = {
        "pharmacy_id": ["PH001", "PH001"],
        "medicine": ["TestMed1", "TestMed2"],
        "stock": [10, 5],
        "price": [100.0, 200.0],
        "sku": ["SKU001", "SKU002"],
        "drug_name": ["Test Medicine 1", "Test Medicine 2"],
        "form": ["tablet", "tablet"],
        "strength": ["500mg", "250mg"]
    }
    
    # Create test zipcode data
    zipcode_data = {
        "pincode": ["400053"],
        "latitude": [19.0760],
        "longitude": [72.8777],
        "city": ["Mumbai"],
        "delivery_zone": ["Western Suburbs"],
        "state": ["Maharashtra"]
    }
    
    # Save test data files
    pharmacy_file = tmpdir.join("pharmacies.json")
    inventory_file = tmpdir.join("inventory.csv")
    zipcode_file = tmpdir.join("zipcodes.csv")
    
    with open(pharmacy_file, 'w') as f:
        json.dump(pharmacies, f)
        
    pd.DataFrame(inventory_data).to_csv(inventory_file, index=False)
    pd.DataFrame(zipcode_data).to_csv(zipcode_file, index=False)
    
    return {
        "pharmacies": str(pharmacy_file),
        "inventory": str(inventory_file),
        "zipcodes": str(zipcode_file)
    }

def test_get_location_from_pincode(test_data_dir):
    agent = PharmacyAgent(
        test_data_dir["pharmacies"],
        test_data_dir["inventory"],
        test_data_dir["zipcodes"]
    )
    
    # Test valid pincode
    location = agent.get_location_from_pincode("400053")
    assert location is not None
    assert location["delivery_zone"] == "Western Suburbs"
    assert abs(location["latitude"] - 19.0760) < 0.0001
    assert abs(location["longitude"] - 72.8777) < 0.0001
    
    # Test invalid pincode
    assert agent.get_location_from_pincode("999999") is None
    assert agent.get_location_from_pincode("") is None
    assert agent.get_location_from_pincode("invalid") is None

def test_agent_run(test_data_dir):
    agent = PharmacyAgent(
        test_data_dir["pharmacies"],
        test_data_dir["inventory"],
        test_data_dir["zipcodes"]
    )
    
    # Test order with available items
    payload = {
        "pincode": "400053",
        "items": [
            {"sku": "SKU001", "qty": 1},
            {"sku": "SKU002", "qty": 1}
        ]
    }
    
    result = agent.run(payload)
    assert result.data is not None
    assert "error" not in result.data
    assert result.data["pharmacy_id"] == "PH001"
    assert result.data["delivery_zone"] == "Western Suburbs"
    assert len(result.data["items"]) == 2
    
    # Test invalid pincode
    result = agent.run({"pincode": "999999", "items": []})
    assert "error" in result.data
    
    # Test empty items
    result = agent.run({"pincode": "400053", "items": []})
    assert "error" in result.data
    
    # Test unavailable items
    result = agent.run({
        "pincode": "400053",
        "items": [{"sku": "INVALID", "qty": 1}]
    })
    assert "error" in result.data
    assert "No pharmacy with stock" in result.data["error"]