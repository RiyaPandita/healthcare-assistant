# agents/pharmacy_agent.py
import pandas as pd
import json
import math
from typing import Dict, Any, List, Optional, Tuple
from agents.base import BaseAgent, AgentResult
from utils.config import Config
import logging

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on Earth"""
    R = 6371  # Earth's radius in kilometers
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

class PharmacyAgent(BaseAgent):
    name = "pharmacy"

    def __init__(self, pharmacies_path: str, inventory_path: str, zipcodes_path: str):
        self.config = Config()
        self.pharmacy_settings = self.config.settings.get('pharmacy', {})
        
        try:
            # Load pharmacy data
            with open(pharmacies_path, "r") as f:
                self.pharmacies = json.load(f)
            self.inventory = pd.read_csv(inventory_path)
            self.zips = pd.read_csv(zipcodes_path)
            
            # Load settings
            self.max_radius = self.pharmacy_settings.get('max_delivery_radius_km', 15)
            self.base_fee = self.pharmacy_settings.get('base_delivery_fee', 25)
            self.fee_per_km = self.pharmacy_settings.get('fee_per_km', 5)
            self.min_order = self.pharmacy_settings.get('min_order_amount', 100)
            self.delivery_speeds = self.pharmacy_settings.get('delivery_speeds', {
                'normal': 30,
                'express': 15
            })
        except Exception as e:
            logging.error(f"Error initializing PharmacyAgent: {str(e)}")
            raise

    def get_location_from_pincode(self, pincode: str) -> Optional[Tuple[float, float]]:
        """Get latitude and longitude from pincode"""
        try:
            if not pincode:
                return None
            rec = self.zips[self.zips["pincode"] == int(pincode)]
            if rec.empty:
                return None
            return float(rec.iloc[0]["lat"]), float(rec.iloc[0]["lon"])
        except Exception as e:
            logging.error(f"Error getting location from pincode: {str(e)}")
            return None

    def calculate_delivery_fee(self, distance_km: float, is_express: bool = False) -> int:
        """Calculate delivery fee based on distance and service type"""
        fee = self.base_fee + (distance_km * self.fee_per_km)
        if is_express:
            fee += self.pharmacy_settings.get('service_charges', {}).get('express', 50)
        return round(fee)

    def estimate_delivery_time(self, distance_km: float, is_express: bool = False) -> int:
        """Estimate delivery time in minutes"""
        base_time = self.delivery_speeds.get('express' if is_express else 'normal', 30)
        return round(base_time + (distance_km * 2))  # 2 min per km

    def run(self, payload: Dict[str, Any]) -> AgentResult:
        events = []
        try:
            pincode = payload.get("pincode")
            items = payload.get("items", [])
            red_flags = payload.get("red_flags", [])
            is_express = bool(red_flags)  # Use express delivery if there are red flags

            # Get user location from pincode
            location = self.get_location_from_pincode(pincode)
            if not location:
                events.append(self.event("location_error", {"pincode": pincode}))
                return AgentResult({"error": "Invalid or unknown pincode"}, events)
                
            lat, lon = location

            # Find pharmacies with stock
            candidates = []
            for pharmacy in self.pharmacies:
                try:
                    dist = haversine(lat, lon, pharmacy["lat"], pharmacy["lon"])
                    if dist <= self.max_radius:
                        reserved_items = []
                        has_stock = True
                        
                        # Check inventory for each item
                        for item in items:
                            inventory = self.inventory[
                                (self.inventory["pharmacy_id"] == pharmacy["id"]) & 
                                (self.inventory["sku"] == item["sku"])
                            ]
                            quantity = int(inventory["qty"].sum()) if not inventory.empty else 0
                            
                            if quantity >= item.get("qty", 1):
                                reserved_items.append({
                                    "sku": item["sku"],
                                    "qty": item.get("qty", 1)
                                })
                            else:
                                has_stock = False
                                break
                                
                        if has_stock:
                            candidates.append({
                                "pharmacy_id": pharmacy["id"],
                                "name": pharmacy.get("name", ""),
                                "distance_km": round(dist, 1),
                                "items": reserved_items,
                                "delivery_fee": self.calculate_delivery_fee(dist, is_express),
                                "eta_min": self.estimate_delivery_time(dist, is_express)
                            })
                            
                except Exception as e:
                    logging.error(f"Error processing pharmacy {pharmacy.get('id')}: {str(e)}")
                    continue
                    
            events.append(self.event("search_complete", {
                "candidates": len(candidates),
                "is_express": is_express
            }))
            
            if not candidates:
                return AgentResult({
                    "error": "No pharmacy with stock in delivery radius",
                    "pincode": pincode,
                    "items_requested": len(items)
                }, events)

            # Sort by delivery time if urgent, otherwise by fee
            if is_express:
                candidates.sort(key=lambda x: x["eta_min"])
            else:
                candidates.sort(key=lambda x: x["delivery_fee"])

            # Select best match
            selected = candidates[0]
            
            output = {
                "pharmacy_id": selected["pharmacy_id"],
                "pharmacy_name": selected["name"],
                "distance_km": selected["distance_km"],
                "delivery_fee": selected["delivery_fee"],
                "eta_min": selected["eta_min"],
                "is_express": is_express,
                "items": selected["items"]
            }

            events.append(self.event("selection_complete", {
                "pharmacy_id": selected["pharmacy_id"],
                "eta": selected["eta_min"]
            }))

            return AgentResult(output, events)
            
        except Exception as e:
            logging.error(f"Error in PharmacyAgent: {str(e)}")
            events.append(self.event("error", {"message": str(e)}))
            return AgentResult({
                "error": "Failed to process pharmacy request",
                "details": str(e)
            }, events)