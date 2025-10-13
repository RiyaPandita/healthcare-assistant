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
        super().__init__()  # Initialize BaseAgent
        self.config = Config()
        self.pharmacy_settings = self.config.settings.get('pharmacy', {})
        
        try:
            # Load pharmacy data
            with open(pharmacies_path, "r") as f:
                self.pharmacies = json.load(f)
            self.inventory = pd.read_csv(inventory_path)
            # Try to load meds catalog for SKU->drug_name mapping (optional)
            try:
                self.meds = pd.read_csv(self.pharmacy_settings.get('meds_path', 'data/meds.csv'))
                # Ensure sku and drug_name columns exist
                if 'sku' in self.meds.columns and 'drug_name' in self.meds.columns:
                    # Build a simple mapping for quick lookups
                    self.sku_to_name = dict(zip(self.meds['sku'].astype(str), self.meds['drug_name'].astype(str)))
                else:
                    self.sku_to_name = {}
            except Exception:
                self.meds = None
                self.sku_to_name = {}
            self.zips = pd.read_csv(zipcodes_path)
            
            # Validate data
            logging.info(f"Loaded {len(self.pharmacies)} pharmacies")
            logging.info(f"Loaded {len(self.inventory)} inventory items")
            logging.info(f"Loaded {len(self.zips)} zipcodes")
            
            # Load settings
            self.max_radius = self.pharmacy_settings.get('max_delivery_radius_km', 15)
            self.base_fee = self.pharmacy_settings.get('base_delivery_fee', 25)
            self.fee_per_km = self.pharmacy_settings.get('fee_per_km', 5)
            self.min_order = self.pharmacy_settings.get('min_order_amount', 100)
            self.delivery_speeds = self.pharmacy_settings.get('delivery_speeds', {
                'normal': 30,
                'express': 15
            })
            # Whether to enforce per-pharmacy minimum delivery amounts. Default: False (do not enforce)
            # This allows the agent to return candidates even if order total is below the
            # pharmacy's `min_delivery_amount`. Set to True in config to enable enforcement.
            self.enforce_min_order = self.pharmacy_settings.get('enforce_min_order', False)
            # Whether to include pharmacies that can partially fulfill the order (default: True)
            # When True, pharmacies that have at least one requested item will be returned with
            # information about which items are available vs missing. When False, only pharmacies
            # that can fulfill all requested items are returned.
            self.allow_partial_fulfillment = self.pharmacy_settings.get('allow_partial_fulfillment', True)
            
            # Pre-process delivery zones - clean zone names
            self.zips['delivery_zone'] = self.zips['delivery_zone'].str.strip()
            self.zone_mapping = self.zips.set_index('pincode')['delivery_zone'].to_dict()
            logging.info(f"DEBUG: Loaded zone mapping: {self.zone_mapping}")
            
        except Exception as e:
            logging.error(f"Error initializing PharmacyAgent: {str(e)}")
            raise

    def get_location_from_pincode(self, pincode: str) -> Optional[Dict[str, Any]]:
        """Get location details from pincode"""
        try:
            logging.info(f"DEBUG: Looking up pincode: {pincode}")
            if not pincode:
                logging.info("DEBUG: Empty pincode provided")
                return None
                
            # Handle pincode as string and ensure proper comparison
            pincode_str = str(pincode)
            self.zips["pincode"] = self.zips["pincode"].astype(str)  # Convert column to string
            
            # Debug zipcodes data
            logging.info(f"DEBUG: First few zipcodes in database:")
            logging.info(f"DEBUG: {self.zips.head().to_dict()}")
            
            rec = self.zips[self.zips["pincode"] == pincode_str]
            logging.info(f"DEBUG: Looking up {pincode_str} in zipcodes: found {len(rec)} records")
            if len(rec) > 0:
                logging.info(f"DEBUG: Found record: {rec.iloc[0].to_dict()}")
            
            if rec.empty:
                logging.warning(f"No location found for pincode: {pincode_str}")
                return None
                
            row = rec.iloc[0]
            location = {
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "city": row["city"],
                "delivery_zone": row["delivery_zone"],
                "state": row["state"]
            }
            
            logging.info(f"Found location for {pincode}: {location['delivery_zone']}")
            return location
            
        except ValueError:
            logging.error(f"Invalid pincode format: {pincode}")
            return None
        except Exception as e:
            logging.error(f"Error getting location from pincode: {str(e)}", exc_info=True)
            return None

    def calculate_delivery_fee(self, distance_km: float, delivery_zone: str, is_express: bool = False) -> Dict[str, float]:
        """Calculate delivery fee with zone-based multipliers and express delivery options"""
        # Zone-based fee multipliers
        zone_multipliers = {
            'Western Suburbs': {'base': 1.0, 'distance': 1.0},   # Standard urban zone
            'Eastern Suburbs': {'base': 1.2, 'distance': 1.3},   # Suburban zone
            'South Mumbai': {'base': 1.5, 'distance': 1.5}      # Premium zone
        }
        
        logging.info(f"Calculating delivery fee for zone: {delivery_zone}, distance: {distance_km}km")
        zone_rates = zone_multipliers.get(delivery_zone, zone_multipliers['Western Suburbs'])        # Calculate base components
        base_fee = self.base_fee * zone_rates['base']
        distance_fee = (distance_km * self.fee_per_km) * zone_rates['distance']
        
        # Additional fees
        service_charges = self.pharmacy_settings.get('service_charges', {})
        express_fee = service_charges.get('express', 50) if is_express else 0
        peak_hours_fee = service_charges.get('peak_hours', 25) if self._is_peak_hours() else 0
        
        # Calculate total
        subtotal = base_fee + distance_fee
        total_additional = express_fee + peak_hours_fee
        total = subtotal + total_additional
        
        return {
            "base_fee": round(base_fee, 2),
            "distance_fee": round(distance_fee, 2),
            "express_charge": round(express_fee, 2),
            "peak_hours_fee": round(peak_hours_fee, 2),
            "subtotal": round(subtotal, 2),
            "total": round(total, 2),
            "zone_multiplier": zone_rates['base']
        }

    def estimate_delivery_time(self, distance_km: float, delivery_zone: str, is_express: bool = False) -> Dict[str, int]:
        """Estimate delivery time with zone-specific calculations and variable conditions"""
        # Zone-specific base processing times (minutes)
        zone_processing = {
            'Western Suburbs': {'base': 10, 'range': 5},     # Urban: 10±5 minutes
            'Eastern Suburbs': {'base': 15, 'range': 10},    # Suburban: 15±10 minutes
            'South Mumbai': {'base': 20, 'range': 15}       # Premium zone: 20±15 minutes
        }
        
        # Zone-specific travel speeds (minutes per km)
        zone_speeds = {
            'Western Suburbs': {'normal': 3, 'express': 1.5},    # Urban areas
            'Eastern Suburbs': {'normal': 4, 'express': 2},      # Suburban areas
            'South Mumbai': {'normal': 5, 'express': 2.5}       # Premium zone
        }
        
        logging.info(f"Calculating delivery time for zone: {delivery_zone}, distance: {distance_km}km")
        zone = zone_processing.get(delivery_zone, zone_processing['Western Suburbs'])
        speeds = zone_speeds.get(delivery_zone, zone_speeds['Western Suburbs'])
        
        try:
            base_time = zone['base']
            min_processing = max(5, base_time - zone['range'])
            max_processing = base_time + zone['range']
            
            speed = speeds['express'] if is_express else speeds['normal']
            travel_time = round(distance_km * speed)
            
            min_total = min_processing + travel_time
            max_total = max_processing + travel_time
            
            logging.info(f"Delivery time estimate: {min_total}-{max_total} minutes")
            
            return {
                "processing_time": {
                    "min": min_processing,
                    "max": max_processing
                },
                "travel_time": travel_time,
                "total_time": {
                    "min": min_total,
                    "max": max_total
                }
            }
        except Exception as e:
            logging.error(f"Error calculating delivery time: {str(e)}")
            # Return default values if calculation fails
            return {
                "processing_time": {"min": 15, "max": 30},
                "travel_time": 30,
                "total_time": {"min": 45, "max": 60}
            }
        
        # Calculate base processing time with variability
        base_time = zone['base']
        min_processing = max(5, base_time - zone['range'])
        max_processing = base_time + zone['range']
        
        # Calculate travel time based on zone and service type
        speed = speeds['express'] if is_express else speeds['normal']
        travel_time = round(distance_km * speed)
        
        # Apply modifiers for conditions
        if self._is_peak_hours():
            travel_time = round(travel_time * 1.3)  # 30% longer during peak hours
        if self._is_adverse_weather():
            travel_time = round(travel_time * 1.5)  # 50% longer in bad weather
        
        # Express delivery processing optimization
        if is_express:
            min_processing = round(min_processing * 0.7)  # 30% faster processing
            max_processing = round(max_processing * 0.7)
        
        min_total = min_processing + travel_time
        max_total = max_processing + travel_time
        
        return {
            "processing_time": {
                "min": min_processing,
                "max": max_processing
            },
            "travel_time": travel_time,
            "total_time": {
                "min": min_total,
                "max": max_total
            },
            "is_peak_hours": self._is_peak_hours(),
            "is_adverse_weather": self._is_adverse_weather()
        }
        
    def _is_peak_hours(self) -> bool:
        """Check if current time is during peak hours"""
        # TODO: Implement actual peak hours logic
        return False
        
    def _is_adverse_weather(self) -> bool:
        """Check if there are adverse weather conditions"""
        # TODO: Implement weather checking logic
        return False

    def run(self, payload: Dict[str, Any]) -> AgentResult:
        events = []
        try:
            pincode = payload.get("pincode")
            items = payload.get("items", [])
            red_flags = payload.get("red_flags", [])
            is_express = bool(red_flags)  # Use express delivery if there are red flags

            # Validate inputs
            if not pincode:
                return AgentResult({"error": "Pincode is required"}, events)
            if not items:
                return AgentResult({"error": "No items in order"}, events)

            # Get user location from pincode
            location_info = self.get_location_from_pincode(pincode)
            logging.info(f"Location info for pincode {pincode}: {location_info}")
            
            if not location_info:
                events.append(self.event("location_error", {"pincode": pincode}))
                return AgentResult({
                    "error": "We are currently not serving in your area",
                    "details": "Delivery not available for this pincode"
                }, events)
                
            lat, lon = location_info["latitude"], location_info["longitude"]
            delivery_zone = location_info.get("delivery_zone")
            
            if not delivery_zone:
                logging.error(f"No delivery zone found for pincode {pincode} in location info: {location_info}")
                return AgentResult({
                    "error": "No delivery zone configured",
                    "details": f"Delivery zone not configured for pincode {pincode}"
                }, events)
                
            logging.info(f"Processing request for delivery zone: {delivery_zone} at coordinates: ({lat}, {lon})")

            # Find pharmacies with stock
            candidates = []
            logging.info(f"Processing pincode {pincode} with location: {location_info}")
            
            # Validate pharmacy data
            valid_pharmacies = []
            for pharmacy in self.pharmacies:
                try:
                    # Basic validation
                    required_fields = ["id", "name", "latitude", "longitude", "services", "delivery_zones"]
                    missing_fields = [f for f in required_fields if f not in pharmacy]
                    if missing_fields:
                        logging.error(f"Pharmacy {pharmacy.get('id')} missing fields: {missing_fields}")
                        continue
                        
                    valid_pharmacies.append(pharmacy)
                except Exception as e:
                    logging.error(f"Invalid pharmacy data: {e}")
            
            logging.info(f"Found {len(valid_pharmacies)} valid pharmacies")
            
            # Process valid pharmacies
            logging.info(f"DEBUG: Processing {len(valid_pharmacies)} pharmacies for zone {delivery_zone}")
            for pharmacy in valid_pharmacies:
                try:
                    pharmacy_id = pharmacy["id"]
                    logging.info(f"\nDEBUG: Checking pharmacy {pharmacy_id} - {pharmacy['name']}")
                    logging.info(f"DEBUG: Full pharmacy data: {pharmacy}")
                    
                    # Check delivery service
                    if "delivery" not in pharmacy["services"]:
                        logging.info(f"DEBUG: Pharmacy {pharmacy_id} filtered - no delivery service")
                        continue
                    
                    # Check zone coverage
                    if not delivery_zone:
                        logging.error(f"DEBUG: No delivery zone found for pincode {pincode}")
                        continue
                        
                    # Debug zone comparison
                    logging.info(f"DEBUG: Zone check - Required: '{delivery_zone}' vs Available: {pharmacy['delivery_zones']}")
                    delivery_zone_clean = delivery_zone.strip()
                    pharmacy_zones = [z.strip() for z in pharmacy["delivery_zones"]]
                    
                    if delivery_zone_clean not in pharmacy_zones:
                        logging.info(f"DEBUG: Pharmacy {pharmacy_id} filtered - zone {delivery_zone_clean} not served")
                        logging.info(f"DEBUG: Available zones (cleaned): {pharmacy_zones}")
                        continue

                    # Calculate distance
                    dist = haversine(lat, lon, pharmacy["latitude"], pharmacy["longitude"])
                    max_delivery = float(pharmacy.get("delivery_km", self.max_radius))
                    
                    logging.info(f"DEBUG: Distance check:")
                    logging.info(f"DEBUG: Customer coords: ({lat}, {lon})")
                    logging.info(f"DEBUG: Pharmacy coords: ({pharmacy['latitude']}, {pharmacy['longitude']})")
                    logging.info(f"DEBUG: Calculated: {dist:.3f} km")
                    logging.info(f"DEBUG: Max allowed: {max_delivery} km")
                    
                    if dist > max_delivery:
                        logging.info(f"DEBUG: Pharmacy {pharmacy_id} filtered - too far ({dist:.1f} > {max_delivery} km)")
                        continue
                    if dist <= max_delivery:
                        reserved_items = []
                        has_stock = True
                        total_cost = 0
                        
                        # Check inventory for each item
                        for item in items:
                            logging.info(f"Checking {pharmacy['id']} inventory for SKU: {item['sku']}")
                            # Convert pharmacy_id to string for comparison
                            pharmacy_id_str = str(pharmacy["id"])
                            logging.info(f"DEBUG: Checking inventory for pharmacy {pharmacy_id_str}")
                            
                            # Create filtered inventory DataFrame to avoid SettingWithCopyWarning
                            inventory_filtered = self.inventory.copy()
                            inventory_filtered.loc[:, "pharmacy_id"] = inventory_filtered["pharmacy_id"].astype(str)
                            
                            # Debug inventory state
                            logging.info(f"DEBUG: Total inventory items: {len(inventory_filtered)}")
                            logging.info(f"DEBUG: Looking for SKU {item['sku']} in pharmacy {pharmacy_id_str}")
                            
                            inventory = inventory_filtered[
                                (inventory_filtered["pharmacy_id"] == pharmacy_id_str) & 
                                (inventory_filtered["sku"] == item["sku"])
                            ]
                            # If SKU-based lookup failed, try to map SKU -> drug_name using meds catalog
                            if inventory.empty:
                                mapped_name = self.sku_to_name.get(str(item.get('sku')))
                                # If coordinator provided drug_name, prefer that
                                name_to_try = item.get('drug_name') or mapped_name
                                if name_to_try:
                                    inventory = inventory_filtered[
                                        (inventory_filtered["pharmacy_id"] == pharmacy_id_str) &
                                        (inventory_filtered["drug_name"].str.contains(str(name_to_try), case=False, na=False))
                                    ]
                            
                            # Debug inventory check
                            pharmacy_inventory = self.inventory[self.inventory["pharmacy_id"] == pharmacy_id_str]
                            logging.info(f"Pharmacy {pharmacy['id']} has {len(pharmacy_inventory)} items in inventory")
                            logging.info(f"Found {len(inventory)} matches for SKU {item['sku']}")
                            
                            if inventory.empty:
                                logging.info(f"No stock found for SKU {item['sku']} at {pharmacy['id']}")
                                has_stock = False
                                break
                                
                            try:
                                # Force numeric conversion of qty column on our filtered copy
                                inventory_copy = inventory.copy()
                                inventory_copy.loc[:, "qty"] = pd.to_numeric(inventory_copy["qty"], errors='coerce')
                                quantity = int(inventory_copy["qty"].sum())
                                min_order = float(pharmacy.get("min_delivery_amount", 0))
                                
                                logging.info(f"DEBUG: Stock check for SKU {item['sku']}:")
                                logging.info(f"DEBUG: Required: {item.get('qty', 1)}")
                                logging.info(f"DEBUG: Available: {quantity}")
                                logging.info(f"DEBUG: Min order amount: {min_order}")
                                
                                if quantity >= item.get("qty", 1):
                                    item_price = float(inventory_copy.iloc[0]["price"])
                                    item_total = item_price * item.get("qty", 1)
                                    total_cost += item_total
                                    logging.info(f"DEBUG: Item available - Price: {item_price}, Total: {item_total}")
                                    reserved_items.append({
                                        "sku": item["sku"],
                                        "qty": item.get("qty", 1),
                                        "price": item_price,
                                        "name": inventory.iloc[0]["drug_name"],
                                        "form": inventory.iloc[0]["form"],
                                        "strength": inventory.iloc[0]["strength"]
                                    })
                                else:
                                    logging.info(f"DEBUG: Insufficient stock for {item['sku']}")
                                    has_stock = False
                                    break
                            except (ValueError, TypeError) as e:
                                logging.error(f"Error processing inventory quantities: {str(e)}")
                                has_stock = False
                                break
                                reserved_items.append({
                                    "sku": item["sku"],
                                    "qty": item.get("qty", 1),
                                    "price": item_price,
                                    "name": inventory.iloc[0]["drug_name"],
                                    "form": inventory.iloc[0]["form"],
                                    "strength": inventory.iloc[0]["strength"]
                                })
                                
                        logging.info(f"DEBUG: Order summary for {pharmacy_id}:")
                        logging.info(f"DEBUG: Has stock: {has_stock}")
                        logging.info(f"DEBUG: Total cost: {total_cost}")
                        logging.info(f"DEBUG: Min delivery amount: {pharmacy.get('min_delivery_amount', 0)}")
                        
                        # If partial fulfillment is allowed, include pharmacies with at least
                        # one matched item; otherwise require all items to be in stock.
                        if has_stock or (self.allow_partial_fulfillment and len(reserved_items) > 0):
                            # Determine per-pharmacy minimum requirement (if any)
                            min_delivery_amount = float(pharmacy.get("min_delivery_amount", 0))
                            # Enforce minimum only when configured to do so. This keeps behaviour
                            # backwards-compatible and allows tests / UIs to receive candidates
                            # even when order totals are below per-pharmacy minima.
                            if self.enforce_min_order and total_cost < min_delivery_amount:
                                logging.info(f"DEBUG: Order total {total_cost} below minimum {pharmacy.get('min_delivery_amount', 0)}")
                                continue
                            # Calculate delivery details
                            delivery_time = self.estimate_delivery_time(dist, delivery_zone, is_express)
                            delivery_fees = self.calculate_delivery_fee(dist, delivery_zone, is_express)
                            
                            candidates.append({
                                "pharmacy_id": pharmacy["id"],
                                "pharmacy_name": pharmacy.get("name", "Unknown Pharmacy"),
                                "distance_km": round(dist, 1),
                                "items": reserved_items,
                                "delivery_time": delivery_time,
                                "delivery_fees": delivery_fees,
                                "total_items_cost": round(total_cost, 2),
                                "delivery_zone": delivery_zone,
                                "features": pharmacy.get("features", []),
                                "rating": pharmacy.get("rating", 0.0),
                                "timings": pharmacy.get("timings", "Unknown"),
                                "contact": pharmacy.get("contact", ""),
                                "payment_methods": pharmacy.get("payment_methods", ["cash"]),
                                "min_order": pharmacy.get("min_delivery_amount", 0)
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
                candidates.sort(key=lambda x: x["delivery_time"]["total_time"]["min"])
            else:
                candidates.sort(key=lambda x: x["delivery_fees"]["total"])

            # Select best match
            selected = candidates[0]
            
            output = {
                "pharmacy_id": selected["pharmacy_id"],
                "pharmacy_name": selected["pharmacy_name"],
                "distance_km": selected["distance_km"],
                "delivery_fees": selected["delivery_fees"],
                "delivery_time": selected["delivery_time"],
                "is_express": is_express,
                "items": selected["items"],
                "total_items_cost": selected["total_items_cost"],
                "delivery_zone": selected["delivery_zone"]
            }

            events.append(self.event("selection_complete", {
                "pharmacy_id": selected["pharmacy_id"],
                "eta": selected["delivery_time"]["total_time"]["min"]
            }))

            # Add candidates to output for UI selection
            output["candidates"] = candidates

            return AgentResult(output, events)
            
        except Exception as e:
            logging.error(f"Error in PharmacyAgent: {str(e)}")
            events.append(self.event("error", {"message": str(e)}))
            return AgentResult({
                "error": "Failed to process pharmacy request",
                "details": str(e)
            }, events)