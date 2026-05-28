"""
Drug name normalization using RxNorm API.
"""
import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Set
import requests
from tqdm import tqdm
import pickle
import os
from ..config import settings


class DrugNormalizer:
    def __init__(self, cache_file: str = "drug_normalization_cache.pkl"):
        """
        Initialize drug normalizer with RxNorm API.

        Args:
            cache_file: Path to cache file for storing normalization results
        """
        self.cache_file = cache_file
        self.base_url = "https://rxnav.nlm.nih.gov/REST"
        self.rate_limit = 10.0  # RxNav allows ~10 requests per second
        self.min_interval = 1.0 / self.rate_limit
        self.last_request_time = 0
        self.cache = self._load_cache()

        # Common drug mappings for quick lookup
        self.common_mappings = {
            "aspirin": "Aspirin",
            "asa": "Aspirin",
            "acetylsalicylic acid": "Aspirin",
            "acetaminophen": "Acetaminophen",
            "paracetamol": "Acetaminophen",
            "ibuprofen": "Ibuprofen",
            "advil": "Ibuprofen",
            "motrin": "Ibuprofen",
            "naproxen": "Naproxen",
            "aleve": "Naproxen",
            "metformin": "Metformin",
            "glucophage": "Metformin",
            "atorvastatin": "Atorvastatin",
            "lipitor": "Atorvastatin",
            "simvastatin": "Simvastatin",
            "zocor": "Simvastatin",
            "lisinopril": "Lisinopril",
            "prinivil": "Lisinopril",
            "zestril": "Lisinopril",
            "metoprolol": "Metoprolol",
            "lopressor": "Metoprolol",
            "toprol": "Metoprolol",
            "amlodipine": "Amlodipine",
            "norvasc": "Amlodipine",
            "omeprazole": "Omeprazole",
            "prilosec": "Omeprazole",
            "warfarin": "Warfarin",
            "coumadin": "Warfarin",
            "clopidogrel": "Clopidogrel",
            "plavix": "Clopidogrel",
            "atorvastatin calcium": "Atorvastatin",
            "rosuvastatin": "Rosuvastatin",
            "crestor": "Rosuvastatin",
        }

    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        if self.min_interval > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

    def _load_cache(self) -> Dict[str, Any]:
        """Load normalization cache from file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
                return {}
        return {}

    def _save_cache(self):
        """Save normalization cache to file."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def _get_cache_key(self, drug_name: str) -> str:
        """Generate cache key for drug name."""
        return hashlib.md5(drug_name.lower().encode()).hexdigest()

    def normalize_drug_name(self, drug_name: str) -> str:
        """
        Normalize a drug name to its standard form using RxNorm.

        Args:
            drug_name: Drug name to normalize

        Returns:
            Normalized drug name (standardized form)
        """
        if not drug_name or not drug_name.strip():
            return drug_name

        drug_name_clean = drug_name.strip().lower()

        # Check common mappings first
        if drug_name_clean in self.common_mappings:
            return self.common_mappings[drug_name_clean]

        # Check cache
        cache_key = self._get_cache_key(drug_name_clean)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Query RxNorm API
        normalized_name = self._query_rxnorm(drug_name_clean)

        # Cache result
        self.cache[cache_key] = normalized_name
        if len(self.cache) % 100 == 0:  # Save periodically
            self._save_cache()

        return normalized_name

    def _query_rxnorm(self, drug_name: str) -> str:
        """
        Query RxNorm API for drug normalization.

        Args:
            drug_name: Drug name to query

        Returns:
            Normalized drug name or original if not found
        """
        self._rate_limit_wait()
        try:
            # First, try to get RxCUI for the drug name
            params = {
                "name": drug_name,
                "srclist": "MTH_SNOMEDCT,MTH_RRXNORM,MTH_MSH,MTH_RXNORM"
            }
            response = requests.get(f"{self.base_url}/rxcui.json", params=params)
            response.raise_for_status()
            data = response.json()
            self.last_request_time = time.time()

            id_group = data.get("idGroup", {})
            rxcuis = id_group.get("rxnormId", [])

            if rxcuis:
                # Get the first RxCUI and get the preferred name
                rxcui = rxcuis[0]
                params = {"rxcui": rxcui}
                response = requests.get(f"{self.base_url}/rxcui/{rxcui}/properties.json", params=params)
                response.raise_for_status()
                data = response.json()
                self.last_request_time = time.time()

                properties = data.get("properties", {})
                preferred_name = properties.get("name", drug_name)
                return preferred_name
            else:
                # Try alternative search
                params = {
                    "name": drug_name,
                    "srclist": "MTH_RRXNORM"
                }
                response = requests.get(f"{self.base_url}/approximateTerm.json", params=params)
                response.raise_for_status()
                data = response.json()
                self.last_request_time = time.time()

                approximate_group = data.get("approximateGroup", {})
                candidates = approximate_group.get("candidate", [])

                if candidates:
                    # Return the first candidate
                    return candidates[0]
                else:
                    # Return original if not found
                    return drug_name.title()  # Return with proper case

        except Exception as e:
            print(f"Error querying RxNorm for '{drug_name}': {e}")
            return drug_name.title()  # Return with proper case on error

    def normalize_drug_names(self, drug_names: List[str]) -> List[str]:
        """
        Normalize a list of drug names.

        Args:
            drug_names: List of drug names to normalize

        Returns:
            List of normalized drug names
        """
        normalized = []
        for drug_name in tqdm(drug_names, desc="Normalizing drug names"):
            normalized.append(self.normalize_drug_name(drug_name))
        return normalized

    def get_rxcui(self, drug_name: str) -> Optional[str]:
        """
        Get RxCUI for a drug name.

        Args:
            drug_name: Drug name to look up

        Returns:
            RxCUI string or None if not found
        """
        if not drug_name or not drug_name.strip():
            return None

        drug_name_clean = drug_name.strip().lower()

        # Check cache first
        cache_key = f"rxcui_{hashlib.md5(drug_name_clean.encode()).hexdigest()}"
        if hasattr(self, '_rxcui_cache'):
            if cache_key in self._rxcui_cache:
                return self._rxcui_cache[cache_key]
        else:
            self._rxcui_cache = {}

        self._rate_limit_wait()
        try:
            params = {
                "name": drug_name_clean,
                "srclist": "MTH_SNOMEDCT,MTH_RRXNORM,MTH_MSH,MTH_RXNORM"
            }
            response = requests.get(f"{self.base_url}/rxcui.json", params=params)
            response.raise_for_status()
            data = response.json()
            self.last_request_time = time.time()

            id_group = data.get("idGroup", {})
            rxcuis = id_group.get("rxnormId", [])

            rxcui = rxcuis[0] if rxcuis else None
            self._rxcui_cache[cache_key] = rxcui
            return rxcui
        except Exception as e:
            print(f"Error getting RxCUI for '{drug_name}': {e}")
            self._rxcui_cache[cache_key] = None
            return None

    def batch_normalize(self, drug_names: List[str], save_cache: bool = True) -> Dict[str, str]:
        """
        Normalize a batch of drug names and return mapping.

        Args:
            drug_names: List of drug names to normalize
            save_cache: Whether to save cache after processing

        Returns:
            Dictionary mapping original names to normalized names
        """
        mapping = {}
        unique_names = list(set(drug_names))  # Remove duplicates

        for drug_name in tqdm(unique_names, desc="Batch normalizing drug names"):
            if drug_name and drug_name.strip():
                mapping[drug_name] = self.normalize_drug_name(drug_name)

        if save_cache:
            self._save_cache()

        return mapping


# Convenience functions
def normalize_drug_name(drug_name: str) -> str:
    """
    Normalize a single drug name.

    Args:
        drug_name: Drug name to normalize

    Returns:
        Normalized drug name
    """
    normalizer = DrugNormalizer()
    return normalizer.normalize_drug_name(drug_name)


def normalize_drug_names(drug_names: List[str]) -> List[str]:
    """
    Normalize a list of drug names.

    Args:
        drug_names: List of drug names to normalize

    Returns:
        List of normalized drug names
    """
    normalizer = DrugNormalizer()
    return normalizer.normalize_drug_names(drug_names)


def get_drug_synonyms(drug_name: str) -> List[str]:
    """
    Get synonyms for a drug name from RxNorm.

    Args:
        drug_name: Drug name to find synonyms for

    Returns:
        List of synonym strings
    """
    normalizer = DrugNormalizer()
    rxcui = normalizer.get_rxcui(drug_name)
    if not rxcui:
        return [drug_name]

    normalizer._rate_limit_wait()
    try:
        response = requests.get(f"{normalizer.base_url}/rxcui/{rxcui}/allproperties.json?prop=everything")
        response.raise_for_status()
        data = response.json()
        normalizer.last_request_time = time.time()

        properties = data.get("properties", {})
        synonyms = []

        # Get various name properties
        for prop_name in ["name", "synonym", "tradename", "fsn"]:
            if prop_name in properties:
                value = properties[prop_name]
                if isinstance(value, list):
                    synonyms.extend([str(v) for v in value if v])
                elif value:
                    synonyms.append(str(value))

        # Also get related concepts
        normalizer._rate_limit_wait()
        response = requests.get(f"{normalizer.base_url}/rxcui/{rxcui}/related.json?tty=in+pty+syn")
        response.raise_for_status()
        data = response.json()
        normalizer.last_request_time = time.time()

        related_group = data.get("relatedGroup", {})
        concept_group = related_group.get("conceptGroup", [])
        for group in concept_group:
            concept_properties = group.get("conceptProperties", [])
            for concept in concept_properties:
                name = concept.get("name", "")
                if name and name not in synonyms:
                    synonyms.append(name)

        return list(set(synonyms))  # Remove duplicates
    except Exception as e:
        print(f"Error getting synonyms for '{drug_name}': {e}")
        return [drug_name]


if __name__ == "__main__":
    # Test the normalizer
    test_drugs = ["aspirin", "asa", "advil", "metformin", "lipitor", "unknown_drug"]
    print("Testing drug normalization:")
    for drug in test_drugs:
        normalized = normalize_drug_name(drug)
        print(f"  {drug} -> {normalized}")