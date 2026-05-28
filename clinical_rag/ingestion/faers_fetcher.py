"""
FDA FAERS data fetcher using OpenFDA API.
"""
import json
import time
from typing import List, Dict, Any, Optional
import requests
from tqdm import tqdm
from ..config import settings


class FAERSFetcher:
    def __init__(self):
        """Initialize FAERS fetcher."""
        self.base_url = "https://api.fda.gov/drug/event.json"
        self.rate_limit = 1.0  # OpenFDA rate limit is conservative
        self.min_interval = 1.0 / self.rate_limit
        self.last_request_time = 0

    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        if self.min_interval > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

    def search_adverse_events(
        self,
        drug_name: str,
        search_years: List[int] = None,
        max_results: int = 10000
    ) -> List[Dict[str, Any]]:
        """
        Search FAERS for adverse event reports for a specific drug.

        Args:
            drug_name: Drug name to search for (will be normalized via RxNorm)
            search_years: List of years to search (e.g., [2020, 2021, 2022])
            max_results: Maximum number of results

        Returns:
            List of adverse event report dictionaries
        """
        # Build search query
        query_parts = [f'patient.drug.medicinalproduct:"{drug_name}"']

        if search_years:
            year_conditions = [f'receivedate:[{year}0101 TO {year}1231]' for year in search_years]
            query_parts.append(f"({' OR '.join(year_conditions)})")

        search_query = " AND ".join(query_parts)

        self._rate_limit_wait()
        try:
            params = {
                "search": search_query,
                "limit": min(max_results, 1000),  # OpenFDA max limit
                "count": "patient.reaction.reactionmeddrapt.exact"  # Aggregate by reaction
            }
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            self.last_request_time = time.time()

            # Process results
            results = data.get("results", [])
            events = []

            # If we have aggregated results, we need to get detailed reports
            if "count" in params and results:
                # Get detailed reports for each reaction type
                for result in results[:50]:  # Limit to top 50 reaction types
                    reaction = result.get("term", "")
                    count = result.get("count", 0)
                    # Fetch detailed reports for this reaction
                    detailed_reports = self._fetch_detailed_reports(
                        drug_name, reaction, search_years, min(count, 100)
                    )
                    events.extend(detailed_reports)
            else:
                # Direct results
                events = data.get("results", [])

            return events
        except Exception as e:
            print(f"Error searching FAERS: {e}")
            return []

    def _fetch_detailed_reports(
        self,
        drug_name: str,
        reaction: str,
        search_years: List[int] = None,
        max_results: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Fetch detailed adverse event reports for drug-reaction pair.

        Args:
            drug_name: Drug name
            reaction: Reaction/MedDRA PT
            search_years: List of years to search
            max_results: Maximum number of results

        Returns:
            List of detailed adverse event report dictionaries
        """
        query_parts = [
            f'patient.drug.medicinalproduct:"{drug_name}"',
            f'patient.reaction.reactionmeddrapt:"{reaction}"'
        ]

        if search_years:
            year_conditions = [f'receivedate:[{year}0101 TO {year}1231]' for year in search_years]
            query_parts.append(f"({' OR '.join(year_conditions)})")

        search_query = " AND ".join(query_parts)

        self._rate_limit_wait()
        try:
            params = {
                "search": search_query,
                "limit": min(max_results, 1000)
            }
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            self.last_request_time = time.time()

            results = data.get("results", [])
            events = []

            for result in results:
                event = self._parse_faers_record(result)
                if event:
                    events.append(event)

            return events
        except Exception as e:
            print(f"Error fetching detailed FAERS reports: {e}")
            return []

    def _parse_faers_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse a FAERS record into standardized format.

        Args:
            record: Raw FAERS record from OpenFDA

        Returns:
            Standardized adverse event dictionary
        """
        try:
            patient = record.get("patient", {})
            drugs = patient.get("drug", [])
            reactions = patient.get("reaction", [])

            # Extract drug information
            drug_names = []
            for drug in drugs:
                if isinstance(drug, dict):
                    medicinal_product = drug.get("medicinalproduct", "")
                    if medicinal_product:
                        drug_names.append(str(medicinal_product))

            # Extract reactions
            reaction_terms = []
            for reaction in reactions:
                if isinstance(reaction, dict):
                    reaction_pt = reaction.get("reactionmeddrapt", "")
                    if reaction_pt:
                        reaction_terms.append(str(reaction_pt))

            # Extract outcome information
            outcome_codes = []
            if "patient" in record and "patientoutcome" in record["patient"]:
                patient_outcome = record["patient"]["patientoutcome"]
                if isinstance(patient_outcome, list):
                    outcome_codes = [str(code) for code in patient_outcome if code]
                else:
                    outcome_codes = [str(patient_outcome)] if patient_outcome else []

            # Map outcome codes to descriptions
            outcome_map = {
                "1": "Death",
                "2": "Life-threatening",
                "3": "Hospitalization - initial or prolonged",
                "4": "Disability",
                "5": "Congenital anomaly",
                "6": "Required intervention to prevent permanent impairment/damage",
                "Other": "Other"
            }
            serious_outcomes = [outcome_map.get(code, code) for code in outcome_codes if code in outcome_map]

            # Extract dates
            receivedate = record.get("receivedate", "")
            receiptyear = None
            if receivedate and len(receivedate) >= 4:
                try:
                    receiptyear = int(receivedate[:4])
                except ValueError:
                    pass

            # Extract primary source country
            primarysourcecountry = record.get("primarysourcecountry", "")

            return {
                "drug_names": drug_names,
                "reactions": reaction_terms,
                "serious_outcomes": serious_outcomes,
                "receivedate": receivedate,
                "receiptyear": receiptyear,
                "primarysourcecountry": primarysourcecountry,
                "source": "faers",
                "raw_record": record  # Keep raw for debugging
            }
        except Exception as e:
            print(f"Error parsing FAERS record: {e}")
            return None


# Convenience functions
def fetch_faers_reports(
    drug_name: str,
    years: List[int] = None,
    max_results: int = 10000
) -> List[Dict[str, Any]]:
    """
    Fetch FAERS reports for a drug.

    Args:
        drug_name: Drug name to search for
        years: List of years to search
        max_results: Maximum number of results

    Returns:
        List of adverse event report dictionaries
    """
    fetcher = FAERSFetcher()
    return fetcher.search_adverse_events(drug_name, years, max_results)


def aggregate_faers_by_outcome(reports: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Aggregate FAERS reports by serious outcome.

    Args:
        reports: List of FAERS report dictionaries

    Returns:
        Dictionary mapping outcome types to counts
    """
    outcome_counts = {}
    for report in reports:
        outcomes = report.get("serious_outcomes", [])
        for outcome in outcomes:
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
    return outcome_counts