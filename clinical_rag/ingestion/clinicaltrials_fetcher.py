"""
ClinicalTrials.gov data fetcher using v2 REST API.
"""
import json
import time
from typing import List, Dict, Any, Optional
import requests
from tqdm import tqdm
from ..config import settings


class ClinicalTrialsFetcher:
    def __init__(self):
        """Initialize ClinicalTrials.gov fetcher."""
        self.base_url = "https://clinicaltrials.gov/api/v2/studies"
        self.rate_limit = settings.CLINICALTRIALS_RATE_LIMIT
        self.min_interval = 1.0 / self.rate_limit if self.rate_limit > 0 else 0
        self.last_request_time = 0

    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        if self.min_interval > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

    def search_trials(
        self,
        query: str = "",
        filter_phases: List[str] = None,
        filter_status: List[str] = None,
        max_results: int = 10000
    ) -> List[str]:
        """
        Search ClinicalTrials.gov for studies matching criteria.

        Args:
            query: Search query string
            filter_phases: List of phases to include (e.g., ["Phase 2", "Phase 3"])
            filter_status: List of statuses to include (e.g., ["Completed", "Active"])
            max_results: Maximum number of results

        Returns:
            List of NCT ID strings
        """
        self._rate_limit_wait()

        # Build query parameters
        params = {
            "format": "json",
            "pageSize": min(max_results, 1000),  # API max page size
        }

        # Build query string
        query_parts = []
        if query:
            query_parts.append(f"(area[title]:\"{query}\" OR area[condition]:\"{query}\" OR area[intervention]:\"{query}\")")

        # Add phase filter
        if filter_phases:
            phase_conditions = [f"area[phase]:\"{phase}\"" for phase in filter_phases]
            if phase_conditions:
                query_parts.append(f"({' OR '.join(phase_conditions)})")

        # Add status filter (we want interventional studies)
        status_conditions = ['area[studyType]:\"Interventional\"']
        if filter_status:
            status_conditions.extend([f"area[overallStatus]:\"{status}\"" for status in filter_status])
        query_parts.append(f"({' AND '.join(status_conditions)})")

        if query_parts:
            params["query.term"] = " AND ".join(query_parts)

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            self.last_request_time = time.time()

            studies = data.get("studies", [])
            nct_ids = [study.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "")
                      for study in studies if study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")]

            # Handle pagination if we need more results
            total_count = data.get("totalCount", 0)
            if total_count > len(nct_ids) and len(nct_ids) < max_results:
                # Fetch remaining pages
                page_size = params["pageSize"]
                for page in range(1, min((total_count + page_size - 1) // page_size, (max_results + page_size - 1) // page_size)):
                    params["pageToken"] = data.get("nextPageToken")
                    if not params["pageToken"]:
                        break
                    self._rate_limit_wait()
                    response = requests.get(self.base_url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    studies = data.get("studies", [])
                    nct_ids.extend([study.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "")
                                  for study in studies if study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")])
                    self.last_request_time = time.time()

            return nct_ids[:max_results]
        except Exception as e:
            print(f"Error searching ClinicalTrials.gov: {e}")
            return []

    def fetch_trial_details(self, nct_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch detailed information for a list of NCT IDs.

        Args:
            nct_ids: List of NCT ID strings

        Returns:
            List of trial dictionaries
        """
        if not nct_ids:
            return []

        trials = []
        batch_size = 100  # Process in batches

        for i in tqdm(range(0, len(nct_ids), batch_size), desc="Fetching ClinicalTrials data"):
            batch_nct_ids = nct_ids[i:i+batch_size]
            self._rate_limit_wait()
            try:
                params = {
                    "format": "json",
                    "ids": ",".join(batch_nct_ids)
                }
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
                self.last_request_time = time.time()

                studies = data.get("studies", [])
                for study in studies:
                    trial = self._parse_trial_record(study)
                    if trial:
                        trials.append(trial)
            except Exception as e:
                print(f"Error fetching batch {i//batch_size}: {e}")
                continue

        return trials

    def _parse_trial_record(self, study: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse a ClinicalTrials.gov record into standardized format.

        Args:
            study: Raw study record from API

        Returns:
            Standardized trial dictionary
        """
        try:
            protocol_section = study.get("protocolSection", {})
            identification_module = protocol_section.get("identificationModule", {})
            status_module = protocol_section.get("statusModule", {})
            conditions_module = protocol_section.get("conditionsModule", {})
            interventions_module = protocol_section.get("interventionsModule", {})
            outcomes_module = protocol_section.get("outcomesModule", {})
            design_module = protocol_section.get("designModule", {})
            sponsor_module = protocol_section.get("sponsorCollaboratorsModule", {})
            enrollment_module = protocol_section.get("enrollmentModule", {})

            # Extract basic info
            nct_id = identification_module.get("nctId", "")
            brief_title = identification_module.get("briefTitle", "")
            official_title = identification_module.get("officialTitle", "")

            # Extract study details
            study_type = status_module.get("studyType", "")
            phase_list = design_module.get("phaseList", {}).get("phases", [])
            phase = phase_list[0] if phase_list else ""
            status = status_module.get("overallStatus", "")

            # Extract conditions
            conditions_list = conditions_module.get("conditionsList", [])
            conditions = [cond.get("condition", "") for cond in conditions_list if cond.get("condition")]

            # Extract interventions
            interventions_list = interventions_module.get("interventionsList", [])
            interventions = []
            for interv in interventions_list:
                if interv.get("interventionName"):
                    interventions.append(interv.get("interventionName"))

            # Extract primary outcome
            primary_outcomes = outcomes_module.get("primaryOutcomesList", [])
            primary_outcome = ""
            if primary_outcomes:
                primary_outcome = primary_outcomes[0].get("outcomeMeasure", "")

            # Extract enrollment
            enrollment = enrollment_module.get("enrollmentCount", None)

            # Extract sponsor
            lead_sponsor = sponsor_module.get("leadSponsor", {})
            sponsor = lead_sponsor.get("leadSponsorName", "")

            # Extract start date
            start_date_struct = status_module.get("startDateStruct", {})
            start_year = start_date_struct.get("year", None)

            return {
                "nct_id": nct_id,
                "brief_title": brief_title,
                "official_title": official_title,
                "study_type": study_type,
                "phase": phase,
                "status": status,
                "conditions": conditions,
                "interventions": interventions,
                "primary_outcome": primary_outcome,
                "enrollment": enrollment,
                "sponsor": sponsor,
                "start_year": start_year,
                "source": "clinicaltrials"
            }
        except Exception as e:
            print(f"Error parsing ClinicalTrials record: {e}")
            return None


# Convenience functions
def fetch_clinicaltrials_data(
    query: str = "",
    phases: List[str] = None,
    statuses: List[str] = None,
    max_results: int = 10000
) -> List[Dict[str, Any]]:
    """
    Fetch ClinicalTrials.gov data for given criteria.

    Args:
        query: Search query string
        phases: List of phases to include (e.g., ["Phase 2", "Phase 3"])
        statuses: List of statuses to include
        max_results: Maximum number of results

    Returns:
        List of trial dictionaries
    """
    fetcher = ClinicalTrialsFetcher()
    # Filter for interventional studies phases II-IV by default
    if phases is None:
        phases = ["Phase 2", "Phase 3", "Phase 4"]
    if statuses is None:
        statuses = ["Completed", "Active", "Recruiting"]

    nct_ids = fetcher.search_trials(
        query=query,
        filter_phases=phases,
        filter_status=statuses,
        max_results=max_results
    )
    return fetcher.fetch_trial_details(nct_ids)