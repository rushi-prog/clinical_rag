"""
PubMed data fetcher using Biopython Entrez API.
"""
import json
import time
from typing import List, Dict, Any, Optional
from Bio import Entrez
from tqdm import tqdm
import asyncio
import aiohttp
from ..config import settings


class PubMedFetcher:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize PubMed fetcher.

        Args:
            api_key: NCBI API key for higher rate limits
        """
        self.api_key = api_key or settings.NCBI_API_KEY
        if self.api_key:
            Entrez.api_key = self.api_key
        Entrez.email = "clinical.trial.rag@example.com"  # Required by NCBI

        # Rate limiting
        self.rate_limit = settings.PUBMED_RATE_LIMIT
        self.min_interval = 1.0 / self.rate_limit if self.rate_limit > 0 else 0
        self.last_request_time = 0

    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        if self.min_interval > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

    def search_pubmed(self, query: str, max_results: int = 10000) -> List[str]:
        """
        Search PubMed for articles matching query.

        Args:
            query: PubMed query string
            max_results: Maximum number of results to return

        Returns:
            List of PMID strings
        """
        self._rate_limit_wait()
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                usehistory="y"
            )
            results = Entrez.read(handle)
            handle.close()
            self.last_request_time = time.time()
            return results["IdList"]
        except Exception as e:
            print(f"Error searching PubMed: {e}")
            return []

    def fetch_article_details(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch detailed information for a list of PMIDs.

        Args:
            pmids: List of PMID strings

        Returns:
            List of article dictionaries
        """
        if not pmids:
            return []

        articles = []
        batch_size = 200  # NCBI recommends batches of 200

        for i in tqdm(range(0, len(pmids), batch_size), desc="Fetching PubMed articles"):
            batch_pmids = pmids[i:i+batch_size]
            self._rate_limit_wait()
            try:
                handle = Entrez.efetch(
                    db="pubmed",
                    id=",".join(batch_pmids),
                    rettype="medline",
                    retmode="xml"
                )
                records = Entrez.read(handle)
                handle.close()
                self.last_request_time = time.time()

                for record in records.get("PubmedArticle", []):
                    article = self._parse_pubmed_record(record)
                    if article:
                        articles.append(article)
            except Exception as e:
                print(f"Error fetching batch {i//batch_size}: {e}")
                continue

        return articles

    def _parse_pubmed_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse a PubMed record into standardized format.

        Args:
            record: Raw PubMed record from Entrez

        Returns:
            Standardized article dictionary
        """
        try:
            medline_citation = record.get("MedlineCitation", {})
            article = medline_citation.get("Article", {})

            # Extract basic info
            pmid = str(medline_citation.get("PMID", ""))
            title = article.get("ArticleTitle", "")

            # Extract abstract
            abstract_parts = article.get("Abstract", {}).get("AbstractText", [])
            if isinstance(abstract_parts, list):
                abstract = " ".join([str(part) for part in abstract_parts])
            else:
                abstract = str(abstract_parts) if abstract_parts else ""

            # Extract journal info
            journal = article.get("Journal", {})
            journal_title = journal.get("Title", "")
            pub_date = journal.get("JournalIssue", {}).get("PubDate", {})
            year = self._extract_year(pub_date)

            # Extract MeSH terms
            mesh_terms = []
            for mesh_heading in medline_citation.get("MeshHeadingList", []):
                descriptor = mesh_heading.get("DescriptorName", "")
                if isinstance(descriptor, dict):
                    descriptor = descriptor.get("#text", "")
                if descriptor:
                    mesh_terms.append(str(descriptor))

            # Extract publication types
            pub_types = []
            for pub_type in article.get("PublicationTypeList", []):
                if isinstance(pub_type, dict):
                    pub_type = pub_type.get("#text", "")
                if pub_type:
                    pub_types.append(str(pub_type))

            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal_title,
                "year": year,
                "mesh_terms": mesh_terms,
                "publication_types": pub_types,
                "source": "pubmed"
            }
        except Exception as e:
            print(f"Error parsing PubMed record: {e}")
            return None

    def _extract_year(self, pub_date: Dict[str, Any]) -> Optional[int]:
        """
        Extract year from PubMed publication date.

        Args:
            pub_date: Publication date dictionary

        Returns:
            Year as integer or None
        """
        # Try different date formats
        for date_field in ["Year", "MedlineDate"]:
            if date_field in pub_date:
                date_str = str(pub_date[date_field])
                # Extract first 4 digits as year
                import re
                year_match = re.search(r'\d{4}', date_str)
                if year_match:
                    return int(year_match.group())
        return None

    async def async_search_and_fetch(self, query: str, max_results: int = 10000) -> List[Dict[str, Any]]:
        """
        Asynchronously search PubMed and fetch articles.

        Args:
            query: PubMed query string
            max_results: Maximum number of results

        Returns:
            List of article dictionaries
        """
        # Search for PMIDs
        pmids = self.search_pubmed(query, max_results)
        if not pmids:
            return []

        # Fetch article details
        return self.fetch_article_details(pmids)


# Convenience functions
def fetch_pubmed_articles(query: str, max_results: int = 10000) -> List[Dict[str, Any]]:
    """
    Fetch PubMed articles for a query.

    Args:
        query: PubMed query string
        max_results: Maximum number of results

    Returns:
        List of article dictionaries
    """
    fetcher = PubMedFetcher()
    return fetcher.fetch_article_details(fetcher.search_pubmed(query, max_results))