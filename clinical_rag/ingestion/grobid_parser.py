"""
GROBID parser for section-aware PDF processing.
"""
import json
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
from ..config import settings


class GrobidParser:
    def __init__(self, grobid_url: str = "http://localhost:8070"):
        """
        Initialize GROBID parser.

        Args:
            grobid_url: URL of the GROBID service
        """
        self.grobid_url = grobid_url.rstrip('/')
        self.timeout = 120  # GROBID can be slow on large PDFs

    def is_available(self) -> bool:
        """
        Check if GROBID service is available.

        Returns:
            True if GROBID is available
        """
        try:
            response = requests.get(f"{self.grobid_url}/api/isalive", timeout=10)
            return response.status_code == 200 and response.text.strip() == "true"
        except Exception:
            return False

    def process_pdf(self, pdf_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Process a PDF file using GROBID to extract structured text.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing structured document data or None if failed
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            print(f"PDF file not found: {pdf_path}")
            return None

        if not self.is_available():
            print("GROBID service is not available. Please ensure GROBID is running.")
            print("You can start it with: docker run -p 8070:8070 grobid/grobid:latest")
            return None

        try:
            with open(pdf_path, 'rb') as f:
                files = {'input': (pdf_path.name, f, 'application/pdf')}
                response = requests.post(
                    f"{self.grobid_url}/api/processFulltextDocument",
                    files=files,
                    timeout=self.timeout
                )

            if response.status_code != 200:
                print(f"GROBID error: {response.status_code} - {response.text}")
                return None

            # Parse the TEI XML response
            tei_xml = response.text
            structured_data = self._parse_tei_xml(tei_xml)
            return structured_data

        except Exception as e:
            print(f"Error processing PDF with GROBID: {e}")
            return None

    def process_pdf_to_tei(self, pdf_path: Union[str, Path]) -> Optional[str]:
        """
        Process a PDF file and return raw TEI XML.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            TEI XML string or None if failed
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            print(f"PDF file not found: {pdf_path}")
            return None

        if not self.is_available():
            print("GROBID service is not available.")
            return None

        try:
            with open(pdf_path, 'rb') as f:
                files = {'input': (pdf_path.name, f, 'application/pdf')}
                response = requests.post(
                    f"{self.grobid_url}/api/processFulltextDocument",
                    files=files,
                    timeout=self.timeout
                )

            if response.status_code != 200:
                print(f"GROBID error: {response.status_code} - {response.text}")
                return None

            return response.text

        except Exception as e:
            print(f"Error processing PDF with GROBID: {e}")
            return None

    def _parse_tei_xml(self, tei_xml: str) -> Dict[str, Any]:
        """
        Parse TEI XML from GROBID into structured format.

        Args:
            tei_xml: TEI XML string from GROBID

        Returns:
            Dictionary with structured document data
        """
        try:
            root = ET.fromstring(tei_xml)

            # Define TEI namespace
            ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

            # Extract title
            title_elem = root.find('.//tei:titleStmt/tei:title', ns)
            title = title_elem.text if title_elem is not None else ""

            # Extract authors
            authors = []
            for author_elem in root.findall('.//tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author', ns):
                pers_name = author_elem.find('.//tei:persName', ns)
                if pers_name is not None:
                    # Extract forename and surname
                    forename = pers_name.find('./tei:forename', ns)
                    surname = pers_name.find('./tei:surname', ns)
                    name_parts = []
                    if forename is not None and forename.text:
                        name_parts.append(forename.text)
                    if surname is not None and surname.text:
                        name_parts.append(surname.text)
                    if name_parts:
                        authors.append(' '.join(name_parts))

            # Extract abstract
            abstract_elem = root.find('.//tei:profileDesc/tei:abstract', ns)
            abstract = ""
            if abstract_elem is not None:
                # Join all p elements in abstract
                abstract_parts = []
                for p_elem in abstract_elem.findall('.//tei:p', ns):
                    if p_elem.text:
                        abstract_parts.append(p_elem.text.strip())
                abstract = ' '.join(abstract_parts)

            # Extract sections (body content)
            sections = self._extract_sections(root, ns)

            # Extract bibliographic info
            bibliographic = self._extract_bibliographic_info(root, ns)

            return {
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "sections": sections,
                "bibliographic": bibliographic,
                "raw_tei": tei_xml  # Keep for debugging
            }

        except Exception as e:
            print(f"Error parsing TEI XML: {e}")
            return {
                "title": "",
                "authors": [],
                "abstract": "",
                "sections": [],
                "bibliographic": {},
                "raw_tei": tei_xml
            }

    def _extract_sections(self, root: ET.Element, ns: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Extract sections from TEI body.

        Args:
            root: TEI XML root element
            ns: Namespace dictionary

        Returns:
            List of section dictionaries
        """
        sections = []
        body_elem = root.find('.//tei:text/tei:body', ns)

        if body_elem is None:
            return sections

        # Find all div elements (sections)
        for div_elem in body_elem.findall('./tei:div', ns):
            # Get section header (head)
            head_elem = div_elem.find('./tei:head', ns)
            section_title = head_elem.text.strip() if head_elem is not None and head_elem.text else "Untitled Section"

            # Get section content (all p elements)
            content_parts = []
            for p_elem in div_elem.findall('.//tei:p', ns):
                if p_elem.text:
                    content_parts.append(p_elem.text.strip())

            section_content = ' '.join(content_parts)

            # Only add non-empty sections
            if section_content:
                sections.append({
                    "title": section_title,
                    "content": section_content,
                    "level": 1  # Could be enhanced to detect nesting
                })

        # If no sections found with div, try to get all p elements directly under body
        if not sections:
            content_parts = []
            for p_elem in body_elem.findall('./tei:p', ns):
                if p_elem.text:
                    content_parts.append(p_elem.text.strip())

            if content_parts:
                sections.append({
                    "title": "Main Content",
                    "content": ' '.join(content_parts),
                    "level": 1
                })

        return sections

    def _extract_bibliographic_info(self, root: ET.Element, ns: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract bibliographic information from TEI header.

        Args:
            root: TEI XML root element
            ns: Namespace dictionary

        Returns:
            Dictionary with bibliographic info
        """
        bibliographic = {}

        # Try to get publication year
        date_elem = root.find('.//tei:sourceDesc/tei:biblStruct/tei:monogr/tei:imprint/tei:date', ns)
        if date_elem is not None:
            # Try to get year from @when attribute or text
            year = date_elem.get('when') or date_elem.text
            if year:
                # Extract first 4 digits
                import re
                year_match = re.search(r'\d{4}', year)
                if year_match:
                    bibliographic['year'] = int(year_match.group())

        # Try to get journal title
        journal_elem = root.find('.//tei:sourceDesc/tei:biblStruct/tei:monogr/tei:title', ns)
        if journal_elem is not None:
            bibliographic['journal'] = journal_elem.text or ""

        # Try to get volume
        volume_elem = root.find('.//tei:sourceDesc/tei:biblStruct/tei:monogr/tei:imprint/tei:biblScope[@unit="volume"]', ns)
        if volume_elem is not None:
            bibliographic['volume'] = volume_elem.text or ""

        # Try to get issue
        issue_elem = root.find('.//tei:sourceDesc/tei:biblStruct/tei:monogr/tei:imprint/tei:biblScope[@unit="issue"]', ns)
        if issue_elem is not None:
            bibliographic['issue'] = issue_elem.text or ""

        # Try to get pages
        pages_elem = root.find('.//tei:sourceDesc/tei:biblStruct/tei:monogr/tei:imprint/tei:biblScope[@unit="page"]', ns)
        if pages_elem is not None:
            bibliographic['pages'] = pages_elem.text or ""

        return bibliographic

    def chunk_sections(self, structured_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert structured GROBID output to chunks suitable for indexing.

        Args:
            structured_data: Output from _parse_tei_xml

        Returns:
            List of chunk dictionaries
        """
        chunks = []

        # Add abstract as a chunk if available
        abstract = structured_data.get("abstract", "").strip()
        if abstract:
            chunks.append({
                "text": abstract,
                "metadata": {
                    "section": "Abstract",
                    "title": structured_data.get("title", ""),
                    "authors": structured_data.get("authors", []),
                    "year": structured_data.get("bibliographic", {}).get("year"),
                    "journal": structured_data.get("bibliographic", {}).get("journal"),
                    "chunk_type": "abstract"
                }
            })

        # Add each section as a chunk
        sections = structured_data.get("sections", [])
        for section in sections:
            section_title = section.get("title", "Untitled")
            section_content = section.get("content", "").strip()

            if section_content:
                chunks.append({
                    "text": section_content,
                    "metadata": {
                        "section": section_title,
                        "title": structured_data.get("title", ""),
                        "authors": structured_data.get("authors", []),
                        "year": structured_data.get("bibliographic", {}).get("year"),
                        "journal": structured_data.get("bibliographic", {}).get("journal"),
                        "chunk_type": "section"
                    }
                })

        return chunks


# Convenience functions
def process_pdf_with_grobid(pdf_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Process a PDF file using GROBID.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Structured document data or None
    """
    parser = GrobidParser()
    return parser.process_pdf(pdf_path)


def process_pdf_to_chunks(pdf_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Process a PDF file and return chunks ready for indexing.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of chunk dictionaries
    """
    parser = GrobidParser()
    structured_data = parser.process_pdf(pdf_path)
    if structured_data:
        return parser.chunk_sections(structured_data)
    return []


if __name__ == "__main__":
    # Test the GROBID parser
    print("GROBID Parser initialized")
    print("To use:")
    print("1. Start GROBID: docker run -p 8070:8070 grobid/grobid:latest")
    print("2. Call process_pdf_with_grobid('path/to/paper.pdf')")