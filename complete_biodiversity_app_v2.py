#!/usr/bin/env python3
"""
Biodiversity Research Pipeline - Streamlit Application
Comprehensive tool for species literature search, data extraction, and mapping

This application combines:
- Multi-database literature search (PubMed, CrossRef, bioRxiv, arXiv, Scopus)
- AI-powered species data extraction using Claude API or local Ollama
- Interactive mapping and visualization
- Comprehensive data analysis and reporting

Author: Biodiversity Research Tool
Version: 1.1 (Professional Edition)
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import time
import re
import xml.etree.ElementTree as ET
import csv
import zipfile
import io
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from streamlit_folium import st_folium

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Biodiversity Research Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
.main-header {
    font-size: 2.2rem;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 2rem;
    padding: 1.5rem;
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    border-radius: 8px;
    border: 1px solid #dee2e6;
    font-weight: 600;
}
.sub-header {
    font-size: 1.4rem;
    color: #495057;
    margin: 1.5rem 0;
    padding: 0.8rem;
    border-left: 4px solid #007bff;
    background: #f8f9fa;
    font-weight: 500;
}
.success-box {
    background: #d1ecf1;
    border: 1px solid #bee5eb;
    border-radius: 6px;
    padding: 1rem;
    margin: 1rem 0;
    color: #0c5460;
}
.warning-box {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 6px;
    padding: 1rem;
    margin: 1rem 0;
    color: #856404;
}
.error-box {
    background: #f8d7da;
    border: 1px solid #f5c6cb;
    border-radius: 6px;
    padding: 1rem;
    margin: 1rem 0;
    color: #721c24;
}
.metric-card {
    background: white;
    padding: 1.2rem;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin: 0.5rem 0;
    border: 1px solid #e9ecef;
}
.info-text {
    color: #6c757d;
    font-style: italic;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'species_data' not in st.session_state:
    st.session_state.species_data = None
if 'maps_created' not in st.session_state:
    st.session_state.maps_created = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

class OllamaExtractor:
    """Species data extractor using local Ollama"""
    
    def __init__(self, model_name="llama3.1:8b", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        
    def test_connection(self):
        """Test if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def extract_species_data(self, papers: List[Dict], max_papers: int = None) -> List[Dict]:
        """Extract species data using local Ollama"""
        if max_papers:
            papers = papers[:max_papers]
        
        st.write(f"**Extracting species data from {len(papers)} papers using Ollama ({self.model_name})**")
        
        all_species_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, paper in enumerate(papers):
            try:
                status_text.write(f"Processing paper {i+1}/{len(papers)}: {paper.get('title', 'Unknown')[:50]}...")
                
                # Create paper text
                text_parts = []
                if paper.get('title'):
                    text_parts.append(f"Title: {paper['title']}")
                if paper.get('abstract'):
                    text_parts.append(f"Abstract: {paper['abstract']}")
                
                for field in ['authors', 'journal', 'year']:
                    if paper.get(field):
                        text_parts.append(f"{field.title()}: {paper[field]}")
                
                if not text_parts:
                    continue
                
                paper_text = "\n\n".join(text_parts)
                
                # Optimized prompt for local LLMs
                prompt = f"""Extract species information from this research paper. Return ONLY a valid JSON array.

For each species mentioned in the study (not just examples or background), extract:
- species: scientific name (Genus species format)
- number: specimen count or "number not specified"
- study_type: "Laboratory", "Field", or "Field+Laboratory"
- location: study location/site

Return format (use simple strings only, no nested objects):
[
  {{
    "species": "Genus species",
    "number": "count or number not specified",
    "study_type": "Laboratory/Field/Field+Laboratory",
    "location": "location description"
  }}
]

Paper: {paper.get('title', 'Unknown')}
DOI: {paper.get('doi', 'Unknown')}

Text to analyze:
{paper_text[:8000]}

JSON:"""

                # Call Ollama API
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "top_p": 0.9,
                            "num_ctx": 4096
                        }
                    },
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    generated_text = result.get('response', '')
                    
                    # Parse JSON from response
                    species_data = self.parse_llm_response(generated_text, paper)
                    all_species_data.extend(species_data)
                    
                    if species_data:
                        status_text.write(f"Successfully extracted {len(species_data)} species from this paper")
                    else:
                        status_text.write(f"No species data extracted from this paper")
                else:
                    status_text.write(f"Ollama API error: {response.status_code}")
                
                # Update progress
                progress_value = min((i + 1), len(papers)) / len(papers)
                progress_bar.progress(progress_value)
                
                # Small delay to prevent overwhelming the local model
                time.sleep(1)
                
            except Exception as e:
                status_text.write(f"Error processing paper: {e}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        st.write(f"**Ollama extraction complete.** Found {len(all_species_data)} species entries")
        return all_species_data
    
    def parse_llm_response(self, response_text: str, paper: Dict) -> List[Dict]:
        """Parse LLM response and extract JSON"""
        try:
            # Find JSON in response
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if not json_match:
                # Try to find single object
                json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
                if json_match:
                    json_text = '[' + json_match.group(0) + ']'
                else:
                    return []
            else:
                json_text = json_match.group(0)
            
            # Clean up common JSON issues
            json_text = re.sub(r',\s*}', '}', json_text)  # Remove trailing commas
            json_text = re.sub(r',\s*]', ']', json_text)  # Remove trailing commas
            
            result = json.loads(json_text)
            
            if isinstance(result, dict):
                result = [result]
            
            # Process results
            processed_data = []
            for item in result:
                if isinstance(item, dict):
                    clean_item = {
                        'query_species': paper.get('title', 'Unknown')[:50],
                        'paper_link': paper.get('doi', paper.get('url', 'Unknown')),
                        'species': str(item.get('species', 'Unknown')).strip(),
                        'number': str(item.get('number', 'unknown')).strip(),
                        'study_type': str(item.get('study_type', 'Unknown')).strip(),
                        'location': str(item.get('location', 'Unknown')).strip(),
                        'doi': paper.get('doi', ''),
                        'paper_title': paper.get('title', 'Unknown')
                    }
                    processed_data.append(clean_item)
            
            return processed_data
        
        except json.JSONDecodeError as e:
            # Try to extract individual fields if JSON parsing fails
            return self.fallback_parsing(response_text, paper)
        except Exception as e:
            return []
    
    def fallback_parsing(self, response_text: str, paper: Dict) -> List[Dict]:
        """Fallback parsing when JSON fails"""
        try:
            # Look for species names in the response
            species_patterns = [
                r'species["\s]*:["\s]*([A-Z][a-z]+\s+[a-z]+)',
                r'"([A-Z][a-z]+\s+[a-z]+)"',
                r'([A-Z][a-z]+\s+[a-z]+)'
            ]
            
            found_species = []
            for pattern in species_patterns:
                matches = re.findall(pattern, response_text)
                found_species.extend(matches)
            
            # Create basic entries for found species
            processed_data = []
            for species in found_species[:3]:  # Limit to 3 to avoid noise
                clean_item = {
                    'query_species': paper.get('title', 'Unknown')[:50],
                    'paper_link': paper.get('doi', paper.get('url', 'Unknown')),
                    'species': species.strip(),
                    'number': 'unknown',
                    'study_type': 'Unknown',
                    'location': 'Unknown',
                    'doi': paper.get('doi', ''),
                    'paper_title': paper.get('title', 'Unknown')
                }
                processed_data.append(clean_item)
            
            return processed_data
        except:
            return []

class BiodiversityPipeline:
    """Main pipeline class that combines all functionality"""
    
    def __init__(self):
        self.search_results = []
        self.species_data = []
        self.maps = {}
        self.analysis_stats = {}
        
    def clean_text_for_csv(self, text):
        """Clean text to prevent CSV parsing issues"""
        if not text or pd.isna(text):
            return ""
        
        text = str(text).strip()
        text = re.sub(r'[\r\n]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('\x00', '')
        
        return text.strip()
    
    def search_pubmed(self, species: str, max_results: int = 25, start_year: int = 2015, end_year: int = 2025) -> List[Dict]:
        """Search PubMed database"""
        st.write(f"Searching PubMed for '{species}'...")
        
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        query = f'("{species}"[Title/Abstract]) AND ("{start_year}"[PDAT] : "{end_year}"[PDAT])'
        
        try:
            # Search for PMIDs
            search_response = requests.get(f"{base_url}/esearch.fcgi", params={
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json',
                'sort': 'relevance'
            }, timeout=30)
            search_response.raise_for_status()
            
            pmids = search_response.json().get('esearchresult', {}).get('idlist', [])
            if not pmids:
                st.write("  No PubMed results found")
                return []
            
            st.write(f"  Found {len(pmids)} PubMed results")
            
            # Fetch details
            results = []
            batch_size = 10
            progress_bar = st.progress(0)
            
            for i in range(0, len(pmids), batch_size):
                batch_pmids = pmids[i:i+batch_size]
                
                fetch_response = requests.get(f"{base_url}/efetch.fcgi", params={
                    'db': 'pubmed',
                    'id': ','.join(batch_pmids),
                    'retmode': 'xml',
                    'rettype': 'abstract'
                }, timeout=30)
                fetch_response.raise_for_status()
                
                root = ET.fromstring(fetch_response.content)
                
                for article in root.findall('.//PubmedArticle'):
                    paper_data = self.parse_pubmed_article(article)
                    if paper_data:
                        results.append(paper_data)
                
                # Fix progress calculation to never exceed 1.0
                progress_value = min((i + batch_size), len(pmids)) / len(pmids)
                progress_bar.progress(progress_value)
                time.sleep(0.5)
            
            progress_bar.empty()
            st.write(f"  Successfully parsed {len(results)} PubMed papers")
            return results
            
        except Exception as e:
            st.error(f"Error searching PubMed: {e}")
            return []
    
    def parse_pubmed_article(self, article_element) -> Optional[Dict]:
        """Parse PubMed article XML"""
        try:
            # Extract PMID
            pmid_elem = article_element.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else ""
            
            # Extract title
            title_elem = article_element.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else ""
            
            # Extract authors
            authors = []
            for author in article_element.findall('.//Author'):
                lastname = author.find('LastName')
                forename = author.find('ForeName')
                if lastname is not None and forename is not None:
                    authors.append(f"{lastname.text}, {forename.text}")
                elif lastname is not None:
                    authors.append(lastname.text)
            authors_str = "; ".join(authors)
            
            # Extract journal
            journal_elem = article_element.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else ""
            
            # Extract year
            year_elem = article_element.find('.//PubDate/Year')
            year = year_elem.text if year_elem is not None else ""
            
            # Extract abstract
            abstract_parts = []
            for abstract_text in article_element.findall('.//AbstractText'):
                if abstract_text.text:
                    abstract_parts.append(abstract_text.text)
            abstract = " ".join(abstract_parts)
            
            # Extract DOI
            doi = ""
            for article_id in article_element.findall('.//ArticleId'):
                if article_id.get('IdType') == 'doi':
                    doi = article_id.text
                    break
            
            return {
                'authors': authors_str,
                'journal': journal,
                'year': year,
                'abstract': abstract,
                'doi': doi,
                'pmid': pmid,
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                'database': 'PubMed',
                'title': title
            }
            
        except Exception as e:
            st.error(f"Error parsing PubMed article: {e}")
            return None
    
    def search_crossref(self, species: str, max_results: int = 25, start_year: int = 2015, end_year: int = 2025) -> List[Dict]:
        """Search CrossRef database"""
        st.write(f"Searching CrossRef for '{species}'...")
        
        try:
            response = requests.get("https://api.crossref.org/works", params={
                'query': species,
                'rows': max_results,
                'filter': f'from-pub-date:{start_year},until-pub-date:{end_year}',
                'sort': 'relevance',
                'select': 'DOI,title,author,published-print,published-online,container-title,abstract,URL'
            }, headers={
                'User-Agent': 'Academic Research Tool (mailto:researcher@example.com)'
            }, timeout=30)
            response.raise_for_status()
            
            items = response.json().get('message', {}).get('items', [])
            if not items:
                st.write("  No CrossRef results found")
                return []
            
            st.write(f"  Found {len(items)} CrossRef results")
            
            results = []
            for item in items:
                paper_data = self.parse_crossref_item(item)
                if paper_data:
                    results.append(paper_data)
            
            st.write(f"  Successfully parsed {len(results)} CrossRef papers")
            return results
            
        except Exception as e:
            st.error(f"Error searching CrossRef: {e}")
            return []
    
    def parse_crossref_item(self, item: Dict) -> Optional[Dict]:
        """Parse CrossRef item"""
        try:
            title_list = item.get('title', [])
            title = title_list[0] if title_list else ""
            
            authors = []
            for author in item.get('author', []):
                given = author.get('given', '')
                family = author.get('family', '')
                if family:
                    if given:
                        authors.append(f"{family}, {given}")
                    else:
                        authors.append(family)
            authors_str = "; ".join(authors)
            
            container_title = item.get('container-title', [])
            journal = container_title[0] if container_title else ""
            
            year = ""
            pub_date = item.get('published-print') or item.get('published-online')
            if pub_date and 'date-parts' in pub_date:
                date_parts = pub_date['date-parts'][0]
                if date_parts:
                    year = str(date_parts[0])
            
            return {
                'authors': authors_str,
                'journal': journal,
                'year': year,
                'abstract': item.get('abstract', ''),
                'doi': item.get('DOI', ''),
                'pmid': '',
                'url': item.get('URL', ''),
                'database': 'CrossRef',
                'title': title
            }
            
        except Exception as e:
            st.error(f"Error parsing CrossRef item: {e}")
            return None
    
    def search_biorxiv(self, species: str, max_results: int = 25, start_year: int = 2015, end_year: int = 2025) -> List[Dict]:
        """Search bioRxiv database"""
        st.write(f"Searching bioRxiv for '{species}'...")
        
        try:
            start_date = f"{start_year}-01-01"
            end_date = f"{end_year}-12-31"
            
            response = requests.get(f"https://api.biorxiv.org/details/biorxiv/{start_date}/{end_date}", timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if 'collection' not in data:
                st.write("  No bioRxiv results found")
                return []
            
            # Filter by species name
            all_papers = data['collection']
            filtered_papers = []
            species_lower = species.lower()
            
            for paper in all_papers:
                title = paper.get('title', '').lower()
                abstract = paper.get('abstract', '').lower()
                
                if species_lower in title or species_lower in abstract:
                    filtered_papers.append(paper)
                    
                if len(filtered_papers) >= max_results:
                    break
            
            if not filtered_papers:
                st.write("  No relevant bioRxiv results found")
                return []
            
            st.write(f"  Found {len(filtered_papers)} relevant bioRxiv results")
            
            results = []
            for paper in filtered_papers:
                results.append({
                    'authors': paper.get('authors', ''),
                    'journal': 'bioRxiv (preprint)',
                    'year': paper.get('date', '')[:4] if paper.get('date') else '',
                    'abstract': paper.get('abstract', ''),
                    'doi': paper.get('doi', ''),
                    'pmid': '',
                    'url': f"https://www.biorxiv.org/content/{paper.get('doi', '')}v1",
                    'database': 'bioRxiv',
                    'title': paper.get('title', '')
                })
            
            st.write(f"  Successfully parsed {len(results)} bioRxiv papers")
            return results
            
        except Exception as e:
            st.error(f"Error searching bioRxiv: {e}")
            return []
    
    def search_arxiv(self, species: str, max_results: int = 25, start_year: int = 2015, end_year: int = 2025) -> List[Dict]:
        """Search arXiv database"""
        st.write(f"Searching arXiv for '{species}'...")
        
        try:
            response = requests.get("http://export.arxiv.org/api/query", params={
                'search_query': f'all:"{species}"',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }, timeout=30)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('atom:entry', namespace)
            
            if not entries:
                st.write("  No arXiv results found")
                return []
            
            st.write(f"  Found {len(entries)} arXiv results")
            
            results = []
            for entry in entries:
                paper_data = self.parse_arxiv_entry(entry, namespace, start_year, end_year)
                if paper_data:
                    results.append(paper_data)
            
            st.write(f"  Successfully parsed {len(results)} arXiv papers")
            return results
            
        except Exception as e:
            st.error(f"Error searching arXiv: {e}")
            return []
    
    def parse_arxiv_entry(self, entry, namespace: Dict, start_year: int, end_year: int) -> Optional[Dict]:
        """Parse arXiv entry"""
        try:
            title_elem = entry.find('atom:title', namespace)
            title = title_elem.text.strip() if title_elem is not None else ""
            
            authors = []
            for author in entry.findall('atom:author', namespace):
                name_elem = author.find('atom:name', namespace)
                if name_elem is not None:
                    authors.append(name_elem.text)
            authors_str = "; ".join(authors)
            
            published_elem = entry.find('atom:published', namespace)
            published = published_elem.text if published_elem is not None else ""
            year = published[:4] if len(published) >= 4 else ""
            
            if year and (int(year) < start_year or int(year) > end_year):
                return None
            
            summary_elem = entry.find('atom:summary', namespace)
            abstract = summary_elem.text.strip() if summary_elem is not None else ""
            
            id_elem = entry.find('atom:id', namespace)
            arxiv_url = id_elem.text if id_elem is not None else ""
            
            return {
                'authors': authors_str,
                'journal': 'arXiv (preprint)',
                'year': year,
                'abstract': abstract,
                'doi': '',
                'pmid': '',
                'url': arxiv_url,
                'database': 'arXiv',
                'title': title
            }
            
        except Exception as e:
            st.error(f"Error parsing arXiv entry: {e}")
            return None
    
    def search_scopus(self, species: str, api_key: str, max_results: int = 25, start_year: int = 2015, end_year: int = 2025, inst_token: str = None) -> List[Dict]:
        """Search Scopus database"""
        if not api_key:
            st.write("  Skipping Scopus search (no API key provided)")
            return []
        
        st.write(f"Searching Scopus for '{species}'...")
        
        try:
            headers = {
                'X-ELS-APIKey': api_key,
                'Accept': 'application/json'
            }
            
            if inst_token:
                headers['X-ELS-Insttoken'] = inst_token
            
            response = requests.get("https://api.elsevier.com/content/search/scopus", headers=headers, params={
                'query': f'TITLE-ABS-KEY("{species}") AND PUBYEAR > {start_year-1} AND PUBYEAR < {end_year+1}',
                'count': max_results,
                'sort': 'relevancy',
                'field': 'dc:title,dc:creator,prism:publicationName,prism:coverDate,dc:description,prism:doi,dc:identifier,prism:url'
            }, timeout=30)
            response.raise_for_status()
            
            entries = response.json().get('search-results', {}).get('entry', [])
            if not entries:
                st.write("  No Scopus results found")
                return []
            
            st.write(f"  Found {len(entries)} Scopus results")
            
            results = []
            for entry in entries:
                results.append({
                    'authors': entry.get('dc:creator', ''),
                    'journal': entry.get('prism:publicationName', ''),
                    'year': entry.get('prism:coverDate', '')[:4] if entry.get('prism:coverDate') else '',
                    'abstract': entry.get('dc:description', ''),
                    'doi': entry.get('prism:doi', ''),
                    'pmid': '',
                    'url': entry.get('prism:url', ''),
                    'database': 'Scopus',
                    'title': entry.get('dc:title', '')
                })
            
            st.write(f"  Successfully parsed {len(results)} Scopus papers")
            return results
            
        except Exception as e:
            st.error(f"Error searching Scopus: {e}")
            return []
    
    def remove_duplicates(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers"""
        if not papers:
            return []
        
        unique_papers = []
        seen_dois = set()
        seen_titles = set()
        
        for paper in papers:
            doi = paper.get('doi', '').strip()
            title = paper.get('title', '').strip().lower()
            
            # Check DOI duplicates
            if doi and doi in seen_dois:
                continue
            
            # Check title similarity
            title_words = set(title.split())
            is_duplicate = False
            
            for seen_title in seen_titles:
                seen_words = set(seen_title.split())
                if title_words and seen_words:
                    overlap = len(title_words & seen_words) / len(title_words | seen_words)
                    if overlap > 0.8:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_papers.append(paper)
                if doi:
                    seen_dois.add(doi)
                if title:
                    seen_titles.add(title)
        
        return unique_papers
    
    def search_all_databases(self, species: str, start_year: int, end_year: int, max_results: int, 
                           scopus_key: str = None, scopus_token: str = None) -> List[Dict]:
        """Search all available databases"""
        st.write(f"**Searching all databases for: {species}**")
        
        all_papers = []
        
        # Define search functions
        search_functions = [
            ("PubMed", lambda: self.search_pubmed(species, max_results, start_year, end_year)),
            ("CrossRef", lambda: self.search_crossref(species, max_results, start_year, end_year)),
            ("bioRxiv", lambda: self.search_biorxiv(species, max_results, start_year, end_year)),
            ("arXiv", lambda: self.search_arxiv(species, max_results, start_year, end_year)),
        ]
        
        if scopus_key:
            search_functions.append(("Scopus", lambda: self.search_scopus(species, scopus_key, max_results, start_year, end_year, scopus_token)))
        
        # Execute searches
        database_results = {}
        progress_container = st.container()
        
        for i, (db_name, search_func) in enumerate(search_functions):
            with progress_container:
                st.write(f"**{db_name}** ({i+1}/{len(search_functions)})")
                try:
                    results = search_func()
                    all_papers.extend(results)
                    database_results[db_name] = len(results)
                    st.success(f"{db_name}: {len(results)} papers found")
                except Exception as e:
                    st.error(f"{db_name}: Error - {e}")
                    database_results[db_name] = 0
                
                if i < len(search_functions) - 1:
                    time.sleep(1)
        
        # Remove duplicates
        st.write(f"**Total papers before deduplication:** {len(all_papers)}")
        unique_papers = self.remove_duplicates(all_papers)
        st.write(f"**Unique papers after deduplication:** {len(unique_papers)}")
        
        # Show database breakdown
        if unique_papers:
            st.write("**Papers by database:**")
            cols = st.columns(len(database_results))
            for i, (db, count) in enumerate(database_results.items()):
                with cols[i]:
                    st.metric(db, count)
        
        return unique_papers
    
    def extract_species_data_with_claude(self, papers: List[Dict], claude_api_key: str, max_papers: int = None) -> List[Dict]:
        """Extract species data using Claude API"""
        if not claude_api_key:
            st.error("Claude API key is required for species extraction")
            return []
        
        if max_papers:
            papers = papers[:max_papers]
        
        st.write(f"**Extracting species data from {len(papers)} papers using Claude API**")
        
        all_species_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        headers = {
            "x-api-key": claude_api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        for i, paper in enumerate(papers):
            try:
                status_text.write(f"Processing paper {i+1}/{len(papers)}: {paper.get('title', 'Unknown')[:50]}...")
                
                # Create paper text
                text_parts = []
                
                if paper.get('title'):
                    text_parts.append(f"Title: {paper['title']}")
                
                if paper.get('abstract'):
                    text_parts.append(f"Abstract: {paper['abstract']}")
                
                for field in ['authors', 'journal', 'year']:
                    if paper.get(field):
                        text_parts.append(f"{field.title()}: {paper[field]}")
                
                if not text_parts:
                    continue
                
                paper_text = "\n\n".join(text_parts)
                
                # Claude API prompt
                prompt = f"""
                Extract species information from this research paper. Return ONLY a JSON array of objects.

                For each species mentioned in the study (not just examples or background), extract:
                - species: scientific name (Genus species format)
                - number: specimen count or "number not specified"
                - study_type: "Laboratory", "Field", or "Field+Laboratory"
                - location: study location/site

                Return format (use simple strings only, no nested objects):
                [
                  {{
                    "species": "Genus species",
                    "number": "count or number not specified",
                    "study_type": "Laboratory/Field/Field+Laboratory",
                    "location": "location description"
                  }}
                ]

                Paper: {paper.get('title', 'Unknown')}
                DOI: {paper.get('doi', 'Unknown')}

                Text to analyze:
                {paper_text[:50000]}
                """
                
                payload = {
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 1500,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0
                }
                
                # Make API request with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = requests.post(
                            "https://api.anthropic.com/v1/messages",
                            headers=headers,
                            json=payload, 
                            timeout=60
                        )
                        
                        if response.status_code == 429:
                            wait_time = min(2 ** attempt, 60)
                            status_text.write(f"Rate limit hit. Waiting {wait_time} seconds...")
                            time.sleep(wait_time)
                            continue
                            
                        if response.status_code != 200:
                            raise Exception(f"API request failed: {response.text}")
                            
                        break
                        
                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait_time = min(2 ** attempt, 60)
                            status_text.write(f"Error: {e}. Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            raise
                else:
                    raise Exception("Max retries exceeded")
                
                # Parse Claude response
                response_data = response.json()
                claude_response = response_data["content"][0]["text"]
                
                # Clean and parse JSON
                try:
                    # Remove markdown formatting
                    claude_response = re.sub(r'```(?:json)?\n', '', claude_response)
                    claude_response = re.sub(r'\n```', '', claude_response)
                    
                    json_match = re.search(r'(\[.*\]|\{.*\})', claude_response, re.DOTALL)
                    if json_match:
                        json_text = json_match.group(1)
                    else:
                        json_text = claude_response
                    
                    result = json.loads(json_text)
                    
                    if isinstance(result, dict):
                        result = [result]
                    
                    # Process results
                    for item in result:
                        if isinstance(item, dict):
                            clean_item = {
                                'query_species': paper.get('title', 'Unknown')[:50],
                                'paper_link': paper.get('doi', paper.get('url', 'Unknown')),
                                'species': str(item.get('species', 'Unknown')).strip(),
                                'number': str(item.get('number', 'unknown')).strip(),
                                'study_type': str(item.get('study_type', 'Unknown')).strip(),
                                'location': str(item.get('location', 'Unknown')).strip(),
                                'doi': paper.get('doi', ''),
                                'paper_title': paper.get('title', 'Unknown')
                            }
                            all_species_data.append(clean_item)
                    
                    if result:
                        status_text.write(f"Successfully extracted {len(result)} species from this paper")
                    
                except json.JSONDecodeError as e:
                    status_text.write(f"Could not parse Claude response for this paper")
                    continue
                
                # Update progress
                progress_value = min((i + 1), len(papers)) / len(papers)
                progress_bar.progress(progress_value)
                
                # Rate limiting
                if i < len(papers) - 1:
                    time.sleep(5)
                    
            except Exception as e:
                status_text.write(f"Error processing paper: {e}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        st.write(f"**Species extraction complete.** Found {len(all_species_data)} species entries")
        return all_species_data
    
    def create_interactive_maps(self, species_data: pd.DataFrame):
        """Create interactive maps from species data"""
        st.write("**Creating interactive maps...**")
        
        if species_data.empty:
            st.warning("No species data available for mapping")
            return {}
        
        # Initialize geocoder
        geolocator = Nominatim(user_agent="biodiversity_app_v1.0")
        
        # Get unique locations
        unique_locations = species_data['location'].unique()
        location_coords = {}
        
        st.write(f"Geocoding {len(unique_locations)} unique locations...")
        geocoding_progress = st.progress(0)
        
        for i, location in enumerate(unique_locations):
            if pd.isna(location) or location in ['unknown', 'Unknown', 'UNSPECIFIED']:
                continue
                
            try:
                result = geolocator.geocode(location, timeout=10)
                if result:
                    location_coords[location] = {
                        'lat': result.latitude,
                        'lon': result.longitude,
                        'display_name': result.address
                    }
                time.sleep(1)  # Rate limiting
            except Exception as e:
                continue
            
            # Fix progress calculation to never exceed 1.0
            progress_value = min((i + 1), len(unique_locations)) / len(unique_locations)
            geocoding_progress.progress(progress_value)
        
        geocoding_progress.empty()
        
        if not location_coords:
            st.warning("Could not geocode any locations")
            return {}
        
        st.write(f"Successfully geocoded {len(location_coords)} locations")
        
        # Create maps
        maps = {}
        
        # 1. Species Distribution Map
        st.write("Creating species distribution map...")
        
        # Calculate center
        lats = [coord['lat'] for coord in location_coords.values()]
        lons = [coord['lon'] for coord in location_coords.values()]
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)
        
        # Create base map
        m_species = folium.Map(location=[center_lat, center_lon], zoom_start=6)
        
        # Add markers for each location
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'pink', 'gray', 'darkblue', 'darkgreen']
        
        location_species_count = species_data.groupby('location')['species'].nunique().to_dict()
        
        for location, coords in location_coords.items():
            if location in location_species_count:
                species_count = location_species_count[location]
                location_data = species_data[species_data['location'] == location]
                
                # Create popup content
                popup_html = f"""
                <div style="width: 300px;">
                    <h4>{location}</h4>
                    <p><strong>Species found:</strong> {species_count}</p>
                    <p><strong>Total records:</strong> {len(location_data)}</p>
                    <hr>
                    <p><strong>Species list:</strong></p>
                    <ul>
                """
                
                for species in location_data['species'].unique()[:10]:  # Show max 10 species
                    popup_html += f"<li>{species}</li>"
                
                if species_count > 10:
                    popup_html += f"<li>... and {species_count - 10} more</li>"
                
                popup_html += "</ul></div>"
                
                # Marker size based on species count
                marker_size = min(5 + species_count * 2, 20)
                
                folium.CircleMarker(
                    location=[coords['lat'], coords['lon']],
                    radius=marker_size,
                    popup=folium.Popup(popup_html, max_width=320),
                    tooltip=f"{location}: {species_count} species",
                    color='white',
                    fillColor='blue',
                    fillOpacity=0.7,
                    weight=2
                ).add_to(m_species)
        
        maps['species_distribution'] = m_species
        
        # 2. Heatmap of species density
        st.write("Creating species density heatmap...")
        
        m_heatmap = folium.Map(location=[center_lat, center_lon], zoom_start=6)
        
        # Prepare heatmap data
        heat_data = []
        max_species = max(location_species_count.values()) if location_species_count else 1
        
        for location, coords in location_coords.items():
            if location in location_species_count:
                weight = (location_species_count[location] / max_species) * 100
                heat_data.append([coords['lat'], coords['lon'], weight])
        
        if heat_data:
            HeatMap(heat_data, radius=30, blur=25, max_zoom=18).add_to(m_heatmap)
        
        maps['heatmap'] = m_heatmap
        
        st.write("**Interactive maps created successfully**")
        return maps

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">Biodiversity Research Pipeline</div>', unsafe_allow_html=True)
    st.markdown("**Comprehensive tool for species literature search, AI-powered data extraction, and interactive mapping**")
    st.markdown('<p class="info-text">Advanced research platform supporting both Claude API and local Ollama models for cost-effective, unlimited analysis</p>', unsafe_allow_html=True)
    
    # Sidebar for inputs
    st.sidebar.markdown("### Configuration")
    
    # API Keys and LLM Configuration
    st.sidebar.markdown("#### AI Configuration")
    llm_option = st.sidebar.radio(
        "Choose AI Backend:",
        ["Claude API", "Ollama (Local)"],
        help="Claude API requires internet connection and API usage costs. Ollama runs locally and is free."
    )
    
    if llm_option == "Claude API":
        claude_api_key = st.sidebar.text_input("Claude API Key", type="password", help="Required for species data extraction")
        scopus_api_key = st.sidebar.text_input("Scopus API Key (Optional)", type="password", help="Optional - provides access to additional academic papers")
        scopus_token = st.sidebar.text_input("Scopus Institutional Token (Optional)", type="password")
    else:  # Ollama
        claude_api_key = None  # Not needed for Ollama
        scopus_api_key = st.sidebar.text_input("Scopus API Key (Optional)", type="password", help="Optional - provides access to additional academic papers")
        scopus_token = st.sidebar.text_input("Scopus Institutional Token (Optional)", type="password")
        
        # Ollama configuration
        st.sidebar.markdown("**Ollama Settings**")
        ollama_model = st.sidebar.selectbox(
            "Ollama Model:",
            ["llama3.1:8b", "llama3.1:13b", "llama3.1:70b", "mistral:7b", "codellama:7b"],
            help="llama3.1:8b is recommended for most users"
        )
        ollama_url = st.sidebar.text_input("Ollama URL:", "http://localhost:11434")
        
        # Test Ollama connection
        if st.sidebar.button("Test Ollama Connection"):
            extractor = OllamaExtractor(ollama_model, ollama_url)
            if extractor.test_connection():
                st.sidebar.success("Ollama is running and accessible")
                # Test if model is available
                try:
                    response = requests.get(f"{ollama_url}/api/tags", timeout=5)
                    if response.status_code == 200:
                        available_models = [model['name'] for model in response.json().get('models', [])]
                        if ollama_model in available_models:
                            st.sidebar.success(f"Model '{ollama_model}' is available")
                        else:
                            st.sidebar.warning(f"Model '{ollama_model}' not found. Available models: {', '.join(available_models)}")
                            st.sidebar.info(f"To download: `ollama pull {ollama_model}`")
                except:
                    st.sidebar.warning("Could not check available models")
            else:
                st.sidebar.error("Cannot connect to Ollama")
                st.sidebar.info("Make sure Ollama is running: `ollama serve`")
        
        # Show setup instructions
        with st.sidebar.expander("Ollama Setup Instructions"):
            st.markdown("""
            **Quick Setup:**
            ```bash
            # 1. Install Ollama
            curl -fsSL https://ollama.ai/install.sh | sh
            
            # 2. Start Ollama
            ollama serve
            
            # 3. Download model (in another terminal)
            ollama pull llama3.1:8b
            
            # 4. Test
            ollama run llama3.1:8b
            ```
            
            **Requirements:**
            - 8GB+ RAM for 8B models
            - 16GB+ RAM for 13B models
            - 40GB+ RAM for 70B models
            """)
    
    # Species input
    st.sidebar.markdown("#### Species Selection")
    input_method = st.sidebar.radio("Choose input method:", ["Single species", "Upload species list"])
    
    if input_method == "Single species":
        species_input = st.sidebar.text_input("Enter species name", placeholder="e.g., Xyrichtys novacula")
        species_list = [species_input] if species_input else []
    else:
        uploaded_file = st.sidebar.file_uploader("Upload species list (.txt)", type=['txt'])
        if uploaded_file:
            species_content = uploaded_file.read().decode('utf-8')
            species_list = [line.strip() for line in species_content.splitlines() if line.strip()]
        else:
            species_list = []
    
    # Search parameters
    st.sidebar.markdown("#### Search Parameters")
    start_year = st.sidebar.number_input("Start Year", min_value=1900, max_value=2025, value=2015)
    end_year = st.sidebar.number_input("End Year", min_value=1900, max_value=2025, value=2025)
    max_results = st.sidebar.number_input("Max papers per database", min_value=1, max_value=100, value=25)
    max_papers_extract = st.sidebar.number_input("Max papers to extract data from", min_value=1, max_value=500, value=50, 
                                                help="Limit to control API usage and processing time")
    
    # Geographic area filter (optional)
    st.sidebar.markdown("#### Geographic Filter (Optional)")
    area_filter = st.sidebar.text_input("Area of interest", placeholder="e.g., Mediterranean, North Atlantic")
    
    # Main content area
    if st.sidebar.button("Start Analysis", type="primary"):
        if not species_list:
            st.error("Please provide at least one species name")
            return
        
        # Validate AI backend
        if llm_option == "Claude API":
            if not claude_api_key:
                st.error("Claude API key is required when using Claude API backend")
                return
        else:  # Ollama
            extractor = OllamaExtractor(ollama_model, ollama_url)
            if not extractor.test_connection():
                st.error("Cannot connect to Ollama. Make sure it's running with: `ollama serve`")
                st.info("See setup instructions in the sidebar")
                return
        
        # Initialize pipeline
        pipeline = BiodiversityPipeline()
        
        # Progress tracking
        total_species = len(species_list)
        species_progress = st.progress(0)
        species_status = st.empty()
        
        all_search_results = []
        all_species_data = []
        
        # Process each species
        for i, species in enumerate(species_list):
            species_status.write(f"**Processing species {i+1}/{total_species}: {species}**")
            
            # Search databases
            search_results = pipeline.search_all_databases(
                species, start_year, end_year, max_results, scopus_api_key, scopus_token
            )
            
            if search_results:
                all_search_results.extend(search_results)
                
                # Extract species data using chosen backend
                if llm_option == "Claude API":
                    species_data = pipeline.extract_species_data_with_claude(
                        search_results, claude_api_key, max_papers_extract
                    )
                else:  # Ollama
                    extractor = OllamaExtractor(ollama_model, ollama_url)
                    species_data = extractor.extract_species_data(search_results, max_papers_extract)
                
                if species_data:
                    all_species_data.extend(species_data)
            
            # Fix progress calculation to never exceed 1.0
            progress_value = min((i + 1), total_species) / total_species
            species_progress.progress(progress_value)
        
        species_progress.empty()
        species_status.empty()
        
        # Store results in session state
        st.session_state.search_results = all_search_results
        st.session_state.species_data = all_species_data
        
        # Display results
        if all_search_results:
            ai_backend = "Claude API" if llm_option == "Claude API" else f"Ollama ({ollama_model})"
            st.success(f"**Analysis complete using {ai_backend}.** Found {len(all_search_results)} papers and extracted {len(all_species_data)} species records")
        else:
            st.warning("No results found. Try different species names or broader search parameters.")
    
    # Display results if available
    if st.session_state.search_results or st.session_state.species_data:
        display_results(st.session_state.search_results, st.session_state.species_data)

def display_results(search_results, species_data):
    """Display analysis results"""
    
    st.markdown('<div class="sub-header">Analysis Results</div>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Papers Found", len(search_results) if search_results else 0)
    
    with col2:
        st.metric("Species Records", len(species_data) if species_data else 0)
    
    with col3:
        unique_species = len(set(item['species'] for item in species_data)) if species_data else 0
        st.metric("Unique Species", unique_species)
    
    with col4:
        unique_locations = len(set(item['location'] for item in species_data if item['location'] != 'Unknown')) if species_data else 0
        st.metric("Locations", unique_locations)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Papers", "Species Data", "Maps", "Downloads"])
    
    with tab1:
        if search_results:
            st.markdown("### Literature Search Results")
            papers_df = pd.DataFrame(search_results)
            
            # Display papers table
            st.dataframe(
                papers_df[['title', 'authors', 'journal', 'year', 'database']],
                use_container_width=True,
                height=400
            )
            
            # Database breakdown
            st.markdown("### Results by Database")
            db_counts = papers_df['database'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            db_counts.plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title('Papers Found by Database')
            ax.set_xlabel('Database')
            ax.set_ylabel('Number of Papers')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("No papers data available. Run the analysis first.")
    
    with tab2:
        if species_data:
            st.markdown("### Extracted Species Data")
            species_df = pd.DataFrame(species_data)
            
            # Display species data table
            st.dataframe(species_df, use_container_width=True, height=400)
            
            # Species summary
            st.markdown("### Species Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                # Top species by occurrence
                species_counts = species_df['species'].value_counts().head(10)
                fig, ax = plt.subplots(figsize=(10, 6))
                species_counts.plot(kind='barh', ax=ax, color='lightcoral')
                ax.set_title('Top 10 Most Studied Species')
                ax.set_xlabel('Number of Records')
                st.pyplot(fig)
            
            with col2:
                # Study types
                study_type_counts = species_df['study_type'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                study_type_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%')
                ax.set_title('Study Types Distribution')
                ax.set_ylabel('')
                st.pyplot(fig)
        else:
            st.info("No species data available. Run the analysis first.")
    
    with tab3:
        if species_data:
            st.markdown("### Interactive Maps")
            
            # Create maps
            pipeline = BiodiversityPipeline()
            species_df = pd.DataFrame(species_data)
            maps = pipeline.create_interactive_maps(species_df)
            
            if maps:
                # Species distribution map
                if 'species_distribution' in maps:
                    st.markdown("#### Species Distribution Map")
                    st.markdown('<p class="info-text">Blue circles show locations with species data. Larger circles indicate more species diversity.</p>', unsafe_allow_html=True)
                    st_folium(maps['species_distribution'], width=700, height=500)
                
                # Heatmap
                if 'heatmap' in maps:
                    st.markdown("#### Species Density Heatmap")
                    st.markdown('<p class="info-text">Heat intensity shows areas with higher species diversity.</p>', unsafe_allow_html=True)
                    st_folium(maps['heatmap'], width=700, height=500)
                
                st.session_state.maps_created = True
            else:
                st.warning("Could not create maps. Check if location data is available and valid.")
        else:
            st.info("No species data available for mapping. Run the analysis first.")
    
    with tab4:
        st.markdown("### Download Results")
        
        if search_results or species_data:
            # Create download files
            downloads = create_download_files(search_results, species_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'papers_csv' in downloads:
                    st.download_button(
                        label="Download Papers CSV",
                        data=downloads['papers_csv'],
                        file_name=f"papers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                if 'species_csv' in downloads:
                    st.download_button(
                        label="Download Species Data CSV",
                        data=downloads['species_csv'],
                        file_name=f"species_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if 'complete_zip' in downloads:
                    st.download_button(
                        label="Download Complete Dataset (ZIP)",
                        data=downloads['complete_zip'],
                        file_name=f"biodiversity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip"
                    )
        else:
            st.info("No data available for download. Run the analysis first.")

def create_download_files(search_results, species_data):
    """Create downloadable files"""
    downloads = {}
    
    # Papers CSV
    if search_results:
        papers_df = pd.DataFrame(search_results)
        papers_csv = papers_df.to_csv(index=False)
        downloads['papers_csv'] = papers_csv
    
    # Species data CSV
    if species_data:
        species_df = pd.DataFrame(species_data)
        species_csv = species_df.to_csv(index=False)
        downloads['species_csv'] = species_csv
    
    # Complete ZIP file
    if search_results or species_data:
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add papers CSV
            if search_results:
                zip_file.writestr("papers.csv", papers_csv)
            
            # Add species data CSV
            if species_data:
                zip_file.writestr("species_data.csv", species_csv)
            
            # Add summary report
            summary_report = create_summary_report(search_results, species_data)
            zip_file.writestr("analysis_summary.txt", summary_report)
        
        downloads['complete_zip'] = zip_buffer.getvalue()
    
    return downloads

def create_summary_report(search_results, species_data):
    """Create a text summary report"""
    report = []
    report.append("BIODIVERSITY RESEARCH PIPELINE - ANALYSIS SUMMARY")
    report.append("=" * 60)
    report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Papers summary
    if search_results:
        papers_df = pd.DataFrame(search_results)
        report.append("LITERATURE SEARCH RESULTS:")
        report.append(f"Total papers found: {len(search_results)}")
        
        # Database breakdown
        db_counts = papers_df['database'].value_counts()
        report.append("\nPapers by database:")
        for db, count in db_counts.items():
            report.append(f"  {db}: {count}")
        
        # Year distribution
        year_counts = papers_df['year'].value_counts().sort_index()
        report.append(f"\nYear range: {year_counts.index.min()} - {year_counts.index.max()}")
    
    # Species summary
    if species_data:
        species_df = pd.DataFrame(species_data)
        report.append("\nSPECIES DATA EXTRACTION:")
        report.append(f"Total species records: {len(species_data)}")
        report.append(f"Unique species: {species_df['species'].nunique()}")
        report.append(f"Unique locations: {species_df['location'].nunique()}")
        
        # Top species
        top_species = species_df['species'].value_counts().head(10)
        report.append("\nTop 10 most studied species:")
        for i, (species, count) in enumerate(top_species.items(), 1):
            report.append(f"  {i}. {species}: {count} records")
        
        # Study types
        study_types = species_df['study_type'].value_counts()
        report.append("\nStudy types:")
        for study_type, count in study_types.items():
            report.append(f"  {study_type}: {count}")
    
    report.append("\n" + "=" * 60)
    report.append("Generated by Biodiversity Research Pipeline")
    
    return "\n".join(report)

if __name__ == "__main__":
    main()
