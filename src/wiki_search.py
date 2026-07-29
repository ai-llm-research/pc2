#!/usr/bin/env python3
"""
Wikipedia Content Fetcher and Processor
"""

import argparse
import json
import re
import time
from pathlib import Path
import sys
import requests
from typing import Optional, List, Dict
from html.parser import HTMLParser

# Configuration
SEARCH_API = "https://en.wikipedia.org/w/api.php"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "content_description" / "individual_content"

# ============================================================
# Wikipedia Fetching Functions
# ============================================================

class ParagraphParser(HTMLParser):
    """HTML parser to extract paragraphs from Wikipedia HTML."""
    
    def __init__(self):
        super().__init__()
        self.paragraphs = []
        self.current_paragraph = []
        self.in_paragraph = False
        self.skip_tags = {'sup', 'style', 'script'}
        self.skip_depth = 0
    
    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self.skip_depth += 1
        elif tag == 'p':
            self.in_paragraph = True
            self.current_paragraph = []
    
    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self.skip_depth -= 1
        elif tag == 'p' and self.in_paragraph:
            self.in_paragraph = False
            text = ''.join(self.current_paragraph).strip()
            text = ' '.join(text.split())
            if text and len(text) > 50:
                self.paragraphs.append(text)
            self.current_paragraph = []
    
    def handle_data(self, data):
        if self.in_paragraph and self.skip_depth == 0:
            self.current_paragraph.append(data)


def extract_paragraphs_from_html(html: str) -> List[str]:
    """Extract paragraphs from Wikipedia HTML."""
    parser = ParagraphParser()
    parser.feed(html)
    return parser.paragraphs


def fetch_wiki_content(title: str, verbose: bool = True) -> Optional[List[str]]:
    """
    Fetch Wikipedia content and return paragraphs.
    
    Args:
        title: Wikipedia page title
        verbose: Print progress messages
        
    Returns:
        List of paragraphs or None if not found
    """
    headers = {
        "accept": "application/json",
        "User-Agent": "ContentBot/1.0 (Educational purpose)"
    }
    
    params = {
        "action": "parse",
        "format": "json",
        "page": title,
        "prop": "text",
        "disableeditsection": True,
        "disabletoc": True,
    }
    
    for _ in range(10):  # max 10 retries
        try:
            r = requests.get(
                SEARCH_API,
                params=params,
                headers=headers,
                timeout=15,
            )
            r.raise_for_status()

            data = r.json()

            if "parse" in data:
                html = data["parse"]["text"]["*"]
                paragraphs = extract_paragraphs_from_html(html)
                if paragraphs:
                    return paragraphs
            return None

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                if verbose:
                    print(f"    ⚠ Rate limited for '{title}', retrying in 10s...")
                time.sleep(10)
                continue
            raise

        except requests.exceptions.RequestException as e:
            if verbose:
                print(f"    ⚠ Request error for '{title}': {e}")
            return None

    return None


def search_wikipedia(query: str, verbose: bool = True) -> Optional[str]:
    """
    Search Wikipedia and return the best matching page title.
    
    Args:
        query: Search query
        verbose: Print progress messages
        
    Returns:
        Best matching page title or None
    """
    headers = {
        "accept": "application/json",
        "User-Agent": "ContentBot/1.0 (Educational purpose)"
    }
    
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 5,
    }
    
    try:
        r = requests.get(SEARCH_API, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        hits = r.json().get("query", {}).get("search", [])
        
        if hits:
            return hits[0]["title"]
        return None
    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"    ⚠ Search error for '{query}': {e}")
        return None


def get_wiki_paragraphs(content: str, verbose: bool = True) -> Optional[List[str]]:
    """
    Get Wikipedia paragraphs for a content item.
    Tries direct fetch first, then search if needed.
    
    Args:
        content: Content name to search for
        verbose: Print progress messages
        
    Returns:
        List of paragraphs or None
    """
    if verbose:
        print(f"    Searching for: {content}")
    
    # Try direct fetch first
    paragraphs = fetch_wiki_content(content, verbose)
    
    if not paragraphs:
        # Try searching
        page_title = search_wikipedia(content, verbose)
        if page_title:
            if verbose:
                print(f"      Found via search: {page_title}")
            paragraphs = fetch_wiki_content(page_title, verbose)
    
    if paragraphs:
        if verbose:
            print(f"      ✓ Retrieved {len(paragraphs)} paragraphs")
        return paragraphs
    else:
        if verbose:
            print(f"      ✗ No content found")
        return None


# ============================================================
# File Handling Functions
# ============================================================

def sanitize_filename(name: str) -> str:
    """
    Convert content name to a safe filename.
    
    Args:
        name: Content name
        
    Returns:
        Sanitized filename (without extension)
    """
    name = re.sub(r'[\\/:*?"<>|]+', '_', name)
    name = re.sub(r'\s+', '_', name)
    name = name.replace('(', '').replace(')', '').replace(',', '')
    name = name.replace("'", '').replace('"', '')
    name = name.strip('_.')
    return name


def content_file_exists(content_name: str, output_dir: Path) -> bool:
    """
    Check if content file already exists in output directory.
    
    Args:
        content_name: Content name
        output_dir: Output directory path
        
    Returns:
        True if file exists, False otherwise
    """
    filename = sanitize_filename(content_name) + ".json"
    filepath = output_dir / filename
    return filepath.exists()


def save_content_json(content_name: str, paragraphs: List[str], output_dir: Path) -> bool:
    """
    Save content paragraphs as JSON file.
    
    Args:
        content_name: Content name
        paragraphs: List of paragraphs
        output_dir: Output directory
        
    Returns:
        True if successful, False otherwise
    """
    filename = sanitize_filename(content_name) + ".json"
    filepath = output_dir / filename
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(paragraphs, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"      ✗ Error saving {filename}: {e}")
        return False


def load_content_json(content_name: str, output_dir: Path) -> Optional[List[str]]:
    """
    Load existing content from JSON file.
    
    Args:
        content_name: Content name
        output_dir: Output directory
        
    Returns:
        List of paragraphs or None if not found
    """
    filename = sanitize_filename(content_name) + ".json"
    filepath = output_dir / filename
    
    if not filepath.exists():
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"    ⚠ Error loading {filename}: {e}")
        return None


# ============================================================
# Main Processing Functions
# ============================================================

def process_single_content(
    content_name: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    verbose: bool = True,
    force: bool = False
) -> Dict[str, any]:
    """
    Process a single content item directly by name.
    Searches Wikipedia and saves paragraphs to individual JSON file.
    
    Args:
        content_name: Name of the content to search (e.g., "Donald Trump", "the nazi flag")
        output_dir: Directory to save the JSON file
        verbose: Print progress messages
        force: If True, re-fetch even if file exists. If False, skip existing files.
        
    Returns:
        Dictionary with processing statistics:
        {
            'success': bool,
            'content_name': str,
            'already_existed': bool,
            'paragraphs_count': int,
            'file_path': str,
            'paragraphs': List[str] (if success and paragraphs found)
        }
    """
    stats = {
        'success': False,
        'content_name': content_name,
        'already_existed': False,
        'paragraphs_count': 0,
        'file_path': '',
        'paragraphs': []
    }
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\nProcessing content: {content_name}")
    
    # Check if already exists
    if not force and content_file_exists(content_name, output_dir):
        if verbose:
            print(f"  ✓ Already exists, skipping (use --force to re-fetch)")
        
        # Load existing to get paragraph count
        existing = load_content_json(content_name, output_dir)
        stats['already_existed'] = True
        stats['success'] = True
        stats['paragraphs_count'] = len(existing) if existing else 0
        
        filename = sanitize_filename(content_name) + ".json"
        stats['file_path'] = str(output_dir / filename)
        stats['paragraphs'] = existing if existing else []
        return stats
    
    # Fetch from Wikipedia
    paragraphs = get_wiki_paragraphs(content_name, verbose)
    
    if paragraphs is None or len(paragraphs) == 0:
        stats['success'] = False
        if verbose:
            print(f"  ✗ No content found to save")
        return stats
    
    # Save to file
    if save_content_json(content_name, paragraphs, output_dir):
        if verbose:
            if paragraphs:
                print(f"  ✓ Saved: {len(paragraphs)} paragraphs")
            else:
                print(f"  ⚠ Saved empty (no content found)")
        
        stats['success'] = True
        stats['paragraphs_count'] = len(paragraphs)
        
        filename = sanitize_filename(content_name) + ".json"
        stats['file_path'] = str(output_dir / filename)
        stats['paragraphs'] = paragraphs
    else:
        if verbose:
            print(f"  ✗ Failed to save")
        stats['success'] = False
    
    return stats


def process_single_country_or_content(
    content_name: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    verbose: bool = True,
    force: bool = False
) -> Dict[str, any]:

    content_mapping = {
        "Georgia": "Georgia (country)",
        "Congo (Democratic Republic)": "Democratic Republic of the Congo",
        "Papua": "Papua New Guinea",
        "Korean" : "Korea",
        "An Austrian politician": "Austria",
        "CDU": "Christian Democratic Union",
        "diversity": "Diversity, equity, and inclusion",
        "inclusion": "Diversity, equity, and inclusion",
        "Gilbert Baker": "Gilbert Baker (artist)"
    }

    if content_name in content_mapping:
        content_name = content_mapping[content_name]

    stats = process_single_content(content_name, output_dir, verbose, force)
    return stats


# ============================================================
# Command-Line Interface
# ============================================================

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Process step1 JSON file OR search single content item and create individual content files"
    )
    
    # Create mutually exclusive group for --file and --content
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--content",
        help="Content name to search directly (e.g., 'Donald Trump', 'the nazi flag')"
    )
    
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--rate_limit",
        type=float,
        default=1.0,
        help="Seconds to wait between Wikipedia requests (default: 1.0, only applies to --file mode)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-fetch even if content already exists (only applies to --content mode)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )
    
    args = parser.parse_args()
    
    # Process single content
    stats = process_single_content(
        content_name=args.content,
        output_dir=Path(args.output_dir),
        verbose=not args.quiet,
        force=args.force
    )
    
    # Exit with appropriate code
    sys.exit(0 if stats.get('success', False) else 1)


if __name__ == "__main__":
    main()

