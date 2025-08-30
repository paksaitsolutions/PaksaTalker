"""
PaksaTalker API Documentation Generator

This script generates static HTML documentation for the PaksaTalker API.
It creates a self-contained documentation website that can be viewed offline.
"""

import json
import os
import shutil
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Configuration
DOCS_DIR = Path("docs")
STATIC_DIR = DOCS_DIR / "static"
REDOC_JS_URL = "https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js"
REDOC_CSS_URL = "https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.css"

class DocumentationError(Exception):
    """Custom exception for documentation generation errors."""
    pass
TEMPLATES = {
    "index.html": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>PaksaTalker API Documentation</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="static/redoc.standalone.css">
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .header {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            padding: 1.5rem 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .header h1 {
            margin: 0;
            font-weight: 600;
            font-size: 1.5rem;
        }
        .header p {
            margin: 0.5rem 0 0;
            opacity: 0.9;
            font-size: 0.95rem;
        }
        .container {
            display: flex;
            min-height: calc(100vh - 80px);
        }
        .sidebar {
            width: 280px;
            background: #f9fafb;
            border-right: 1px solid #e5e7eb;
            padding: 1.5rem;
            overflow-y: auto;
        }
        .content {
            flex: 1;
            overflow-y: auto;
        }
        .nav-item {
            display: block;
            padding: 0.5rem 0;
            color: #374151;
            text-decoration: none;
            border-radius: 0.375rem;
            transition: all 0.2s;
        }
        .nav-item:hover {
            color: #4f46e5;
            background-color: #eef2ff;
        }
        .nav-section {
            margin-top: 1.5rem;
        }
        .nav-section-title {
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            color: #6b7280;
            margin-bottom: 0.5rem;
            letter-spacing: 0.05em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>PaksaTalker API</h1>
        <p>Professional AI Video Generation Platform</p>
    </div>
    <div class="container">
        <div class="sidebar">
            <div class="nav-section">
                <div class="nav-section-title">Documentation</div>
                <a href="#" class="nav-item" onclick="loadSpec('openapi.json')">API Reference</a>
                <a href="#getting-started" class="nav-item">Getting Started</a>
                <a href="#authentication" class="nav-item">Authentication</a>
                <a href="#rate-limiting" class="nav-item">Rate Limiting</a>
                <a href="#changelog" class="nav-item">Changelog</a>
            </div>
            <div class="nav-section">
                <div class="nav-section-title">Resources</div>
                <a href="https://github.com/paksaitsolutions/PaksaTalker" target="_blank" class="nav-item">GitHub Repository</a>
                <a href="https://paksa.com.pk" target="_blank" class="nav-item">Paksa IT Solutions</a>
            </div>
        </div>
        <div class="content">
            <div id="redoc-container"></div>
        </div>
    </div>

    <script src="static/redoc.standalone.js"></script>
    <script>
        // Initialize ReDoc
        Redoc.init('openapi.json', {
            scrollYOffset: 60,
            hideDownloadButton: false,
            expandSingleSchemaField: true,
            menuToggle: true,
            hideLoading: true,
            pathInMiddlePanel: true,
            nativeScrollbars: true,
            hideHostname: false,
            requiredPropsFirst: true,
            sortPropsAlphabetically: true,
        }, document.getElementById('redoc-container'));

        // Simple router for navigation
        function loadSpec(specUrl) {
            Redoc.init(specUrl, {
                scrollYOffset: 60,
                hideDownloadButton: false,
                expandSingleSchemaField: true
            }, document.getElementById('redoc-container'));
            return false;
        }

        // Handle anchor links
        document.addEventListener('DOMContentLoaded', function() {
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {
                anchor.addEventListener('click', function (e) {
                    e.preventDefault();
                    const targetId = this.getAttribute('href').substring(1);
                    if (targetId) {
                        const targetElement = document.getElementById(targetId);
                        if (targetElement) {
                            targetElement.scrollIntoView({ behavior: 'smooth' });
                        }
                    }
                });
            });
        });
    </script>
</body>
</html>""",
    ".gitignore": "# Ignore everything in this directory\n*\n# Except this file\n!.gitignore"
}

def setup_directories():
    """Create necessary directories for documentation."""
    # Remove existing docs directory if it exists
    if DOCS_DIR.exists():
        shutil.rmtree(DOCS_DIR)
    
    # Create fresh directories
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url: str, path: Path, timeout: int = 10) -> None:
    """Download a file from a URL to the specified path.
    
    Args:
        url: The URL of the file to download
        path: The local path where the file should be saved
        timeout: Timeout in seconds for the download request
        
    Raises:
        DocumentationError: If the download fails
    """
    try:
        os.makedirs(path.parent, exist_ok=True)
        with urllib.request.urlopen(url, timeout=timeout) as response, open(path, 'wb') as out_file:
            if response.status != 200:
                raise DocumentationError(f"Failed to download {url}: HTTP {response.status}")
            shutil.copyfileobj(response, out_file)
    except (urllib.error.URLError, OSError) as e:
        raise DocumentationError(f"Error downloading {url}: {str(e)}")

def download_redoc_assets() -> None:
    """Download ReDoc assets for offline use.
    
    Raises:
        DocumentationError: If any asset download fails
    """
    print("Downloading ReDoc assets...")
    try:
        download_file(REDOC_JS_URL, STATIC_DIR / "redoc.standalone.js")
        download_file(REDOC_CSS_URL, STATIC_DIR / "redoc.standalone.css")
    except DocumentationError as e:
        raise DocumentationError(f"Failed to download ReDoc assets: {str(e)}")

def generate_documentation(app) -> Tuple[bool, str]:
    """Generate the complete documentation website.
    
    Args:
        app: The FastAPI application instance
        
    Returns:
        Tuple[bool, str]: Success status and message
    """
    print("Generating documentation...")
    
    try:
        # Create HTML files from templates
        for filename, content in TEMPLATES.items():
            try:
                with open(DOCS_DIR / filename, 'w', encoding='utf-8') as f:
                    f.write(content)
            except IOError as e:
                return False, f"Failed to write {filename}: {str(e)}"
        
        # Generate OpenAPI spec
        try:
            openapi_spec = app.openapi()
            with open(DOCS_DIR / 'openapi.json', 'w', encoding='utf-8') as f:
                json.dump(openapi_spec, f, indent=2, ensure_ascii=False)
        except Exception as e:
            return False, f"Failed to generate OpenAPI spec: {str(e)}"
        
        # Download ReDoc assets
        try:
            download_redoc_assets()
        except DocumentationError as e:
            return False, str(e)
            
        return True, f"Documentation generated successfully in {DOCS_DIR.absolute()}"
    except Exception as e:
        return False, f"Unexpected error generating documentation: {str(e)}"

def main() -> int:
    """Main function to generate the documentation.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        setup_directories()
        
        # Import app here to avoid circular imports
        from app import app
        
        success, message = generate_documentation(app)
        if success:
            print(message)
            return 0
        else:
            print(f"Error: {message}", file=sys.stderr)
            return 1
            
    except ImportError as e:
        print(f"Failed to import application: {str(e)}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
