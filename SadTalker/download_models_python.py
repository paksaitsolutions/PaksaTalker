import os
import urllib.request
from pathlib import Path
import requests
from bs4 import BeautifulSoup

def download_file(url, destination):
    """Download a file from URL to destination with progress"""
    try:
        print(f"Downloading {url}...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download with progress
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\rProgress: {percent}%", end='', flush=True)
        
        urllib.request.urlretrieve(url, destination, reporthook=reporthook)
        print("\n✅ Download complete!")
        return True
    except Exception as e:
        print(f"\n❌ Error downloading {url}: {e}")
        return False

def main():
    url = "https://huggingface.co/spaces/vinthony/SadTalker/tree/main/checkpoints"
    checkpoints_dir = Path("D:/SadTalker/checkpoints")
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Find all the links to the checkpoint files
        file_links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith("/spaces/vinthony/SadTalker/blob/main/checkpoints/"):
                file_links.append("https://huggingface.co" + href.replace("/blob/", "/resolve/"))
        
        print("Starting download of model files...\n")

        # Download each file
        for file_url in file_links:
            filename = file_url.split("/")[-1]
            dest_path = checkpoints_dir / filename
            if dest_path.exists():
                print(f"{filename} already exists, skipping...")
                continue
                
            print(f"\n--- Downloading {filename} ---")
            success = download_file(file_url, dest_path)
            
            if not success:
                print(f"Failed to download {filename}")
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
    except Exception as e:
        print(f"Error parsing the HTML: {e}")
    
    # List all files in checkpoints directory
    print("\nCurrent contents of checkpoints directory:")
    for item in checkpoints_dir.glob("*"):
        print(f"- {item.name} ({item.stat().st_size / (1024*1024):.1f} MB)")

if __name__ == "__main__":
    main()
