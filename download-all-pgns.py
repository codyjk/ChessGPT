import requests
from bs4 import BeautifulSoup
import threading
import os
from tqdm import tqdm
import time

# Set up semaphore for rate limiting
semaphore = threading.Semaphore(5)

# Function to download a single file
def download_file(url, retry_count=3):
    with semaphore:
        for attempt in range(retry_count):
            try:
                response = requests.get(url, stream=True, allow_redirects=True)
                response.raise_for_status()
                filename = url.split('/')[-1]
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {url}: {e}")
                if attempt < retry_count - 1:
                    time.sleep(5)  # Wait before retrying
                else:
                    return False

# Fetch the list of .pgn.zst files
url = "https://database.lichess.org"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
file_links = [link['href'] for link in soup.find_all('a') if link['href'].endswith('.pgn.zst')]

# Download files with progress bar
successful_downloads = 0
failed_downloads = 0

with tqdm(total=len(file_links), desc="Downloading files", unit="file") as pbar:
    def download_and_update(url):
        global successful_downloads, failed_downloads
        if download_file(url):
            successful_downloads += 1
        else:
            failed_downloads += 1
        pbar.update(1)

    threads = []
    for file_url in file_links:
        full_url = url + file_url if not file_url.startswith('http') else file_url
        thread = threading.Thread(target=download_and_update, args=(full_url,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

print(f"Download complete. Successful: {successful_downloads}, Failed: {failed_downloads}")
