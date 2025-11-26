import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
CREDENTIALS_PATH = os.getenv("CREDENTIALS_PATH")
FOLDER_ID = os.getenv("FOLDER_ID")


def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        CREDENTIALS_PATH,
        scopes=['https://www.googleapis.com/auth/drive.readonly'])
    return build('drive', 'v3', credentials=creds)


def list_files_in_folder(service, folder_id: str = FOLDER_ID):
    """List all files in a Google Drive folder"""
    results = service.files().list(q=f"'{folder_id}' in parents",
                                   fields="files(id, name, size)").execute()
    return results.get('files', [])


def download_file(service, file_id: str, output_path: str):
    """Download a single file"""
    request = service.files().get_media(fileId=file_id)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"  {int(status.progress() * 100)}%", end='\r')
    print()


def download_folder(folder_id: str = FOLDER_ID,
                    output_dir: str = "src/KnowledgeDb"):
    """Download an entire folder, including subfolders"""
    service = get_drive_service()

    results = service.files().list(
        q=f"'{folder_id}' in parents",
        fields="files(id, name, size, mimeType)").execute()
    files = results.get('files', [])

    print(f"Found {len(files)} items in folder")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for f in files:
        output_path = Path(output_dir) / f['name']

        if f['mimeType'] == 'application/vnd.google-apps.folder':
            print(f"📁 Entering folder: {f['name']}")
            download_folder(f['id'], str(output_path))
        else:
            print(f"📄 Downloading: {f['name']}")
            download_file(service, f['id'], str(output_path))

    print(f"✓ Done with {output_dir}")
