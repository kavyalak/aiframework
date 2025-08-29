import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

_ACCOUNT_URL = os.getenv("AZURE_BLOB_ACCOUNT_URL")  # e.g., https://<acct>.blob.core.windows.net
_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")  # optional if using account URL + key
_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER")  # required

if not _CONTAINER:
    raise RuntimeError("AZURE_BLOB_CONTAINER is not set")

def _client():
    if _CONNECTION_STRING:
        return BlobServiceClient.from_connection_string(_CONNECTION_STRING)
    if not _ACCOUNT_URL or not os.getenv("AZURE_BLOB_SAS_TOKEN") and not os.getenv("AZURE_BLOB_ACCOUNT_KEY"):
        # If using AAD, you may rely on DefaultAzureCredential via azure-identity (not shown here)
        # For simplicity we accept SAS (append to ACCOUNT_URL) or connection string
        pass
    return BlobServiceClient(account_url=_ACCOUNT_URL, credential=os.getenv("AZURE_BLOB_SAS_TOKEN") or os.getenv("AZURE_BLOB_ACCOUNT_KEY"))

def _container_client():
    return _client().get_container_client(_CONTAINER)

def list_blobs(prefix: str | None = None) -> list[str]:
    cc = _container_client()
    return [b.name for b in cc.list_blobs(name_starts_with=prefix or "")]

def upload_to_blob(uploaded_file, blob_name: str | None = None) -> str:
    """Upload a Streamlit UploadedFile (or file-like) to blob. Returns the blob name used."""
    data = uploaded_file.read()
    name = blob_name or getattr(uploaded_file, "name", "uploaded.bin")
    cc = _container_client()
    bc = cc.get_blob_client(name)
    bc.upload_blob(data, overwrite=False)
    return name

def download_blob(name: str) -> bytes:
    cc = _container_client()
    bc = cc.get_blob_client(name)
    return bc.download_blob().readall()
