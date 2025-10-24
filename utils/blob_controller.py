import os
import io
import mimetypes
import tempfile
from pathlib import Path
from urllib.parse import quote
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from datetime import datetime, timedelta
from azure.storage.blob import (
    BlobServiceClient,
    ContainerClient,
    BlobClient,
    generate_blob_sas,
    BlobSasPermissions,
    ContentSettings,
)
import fitz

class blob_controller:
    def __init__(self, conn: str, container: str):
        if not conn:
            raise ValueError("Azure connection string이 비어 있습니다.")
        if not container or container.lower() != container:
            raise ValueError("컨테이너 이름은 소문자여야 합니다. (예: 'meritz-data')")

        self.conn = conn
        self.container = container
        self.blob_service = BlobServiceClient.from_connection_string(conn)
        self.container_client = self.blob_service.get_container_client(container)

        # 컨테이너 없으면 생성, 있으면 무시
        self.ensure_container_exists()

    def ensure_container_exists(self):
        try:
            self.container_client.create_container()
        except ResourceExistsError:
            pass 


    def upload_pdf_to_blob(self, file_path: str) -> str:
        """
        로컬 PDF 파일을 업로드. blob_name을 넘기지 않으면  basename으로 업로드.
        """
        name = os.path.basename(file_path)
        blob_client: BlobClient = self.container_client.get_blob_client(name)
        with open(file_path, "rb") as f:
            blob_client.upload_blob(
                f,
                overwrite=True,
                max_concurrency=3,
                timeout=600,
                content_settings=ContentSettings(content_type="application/pdf"),
            )
        return name
    
    
    def list_files(self):
        """
        특정 prefix 하위의 blob 목록 반환
        """
        blobs = self.container_client.list_blobs()
        file_list = [b.name for b in blobs]
        return file_list
    
    
    def download_to_temp(self, blob_name: str, tmp_dir: Path | str) -> Path:
        """
        blob_name(컨테이너 내 경로)을 tmp_dir에 파일로 저장하고 그 경로(Path)를 반환.
        tmp_dir은 str 또는 Path 모두 허용.
        """
        # str 이 들어와도 Path 로 정규화
        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        local_path = tmp_dir / Path(blob_name).name

        bc: BlobClient = self.container_client.get_blob_client(blob_name)
        try:
            stream = bc.download_blob(max_concurrency=4)
            with open(local_path, "wb") as f:
                for chunk in stream.chunks():
                    if chunk:
                        f.write(chunk)
        except ResourceNotFoundError:
            raise FileNotFoundError(f"Blob not found: {blob_name}")

        return local_path
    
    
    
    def open_pdf_from_blob_stream(self, blob_name: str):
        """
        디스크 임시파일 없이 Blob을 스트리밍으로 읽어 메모리/스풀 버퍼에 적재한 뒤,
        fitz.open(stream=..., filetype="pdf")로 바로 연다.
        - 작은/중간 사이즈 PDF: 메모리에서 처리
        - 큰 PDF: SpooledTemporaryFile 이 자동으로 디스크로 스필오버
        """
        spooled = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024)  # 64MB까지 메모리, 초과 시 디스크 스필오버
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            stream = blob_client.download_blob(max_concurrency=4)
            for chunk in stream.chunks():
                if chunk:
                    spooled.write(chunk)
            spooled.seek(0)
            data = spooled.read()
            return fitz.open(stream=data, filetype="pdf")
        finally:
            try:
                spooled.close()
            except Exception:
                pass

    def make_pdf_sas_url(self, blob_name: str, expiry_hours: int = 1) -> str:
        sas = generate_blob_sas(
            account_name=self.blob_service.account_name,
            container_name=self.container,
            blob_name=blob_name,
            account_key=self.blob_service.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=expiry_hours),
        )
        base_url = self.container_client.url  # https://{account}.blob.core.windows.net/{container}
        # blob_name 안에 폴더/한글/공백 등이 있을 수 있으니 URL 인코딩
        return f"{base_url}/{quote(blob_name, safe='/')}?{sas}"