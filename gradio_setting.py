from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set
import pandas as pd
from utils.blob_controller import blob_controller
import numpy as np
from pathlib import Path
import io
from azure.core.exceptions import ResourceNotFoundError



LAST_NUMBER_BLOB = "last_number.txt"

@dataclass
class AppState:
    conn_str: str
    data_container: str
    image_container: str
    markdown_container: str
    history_container: str
    output_container: str = "output"   # ✅ output 컨테이너 기본값 추가

    # Blob controllers
    data_bc: Optional[blob_controller] = None
    img_bc: Optional[blob_controller] = None
    md_bc: Optional[blob_controller] = None
    hist_bc: Optional[blob_controller] = None
    output_bc: Optional[blob_controller] = None   # ✅ output controller 추가

    # Labeling 진행 관련
    category_code: str = "I"
    data_index: int = 0
    seg_index: int = 0
    extending: bool = False
    current_group_items: List[Tuple[dict, bytes, str]] = field(default_factory=list)

    # Progress
    df: Optional[pd.DataFrame] = None
    cursor: int = 0
    processed_row_indices: List[int] = field(default_factory=list)
    processed_sources: Set[Tuple[str, str]] = field(default_factory=set)
    processed_png_keys: List[str] = field(default_factory=list)  # ✅ 추가
    # Output
    output_dir: Path = Path("output")
    


def get_sas_url_for_blob(bc: blob_controller, blob_name: str, expiry_hours: int = 1) -> str:
    return bc.make_pdf_sas_url(blob_name, expiry_hours=expiry_hours)

def download_png(img_bc: blob_controller, png_name: str) -> bytes:
    bc = img_bc.container_client.get_blob_client(png_name)
    return bc.download_blob(max_concurrency=2).readall()


def download_md(md_bc: blob_controller, md_name: str) -> str:
    bc = md_bc.container_client.get_blob_client(md_name)
    return bc.download_blob(max_concurrency=2).readall().decode("utf-8")













