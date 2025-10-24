from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import pandas as pd
from utils.blob_controller import blob_controller

@dataclass
class AppState:
    # Azure
    conn_str: str
    data_container: str
    image_container: str
    markdown_container: str
    history_container: str
    output_container: str

    # Label naming
    category_code: str = "I"
    data_index: int = 0
    seg_index: int = 0
    extending: bool = False
    current_group_items: List[Tuple[dict, bytes, str]] = field(default_factory=list)

    # Progress
    df: Optional[pd.DataFrame] = None
    cursor: int = 0
    processed_row_indices: List[int] = field(default_factory=list)

    # Azure controllers
    data_bc: Optional[blob_controller] = None
    img_bc: Optional[blob_controller] = None
    md_bc: Optional[blob_controller] = None
    hist_bc: Optional[blob_controller] = None
    out_bc: Optional[blob_controller] = None