from pathlib import Path
import gradio as gr
import json


src_path = "./src"
pdfjs_path = "./src/pdfjs-4.0.379-dist"

gr.set_static_paths({"/pdfjs": pdfjs_path})


## css,js settings
css = ''
js = ''
pdf_view_js = ''


external_js = (
    "<script type='module' "
    "src='https://cdn.skypack.dev/pdfjs-viewer-element'>"
    "</script>"
)


dir_assets = Path(src_path)
with (dir_assets / "css" / "main.css").open() as fi:
    _css = fi.read()
    css = _css
with (dir_assets / "js" / "main.js").open() as fi:
    _js = fi.read()
with (dir_assets / "js" / "pdf_viewer.js").open() as fi:
    pdf_view_js = fi.read()

# ⬇️ PDF 캐시 디렉토리(서버 로컬)에 저장해 같은 오리진으로 서빙
PDF_CACHE_DIR = Path("./.pdf_cache")
PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 정적 경로 등록: pdf.js 뷰어와 캐시 폴더 둘 다 같은 오리진에서 접근하게 함
gr.set_static_paths([
    str(pdfjs_path),
    str(PDF_CACHE_DIR),
])



def pdf_maker_local(local_pdf_path: Path, page: int) -> str:
    """
    pdf_cache 아래에 저장된 로컬 파일을 같은 오리진으로 로드.
    Gradio는 set_static_paths로 등록된 실제 파일을 /gradio_api/file=<절대경로> 로 서빙 가능.
    """
    abs_path = str(local_pdf_path.resolve())
    return f"""
    <div id="pdf-modal" class="modal" style="display: block; height:80vh;">
      <div class="modal-body" style="height:100%;">
        <pdfjs-viewer-element id="pdf-viewer"
          viewer-path="/gradio_api/file={pdfjs_path}"
          locale="en"
          phrase="true"
          src="/gradio_api/file={abs_path}"
          page="{int(page)}"
          style="width:100%; height:100%;"
          crossorigin="anonymous">
        </pdfjs-viewer-element>
      </div>
    </div>
    """