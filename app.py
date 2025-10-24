import os
import io
from pathlib import Path

from typing import List, Tuple, Optional, Set
import gradio as gr
import pandas as pd
import numpy as np
from PIL import Image

from utils.blob_controller import blob_controller
from azure.core.exceptions import ResourceNotFoundError

from render_pdf import *
from gradio_setting import *
from pathlib import Path
from azure.storage.blob import ContentSettings

import dotenv
dotenv.load_dotenv()




# ========= Gradio 핸들러 =========
def _write_last_number(hist_bc: blob_controller, last_number: int) -> Tuple[bool, str]:
    """last_number.txt에 마지막 번호 쓰기. (ok, msg)"""
    try:
        bc = hist_bc.container_client.get_blob_client(LAST_NUMBER_BLOB)
        payload = f"{last_number}\n".encode("utf-8")
        bc.upload_blob(payload, overwrite=True, max_concurrency=2, timeout=30)
        print(f"[LAST] last_number.txt updated -> {last_number}")
        return True, f"last_number updated: {last_number}"
    except Exception as e:
        print(f"[LAST][ERROR] write failed: {e}")
        return False, f"last_number write failed: {e}"

def _commit_extending_all(state: AppState):
    """
    extend 모드의 스테이징된 모든 조각 확정:
      - (출력 저장은 기존 로직 유지)
      - processed 추적에 그룹 내 모든 행의 png_name/md_name 추가
      - data_index 증가, extend 상태 초기화
      - Exit 시 image / markdown blob 삭제 대상 등록
    """
    # 1️⃣ output 컨테이너에 저장
    for seg_idx, (row, png_bytes, md_text) in enumerate(state.current_group_items):
        _save_pair_local(
            state.category_code,
            state.data_index,
            png_bytes,
            md_text,
            state.output_bc,
            seg_idx,
        )

    # 2️⃣ Exit 시 삭제 대상 등록 (image / markdown)
    for row, _, _ in state.current_group_items:
        png_name = row.get("png_name", "")
        md_name  = row.get("md_name", "")
        if png_name:
            state.processed_png_keys.append(png_name)
            state.processed_sources.add((png_name, md_name))  # ✅ Exit 시 삭제

    # 3️⃣ 상태 초기화
    state.data_index += 1
    state.seg_index = 0
    state.extending = False
    state.current_group_items.clear()

    print(f"[EXTEND_END] committed group, next DATA={state.data_index}")


def _remove_processed_rows_from_history(state: AppState) -> Tuple[bool, str]:
    """
    processed_png_keys 목록에 든 행들을 history.csv에서 제거하고 업로드. (ok, msg)
    """
    if not state.processed_png_keys:
        print("[EXIT] nothing to commit (no processed rows).")
        return True, "Nothing to commit."

    df = state.df if state.df is not None else pd.DataFrame()
    before = len(df)
    keyset: Set[str] = set(state.processed_png_keys)
    new_df = df[~df["png_name"].astype(str).isin(keyset)].reset_index(drop=True)

    ok, msg = save_history_csv(state.hist_bc, new_df)
    if ok:
        state.df = new_df
        state.processed_png_keys.clear()
        print(f"[EXIT] history rows removed: {before - len(new_df)}")
    return ok, msg

def _delete_source_blobs(img_bc: blob_controller, md_bc: blob_controller, png_name: str, md_name: str) -> Tuple[bool, str]:
    """이미지/마크다운 원본 블랍 삭제(없으면 무시)."""
    msgs = []
    ok_all = True
    try:
        bc = img_bc.container_client.get_blob_client(png_name)
        bc.delete_blob(delete_snapshots="include")
        msgs.append(f"del {png_name}")
    except ResourceNotFoundError:
        msgs.append(f"skip-del {png_name}(not found)")
    except Exception as e:
        msgs.append(f"err-del {png_name}:{e}")
        ok_all = False
    try:
        bc = md_bc.container_client.get_blob_client(md_name)
        bc.delete_blob(delete_snapshots="include")
        msgs.append(f"del {md_name}")
    except ResourceNotFoundError:
        msgs.append(f"skip-del {md_name}(not found)")
    except Exception as e:
        msgs.append(f"err-del {md_name}:{e}")
        ok_all = False
    print("[DEL]", " | ".join(msgs))
    return ok_all, " | ".join(msgs)

def get_current_row(state: AppState) -> Optional[dict]:
    if state.df is None or len(state.df) == 0:
        return None
    if state.cursor >= len(state.df):
        return None
    return state.df.iloc[state.cursor].to_dict()

def _advance_cursor(state: AppState):
    state.cursor += 1
    print(f"[NAV] cursor -> {state.cursor}")
    
def load_history_csv(hist_bc: blob_controller) -> pd.DataFrame:
    """history.csv 다운로드 → DataFrame. 없으면 빈 DF."""
    try:
        bc = hist_bc.container_client.get_blob_client("history.csv")
        content = bc.download_blob(max_concurrency=2).readall().decode("utf-8")
        df = pd.read_csv(io.StringIO(content))
        expected = ["original_pdf", "png_name", "md_name", "page", "category", "table_index"]
        for col in expected:
            if col not in df.columns:
                df[col] = ""
        df = df[expected]
        print(f"[INIT] history.csv loaded. rows={len(df)}")
        return df
    except ResourceNotFoundError:
        print("[INIT] history.csv not found. starting empty.")
        return pd.DataFrame(columns=["original_pdf", "png_name", "md_name", "page", "category", "table_index"])
    except Exception as e:
        print(f"[INIT] failed to load history.csv: {e}")
        return pd.DataFrame(columns=["original_pdf", "png_name", "md_name", "page", "category", "table_index"])


def save_history_csv(hist_bc: blob_controller, df: pd.DataFrame) -> Tuple[bool, str]:
    """history.csv 업로드(덮어쓰기). (ok, msg) 반환."""
    try:
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        data = csv_buf.getvalue().encode("utf-8")
        bc = hist_bc.container_client.get_blob_client("history.csv")
        bc.upload_blob(
            data, overwrite=True, max_concurrency=2, timeout=60,
            content_settings=None
        )
        print(f"[HIST] history.csv uploaded. rows={len(df)}")
        return True, "history.csv uploaded"
    except Exception as e:
        print(f"[HIST][ERROR] history.csv upload failed: {e}")
        return False, f"upload failed: {e}"
    
    
def _save_pair_local(cat: str, data_idx: int, png_bytes: bytes, md_text: str, bc_output, seg_idx: Optional[int] = None):
    """
    PNG/MD 한 벌을 Azure Blob의 output 컨테이너로 업로드.
    bc_output: blob_controller (output 컨테이너)
    """
    if seg_idx is None:
        base = f"{cat}_table_{data_idx}"
    else:
        base = f"{cat}_table_{data_idx}_{seg_idx}"

    # blob 이름 설정
    png_name = f"{base}.png"
    md_name = f"{base}.md"

    # PNG 업로드
    png_client = bc_output.container_client.get_blob_client(png_name)
    png_client.upload_blob(
        png_bytes,
        overwrite=True,
        max_concurrency=3,
        timeout=600,
        content_settings=ContentSettings(content_type="image/png")
    )

    # Markdown 업로드
    md_client = bc_output.container_client.get_blob_client(md_name)
    md_client.upload_blob(
        md_text.encode("utf-8"),
        overwrite=True,
        max_concurrency=3,
        timeout=600,
        content_settings=ContentSettings(content_type="text/markdown")
    )

    print(f"[UPLOAD] {png_name}, {md_name} → {bc_output.container}")
    return png_name, md_name

def _commit_single(state: AppState, row: dict, png_bytes: bytes, md_text: str):
    _save_pair_local(
        cat=state.category_code,
        data_idx=state.data_index,
        png_bytes=png_bytes,
        md_text=md_text,
        bc_output=state.output_bc
    )

    png_name = row.get("png_name", "")
    md_name  = row.get("md_name", "")
    if png_name:
        state.processed_sources.add((png_name, md_name))  # ✅ Exit 시 삭제대상 등록

    state.data_index += 1
    state.processed_row_indices.append(state.cursor)
    print(f"[PASS] committed row idx={state.cursor}, next DATA={state.data_index}")


def _bytes_to_numpy_png(png_bytes: Optional[bytes]) -> Optional[np.ndarray]:
    if not png_bytes:
        return None
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        return np.array(img)
    except Exception as e:
        print(f"[IMG][ERROR] failed to convert png bytes: {e}")
        return None
    

def _read_last_number(hist_bc: blob_controller) -> Optional[int]:
    """last_number.txt에서 마지막 번호 읽기. 없으면 None."""
    try:
        bc = hist_bc.container_client.get_blob_client(LAST_NUMBER_BLOB)
        data = bc.download_blob(max_concurrency=2).readall().decode("utf-8").strip()
        val = int(data)
        print(f"[LAST] last_number.txt read: {val}")
        return val
    except ResourceNotFoundError:
        print("[LAST] last_number.txt not found.")
        return None
    except Exception as e:
        print(f"[LAST][ERROR] read failed: {e}")
        return None

def _ensure_last_number_initialized(hist_bc: blob_controller, start_index: int) -> int:
    """
    last_number.txt가 없으면 (start_index-1)로 생성.
    반환값: 읽은(or 생성한) 마지막 번호(int).
    """
    val = _read_last_number(hist_bc)
    if val is not None:
        return val
    init_last = int(start_index) - 1
    ok, msg = _write_last_number(hist_bc, init_last)
    print(f"[LAST] initialized: {init_last} | {msg}")
    return init_last
    
def _lower_or_none(x: Optional[str]) -> Optional[str]:
    return x.strip().lower() if x else x


def on_init(data_container, image_container, markdown_container, history_container, category_code, start_index, output_dir):
    print("[INIT] clicked")
    conn = os.getenv("azure-blob-connection-string")
    data_container = _lower_or_none(data_container)
    image_container = _lower_or_none(image_container)
    markdown_container = _lower_or_none(markdown_container)
    history_container = _lower_or_none(history_container)

    # ⬇️ gr.State()와 이름이 겹치지 않도록 AppState는 st로
    st = AppState(
        conn_str=conn,
        data_container=data_container,
        image_container=image_container,
        markdown_container=markdown_container,
        history_container=history_container,
        output_container="output",
        category_code=category_code.strip(),
        data_index=0,
        seg_index=0,
        extending=False,
    )
    st.data_bc = blob_controller(conn=conn, container=data_container)
    st.img_bc  = blob_controller(conn=conn, container=image_container)
    st.md_bc   = blob_controller(conn=conn, container=markdown_container)
    st.hist_bc = blob_controller(conn=conn, container=history_container)
    st.output_bc = blob_controller(conn=conn, container="output")

    # last_number 로드
    last_used = _ensure_last_number_initialized(st.hist_bc, int(start_index) if start_index is not None else 0)
    st.data_index = int(last_used) + 1
    print(f"[INIT] resolved DATA start index -> {st.data_index} (last={last_used})")

    # history.csv 로드
    df = load_history_csv(st.hist_bc)
    st.df = df


    # ✅ last_cursor.txt 로드
    try:
        bc_cursor = st.hist_bc.container_client.get_blob_client("last_cursor.txt")
        data = bc_cursor.download_blob(max_concurrency=2).readall().decode("utf-8").strip()
        st.cursor = int(data)
        print(f"[INIT] restored cursor={st.cursor}")
    except Exception:
        st.cursor = 0
        print("[INIT] no cursor info found, starting from top")    

    row = get_current_row(st)
    if row is None:
        info_md = (f"**Category**: `{st.category_code}` &nbsp; **DATA index(next)**: `{st.data_index}` "
                   f"&nbsp; **extending**: `{st.extending}`")
        # pdf_render까지 반환하는 버전이라면 빈 html 추가
        return st, None, "### No rows in history.csv", "<div id='pdf-modal'></div>", info_md, "0 / 0"

    try:
        png_bytes = download_png(st.img_bc, row["png_name"])
        md_text   = download_md(st.md_bc, row["md_name"])

        # PDF HTML (있다면)
        orig = str(row.get("original_pdf", "") or "").strip()
        page = int(row.get("page", 1) or 1)
        
        pdf_html = ""
        if orig:
            local_pdf = st.data_bc.download_to_temp(orig, PDF_CACHE_DIR)
            pdf_html = pdf_maker_local(local_pdf_path=local_pdf, page=page)


    except Exception as e:
        print(f"[INIT][ERROR] failed to load first row: {e}")
        png_bytes, md_text, pdf_html = None, f"Error loading: {e}", "<div id='pdf-modal'></div>"

    status = f"{st.cursor+1} / {len(st.df)}"
    info_md = (f"**Category**: `{st.category_code}` &nbsp; **DATA index(next)**: `{st.data_index}` "
               f"&nbsp; **extending**: `{st.extending}`")

    return st, _bytes_to_numpy_png(png_bytes), md_text, pdf_html, info_md, status

def _render_current(state: AppState):
    row = get_current_row(state)
    if row is None:
        info = (f"All processed staged: {len(state.processed_sources)}. "
                f"Next DATA index: `{state.data_index}`. Press **Exit** to commit.")
        status = f"{state.cursor} / {len(state.df) if state.df is not None else 0}"
        return None, "### Done.", "", info, status

    try:
        png_bytes = download_png(state.img_bc, row["png_name"])
        md_text   = download_md(state.md_bc, row["md_name"])

        # PDF HTML (있다면)
        orig = str(row.get("original_pdf", "") or "").strip()
        page = int(row.get("page", 1) or 1)

        pdf_html = ""
        if orig:
            local_pdf = state.data_bc.download_to_temp(orig, PDF_CACHE_DIR)
            pdf_html = pdf_maker_local(local_pdf_path=local_pdf, page=page)

    except Exception as e:
        print(f"[RENDER][ERROR] row load failed: {e}")
        info_md = (f"**Category**: `{state.category_code}` &nbsp; **DATA index(next)**: `{state.data_index}` "
                   f"&nbsp; **extending**: `{state.extending}`")
        return None, f"Error loading row: {e}", "", info_md, f"{state.cursor+1} / {len(state.df)}"

    info_md = (f"**Category**: `{state.category_code}` &nbsp; **DATA index(next)**: `{state.data_index}` "
               f"&nbsp; **extending**: `{state.extending}`")
    status = f"{state.cursor+1} / {len(state.df)}"

    return _bytes_to_numpy_png(png_bytes), md_text, pdf_html, info_md, status


def on_pass(state: AppState):
    print("[PASS] clicked")
    row = get_current_row(state)
    if row is None:
        return state, *_render_current(state)
    png_bytes = download_png(state.img_bc, row["png_name"])
    md_text = download_md(state.md_bc, row["md_name"])
    _commit_single(state, row, png_bytes, md_text)
    _advance_cursor(state)
    return state, *_render_current(state)


def on_extend(state: AppState):
    print("[EXTEND] clicked")
    row = get_current_row(state)
    if row is None:
        return state, *_render_current(state)
    png_bytes = download_png(state.img_bc, row["png_name"])
    md_text = download_md(state.md_bc, row["md_name"])
    state.extending = True
    state.current_group_items.append((row, png_bytes, md_text))
    state.seg_index += 1
    print(f"[EXTEND] staged seg={state.seg_index} (group size={len(state.current_group_items)})")
    _advance_cursor(state)
    return state, *_render_current(state)


def on_extend_end(state: AppState):
    print("[EXTEND_END] clicked")
    if not state.extending or len(state.current_group_items) == 0:
        print("[EXTEND_END] nothing to finalize")
        return state, *_render_current(state)
    _commit_extending_all(state)
    return state, *_render_current(state)


def on_next(state: AppState):
    """
    Skip(잘못된 데이터): 현재 행을 즉시 삭제.
      - 이미지/마크다운 원본 블랍 즉시 삭제
      - history.csv에서 해당 행 즉시 제거 & 업로드
      - 커서는 '같은 인덱스'에 유지(삭제 후 다음 행이 그 자리를 채움)
      - data_index/last_number 변화 없음
    """
    print("[SKIP] clicked")
    row = get_current_row(state)
    if row is None:
        return state, *_render_current(state)

    png_name = str(row.get("png_name", "") or "")
    md_name  = str(row.get("md_name", "") or "")
    # 1) 소스 삭제(베스트 에포트)
    _delete_source_blobs(state.img_bc, state.md_bc, png_name, md_name)

    # 2) history DataFrame에서 제거 후 업로드
    df = state.df if state.df is not None else pd.DataFrame()
    before = len(df)
    if before > 0:
        # 현재 커서의 행을 제거
        new_df = df.drop(index=state.cursor).reset_index(drop=True)
        ok, msg = save_history_csv(state.hist_bc, new_df)
        if ok:
            state.df = new_df
            # 커서는 그대로(같은 위치에 다음 행이 왔음). 만약 마지막을 지웠다면 get_current_row가 None 반환.
            print(f"[SKIP] removed row at cursor={state.cursor} | rows {before} -> {len(new_df)}")
        else:
            print(f"[SKIP][ERROR] history upload failed after drop: {msg}")

    return state, *_render_current(state)


def on_exit(state: AppState):
    """
    Exit:
      - 확정(패스/extend_end)된 행들만 모아 history.csv에서 제거
      - 그 행들의 원본 PNG/MD 블랍도 일괄 삭제
      - last_number.txt 갱신(data_index - 1)
    """
    print(f"[EXIT] clicked | processed_keys={len(state.processed_png_keys)}")
    ui_msgs = []

    # 1) history.csv 커밋(확정 행 제거)
    try:
        ok_hist, upload_msg = _remove_processed_rows_from_history(state)
        base_msg = "Committed" if ok_hist else "Commit failed"
        ui_msgs.append(f"{base_msg}: {upload_msg}")
    except Exception as e:
        print(f"[EXIT][ERROR] history commit: {e}")
        ui_msgs.append(f"Commit failed: {e}")

    # 2) 확정 행들의 소스 PNG/MD 일괄 삭제
    del_msgs = []
    for (png_name, md_name) in list(state.processed_sources):
        ok_del, msg = _delete_source_blobs(state.img_bc, state.md_bc, png_name, md_name)
        del_msgs.append(msg)
    if state.processed_sources:
        ui_msgs.append("deleted sources for committed rows")
    state.processed_sources.clear()

    # 3) last_number.txt 업데이트 (마지막으로 사용한 번호 = data_index - 1)
    try:
        last_used = max(0, state.data_index - 1)
        ok_last, msg_last = _write_last_number(state.hist_bc, last_used)
        ui_msgs.append(msg_last if ok_last else f"last_number update failed: {msg_last}")
    except Exception as e:
        ui_msgs.append(f"last_number update failed: {e}")

    try:
        bc_cursor = state.hist_bc.container_client.get_blob_client("last_cursor.txt")
        bc_cursor.upload_blob(str(state.cursor).encode("utf-8"), overwrite=True)
        print(f"[EXIT] saved cursor={state.cursor}")
    except Exception as e:
        print(f"[EXIT][ERROR] failed to save cursor: {e}")

    msg = "### " + " | ".join(ui_msgs) + f"\nRemaining rows: {len(state.df) if state.df is not None else 0}\nNext DATA index: `{state.data_index}`"

    # 화면 갱신
    png, md_text, pdf_html, info_md, status = _render_current(state)
    if state.df is None or len(state.df) == 0:
        return (
            state,         # state
            None,          # image (numpy or None)
            "",            # markdown (md display)
            "",            # html (pdf_render)  ← 빠져 있던 요소!
            f"{msg}\nAll done.",  # info markdown
            "0 / 0",       # status markdown
        )

    # ✅ 일반 분기도 6개 모두 반환
    return (
        state,
        png,                          # image
        md_text,                      # markdown
        pdf_html,                     # html
        f"{msg}\n" + info_md,         # info markdown
        status,                       # status markdown
    )


# ===== Gradio UI =====
with gr.Blocks(title="Table Labeler",
                analytics_enabled=False,
                js=js,
                css=css,
                head=external_js) as demo:
    gr.Markdown("## Table Labeling UI (extend / extend_end / pass / skip / exit)")

    with gr.Row():
        data_container = gr.Textbox(label="Data (PDF) Container", value="pdf")
        image_container = gr.Textbox(label="Image Container", value="image")
        markdown_container = gr.Textbox(label="Markdown Container", value="markdown")
        history_container = gr.Textbox(label="History Container", value="history")

    with gr.Row():
        category_code = gr.Textbox(label="Category Code (e.g., I or F)", value="I")
        start_index = gr.Number(label="Start Data Index", value=0, precision=0)
        output_dir = gr.Textbox(label="(Optional) Local Output Dir (debug)", value="output")

    init_btn = gr.Button("Initialize / Load History")
    app_state = gr.State()    # AppState

    with gr.Row():
        img = gr.Image(label="PNG", type="numpy")
        md = gr.Markdown("Markdown will appear here")
    with gr.Row(elem_id="chat-info-panel"):
        pdf_render = gr.HTML("<div id='pdf-modal'></div>")
        

    info = gr.Markdown()
    status = gr.Markdown()

    with gr.Row():
        btn_pass = gr.Button("Pass (single)")
        btn_extend = gr.Button("Extend (add segment)")
        btn_extend_end = gr.Button("Extend End (finalize group)")
        btn_next = gr.Button("Skip (delete & next)")
        btn_exit = gr.Button("Exit (commit confirmed & cleanup)")

# 기존: outputs=[state, img, md, info, status]
# 변경: outputs=[state, img, md, pdf_render, info, status]

    init_btn.click(
        fn=on_init,
        inputs=[data_container, image_container, markdown_container, history_container, category_code, start_index, output_dir],
        outputs=[app_state, img, md, pdf_render, info, status]
    )

    # 나머지 핸들러도 inputs=[app_state], outputs 첫 번째는 app_state
    btn_pass.click(fn=on_pass, inputs=[app_state], outputs=[app_state, img, md, pdf_render, info, status])
    btn_extend.click(fn=on_extend, inputs=[app_state], outputs=[app_state, img, md, pdf_render, info, status])
    btn_extend_end.click(fn=on_extend_end, inputs=[app_state], outputs=[app_state, img, md, pdf_render, info, status])
    btn_next.click(fn=on_next, inputs=[app_state], outputs=[app_state, img, md, pdf_render, info, status])
    btn_exit.click(fn=on_exit, inputs=[app_state], outputs=[app_state, img, md, pdf_render, info, status])

# 이벤트 큐
demo.load(js=pdf_view_js)
demo.queue(max_size=32)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        allowed_paths=[src_path, pdfjs_path],
        server_port=int(os.getenv("PORT", "7860")),
        share=True,
        show_api=False
    )