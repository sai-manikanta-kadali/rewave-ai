from __future__ import annotations

import json
from collections import Counter
from io import BytesIO
from pathlib import Path

import easyocr
import pandas as pd
import streamlit as st
from PIL import Image

from reweave_core import (
    confirm_inspection,
    fetch_inspections,
    image_diagnostics,
    init_db,
    prepare_image,
    save_inspection,
)
from reweave_decision import (
    PATHWAY_MANUAL,
    PATHWAY_RECOVERY,
    PATHWAY_REPAIR,
    PATHWAY_REUSE,
    evaluate_garment,
)
from reweave_ocr import run_label_ocr
from reweave_vision import DEFAULT_MODEL_PATH, detect_defects, load_defect_model

BASE_DIR = Path(__file__).resolve().parent
EVAL_REPORT_PATH = BASE_DIR / "reports" / "model_evaluation.json"
DEMO_DIR = BASE_DIR / "demo_samples"
ALL_PATHWAYS = [PATHWAY_REUSE, PATHWAY_REPAIR, PATHWAY_RECOVERY, PATHWAY_MANUAL]
RECOVERY_LABELS = {PATHWAY_RECOVERY, "MATERIAL TESTING / RECYCLING ASSESSMENT"}

st.set_page_config(
    page_title="ReWeave AI | Circular Textile Intelligence",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize and safely migrate SQLite schema in-place (preserves existing records).
init_db()

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1280px;}
      .rw-kicker {color:#14b8a6; font-weight:700; letter-spacing:.12em; font-size:.75rem;}
      .rw-sub {color:#94a3b8; font-size:1rem;}
      div[data-testid="stMetric"] {border:1px solid rgba(148,163,184,.25); padding:12px; border-radius:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading English care-label OCR model...")
def get_ocr_reader():
    return easyocr.Reader(["en"], gpu=False)


@st.cache_resource(show_spinner="Loading trained ReWeave YOLO defect detector...")
def get_yolo_model():
    return load_defect_model(DEFAULT_MODEL_PATH)


def _safe_json_list(raw_val) -> list:
    if isinstance(raw_val, list):
        return raw_val
    if not raw_val or (isinstance(raw_val, float) and pd.isna(raw_val)):
        return []
    try:
        parsed = json.loads(str(raw_val))
        return parsed if isinstance(parsed, list) else []
    except (ValueError, TypeError):
        return []


def show_pathway_banner(pathway: str) -> None:
    if pathway == PATHWAY_REUSE:
        st.success(f"**Recommended Pathway:** {pathway}")
    elif pathway == PATHWAY_REPAIR:
        st.info(f"**Recommended Pathway:** {pathway}")
    elif pathway == PATHWAY_MANUAL:
        st.warning(f"**Recommended Pathway:** {pathway}")
    else:
        st.error(f"**Recommended Pathway:** {pathway}")


def show_dashboard() -> None:
    st.subheader("Sorting analytics dashboard")
    df = fetch_inspections()
    if df.empty:
        st.info(
            "No inspections recorded yet. Go to **New inspection**, click a **1-Click Judge Demo Sample**, "
            "and confirm an inspection to see live analytics populate here."
        )
        return

    total = len(df)
    reuse = int((df["recommendation"] == PATHWAY_REUSE).sum())
    repair = int((df["recommendation"] == PATHWAY_REPAIR).sum())
    testing = int(df["recommendation"].isin(RECOVERY_LABELS).sum())
    confirmed = int((df["confirmation_status"] == "CONFIRMED").sum())
    pending = total - confirmed

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Inspected", total)
    c2.metric("Reuse", reuse)
    c3.metric("Repair", repair)
    c4.metric("Material Testing", testing)
    c5.metric("Confirmed", confirmed)
    c6.metric("Pending Review", pending)

    chart_col1, chart_col2 = st.columns(2, gap="large")
    with chart_col1:
        st.markdown("#### Recommended pathways")
        normalized_pathways = df["recommendation"].replace(
            {"MATERIAL TESTING / RECYCLING ASSESSMENT": PATHWAY_RECOVERY}
        )
        pathway_counts = normalized_pathways.value_counts().rename_axis("Pathway").to_frame("Records")
        st.bar_chart(pathway_counts)

    with chart_col2:
        st.markdown("#### Defect category distribution (YOLO + Worker)")
        defect_counter: Counter[str] = Counter()
        for raw_yolo in df["yolo_detections"]:
            for det in _safe_json_list(raw_yolo):
                cls_name = det.get("class_name")
                if cls_name:
                    defect_counter[f"YOLO: {cls_name}"] += 1
        for raw_worker in df["observed_damage"]:
            for dmg in _safe_json_list(raw_worker):
                if dmg not in {"No visible damage", "Other / unsure"}:
                    defect_counter[f"Worker: {dmg}"] += 1

        if defect_counter:
            defect_df = pd.DataFrame(
                defect_counter.most_common(), columns=["Defect", "Count"]
            ).set_index("Defect")
            st.bar_chart(defect_df)
        else:
            st.info("No defects logged yet across saved inspections.")

    st.markdown("#### Confirmation status breakdown")
    status_counts = df["confirmation_status"].value_counts().rename_axis("Status").to_frame("Records")
    st.bar_chart(status_counts, horizontal=True)

    pending_df = df[df["confirmation_status"] != "CONFIRMED"]
    if not pending_df.empty:
        with st.expander(f"Review & confirm pending inspections ({len(pending_df)})", expanded=False):
            selected_id = st.selectbox(
                "Select pending inspection ID",
                pending_df["id"].tolist(),
                format_func=lambda x: f"{x} — {pending_df.loc[pending_df['id'] == x, 'garment_type'].iloc[0]} ({pending_df.loc[pending_df['id'] == x, 'recommendation'].iloc[0]})",
            )
            chosen_final = st.selectbox("Final worker-confirmed pathway", ALL_PATHWAYS, key="dash_confirm_pathway")
            if st.button("Confirm selected inspection", type="primary"):
                confirm_inspection(selected_id, chosen_final)
                st.success(f"Inspection {selected_id} confirmed as {chosen_final}.")
                st.rerun()

    st.markdown("#### Inspection log")
    display_cols = [
        "id",
        "created_at",
        "image_name",
        "garment_type",
        "condition",
        "composition",
        "yolo_max_confidence",
        "recommendation",
        "confirmation_status",
        "confirmed_pathway",
    ]
    st.dataframe(df[display_cols], width="stretch", hide_index=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download full inspection log (CSV)",
        data=csv_bytes,
        file_name="reweave_inspection_log.csv",
        mime="text/csv",
    )


def show_about_page() -> None:
    st.subheader("System architecture, real model metrics & scientific honesty")

    st.markdown(
        """
        **Active pipeline in this build**
        1. **Image Intake (OpenCV):** EXIF orientation fix, RGB normalization, brightness & Laplacian sharpness diagnostics.
        2. **Visible Defect Detection (Custom YOLO):** Fine-tuned `yolo26n` (5.2 MB) on audited textile defect data (`hole`, `stain`, `yarn missing`) running locally on Apple Silicon / CPU.
        3. **Care-Label OCR (OpenCV LAB-CLAHE + EasyOCR):** Enhances contrast on optional tag photos, extracts explicit `% + fibre` pairs & care instructions, and keeps missing/unreadable composition strictly **`UNKNOWN`**.
        4. **Explainable Decision Engine:** Separates YOLO suggestions, OCR evidence, and worker observations; triggers a **Human–AI conflict guard** if a worker marks *"Ready to wear"* while YOLO flags a high-confidence defect.
        5. **Human-in-the-Loop Confirmation & SQLite Log:** Every record tracks `PENDING` vs `CONFIRMED` status in `reweave.db` with safe in-place schema migration.
        """
    )

    if EVAL_REPORT_PATH.is_file():
        eval_data = json.loads(EVAL_REPORT_PATH.read_text(encoding="utf-8"))
        st.markdown("#### Verified YOLO baseline metrics (Held-out `test` split)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Unseen Test Images", eval_data.get("images_in_split", "—"))
        m2.metric("Precision", f"{eval_data.get('precision_mean', 0):.1%}")
        m3.metric("Recall", f"{eval_data.get('recall_mean', 0):.1%}")
        m4.metric("mAP@50", f"{eval_data.get('map50', 0):.1%}")
        st.caption(
            f"Dataset: Roboflow Defect Detection (CC BY 4.0) | Classes: {', '.join(eval_data.get('class_names', {}).values())} | "
            f"mAP@50-95: {eval_data.get('map50_95', 0):.1%}"
        )

    st.warning(
        "**Known limitations (Honest engineering):**\n"
        "- Ordinary RGB photographs **cannot** identify chemical fibre blends — missing labels stay `UNKNOWN`.\n"
        "- The baseline YOLO detector was trained on flat fabric defect images; real worn/patterned garments (e.g., dark shirt buttons) can cause domain-gap false positives or misses.\n"
        "- Absence of a YOLO detection does **not** prove a garment is defect-free.\n"
        "- Worker confirmation is mandatory before final sorting or recycling."
    )


st.markdown('<div class="rw-kicker">CIRCULAR TEXTILE INTELLIGENCE • INNOVATE INDIA 2026 • PANIPAT ECOSYSTEM</div>', unsafe_allow_html=True)
st.title("🧵 ReWeave AI")
st.markdown(
    '<div class="rw-sub">Human-in-the-loop textile sorting assistant prioritizing REUSE and REPAIR before material recovery.</div>',
    unsafe_allow_html=True,
)
st.write("")

yolo_model = get_yolo_model()

with st.sidebar:
    st.header("ReWeave AI")
    st.caption("Open-Source AI • Vision + OCR + HITL")
    page = st.radio("Workspace", ["New inspection", "Sorting dashboard", "Architecture & model metrics"])
    st.divider()
    conf_threshold = st.slider(
        "YOLO confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Minimum confidence required to display a YOLO defect suggestion.",
    )
    if yolo_model is not None:
        st.success("YOLO model: Loaded (`hole`, `stain`, `yarn missing`)")
    else:
        st.warning("YOLO weights not found in `models/reweave_defect_yolo.pt`.")
    st.divider()
    st.caption("Core safeguards")
    st.write("• Unknown fibre stays **UNKNOWN**\n• YOLO suggests; **worker confirms**\n• Human–AI conflict guard active")

if page == "New inspection":
    # 1-Click Judge Demo Samples bar
    stain_sample = DEMO_DIR / "sample_stain.jpg"
    hole_sample = DEMO_DIR / "sample_hole.jpg"
    tag_sample = DEMO_DIR / "sample_care_label.png"

    if stain_sample.is_file() or hole_sample.is_file():
        st.markdown("##### ⚡ Quick Test for Judges (1-Click Real Test Samples)")
        d1, d2, d3 = st.columns([1.2, 1.3, 0.7])
        if d1.button("🧪 Demo 1: Stained Fabric + Cotton Tag", width="stretch"):
            st.session_state["demo_garment"] = str(stain_sample)
            st.session_state["demo_label"] = str(tag_sample) if tag_sample.is_file() else None
            st.session_state["demo_condition_idx"] = 1  # Minor repair needed
            st.session_state["demo_note"] = "Loaded unseen test stain + sample 100% Cotton tag"
            st.rerun()
        if d2.button("⚠️ Demo 2: Hole Defect (Conflict Guard Test)", width="stretch"):
            st.session_state["demo_garment"] = str(hole_sample) if hole_sample.is_file() else str(stain_sample)
            st.session_state["demo_label"] = None
            st.session_state["demo_condition_idx"] = 0  # Ready to wear -> triggers Human-AI conflict!
            st.session_state["demo_note"] = "Simulating worker marking 'Ready to wear' on a damaged item"
            st.rerun()
        if d3.button("🧹 Clear Demo", width="stretch"):
            st.session_state.pop("demo_garment", None)
            st.session_state.pop("demo_label", None)
            st.session_state.pop("demo_condition_idx", None)
            st.session_state.pop("demo_note", None)
            st.rerun()
        st.write("")

    st.subheader("Step 1: Capture or upload garment & optional care-label photos")
    capture_mode = st.radio(
        "Garment input method",
        ["Upload photo", "Take photo (Mobile / Webcam)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    up1, up2 = st.columns([1.1, 0.9], gap="large")
    with up1:
        if capture_mode == "Upload photo":
            garment_file = st.file_uploader(
                "1. Garment photo (required)",
                type=["jpg", "jpeg", "png", "webp"],
                help="Upload a clear, well-lit photo of the garment or textile item.",
            )
        else:
            garment_file = st.camera_input("1. Snap garment photo directly")
    with up2:
        label_file = st.file_uploader(
            "2. Care-label photo (optional)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a close-up photo of the care/composition tag for OCR.",
        )

    garment_image = None
    garment_name = None
    label_image = None

    # Uploaded / captured image takes priority over demo sample
    if garment_file is not None:
        try:
            garment_image = prepare_image(Image.open(BytesIO(garment_file.getvalue())))
            garment_name = garment_file.name
        except Exception as exc:
            st.error(f"Could not open garment image: {exc}")
    elif st.session_state.get("demo_garment") and Path(st.session_state["demo_garment"]).is_file():
        demo_p = Path(st.session_state["demo_garment"])
        garment_image = prepare_image(Image.open(demo_p))
        garment_name = f"demo/{demo_p.name}"
        st.info(f"Using 1-click demo sample: `{demo_p.name}` — click **Run AI Inspection** below!")

    if label_file is not None:
        try:
            label_image = prepare_image(Image.open(BytesIO(label_file.getvalue())))
        except Exception as exc:
            st.warning(f"Could not open care-label image: {exc}. Composition will remain UNKNOWN.")
    elif garment_file is None and st.session_state.get("demo_label") and Path(st.session_state["demo_label"]).is_file():
        label_image = prepare_image(Image.open(Path(st.session_state["demo_label"])))

    if garment_image is not None:
        preview_col, diag_col = st.columns([1, 1], gap="large")
        preview_col.image(garment_image, caption=f"Input: {garment_name}", width="stretch")
        diagnostics = image_diagnostics(garment_image)
        diag_col.markdown("**OpenCV intake diagnostics (image quality only)**")
        diag_col.write(f"Resolution: **{diagnostics['width']} × {diagnostics['height']} px**")
        diag_col.write(f"Mean brightness: **{diagnostics['mean_brightness']} / 255**")
        diag_col.write(f"Sharpness proxy: **{diagnostics['sharpness_proxy']}**")
        if diagnostics["mean_brightness"] < 45:
            diag_col.warning("Low lighting detected — better lighting improves YOLO and visual checks.")
        if diagnostics["sharpness_proxy"] < 30:
            diag_col.warning("Photo may be blurry — consider retaking a sharper image.")

    if label_image is not None:
        with st.expander("Preview loaded care-label photo", expanded=False):
            st.image(label_image, caption="Care-label input for LAB-CLAHE OCR", width=320)

    st.divider()
    st.subheader("Step 2: Worker observation & AI analysis")

    with st.form("inspection_form", clear_on_submit=False):
        left, right = st.columns([1, 1], gap="large")
        with left:
            garment_type = st.selectbox(
                "Garment / item type",
                ["Shirt", "T-shirt", "Trousers", "Jeans", "Dress", "Jacket", "Bedsheet", "Fabric scrap", "Other"],
            )
            observed_damage = st.multiselect(
                "Worker-observed damage",
                ["Hole", "Tear", "Stain", "Loose seam", "Missing button", "Colour fading", "No visible damage", "Other / unsure"],
            )
            worker_notes = st.text_area(
                "Worker notes (optional)",
                value=st.session_state.get("demo_note", ""),
                placeholder="Example: small stain near collar, easily washable",
            )
        with right:
            condition = st.radio(
                "Worker-assessed condition",
                ["Ready to wear", "Minor repair needed", "Beyond repair", "Unsure — needs inspection"],
                index=st.session_state.get("demo_condition_idx", 0),
                help="Combined with YOLO detections and OCR evidence in the explainable decision engine.",
            )

        submitted = st.form_submit_button("Run AI Inspection & Generate Recommendation", type="primary", width="stretch")

    if submitted:
        if garment_image is None or garment_name is None:
            st.error("Please upload a garment photo, use the camera, or click a 1-Click Judge Demo Sample above.")
        else:
            with st.spinner("Running YOLO defect detection, care-label OCR, and explainable rules..."):
                vision_out = detect_defects(garment_image, yolo_model, conf_threshold=conf_threshold)

                if label_image is not None:
                    try:
                        ocr_out = run_label_ocr(label_image, get_ocr_reader())
                    except Exception as exc:
                        st.warning(f"Care-label OCR failed: {exc}. Keeping composition UNKNOWN.")
                        ocr_out = {
                            "raw_text": "",
                            "composition": "UNKNOWN",
                            "composition_evidence": "OCR error — composition kept UNKNOWN",
                            "care_instructions": [],
                            "ocr_confidence": None,
                        }
                else:
                    ocr_out = {
                        "raw_text": "",
                        "composition": "UNKNOWN",
                        "composition_evidence": "No care-label photo provided",
                        "care_instructions": [],
                        "ocr_confidence": None,
                    }

                decision = evaluate_garment(
                    worker_condition=condition,
                    observed_damage=observed_damage,
                    yolo_detections=vision_out["detections"],
                    ocr_result=ocr_out,
                )

                record = {
                    "image_name": garment_name,
                    "garment_type": garment_type,
                    "condition": condition,
                    "observed_damage": observed_damage,
                    "care_label_text": ocr_out["raw_text"],
                    "composition": decision["composition"],
                    "composition_evidence": decision["composition_evidence"],
                    "care_instructions": decision["care_instructions"],
                    "ocr_confidence": ocr_out.get("ocr_confidence"),
                    "yolo_detections": vision_out["detections"],
                    "recommendation": decision["recommended_pathway"],
                    "rationale": decision["rationale"],
                    "worker_notes": worker_notes,
                    "confirmation_status": "PENDING",
                }
                inspection_id = save_inspection(record)

                st.session_state["last_inspection"] = {
                    "id": inspection_id,
                    "vision": vision_out,
                    "ocr": ocr_out,
                    "decision": decision,
                    "worker_notes": worker_notes,
                    "confirmed": False,
                    "confirmed_pathway": "",
                }

    active = st.session_state.get("last_inspection")
    if active:
        st.divider()
        st.subheader(f"Step 3: Evidence review & worker confirmation — ID `{active['id']}`")

        decision = active["decision"]
        vision_out = active["vision"]
        ocr_out = active["ocr"]

        if decision["conflict_flag"]:
            st.error(
                "⚠️ **Human–AI Conflict Detected:** Worker selected 'Ready to wear', but visible defect evidence was flagged. "
                "Please visually re-check before confirming."
            )

        col_vis, col_rec = st.columns([1.05, 0.95], gap="large")
        with col_vis:
            st.markdown("#### 1. YOLO Defect Detection (AI Suggestions)")
            st.image(vision_out["annotated_image"], caption=vision_out["summary_text"], width="stretch")
            if vision_out["detections"]:
                det_rows = [
                    {
                        "Defect": d["class_name"],
                        "Confidence": f"{d['confidence']:.1%}",
                        "Box Area %": f"{d['area_ratio'] * 100:.2f}%",
                    }
                    for d in vision_out["detections"]
                ]
                st.dataframe(pd.DataFrame(det_rows), width="stretch", hide_index=True)
            else:
                st.info("No YOLO defect boxes above threshold (not proof garment is defect-free).")

        with col_rec:
            st.markdown("#### 2. Recommended Pathway (Rules Engine)")
            show_pathway_banner(decision["recommended_pathway"])
            for reason in decision["reasons"]:
                st.write(f"• {reason}")

            st.markdown("#### 3. Care-Label OCR Evidence")
            st.write(f"**Fibre composition:** `{decision['composition']}`")
            st.caption(f"Evidence: {decision['composition_evidence']}")
            if decision["care_instructions"]:
                st.write("**Care instructions:** " + ", ".join(decision["care_instructions"]))
            if ocr_out["raw_text"]:
                with st.expander("View raw OCR text", expanded=False):
                    st.code(ocr_out["raw_text"])
                    if ocr_out.get("ocr_confidence") is not None:
                        st.caption(f"Mean OCR text confidence: {ocr_out['ocr_confidence']:.1%}")

            st.markdown("#### 4. Worker Final Confirmation (Required)")
            if active["confirmed"]:
                st.success(f"✅ Confirmed by worker as **{active['confirmed_pathway']}** and saved to `reweave.db`.")
            else:
                default_idx = (
                    ALL_PATHWAYS.index(decision["recommended_pathway"])
                    if decision["recommended_pathway"] in ALL_PATHWAYS
                    else 0
                )
                final_choice = st.selectbox(
                    "Confirm or override final sorting pathway",
                    ALL_PATHWAYS,
                    index=default_idx,
                    key=f"confirm_select_{active['id']}",
                )
                confirm_notes = st.text_input(
                    "Final confirmation note (optional)",
                    value=active.get("worker_notes", ""),
                    key=f"confirm_notes_{active['id']}",
                )
                if st.button("✅ Confirm Final Pathway", type="primary", width="stretch"):
                    confirm_inspection(active["id"], final_choice, worker_notes=confirm_notes)
                    st.session_state["last_inspection"]["confirmed"] = True
                    st.session_state["last_inspection"]["confirmed_pathway"] = final_choice
                    st.rerun()

elif page == "Sorting dashboard":
    show_dashboard()

else:
    show_about_page()
