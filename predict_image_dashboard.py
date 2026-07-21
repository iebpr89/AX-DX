from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import AutoModelForImageClassification

try:
    from transformers import AutoImageProcessor as ProcessorAutoClass
except ImportError:
    from transformers import AutoFeatureExtractor as ProcessorAutoClass


def apply_theme() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #f3f7fb 0%, #e8f0f7 100%);
            }
            .block-container {
                padding-top: 1.6rem;
                padding-bottom: 2rem;
            }
            .result-card {
                border-radius: 14px;
                padding: 18px;
                margin-top: 10px;
                box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
                border: 1px solid rgba(14, 116, 144, 0.2);
                background: #ffffff;
            }
            .result-title {
                color: #0f172a;
                font-size: 0.95rem;
                font-weight: 700;
                margin-bottom: 8px;
            }
            .result-value {
                color: #0e7490;
                font-size: 1.45rem;
                font-weight: 800;
            }
            .confidence-value {
                color: #b45309;
                font-size: 1.45rem;
                font-weight: 800;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def discover_model_dirs(workspace_root: Path) -> list[str]:
    candidates: list[str] = []
    for config_path in workspace_root.rglob("config.json"):
        model_dir = config_path.parent
        has_weights = (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()
        has_processor = (model_dir / "preprocessor_config.json").exists() or (model_dir / "feature_extractor_config.json").exists()
        if has_weights and has_processor:
            try:
                candidates.append(str(model_dir.relative_to(workspace_root)).replace("\\", "/"))
            except ValueError:
                candidates.append(str(model_dir))
    return sorted(set(candidates))


@st.cache_resource(show_spinner=False)
def load_model(model_dir: str):
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"모델 폴더를 찾을 수 없습니다: {model_path}")

    processor = ProcessorAutoClass.from_pretrained(str(model_path))
    model = AutoModelForImageClassification.from_pretrained(str(model_path))
    model.eval()
    return processor, model


def predict_image(image: Image.Image, processor, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    pred_id = int(torch.argmax(probs).item())
    confidence = float(probs[pred_id].item())
    label_name = model.config.id2label.get(pred_id, str(pred_id))

    top_k = min(3, probs.shape[0])
    top_vals, top_ids = torch.topk(probs, k=top_k)
    top_rows = []
    for idx, prob in zip(top_ids.tolist(), top_vals.tolist()):
        top_rows.append(
            {
                "label": model.config.id2label.get(int(idx), str(int(idx))),
                "confidence": float(prob),
            }
        )

    return label_name, confidence, pd.DataFrame(top_rows)


def main() -> None:
    st.set_page_config(page_title="AI 이미지 분류 대시보드", layout="wide")
    apply_theme()

    st.title("AI 이미지 분류 대시보드")
    st.write("학습된 모델로 이미지 1장을 업로드해 상태를 분류합니다.")

    workspace_root = Path.cwd()
    discovered_dirs = discover_model_dirs(workspace_root)

    with st.sidebar:
        st.header("설정")
        default_dir = discovered_dirs[0] if discovered_dirs else "outputs_vit/best_model"
        if discovered_dirs:
            model_dir = st.selectbox("탐지된 모델 폴더", options=discovered_dirs, index=0)
            st.caption("자동 탐지된 모델 폴더를 선택하세요.")
        else:
            model_dir = st.text_input("모델 폴더 경로", value=default_dir)
            st.caption("예: outputs_vit/best_model 또는 outputs_vit_tiny/best_model")

    try:
        with st.spinner("모델 로딩 중..."):
            processor, model = load_model(model_dir)
    except Exception as exc:
        st.error(f"모델 로딩 실패: {exc}")
        if not discovered_dirs:
            st.info("현재 워크스페이스에서 학습 완료 모델 폴더를 찾지 못했습니다.")
            st.code(
                "python train_vit_defect_classifier.py --cpu-optimized --tiny-test --early-stopping --output-dir outputs_vit_tiny",
                language="bash",
            )
            st.caption("학습 후 모델 경로 예시: outputs_vit_tiny/best_model")
        st.stop()

    uploaded_file = st.file_uploader(
        "이미지 파일 업로드",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="한 장의 이미지 파일을 업로드하세요.",
    )

    if uploaded_file is None:
        st.info("분류할 이미지를 업로드해주세요.")
        st.stop()

    image = Image.open(BytesIO(uploaded_file.read())).convert("RGB")

    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        st.subheader("입력 이미지")
        st.image(image, caption=uploaded_file.name, use_container_width=True)

    with st.spinner("분류 실행 중..."):
        try:
            label_name, confidence, top_df = predict_image(image, processor, model)
        except Exception as exc:
            st.error(f"분류 실패: {exc}")
            st.stop()

    with right_col:
        st.subheader("분류 결과")
        st.markdown(
            f"""
            <div class='result-card'>
              <div class='result-title'>분류 결과</div>
              <div class='result-value'>{label_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class='result-card'>
              <div class='result-title'>신뢰도</div>
              <div class='confidence-value'>{confidence * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 상위 예측")
        chart_df = top_df.copy()
        chart_df["confidence_pct"] = chart_df["confidence"] * 100
        st.bar_chart(chart_df, x="label", y="confidence_pct", color="#0E7490")

        display_df = chart_df.rename(columns={"label": "클래스", "confidence_pct": "신뢰도(%)"})
        display_df["신뢰도(%)"] = display_df["신뢰도(%)"].map(lambda x: f"{x:.2f}")
        st.table(display_df[["클래스", "신뢰도(%)"]])


if __name__ == "__main__":
    main()
