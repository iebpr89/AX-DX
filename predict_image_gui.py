import argparse
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import torch
from PIL import Image, ImageTk
from transformers import AutoImageProcessor, AutoModelForImageClassification


class ImageClassifierApp:
    def __init__(self, root: tk.Tk, model_dir: str):
        self.root = root
        self.root.title("AI Image Classifier")
        self.root.geometry("900x700")

        self.model_dir_var = tk.StringVar(value=model_dir)
        self.image_path_var = tk.StringVar(value="")
        self.result_var = tk.StringVar(value="분류 결과: -")
        self.confidence_var = tk.StringVar(value="신뢰도: -")
        self.status_var = tk.StringVar(value="상태: 모델 로딩 대기")

        self.processor = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preview_photo = None

        self._build_ui()
        self._load_model_async()

    def _build_ui(self) -> None:
        top_frame = tk.Frame(self.root, padx=12, pady=12)
        top_frame.pack(fill="x")

        tk.Label(top_frame, text="모델 폴더:").grid(row=0, column=0, sticky="w")
        tk.Entry(top_frame, textvariable=self.model_dir_var, width=70).grid(row=0, column=1, padx=8)
        tk.Button(top_frame, text="찾기", command=self._choose_model_dir).grid(row=0, column=2)
        tk.Button(top_frame, text="모델 다시 로드", command=self._load_model_async).grid(row=0, column=3, padx=(8, 0))

        mid_frame = tk.Frame(self.root, padx=12, pady=4)
        mid_frame.pack(fill="x")

        tk.Label(mid_frame, text="이미지 파일:").grid(row=0, column=0, sticky="w")
        tk.Entry(mid_frame, textvariable=self.image_path_var, width=70).grid(row=0, column=1, padx=8)
        tk.Button(mid_frame, text="열기", command=self._choose_image).grid(row=0, column=2)
        tk.Button(mid_frame, text="분류 실행", command=self._classify_async).grid(row=0, column=3, padx=(8, 0))

        result_frame = tk.Frame(self.root, padx=12, pady=12)
        result_frame.pack(fill="x")

        tk.Label(result_frame, textvariable=self.result_var, font=("Segoe UI", 13, "bold")).pack(anchor="w")
        tk.Label(result_frame, textvariable=self.confidence_var, font=("Segoe UI", 12)).pack(anchor="w", pady=(4, 0))
        tk.Label(result_frame, textvariable=self.status_var, fg="#555").pack(anchor="w", pady=(8, 0))

        preview_frame = tk.Frame(self.root, padx=12, pady=12)
        preview_frame.pack(fill="both", expand=True)

        self.image_label = tk.Label(preview_frame, text="이미지를 선택하면 미리보기가 표시됩니다.", bg="#f5f5f5")
        self.image_label.pack(fill="both", expand=True)

    def _choose_model_dir(self) -> None:
        selected = filedialog.askdirectory(title="모델 폴더 선택")
        if selected:
            self.model_dir_var.set(selected)

    def _choose_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="이미지 선택",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if not selected:
            return

        self.image_path_var.set(selected)
        self._show_image_preview(selected)

    def _show_image_preview(self, image_path: str) -> None:
        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((760, 460))
            self.preview_photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.preview_photo, text="")
        except Exception as exc:
            self.image_label.configure(image="", text=f"미리보기 실패: {exc}")

    def _load_model_async(self) -> None:
        worker = threading.Thread(target=self._load_model, daemon=True)
        worker.start()

    def _load_model(self) -> None:
        model_dir = Path(self.model_dir_var.get().strip())
        if not model_dir.exists():
            self.status_var.set(f"상태: 모델 폴더가 없습니다 - {model_dir}")
            return

        self.status_var.set("상태: 모델 로딩 중...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(str(model_dir))
            self.model = AutoModelForImageClassification.from_pretrained(str(model_dir)).to(self.device)
            self.model.eval()
            self.status_var.set(f"상태: 모델 로딩 완료 (device={self.device})")
        except Exception as exc:
            self.processor = None
            self.model = None
            self.status_var.set(f"상태: 모델 로딩 실패 - {exc}")

    def _classify_async(self) -> None:
        worker = threading.Thread(target=self._classify, daemon=True)
        worker.start()

    def _classify(self) -> None:
        if self.model is None or self.processor is None:
            messagebox.showwarning("알림", "먼저 모델을 로딩해주세요.")
            return

        image_path = self.image_path_var.get().strip()
        if not image_path:
            messagebox.showwarning("알림", "분류할 이미지를 선택해주세요.")
            return
        if not Path(image_path).exists():
            messagebox.showerror("오류", f"이미지 파일이 없습니다: {image_path}")
            return

        self.status_var.set("상태: 분류 중...")

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]

            pred_id = int(torch.argmax(probs).item())
            confidence = float(probs[pred_id].item())
            label_name = self.model.config.id2label.get(pred_id, str(pred_id))

            self.result_var.set(f"분류 결과: {label_name}")
            self.confidence_var.set(f"신뢰도: {confidence * 100:.2f}%")
            self.status_var.set("상태: 분류 완료")
        except Exception as exc:
            self.status_var.set("상태: 분류 실패")
            messagebox.showerror("오류", f"분류 중 오류가 발생했습니다:\n{exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-image classifier GUI for trained ViT model")
    parser.add_argument(
        "--model-dir",
        default="outputs_vit/best_model",
        help="Path to trained model directory (contains config.json and model weights)",
    )
    args = parser.parse_args()

    root = tk.Tk()
    app = ImageClassifierApp(root=root, model_dir=args.model_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
