from __future__ import annotations

import math
import os
import queue
import threading
import zlib
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from paflab.experiment_images import (
    EFFECT_LABELS,
    METHOD_LABELS,
    PROJECT_ROOT,
    ExperimentImageGenerator,
    load_manifest,
)


LABEL_TO_EFFECT = {label: key for key, label in EFFECT_LABELS.items()}


class ExperimentImageApp:
    """単体実験画像と中間処理一式を対話的に書き出すGUI。"""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("PAF 単体実験画像ジェネレーター")
        self.root.geometry("1180x960")
        self.root.minsize(900, 780)

        self.dataset_var = tk.StringVar(
            value=str(PROJECT_ROOT / "output/datasets/research_base_v1")
        )
        self.output_var = tk.StringVar(
            value=str(PROJECT_ROOT / "output/gui_exports")
        )
        self.filter_var = tk.StringVar()
        self.sample_var = tk.StringVar()
        self.effect_var = tk.StringVar(value=EFFECT_LABELS["clean"])
        self.severity_var = tk.DoubleVar(value=25.0)
        self.severity_text = tk.StringVar(value="25%")
        self.seed_var = tk.StringVar(value="20260716")
        self.status_var = tk.StringVar(value="データセットを読み込み中...")
        self.method_vars = {
            method: tk.BooleanVar(value=True) for method in METHOD_LABELS
        }

        self.samples: dict[str, dict] = {}
        self.sample_ids: list[str] = []
        self.generator: ExperimentImageGenerator | None = None
        self.preview_image: tk.PhotoImage | None = None
        self.last_output_dir: Path | None = None
        self.worker_results: queue.Queue = queue.Queue()

        self._build_layout()
        self._effect_changed()
        self._load_dataset()

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(10, weight=1)

        ttk.Label(outer, text="データセット").grid(
            row=0, column=0, sticky="w", pady=4
        )
        ttk.Entry(outer, textvariable=self.dataset_var).grid(
            row=0, column=1, sticky="ew", padx=8, pady=4
        )
        ttk.Button(
            outer,
            text="選択...",
            command=self._choose_dataset,
        ).grid(row=0, column=2, pady=4)
        ttk.Button(
            outer,
            text="再読込",
            command=self._load_dataset,
        ).grid(row=0, column=3, padx=(8, 0), pady=4)

        ttk.Label(outer, text="sample_id絞り込み").grid(
            row=1, column=0, sticky="w", pady=4
        )
        filter_entry = ttk.Entry(outer, textvariable=self.filter_var)
        filter_entry.grid(row=1, column=1, sticky="ew", padx=8, pady=4)
        filter_entry.bind("<KeyRelease>", lambda _: self._apply_filter())
        ttk.Button(
            outer,
            text="解除",
            command=self._clear_filter,
        ).grid(row=1, column=2, pady=4)

        ttk.Label(outer, text="入力サンプル").grid(
            row=2, column=0, sticky="w", pady=4
        )
        self.sample_combo = ttk.Combobox(
            outer,
            textvariable=self.sample_var,
            state="readonly",
        )
        self.sample_combo.grid(
            row=2,
            column=1,
            columnspan=3,
            sticky="ew",
            padx=8,
            pady=4,
        )

        ttk.Separator(outer).grid(
            row=3,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=10,
        )

        ttk.Label(outer, text="付与する効果").grid(
            row=4, column=0, sticky="w", pady=4
        )
        effect_combo = ttk.Combobox(
            outer,
            textvariable=self.effect_var,
            values=list(LABEL_TO_EFFECT),
            state="readonly",
        )
        effect_combo.grid(row=4, column=1, sticky="ew", padx=8, pady=4)
        effect_combo.bind("<<ComboboxSelected>>", lambda _: self._effect_changed())

        ttk.Label(outer, text="強度").grid(
            row=5, column=0, sticky="w", pady=4
        )
        self.severity_scale = ttk.Scale(
            outer,
            from_=0,
            to=100,
            variable=self.severity_var,
            command=self._severity_changed,
        )
        self.severity_scale.grid(
            row=5,
            column=1,
            sticky="ew",
            padx=8,
            pady=4,
        )
        ttk.Label(outer, textvariable=self.severity_text, width=6).grid(
            row=5, column=2, sticky="w", pady=4
        )

        ttk.Label(outer, text="使用する手法").grid(
            row=6, column=0, sticky="w", pady=4
        )
        method_frame = ttk.Frame(outer)
        method_frame.grid(
            row=6,
            column=1,
            columnspan=3,
            sticky="w",
            padx=8,
            pady=4,
        )
        for column, (method, label) in enumerate(METHOD_LABELS.items()):
            ttk.Checkbutton(
                method_frame,
                text=label,
                variable=self.method_vars[method],
            ).grid(row=0, column=column, sticky="w", padx=(0, 18))

        ttk.Label(outer, text="乱数seed").grid(
            row=7, column=0, sticky="w", pady=4
        )
        ttk.Entry(outer, textvariable=self.seed_var, width=18).grid(
            row=7, column=1, sticky="w", padx=8, pady=4
        )

        ttk.Label(outer, text="保存先").grid(
            row=8, column=0, sticky="w", pady=4
        )
        ttk.Entry(outer, textvariable=self.output_var).grid(
            row=8, column=1, sticky="ew", padx=8, pady=4
        )
        ttk.Button(
            outer,
            text="選択...",
            command=self._choose_output,
        ).grid(row=8, column=2, pady=4)

        result_frame = ttk.LabelFrame(
            outer,
            text="検出結果（成功基準: IoU ≥ 0.80）",
            padding=8,
        )
        result_frame.grid(
            row=9,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(10, 0),
        )
        result_frame.columnconfigure(0, weight=1)
        self.result_tree = ttk.Treeview(
            result_frame,
            columns=("method", "iou", "judgment"),
            show="headings",
            height=3,
        )
        self.result_tree.heading("method", text="手法")
        self.result_tree.heading("iou", text="IoU")
        self.result_tree.heading("judgment", text="判定")
        self.result_tree.column("method", width=430, anchor="w")
        self.result_tree.column("iou", width=100, anchor="center")
        self.result_tree.column("judgment", width=130, anchor="center")
        self.result_tree.tag_configure("success", foreground="#137333")
        self.result_tree.tag_configure("failure", foreground="#b3261e")
        self.result_tree.tag_configure("none", foreground="#5f6368")
        self.result_tree.grid(row=0, column=0, sticky="ew")

        preview_frame = ttk.LabelFrame(
            outer,
            text="生成結果プレビュー（保存時は各段階を個別PNGでも出力）",
            padding=8,
        )
        preview_frame.grid(
            row=10,
            column=0,
            columnspan=4,
            sticky="nsew",
            pady=(12, 8),
        )
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        self.preview_label = ttk.Label(
            preview_frame,
            text="「生成して保存」を押すと、ここに中間処理一覧を表示します。",
            anchor="center",
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        actions = ttk.Frame(outer)
        actions.grid(row=11, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        actions.columnconfigure(0, weight=1)
        ttk.Label(actions, textvariable=self.status_var).grid(
            row=0, column=0, sticky="w"
        )
        self.open_button = ttk.Button(
            actions,
            text="保存フォルダーを開く",
            command=self._open_output,
            state=tk.DISABLED,
        )
        self.open_button.grid(row=0, column=1, padx=8)
        self.generate_button = ttk.Button(
            actions,
            text="生成して保存",
            command=self._start_generation,
        )
        self.generate_button.grid(row=0, column=2)

    def _choose_dataset(self) -> None:
        selected = filedialog.askdirectory(
            title="manifest.jsonを含むデータセットを選択",
            initialdir=self.dataset_var.get(),
        )
        if selected:
            self.dataset_var.set(selected)
            self._load_dataset()

    def _choose_output(self) -> None:
        selected = filedialog.askdirectory(
            title="出力先を選択",
            initialdir=self.output_var.get(),
        )
        if selected:
            self.output_var.set(selected)

    def _load_dataset(self) -> None:
        try:
            _, manifest, self.samples = load_manifest(self.dataset_var.get())
        except Exception as error:
            messagebox.showerror("データセット読込エラー", str(error))
            self.status_var.set("データセットを読み込めませんでした。")
            return
        self.sample_ids = sorted(self.samples)
        self._apply_filter()
        self.status_var.set(
            f"{manifest.get('experiment_id', 'dataset')}: "
            f"{len(self.sample_ids):,}サンプルを読み込みました。"
        )

    def _apply_filter(self) -> None:
        query = self.filter_var.get().strip().lower()
        values = [
            sample_id
            for sample_id in self.sample_ids
            if not query or query in sample_id.lower()
        ]
        self.sample_combo["values"] = values
        if self.sample_var.get() not in values:
            self.sample_var.set(values[0] if values else "")
        self.status_var.set(f"表示候補: {len(values):,}件")

    def _clear_filter(self) -> None:
        self.filter_var.set("")
        self._apply_filter()

    def _severity_changed(self, value: str) -> None:
        percent = int(round(float(value)))
        self.severity_text.set(f"{percent}%")

    def _effect_changed(self) -> None:
        clean = LABEL_TO_EFFECT[self.effect_var.get()] == "clean"
        self.severity_scale.configure(state=tk.DISABLED if clean else tk.NORMAL)
        if clean:
            self.severity_text.set("0%")
        else:
            self._severity_changed(str(self.severity_var.get()))

    def _start_generation(self) -> None:
        sample_id = self.sample_var.get()
        if not sample_id:
            messagebox.showwarning("入力未選択", "入力サンプルを選択してください。")
            return
        try:
            seed = int(self.seed_var.get())
        except ValueError:
            messagebox.showwarning("seedエラー", "乱数seedは整数で指定してください。")
            return
        effect = LABEL_TO_EFFECT[self.effect_var.get()]
        severity = 0.0 if effect == "clean" else self.severity_var.get() / 100.0
        selected_methods = [
            method
            for method, selected in self.method_vars.items()
            if selected.get()
        ]
        if not selected_methods:
            messagebox.showwarning(
                "手法未選択",
                "使用する手法を1つ以上選択してください。",
            )
            return
        output_root = Path(self.output_var.get())
        digest = zlib.crc32(sample_id.encode("utf-8")) % (2**32)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = (
            output_root
            / f"{timestamp}__{effect}_s{round(severity * 1000):04d}_{digest:08x}"
        )

        self.generate_button.configure(state=tk.DISABLED)
        self.open_button.configure(state=tk.DISABLED)
        self._clear_results()
        if "cnn_ransac" in selected_methods:
            self.status_var.set(
                "生成中です。初回はCNNモデル読込に数秒かかります..."
            )
        else:
            self.status_var.set("選択した手法で生成中です...")
        worker = threading.Thread(
            target=self._generate_worker,
            args=(
                self.dataset_var.get(),
                sample_id,
                effect,
                severity,
                seed,
                output_dir,
                selected_methods,
            ),
            daemon=True,
        )
        worker.start()
        self.root.after(100, self._poll_worker)

    def _generate_worker(
        self,
        dataset_dir: str,
        sample_id: str,
        effect: str,
        severity: float,
        seed: int,
        output_dir: Path,
        selected_methods: list[str],
    ) -> None:
        try:
            if self.generator is None:
                self.generator = ExperimentImageGenerator()
            result = self.generator.generate(
                dataset_dir,
                sample_id,
                effect=effect,
                severity=severity,
                seed=seed,
                output_dir=output_dir,
                selected_methods=selected_methods,
            )
        except Exception as error:
            self.worker_results.put(("error", error))
            return
        self.worker_results.put(("success", result))

    def _poll_worker(self) -> None:
        try:
            status, payload = self.worker_results.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_worker)
            return
        if status == "success":
            self._generation_finished(payload)
        else:
            self._generation_failed(payload)

    def _generation_failed(self, error: Exception) -> None:
        self.generate_button.configure(state=tk.NORMAL)
        self.status_var.set("生成に失敗しました。")
        messagebox.showerror("生成エラー", str(error))

    def _generation_finished(self, result: dict) -> None:
        self.generate_button.configure(state=tk.NORMAL)
        self.last_output_dir = Path(result["output_dir"])
        self.open_button.configure(state=tk.NORMAL)
        self._show_preview(Path(result["overview"]))
        self._show_method_results(result["method_results"])
        image_count = len(result["stage_files"])
        self.status_var.set(
            f"完了: {self.last_output_dir} "
            f"（個別PNG {image_count}枚、overview.png、metadata.json）"
        )

    def _clear_results(self) -> None:
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)

    def _show_method_results(self, method_results: dict) -> None:
        """選択した各手法のIoUと成功・失敗をGUIへ表示する。"""
        self._clear_results()
        for method in METHOD_LABELS:
            if method not in method_results:
                continue
            result = method_results[method]
            evaluation = result.get("evaluation")
            if evaluation is None:
                iou_text = "—"
                judgment = "検出なし"
                tag = "none"
            else:
                iou = float(evaluation["ellipse_iou"])
                iou_text = f"{iou:.3f}"
                success = iou >= 0.80
                judgment = "成功" if success else "失敗"
                tag = "success" if success else "failure"
            self.result_tree.insert(
                "",
                tk.END,
                values=(result["label"], iou_text, judgment),
                tags=(tag,),
            )

    def _show_preview(self, path: Path) -> None:
        preview = tk.PhotoImage(file=str(path))
        factor = max(
            1,
            math.ceil(preview.width() / 1050),
            math.ceil(preview.height() / 590),
        )
        if factor > 1:
            preview = preview.subsample(factor, factor)
        self.preview_image = preview
        self.preview_label.configure(image=preview, text="")

    def _open_output(self) -> None:
        if self.last_output_dir is not None:
            os.startfile(self.last_output_dir)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    ExperimentImageApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
