"""
Tkinter GUI for Live Deepfake Detection.

Layout
------

┌────────────────────────────────────────────────────────────┐
│  LiveDeepfakeDetection                            [toolbar] │
├──────────────────────────┬─────────────────────────────────┤
│                          │  STATUS PANEL                   │
│   CAMERA FEED            │  ─────────────────────────────  │
│   (with overlay)         │  Detection:  ██████  REAL/FAKE  │
│                          │  Confidence: ██████  92.3 %     │
│                          │  FPS:        28                 │
│                          │  Temporal:   ██████  0.82       │
│                          │  ─────────────────────────────  │
│                          │  ALERT LOG                      │
│                          │  [scrollable text]              │
│                          │                                 │
├──────────────────────────┴─────────────────────────────────┤
│  [Start Camera]  [Stop Camera]  [Load Model]  [Clear Log]  │
│  Status bar                                     FPS: 0     │
└────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import os
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

# Silence TF logs before importing
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from src.detection.detector import DeepfakeDetector, DetectionResult

# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------

COLOUR_REAL = "#00e676"    # bright green
COLOUR_FAKE = "#ff1744"    # bright red
COLOUR_UNKNOWN = "#ffd600" # amber
COLOUR_BG = "#1a1a2e"      # dark navy
COLOUR_PANEL = "#16213e"
COLOUR_ACCENT = "#0f3460"
COLOUR_TEXT = "#e0e0e0"
FONT_FAMILY = "Helvetica"

# ---------------------------------------------------------------------------
# Overlay helper
# ---------------------------------------------------------------------------

def _draw_overlay(
    bgr_frame: np.ndarray,
    result: DetectionResult,
) -> np.ndarray:
    """Draw face bounding boxes and labels on *bgr_frame* (in-place copy).

    Args:
        bgr_frame: Raw BGR camera frame.
        result:    :class:`~src.detection.detector.DetectionResult` for this frame.

    Returns:
        Annotated BGR frame.
    """
    frame = bgr_frame.copy()
    for face in result.faces:
        x, y, w, h = face.bbox
        colour_bgr = (0, 73, 255) if face.label == "FAKE" else (0, 230, 118)

        # Bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour_bgr, 2)

        # Label background
        label = f"{face.label}  {face.p_fake * 100:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 4, y), colour_bgr, -1)
        cv2.putText(
            frame, label,
            (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

    # FPS watermark
    cv2.putText(
        frame, f"FPS: {result.fps:.1f}",
        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (180, 180, 180), 1, cv2.LINE_AA,
    )

    # Temporal alert banner
    if result.alert:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (0, 30, 220), -1)
        cv2.putText(
            frame, "⚠  DEEPFAKE ALERT  ⚠",
            (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
            (255, 255, 255), 2, cv2.LINE_AA,
        )

    return frame


# ---------------------------------------------------------------------------
# Confidence bar widget
# ---------------------------------------------------------------------------

class ConfidenceBar(tk.Frame):
    """A simple labelled progress-bar widget.

    Args:
        parent:   Parent widget.
        label:    Left-hand label text.
        color:    Bar fill colour.
    """

    def __init__(self, parent: tk.Widget, label: str, color: str = COLOUR_REAL) -> None:
        super().__init__(parent, bg=COLOUR_PANEL)
        self._color = color

        tk.Label(
            self, text=label, width=12, anchor="w",
            bg=COLOUR_PANEL, fg=COLOUR_TEXT,
            font=(FONT_FAMILY, 10),
        ).pack(side=tk.LEFT)

        self._bar = ttk.Progressbar(self, length=140, mode="determinate")
        self._bar.pack(side=tk.LEFT, padx=(0, 6))

        self._value_lbl = tk.Label(
            self, text="0.0 %", width=7, anchor="e",
            bg=COLOUR_PANEL, fg=COLOUR_TEXT,
            font=(FONT_FAMILY, 10, "bold"),
        )
        self._value_lbl.pack(side=tk.LEFT)

    def set(self, fraction: float, text: Optional[str] = None) -> None:
        """Update bar to *fraction* in [0, 1].

        Args:
            fraction: Value between 0 and 1.
            text:     Override the right-hand label (defaults to percentage).
        """
        pct = max(0.0, min(1.0, fraction)) * 100.0
        self._bar["value"] = pct
        self._value_lbl.config(text=text or f"{pct:.1f} %")


# ---------------------------------------------------------------------------
# Main Application Window
# ---------------------------------------------------------------------------

class DeepfakeDetectionApp(tk.Tk):
    """Main Tkinter application for live deepfake detection.

    Args:
        model_path: Optional path to a saved Keras or TFLite model to load
                    at startup.
    """

    _FEED_WIDTH = 640
    _FEED_HEIGHT = 480
    _QUEUE_MAXSIZE = 2  # keep GUI responsive; drop stale frames

    def __init__(self, model_path: Optional[str] = None) -> None:
        super().__init__()
        self.title("LiveDeepfakeDetection — Synthetic Face Monitor")
        self.configure(bg=COLOUR_BG)
        self.resizable(True, True)

        # Core components
        self._detector = DeepfakeDetector()
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=self._QUEUE_MAXSIZE)
        self._main_thread_queue: queue.Queue = queue.Queue()  # for cross-thread UI ops
        self._last_result: Optional[DetectionResult] = None
        self._alert_count = 0

        self._build_ui()
        self._update_loop()  # start polling

        if model_path:
            self._load_model(model_path)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build all widgets."""
        # ---- Menu bar ------------------------------------------------
        menubar = tk.Menu(self, bg=COLOUR_ACCENT, fg=COLOUR_TEXT)
        file_menu = tk.Menu(menubar, tearoff=0, bg=COLOUR_ACCENT, fg=COLOUR_TEXT)
        file_menu.add_command(label="Load Model …", command=self._on_load_model)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0, bg=COLOUR_ACCENT, fg=COLOUR_TEXT)
        help_menu.add_command(label="About", command=self._on_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.config(menu=menubar)

        # ---- Main content frame -------------------------------------
        content = tk.Frame(self, bg=COLOUR_BG)
        content.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 0))

        # Left: camera feed
        left = tk.Frame(content, bg=COLOUR_BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._feed_label = tk.Label(
            left, bg="#000000",
            width=self._FEED_WIDTH, height=self._FEED_HEIGHT,
        )
        self._feed_label.pack(fill=tk.BOTH, expand=True)

        # Right: status panel
        right = tk.Frame(content, bg=COLOUR_PANEL, width=300)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        right.pack_propagate(False)

        self._build_status_panel(right)

        # ---- Bottom toolbar -----------------------------------------
        toolbar = tk.Frame(self, bg=COLOUR_ACCENT, pady=6)
        toolbar.pack(fill=tk.X, side=tk.BOTTOM)

        btn_style = {"bg": "#0f3460", "fg": COLOUR_TEXT, "activebackground": "#1a4a80",
                     "font": (FONT_FAMILY, 10, "bold"), "relief": tk.FLAT,
                     "padx": 12, "pady": 6, "cursor": "hand2"}

        self._btn_start = tk.Button(
            toolbar, text="▶  Start Camera", command=self._on_start, **btn_style
        )
        self._btn_start.pack(side=tk.LEFT, padx=4)

        self._btn_stop = tk.Button(
            toolbar, text="■  Stop Camera", command=self._on_stop,
            state=tk.DISABLED, **btn_style
        )
        self._btn_stop.pack(side=tk.LEFT, padx=4)

        tk.Button(
            toolbar, text="📂  Load Model", command=self._on_load_model, **btn_style
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            toolbar, text="🗑  Clear Log", command=self._on_clear_log, **btn_style
        ).pack(side=tk.LEFT, padx=4)

        # Status bar
        status_bar = tk.Frame(self, bg=COLOUR_ACCENT, pady=2)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self._status_var = tk.StringVar(value="Ready — load a model and start the camera.")
        tk.Label(
            status_bar, textvariable=self._status_var,
            bg=COLOUR_ACCENT, fg=COLOUR_TEXT,
            font=(FONT_FAMILY, 9), anchor="w",
        ).pack(side=tk.LEFT, padx=8)

        self._fps_var = tk.StringVar(value="FPS: –")
        tk.Label(
            status_bar, textvariable=self._fps_var,
            bg=COLOUR_ACCENT, fg=COLOUR_TEXT,
            font=(FONT_FAMILY, 9, "bold"), anchor="e",
        ).pack(side=tk.RIGHT, padx=8)

    # ------------------------------------------------------------------
    def _build_status_panel(self, parent: tk.Widget) -> None:
        """Populate the right-hand status panel."""
        tk.Label(
            parent, text="DETECTION STATUS",
            bg=COLOUR_PANEL, fg=COLOUR_TEXT,
            font=(FONT_FAMILY, 12, "bold"), pady=8,
        ).pack(fill=tk.X)

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, padx=8, pady=4)

        # Detection label (REAL / FAKE / –)
        det_frame = tk.Frame(parent, bg=COLOUR_PANEL)
        det_frame.pack(fill=tk.X, padx=10, pady=4)
        tk.Label(
            det_frame, text="Decision:", width=12, anchor="w",
            bg=COLOUR_PANEL, fg=COLOUR_TEXT, font=(FONT_FAMILY, 10),
        ).pack(side=tk.LEFT)
        self._detection_lbl = tk.Label(
            det_frame, text="–", width=8,
            bg=COLOUR_PANEL, fg=COLOUR_UNKNOWN,
            font=(FONT_FAMILY, 14, "bold"),
        )
        self._detection_lbl.pack(side=tk.LEFT)

        # Confidence bars
        bars_frame = tk.Frame(parent, bg=COLOUR_PANEL)
        bars_frame.pack(fill=tk.X, padx=10, pady=2)

        self._bar_fake = ConfidenceBar(bars_frame, "P(Fake):", COLOUR_FAKE)
        self._bar_fake.pack(fill=tk.X, pady=2)

        self._bar_real = ConfidenceBar(bars_frame, "P(Real):", COLOUR_REAL)
        self._bar_real.pack(fill=tk.X, pady=2)

        self._bar_temporal = ConfidenceBar(bars_frame, "Consensus:", COLOUR_UNKNOWN)
        self._bar_temporal.pack(fill=tk.X, pady=2)

        self._bar_consistency = ConfidenceBar(bars_frame, "Consistency:", "#29b6f6")
        self._bar_consistency.pack(fill=tk.X, pady=2)

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, padx=8, pady=8)

        # Alert count
        alert_frame = tk.Frame(parent, bg=COLOUR_PANEL)
        alert_frame.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(
            alert_frame, text="Alerts raised:", width=14, anchor="w",
            bg=COLOUR_PANEL, fg=COLOUR_TEXT, font=(FONT_FAMILY, 10),
        ).pack(side=tk.LEFT)
        self._alert_count_var = tk.StringVar(value="0")
        tk.Label(
            alert_frame, textvariable=self._alert_count_var, width=6,
            bg=COLOUR_PANEL, fg=COLOUR_FAKE,
            font=(FONT_FAMILY, 13, "bold"),
        ).pack(side=tk.LEFT)

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, padx=8, pady=8)

        # Alert log
        tk.Label(
            parent, text="ALERT LOG",
            bg=COLOUR_PANEL, fg=COLOUR_TEXT,
            font=(FONT_FAMILY, 11, "bold"),
        ).pack(anchor="w", padx=10)

        log_frame = tk.Frame(parent, bg=COLOUR_PANEL)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._log_text = tk.Text(
            log_frame,
            bg="#0d0d1a", fg=COLOUR_TEXT,
            font=(FONT_FAMILY, 9),
            yscrollcommand=scrollbar.set,
            state=tk.DISABLED,
            wrap=tk.WORD,
            relief=tk.FLAT,
        )
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self._log_text.yview)

        self._log_text.tag_config("alert", foreground=COLOUR_FAKE)
        self._log_text.tag_config("info", foreground=COLOUR_UNKNOWN)

    # ------------------------------------------------------------------
    # Camera capture thread
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Background thread: read frames from camera and push to queue."""
        while self._running and self._cap is not None and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                break
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # drop stale frame to keep GUI responsive

        self._running = False

    # ------------------------------------------------------------------
    # GUI update loop (runs on main thread via after())
    # ------------------------------------------------------------------

    def _update_loop(self) -> None:
        """Poll the frame queue and update the GUI.  Reschedules itself."""
        # Drain any pending cross-thread callables (from background threads).
        try:
            while True:
                fn = self._main_thread_queue.get_nowait()
                fn()
        except queue.Empty:
            pass

        try:
            frame = self._frame_queue.get_nowait()
            result = self._detector.process_frame(frame)
            self._last_result = result
            self._refresh_feed(frame, result)
            self._refresh_status(result)
        except queue.Empty:
            pass

        self.after(16, self._update_loop)  # ~60 Hz polling

    # ------------------------------------------------------------------
    def _refresh_feed(self, bgr_frame: np.ndarray, result: DetectionResult) -> None:
        """Update the camera-feed label with the annotated frame."""
        annotated = _draw_overlay(bgr_frame, result)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        # Resize to fit the label
        lw = self._feed_label.winfo_width() or self._FEED_WIDTH
        lh = self._feed_label.winfo_height() or self._FEED_HEIGHT
        img = Image.fromarray(rgb).resize((lw, lh), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self._feed_label.config(image=imgtk)
        self._feed_label.imgtk = imgtk  # prevent GC

    # ------------------------------------------------------------------
    def _refresh_status(self, result: DetectionResult) -> None:
        """Refresh all status-panel widgets with the latest result."""
        self._fps_var.set(f"FPS: {result.fps:.1f}")

        if not result.faces:
            self._detection_lbl.config(text="–", fg=COLOUR_UNKNOWN)
            self._bar_fake.set(0)
            self._bar_real.set(0)
            return

        # Use the first (primary) face
        face = result.faces[0]
        colour = COLOUR_FAKE if face.label == "FAKE" else COLOUR_REAL
        self._detection_lbl.config(text=face.label, fg=colour)
        self._bar_fake.set(face.p_fake)
        self._bar_real.set(face.p_real)

        if result.temporal is not None:
            self._bar_temporal.set(result.temporal.consensus_p_fake)
            self._bar_consistency.set(result.temporal.consistency_score)

        if result.alert:
            self._alert_count += 1
            self._alert_count_var.set(str(self._alert_count))
            ts = datetime.fromtimestamp(result.timestamp).strftime("%H:%M:%S")
            self._append_log(
                f"[{ts}] DEEPFAKE ALERT  P(fake)={face.p_fake:.3f}  "
                f"consensus={result.temporal.consensus_p_fake:.3f}\n",
                tag="alert",
            )

    # ------------------------------------------------------------------
    def _append_log(self, message: str, tag: str = "info") -> None:
        """Append *message* to the alert log widget."""
        self._log_text.config(state=tk.NORMAL)
        self._log_text.insert(tk.END, message, tag)
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _on_start(self) -> None:
        """Start camera capture."""
        if self._running:
            return
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            messagebox.showerror("Camera Error", "Cannot open camera (index 0).")
            return
        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self._capture_thread.start()
        self._btn_start.config(state=tk.DISABLED)
        self._btn_stop.config(state=tk.NORMAL)
        self._status_var.set("Camera running …")
        self._append_log(f"[{datetime.now().strftime('%H:%M:%S')}] Camera started.\n")

    def _on_stop(self) -> None:
        """Stop camera capture."""
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self._btn_start.config(state=tk.NORMAL)
        self._btn_stop.config(state=tk.DISABLED)
        self._detector.reset_temporal()
        self._status_var.set("Camera stopped.")
        self._append_log(f"[{datetime.now().strftime('%H:%M:%S')}] Camera stopped.\n")
        # Clear feed
        self._feed_label.config(image="")

    def _on_load_model(self) -> None:
        """Open file dialog to load a model."""
        path = filedialog.askopenfilename(
            title="Load Deepfake Detection Model",
            filetypes=[
                ("Keras model", "*.keras *.h5"),
                ("TFLite model", "*.tflite"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._load_model(path)

    def _load_model(self, path: str) -> None:
        """Load a model from *path* in background to avoid UI freeze."""
        self._status_var.set(f"Loading model: {os.path.basename(path)} …")
        self.update_idletasks()

        def _schedule(fn):
            """Queue a callable to run on the main thread."""
            self._main_thread_queue.put(fn)

        def _do_load():
            try:
                import tensorflow as tf
                if path.endswith(".tflite"):
                    from src.model.lightweight_cnn import TFLiteInferenceModel
                    model = TFLiteInferenceModel(path)
                else:
                    model = tf.keras.models.load_model(path)
                self._detector.model = model
                _schedule(lambda: self._status_var.set(
                    f"Model loaded: {os.path.basename(path)}"
                ))
                _schedule(lambda: self._append_log(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"Model loaded: {os.path.basename(path)}\n",
                    tag="info",
                ))
            except Exception as exc:
                _schedule(lambda: self._status_var.set(f"Model load failed: {exc}"))
                _schedule(lambda: messagebox.showerror("Model Load Error", str(exc)))

        threading.Thread(target=_do_load, daemon=True).start()

    def _on_clear_log(self) -> None:
        """Clear the alert log."""
        self._log_text.config(state=tk.NORMAL)
        self._log_text.delete("1.0", tk.END)
        self._log_text.config(state=tk.DISABLED)
        self._alert_count = 0
        self._alert_count_var.set("0")

    def _on_about(self) -> None:
        """Show About dialog."""
        messagebox.showinfo(
            "About",
            "LiveDeepfakeDetection\n\n"
            "Live Detection of Synthetic Faces in Video Conferencing\n"
            "Using Lightweight CNN Models\n\n"
            "Architecture: Depthwise-Separable CNN (6 DSC blocks)\n"
            "Accuracy: 92.3%  |  Model size: ~3.2 MB\n"
            "Target FPS: 28–35 (Raspberry Pi 4)\n\n"
            "Rohan Mishra · Amity University, 2024",
        )

    def _on_close(self) -> None:
        """Clean up and close the application."""
        self._running = False
        if self._cap:
            self._cap.release()
        self.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def launch(model_path: Optional[str] = None) -> None:
    """Create and run the application.

    Args:
        model_path: Optional path to a pre-trained model to load at startup.
    """
    app = DeepfakeDetectionApp(model_path=model_path)
    app.mainloop()


if __name__ == "__main__":
    launch()
