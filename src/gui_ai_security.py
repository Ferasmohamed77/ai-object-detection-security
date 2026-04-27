#!/usr/bin/env python3
import os
import time
import json
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageOps

# GPIO sensor
try:
    from gpiozero import Button
    GPIOZERO_AVAILABLE = True
except Exception:
    GPIOZERO_AVAILABLE = False

# CSI camera preview
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except Exception:
    PICAMERA2_AVAILABLE = False

# USB camera preview
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False


# ---------------- Paths / Config ----------------
PROJECT_DIR = os.path.expanduser("~/ai_project")
AI_SCRIPT   = os.path.join(PROJECT_DIR, "ai_security.py")
MODEL_PATH  = os.path.join(PROJECT_DIR, "best.onnx")
CLASS_NAMES = os.path.join(PROJECT_DIR, "custom_classes.txt")

CAPTURE_PATH = os.path.join(PROJECT_DIR, "capture.jpg")
RESULT_PATH  = os.path.join(PROJECT_DIR, "result.jpg")
LOGO_PATH    = os.path.join(PROJECT_DIR, "control_easy.png")

CAMERA_MODE = "usb"   # "usb" or "csi"
USB_DEVICE  = 0
GPIO_PIN    = 17      # set None to disable sensor

PREVIEW_W = 640
PREVIEW_H = 480
PREVIEW_INTERVAL_MS = 33  # ~30 FPS


def parse_ai_json(text: str):
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("AI_RESULT_JSON="):
            try:
                return json.loads(line.split("=", 1)[1].strip())
            except Exception:
                return None
    return None


def load_logo(path, size=(170, 55)):
    if not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("RGBA")
        img = ImageOps.contain(img, size)
        return ImageTk.PhotoImage(img)
    except Exception:
        return None


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Machine Vision & Control Automation")
        self.configure(bg="#e9e9e9")
        self.geometry("1200x650")
        self.minsize(900, 550)

        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Stats + status
        self.var_resolution = tk.StringVar(value="—")
        self.var_process    = tk.StringVar(value="—")
        self.var_object     = tk.StringVar(value="—")
        self.var_conf       = tk.StringVar(value="—")
        self.var_status     = tk.StringVar(value="Ready.")

        # Image buffers (PIL + Tk)
        self._cap_pil = None
        self._res_pil = None
        self._cap_tk = None
        self._res_tk = None
        self.logo_img = None

        # State
        self._busy = False
        self._pending_trigger = False
        self._preview_on = False
        self._resize_after = None

        # CSI preview
        self.picam2 = None
        self._csi_swap_rb = None   # AUTO: True = swap R/B, False = keep, None = not decided yet

        # USB preview
        self._usb_cap = None
        self._usb_thread = None
        self._usb_lock = threading.Lock()
        self._usb_last_bgr = None
        self._usb_stop_evt = threading.Event()

        # Sensor
        self._sensor = None

        self.build_ui()
        self.bind("<Configure>", self._on_resize)
        self.after(150, self.refresh_images)

        if GPIO_PIN is not None:
            self._init_sensor()

        self._update_status_ready()

    # ---------------- UI ----------------
    def build_ui(self):
        self.header = tk.Frame(self, bg="#1f1f1f", pady=10)
        self.header.pack(side="top", fill="x")
        self.header.grid_columnconfigure(0, weight=0)
        self.header.grid_columnconfigure(1, weight=1)

        self.logo_img = load_logo(LOGO_PATH, size=(170, 55))
        if self.logo_img:
            tk.Label(self.header, image=self.logo_img, bg="#1f1f1f").grid(
                row=0, column=0, padx=(10, 12), sticky="w"
            )
        else:
            tk.Label(self.header, text="Control Easy!", fg="white", bg="#1f1f1f",
                     font=("Arial", 12, "bold")).grid(row=0, column=0, padx=(10, 12), sticky="w")

        tk.Label(self.header, text="Machine Vision & Control Automation",
                 fg="white", bg="#1f1f1f", font=("Arial", 18, "bold")).grid(row=0, column=1, sticky="ew")

        self.footer = tk.Frame(self, bg="#e9e9e9", height=90, padx=10, pady=8)
        self.footer.pack(side="bottom", fill="x")
        self.footer.pack_propagate(False)

        tk.Label(self.footer, textvariable=self.var_status,
                 bg="#e9e9e9", fg="black", anchor="w").pack(side="top", fill="x", pady=(0, 8))

        row = tk.Frame(self.footer, bg="#e9e9e9")
        row.pack(side="left", anchor="w")

        self.btn_live = ttk.Button(row, text="Live Preview (Start)", command=self.toggle_preview)
        self.btn_live.pack(side="left", padx=(0, 8))

        self.btn_run = ttk.Button(row, text="Capture + Detect", command=self.capture_detect)
        self.btn_run.pack(side="left", padx=(0, 8))

        self.btn_refresh = ttk.Button(row, text="Refresh Images", command=self.refresh_images)
        self.btn_refresh.pack(side="left", padx=(0, 8))

        ttk.Button(self.footer, text="Quit", command=self.on_quit).pack(side="right")

        self.body = tk.Frame(self, bg="#e9e9e9", padx=10, pady=10)
        self.body.pack(side="top", fill="both", expand=True)
        self.body.grid_rowconfigure(0, weight=0)
        self.body.grid_rowconfigure(1, weight=1)
        self.body.grid_rowconfigure(2, weight=0)
        self.body.grid_columnconfigure(0, weight=1, uniform="cols")
        self.body.grid_columnconfigure(1, weight=1, uniform="cols")

        tk.Label(self.body, text="Captured Frame", bg="#e9e9e9",
                 fg="black", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 10))
        tk.Label(self.body, text="Result (Detection)", bg="#e9e9e9",
                 fg="black", font=("Arial", 12, "bold")).grid(row=0, column=1, sticky="w", padx=(10, 0))

        left_box = tk.Frame(self.body, bg="white", relief="solid", bd=1)
        left_box.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=6)
        left_box.grid_rowconfigure(0, weight=1)
        left_box.grid_columnconfigure(0, weight=1)

        self.cap_label = tk.Label(left_box, bg="white", text="No capture yet")
        self.cap_label.grid(row=0, column=0, sticky="nsew")

        right_box = tk.Frame(self.body, bg="white", relief="solid", bd=1)
        right_box.grid(row=1, column=1, sticky="nsew", padx=(10, 0), pady=6)
        right_box.grid_rowconfigure(0, weight=1)
        right_box.grid_columnconfigure(0, weight=1)

        self.res_label = tk.Label(right_box, bg="white", text="No result yet")
        self.res_label.grid(row=0, column=0, sticky="nsew")

        self.stats = tk.Frame(self.body, bg="#e9e9e9")
        self.stats.grid(row=2, column=1, sticky="ew", padx=(10, 0))
        self.stats.grid_columnconfigure(0, minsize=160)
        self.stats.grid_columnconfigure(1, weight=1)

        self._add_stat(0, "Resolution", self.var_resolution)
        self._add_stat(1, "Processing Time", self.var_process)
        self._add_stat(2, "Object Detected", self.var_object)
        self._add_stat(3, "Confidence", self.var_conf)

    def _add_stat(self, r, label, var):
        tk.Label(self.stats, text=label, anchor="w",
                 bg="#e9e9e9", fg="black", font=("Arial", 10, "bold")).grid(row=r, column=0, sticky="w", pady=3)
        tk.Label(self.stats, textvariable=var, bg="white", fg="black",
                 relief="solid", bd=1, padx=6, pady=3).grid(row=r, column=1, sticky="ew", pady=3)

    # ---------------- Status helper ----------------
    def _update_status_ready(self):
        if GPIO_PIN is not None and GPIOZERO_AVAILABLE and self._sensor is not None:
            self.var_status.set(f"Ready. (Sensor armed on GPIO{GPIO_PIN})")
        else:
            self.var_status.set("Ready.")

    # ---------------- Resize redraw ----------------
    def _on_resize(self, event):
        if self._resize_after:
            self.after_cancel(self._resize_after)
        self._resize_after = self.after(120, self._redraw_to_fit)

    def _redraw_to_fit(self):
        if self._cap_pil is not None:
            self._draw_pil(self._cap_pil, self.cap_label, True)
        if self._res_pil is not None:
            self._draw_pil(self._res_pil, self.res_label, False)

    def _draw_pil(self, pil_img, label, is_capture: bool):
        w = max(50, label.winfo_width())
        h = max(50, label.winfo_height())
        fitted = ImageOps.contain(pil_img, (w - 10, h - 10))
        tkimg = ImageTk.PhotoImage(fitted)
        label.configure(image=tkimg, text="")
        label.image = tkimg
        if is_capture:
            self._cap_tk = tkimg
        else:
            self._res_tk = tkimg

    # ---------------- Refresh disk images ----------------
    def refresh_images(self):
        if not self._preview_on and os.path.exists(CAPTURE_PATH):
            try:
                self._cap_pil = ImageOps.exif_transpose(Image.open(CAPTURE_PATH).convert("RGB"))
                self._draw_pil(self._cap_pil, self.cap_label, True)
            except Exception:
                pass

        if os.path.exists(RESULT_PATH):
            try:
                self._res_pil = ImageOps.exif_transpose(Image.open(RESULT_PATH).convert("RGB"))
                self._draw_pil(self._res_pil, self.res_label, False)
            except Exception:
                pass

    # ---------------- Live Preview ----------------
    def toggle_preview(self):
        if self._busy:
            return
        if self._preview_on:
            self.stop_preview()
        else:
            self.start_preview()

    def start_preview(self):
        if CAMERA_MODE == "csi":
            self._start_preview_csi()
        else:
            self._start_preview_usb()

    # ---------- CSI preview (Picamera2) ----------
    def _start_preview_csi(self):
        if not PICAMERA2_AVAILABLE:
            messagebox.showerror(
                "Preview Error",
                "Picamera2 not available.\nInstall:\n  sudo apt install -y python3-picamera2"
            )
            return

        try:
            self.picam2 = Picamera2()
            # Keep it simple: request RGB888, then AUTO decide if swap is needed.
            config = self.picam2.create_preview_configuration(
                main={"format": "RGB888", "size": (PREVIEW_W, PREVIEW_H)}
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(0.15)
            self._csi_swap_rb = None  # reset decision each time preview starts
        except Exception as e:
            self.picam2 = None
            messagebox.showerror("Preview Error", f"Failed to start camera:\n{e}")
            return

        self._preview_on = True
        self.btn_live.configure(text="Live Preview (Stop)")
        self.var_status.set("Live preview running (CSI)...")
        self._preview_tick_csi()

    def _score_rgb_candidate(self, rgb):
        # Heuristic scoring: in typical scenes, G tends to be strongest, and R often > B.
        r = float(rgb[:, :, 0].mean())
        g = float(rgb[:, :, 1].mean())
        b = float(rgb[:, :, 2].mean())
        score = 0
        if g >= r and g >= b:
            score += 3
        if r >= b:
            score += 2
        # avoid extreme weirdness
        if 10 < r < 245 and 10 < g < 245 and 10 < b < 245:
            score += 1
        return score

    def _preview_tick_csi(self):
        if not self._preview_on or self.picam2 is None:
            return
        try:
            frame = self.picam2.capture_array()

            # Some configs can return 4 channels; drop alpha if present
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]

            # AUTO decide swap on first frame
            if self._csi_swap_rb is None:
                cand1 = frame  # treat as RGB
                cand2 = frame[:, :, ::-1]  # swap
                s1 = self._score_rgb_candidate(cand1)
                s2 = self._score_rgb_candidate(cand2)
                self._csi_swap_rb = (s2 > s1)

            rgb = frame[:, :, ::-1] if self._csi_swap_rb else frame
            pil = Image.fromarray(rgb)

            self._cap_pil = pil
            self._draw_pil(pil, self.cap_label, True)

        except Exception as e:
            self.var_status.set(f"Preview error: {e}")

        self.after(PREVIEW_INTERVAL_MS, self._preview_tick_csi)

    # ---------- USB preview (OpenCV embedded) ----------
    def _start_preview_usb(self):
        if not CV2_AVAILABLE:
            messagebox.showerror(
                "Preview Error",
                "OpenCV (cv2) not installed in venv.\nInstall:\n  pip install opencv-python"
            )
            return

        cap = cv2.VideoCapture(USB_DEVICE)
        if not cap.isOpened():
            messagebox.showerror("USB Preview", f"Cannot open USB camera device {USB_DEVICE}.")
            cap.release()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_H)

        self._usb_cap = cap
        self._usb_stop_evt.clear()
        self._preview_on = True
        self.btn_live.configure(text="Live Preview (Stop)")
        self.var_status.set("Live preview running (USB)...")

        self._usb_thread = threading.Thread(target=self._usb_grab_loop, daemon=True)
        self._usb_thread.start()
        self._preview_tick_usb()

    def _usb_grab_loop(self):
        while not self._usb_stop_evt.is_set() and self._usb_cap is not None:
            ok, frame_bgr = self._usb_cap.read()
            if ok and frame_bgr is not None:
                with self._usb_lock:
                    self._usb_last_bgr = frame_bgr
            time.sleep(0.01)

    def _preview_tick_usb(self):
        if not self._preview_on or self._usb_cap is None:
            return
        frame = None
        with self._usb_lock:
            if self._usb_last_bgr is not None:
                frame = self._usb_last_bgr.copy()

        if frame is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                self._cap_pil = pil
                self._draw_pil(pil, self.cap_label, True)
            except Exception:
                pass

        self.after(PREVIEW_INTERVAL_MS, self._preview_tick_usb)

    def stop_preview(self):
        self._preview_on = False
        self.btn_live.configure(text="Live Preview (Start)")

        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass
            try:
                self.picam2.close()
            except Exception:
                pass
            self.picam2 = None

        self._usb_stop_evt.set()
        if self._usb_cap is not None:
            try:
                self._usb_cap.release()
            except Exception:
                pass
        self._usb_cap = None
        with self._usb_lock:
            self._usb_last_bgr = None

        self._update_status_ready()
        self.refresh_images()

    # ---------------- Capture + Detect ----------------
    def capture_detect(self):
        if self._busy:
            self._pending_trigger = True
            return

        if self._preview_on:
            self.stop_preview()
            time.sleep(0.25)

        self._busy = True
        self._pending_trigger = False
        self.btn_run.configure(state="disabled")
        self.btn_live.configure(state="disabled")
        self.var_status.set("Running capture + detection...")
        threading.Thread(target=self._capture_worker, daemon=True).start()

    def _capture_worker(self):
        start = time.time()

        cmd = [
            "python3", AI_SCRIPT,
            "--model", MODEL_PATH,
            "--class-names", CLASS_NAMES,
            "--camera", CAMERA_MODE,
            "--no-wait",
            "--headless",
            "--json",
            "--capture-output", CAPTURE_PATH,
            "--output", RESULT_PATH
        ]
        if CAMERA_MODE == "usb":
            cmd += ["--device", str(USB_DEVICE)]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        end = time.time()

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        payload = parse_ai_json(stdout) or parse_ai_json(stderr)

        def ui_update():
            self._busy = False
            self.btn_run.configure(state="normal")
            self.btn_live.configure(state="normal")

            if proc.returncode != 0:
                msg = (stderr.strip() or stdout.strip() or "Unknown error")
                self.var_status.set("Detection failed.")
                messagebox.showerror("AI Script Error", msg[-2000:])
                self._update_status_ready()
                return

            if payload and payload.get("ok"):
                res = payload.get("resolution")
                if isinstance(res, list) and len(res) == 2:
                    self.var_resolution.set(f"{res[0]} x {res[1]}")
                self.var_object.set(str(payload.get("class") or "—"))
                conf = payload.get("confidence")
                self.var_conf.set(f"{conf*100:.1f}%" if isinstance(conf, (int, float)) else "—")
                pt = payload.get("processing_time")
                self.var_process.set(f"{pt:.3f}s" if isinstance(pt, (int, float)) else f"{(end-start):.3f}s")
            else:
                self.var_process.set(f"{(end-start):.3f}s")

            self.refresh_images()
            self._update_status_ready()

            if self._pending_trigger:
                self._pending_trigger = False
                self.after(50, self.capture_detect)

        self.after(0, ui_update)

    # ---------------- Sensor trigger ----------------
    def _init_sensor(self):
        if not GPIOZERO_AVAILABLE:
            self.var_status.set("Ready. (gpiozero missing; sensor disabled)")
            return
        try:
            self._sensor = Button(GPIO_PIN, pull_up=True, bounce_time=0.05)

            def on_press():
                self.after(0, self.capture_detect)

            self._sensor.when_pressed = on_press
        except Exception as e:
            self._sensor = None
            self.var_status.set(f"Ready. (Sensor init failed: {e})")

    # ---------------- Quit ----------------
    def on_quit(self):
        try:
            self.stop_preview()
        except Exception:
            pass
        try:
            if self._sensor:
                self._sensor.close()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    App().mainloop()