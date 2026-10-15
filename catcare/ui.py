"""A small desktop setup dialog; stream credentials stay in memory."""
from .config import load_settings, save_settings, validate_settings
from .security import validate_rtsp


def connect_dialog():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    saved = load_settings()
    root = tk.Tk()
    root.title("Cat Care Monitor - connect")
    root.geometry("720x550")
    root.minsize(650, 520)
    panel = ttk.Frame(root, padding=18)
    panel.pack(fill="both", expand=True)
    panel.columnconfigure(1, weight=1)
    ttk.Label(panel, text="One cat · Three stations · Local camera", font=("Segoe UI", 13, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))
    ttk.Label(panel, text="The RTSP URL is used for this session and is never saved.").grid(
        row=1, column=0, columnspan=3, sticky="w", pady=(0, 12))
    url = tk.StringVar()
    ttk.Label(panel, text="RTSP URL").grid(row=2, column=0, sticky="w", padx=(0, 12))
    entry = ttk.Entry(panel, textvariable=url, show="*")
    entry.grid(row=2, column=1, sticky="ew")
    status = tk.StringVar(value="Paste the camera's complete local RTSP URL.")

    def paste(event=None):
        try:
            url.set(root.clipboard_get().strip())
            status.set(f"URL received: {len(url.get())} characters.")
            entry.focus_set()
        except tk.TclError:
            status.set("The clipboard does not contain text.")
        return "break"

    entry.bind("<Control-v>", paste)
    entry.bind("<Control-V>", paste)
    ttk.Button(panel, text="Paste", command=paste).grid(row=2, column=2, padx=(8, 0))
    show = tk.BooleanVar(value=False)
    ttk.Checkbutton(panel, text="Show URL", variable=show,
                    command=lambda: entry.configure(show="" if show.get() else "*")).grid(
                        row=3, column=1, sticky="w", pady=(5, 0))
    ttk.Label(panel, textvariable=status).grid(row=4, column=0, columnspan=3, sticky="w", pady=(5, 12))
    variables = {}
    for row, (key, title, choices) in enumerate([
        ("cat_label", "Cat name", None),
        ("model", "Detector", ["yolo26n.pt", "yolo26s.pt"]),
        ("device", "Processor", ["auto", "cpu", "0"]),
        ("image_size", "Image size", [640, 960, 1280, 1920]),
        ("confidence", "Confidence", None),
    ], start=5):
        variables[key] = tk.StringVar(value=str(saved[key]))
        ttk.Label(panel, text=title).grid(row=row, column=0, sticky="w", pady=5)
        widget = (ttk.Combobox(panel, textvariable=variables[key], values=choices)
                  if choices else ttk.Entry(panel, textvariable=variables[key]))
        widget.grid(row=row, column=1, sticky="ew", pady=5)

    def choose_weights():
        path = filedialog.askopenfilename(title="Select your trained detector", filetypes=[("PyTorch weights", "*.pt")])
        if path:
            variables["model"].set(path)

    ttk.Button(panel, text="Browse", command=choose_weights).grid(row=6, column=2, padx=(8, 0))
    select_zones = tk.BooleanVar(value=not bool(saved["zones"]))
    crop = tk.BooleanVar(value=saved["crop"])
    ttk.Checkbutton(panel, text="Select or change station zones", variable=select_zones).grid(row=10, column=1, sticky="w", pady=4)
    ttk.Checkbutton(panel, text="Focus detection around the selected stations", variable=crop).grid(row=11, column=1, sticky="w", pady=4)
    result = []

    def connect():
        try:
            source = validate_rtsp(url.get())
            values = {key: variable.get() for key, variable in variables.items()}
            values.update(crop=crop.get(), zones=saved["zones"])
            values = validate_settings(values)
            changed_profile = values["cat_label"] != saved["cat_label"]
            if select_zones.get() or changed_profile:
                values["zones"] = {}
            save_settings(values)
        except (ValueError, OSError) as error:
            messagebox.showerror("Check the settings", str(error), parent=root)
            return
        result.append((values, source))
        root.destroy()

    ttk.Button(panel, text="Connect", command=connect).grid(row=12, column=1, sticky="e", pady=(14, 0))
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    entry.focus_set()
    root.mainloop()
    return result[0] if result else None
