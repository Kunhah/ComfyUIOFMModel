#!/usr/bin/env python3
"""configure.py as a window: one box per setting in .env, a "Get key" button for each key, Save.

    python custom_nodes/ai_influencer_toolkit/tools/configure_gui.py     # SETTINGS_API_KEYS.bat runs this

Keys are hidden (tick "Show keys" to see them). Saving writes .env the same way configure.py does, so lines in
.env that aren't toolkit settings are kept. Restart ComfyUI afterwards so the nodes see the new values.
"""
from __future__ import annotations

import os
import sys
import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_config  # noqa: E402
from configure import write_env  # noqa: E402


class SettingsWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.current = env_config.parse_env_file()
        self.vars: dict[str, tk.StringVar] = {}
        self.secret_entries: list[ttk.Entry] = []

        root.title("AI Influencer - API keys and settings")
        root.geometry("760x640")
        root.minsize(560, 360)

        ttk.Label(root, padding=(12, 10, 12, 4), wraplength=720, justify="left",
                  text="Fill in only the keys for what you use; everything can stay empty. Your keys are saved on "
                       "this computer only (in the .env file) and are never uploaded with the project. "
                       "Restart ComfyUI after saving.").pack(fill="x")

        # Scrollable list of settings, for small screens.
        outer = ttk.Frame(root)
        outer.pack(fill="both", expand=True, padx=12)
        canvas = tk.Canvas(outer, highlightthickness=0)
        bar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        body = ttk.Frame(canvas)
        body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window = canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))
        body.columnconfigure(0, weight=1)

        for row, s in enumerate(env_config.SETTINGS):
            box = ttk.Frame(body, padding=(0, 8))
            box.grid(row=row, column=0, sticky="ew")
            box.columnconfigure(0, weight=1)
            ttk.Label(box, text=s.name, font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, sticky="w")
            ttk.Label(box, text=s.help + (f" Leave empty for the default ({s.default})." if s.default else ""),
                      wraplength=640, justify="left", foreground="#555").grid(row=1, column=0, columnspan=2, sticky="w")
            var = tk.StringVar(value=self.current.get(s.name, ""))
            self.vars[s.name] = var
            entry = ttk.Entry(box, textvariable=var, show="•" if s.secret else "")
            entry.grid(row=2, column=0, sticky="ew", pady=(4, 0))
            if s.secret:
                self.secret_entries.append(entry)
            if s.url:
                ttk.Button(box, text="Get key", command=lambda u=s.url: webbrowser.open(u)).grid(
                    row=2, column=1, padx=(8, 0), pady=(4, 0))

        bottom = ttk.Frame(root, padding=12)
        bottom.pack(fill="x")
        self.show = tk.BooleanVar(value=False)
        ttk.Checkbutton(bottom, text="Show keys", variable=self.show, command=self.toggle_show).pack(side="left")
        ttk.Button(bottom, text="Cancel", command=root.destroy).pack(side="right")
        ttk.Button(bottom, text="Save", command=self.save).pack(side="right", padx=8)

    def toggle_show(self) -> None:
        for entry in self.secret_entries:
            entry.configure(show="" if self.show.get() else "•")

    def save(self) -> None:
        updates = {name: var.get().strip() for name, var in self.vars.items()
                   if var.get().strip() != self.current.get(name, "")}
        if updates or not os.path.exists(env_config.ENV_FILE):
            write_env(updates)
        messagebox.showinfo("Saved", "Settings saved. If ComfyUI is running, close its window and start it again.")
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    SettingsWindow(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
