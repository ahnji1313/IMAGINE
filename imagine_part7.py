        """Load a previously saved draft from the draft directory.

        Presents a file selection dialog to the user listing available drafts.
        Upon selection, the snapshot is loaded and the editor state is restored.
        """
        import pickle
        # List all draft files in the directory
        if not os.path.isdir(self.draft_dir):
            messagebox.showinfo("No Drafts", "Draft directory does not exist.")
            return
        draft_files = [f for f in os.listdir(self.draft_dir) if f.endswith(".pkl")]
        if not draft_files:
            messagebox.showinfo("No Drafts", "There are no saved drafts to load.")
            return
        # Ask user to choose a draft; use file dialog for convenience
        filepath = filedialog.askopenfilename(
            title="Load Draft",
            initialdir=self.draft_dir,
            filetypes=[("Draft files", "*.pkl")],
        )
        if not filepath:
            return
        try:
            with open(filepath, "rb") as f:
                snapshot = pickle.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load draft: {e}")
            return
        # Restore state
        self._restore_history_state(snapshot)
        # Reset history to just this snapshot
        self.history = [snapshot]
        self.history_index = 0
        messagebox.showinfo("Draft Loaded", f"Draft loaded from {os.path.basename(filepath)}.")

    def _delete_all_drafts(self) -> None:
        """Delete all saved draft files after user confirmation."""
        if not os.path.isdir(self.draft_dir):
            messagebox.showinfo("No Drafts", "There are no drafts to delete.")
            return
        confirm = messagebox.askyesno(
            "Delete All Drafts",
            "Are you sure you want to delete all saved drafts? This action cannot be undone.",
        )
        if not confirm:
            return
        deleted = 0
        for filename in os.listdir(self.draft_dir):
            path = os.path.join(self.draft_dir, filename)
            try:
                os.remove(path)
                deleted += 1
            except Exception:
                pass
        messagebox.showinfo("Drafts Deleted", f"Deleted {deleted} draft(s).")

    # ------------------------------------------------------------------
    # Macro Recording Methods
    # ------------------------------------------------------------------
    def _start_macro_recording(self):
        """Start recording a new macro."""
        macro_name = simpledialog.askstring("Macro Name", "Enter name for the macro:")
        if not macro_name:
            return
            
        self.macro_recorder.start_recording(macro_name)
        messagebox.showinfo("Macro Recording", f"Started recording macro: {macro_name}")

    def _stop_macro_recording(self):
        """Stop recording the current macro."""
        if not self.macro_recorder.is_recording:
            messagebox.showwarning("Not Recording", "No macro is currently being recorded.")
            return
            
        actions = self.macro_recorder.stop_recording()
        messagebox.showinfo("Macro Recording", f"Stopped recording. Captured {len(actions)} actions.")

    def _play_macro(self):
        """Play back a recorded macro."""
        if not self.macro_recorder.actions:
            messagebox.showwarning("No Macro", "No macro has been recorded yet.")
            return
            
        # Ask user to confirm playback
        confirm = messagebox.askyesno("Play Macro", f"Play back macro with {len(self.macro_recorder.actions)} actions?")
        if not confirm:
            return
            
        self.macro_recorder.play_macro({
            "name": self.macro_recorder.current_macro,
            "actions": self.macro_recorder.actions
        }, self)
        
        self._update_composite()
        messagebox.showinfo("Macro Complete", "Macro playback completed.")

    def _save_macro(self):
        """Save the current macro to a file."""
        if not self.macro_recorder.actions:
            messagebox.showwarning("No Macro", "No macro has been recorded yet.")
            return
            
        filepath = filedialog.asksaveasfilename(
            title="Save Macro",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath:
            return
            
        if self.macro_recorder.save_macro(filepath):
            messagebox.showinfo("Macro Saved", f"Macro saved to {filepath}")
        else:
            messagebox.showerror("Error", "Failed to save macro.")

    def _load_macro(self):
        """Load a macro from a file."""
        filepath = filedialog.askopenfilename(
            title="Load Macro",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath:
            return
            
        macro_data = self.macro_recorder.load_macro(filepath)
        if macro_data:
            self.macro_recorder.actions = macro_data.get("actions", [])
            self.macro_recorder.current_macro = macro_data.get("name", "Loaded Macro")
            messagebox.showinfo("Macro Loaded", f"Loaded macro: {self.macro_recorder.current_macro}")
        else:
            messagebox.showerror("Error", "Failed to load macro.")

    def _show_help(self) -> None:
        """Display a help dialog summarising the available features and how to use them.

        This method creates a scrollable window outlining the main actions in the editor:
        loading/saving images, using layers and groups, selection and mask tools,
        filters and adjustments, auto enhancements, transformations, collage
        creation, history management and other utilities.  The help text is
        designed to give newcomers an overview and remind experienced users of
        the full capabilities of the application.
        """
        help_text = (
            "IMAGINE IMAGE EDITOR - User Guide\n\n"
            "Loading and Saving:\n"
            "  • Use File → Open Image to load a new image as a layer. Multiple images may be loaded as separate layers.\n"
            "  • Use File → Save As to export the current composite image.\n\n"
            "Layers:\n"
            "  • Layers appear in the list at the left. Click to select. Use Edit → Duplicate/Delete/Move Up/Move Down or right‑click for context actions.\n"
            "  • New layers can be blank, filled with a colour, gradient, pattern, noise or shapes. Use the New Layer button to create them.\n"
            "  • Group multiple selected layers via Edit → Group Selected Layers; layers inside a group behave like a folder.\n\n"
            "Text and Fonts:\n"
            "  • Add editable Text Layers by selecting the Text tool and clicking the canvas. Each Text Layer stores its string, font (family name or TTF/OTF path), size, colour, position and effects.\n"
            "  • Fonts may be specified by family name (e.g. Arial) or a full .ttf/.otf file path. Fonts are cached to improve performance.\n"
            "  • Korean (Hangul) characters are supported. If the chosen font does not contain Hangul glyphs, the editor will automatically fall back to a Korean-capable font (for example Malgun Gothic or Noto Sans KR) so characters render correctly.\n"
            "  • Double‑click a Text Layer on the canvas to edit its content, font, size, colour and effects. Effects are re-rendered automatically.\n\n"
            "Text Effects:\n"
            "  • Outline (stroke): choose colour and thickness.\n"
            "  • Shadow: set offset, opacity and colour for a drop shadow.\n"
            "  • Styles: calligraphy and comic-like effects are available as post-render styles.\n\n"
            "Transformations:\n"
            "  • Rotate, flip and resize the canvas with or without scaling layers.\n"
            "  • Rotate/Scale operations preserve layer properties (visibility, opacity, blend mode, mask and z-order) so layers remain visible and retain their opacity after transforms — this fixes an earlier bug where Rotate could make a layer disappear or reset its opacity.\n"
            "  • Perspective Transform, Warp (sine), Face Ratio Adjust and Free Distort allow advanced warping.\n"
            "  • Upscale or downscale a layer in the Transform menu. Text Layers remain editable after scaling (the font size is adjusted and the text re-rendered).\n\n"
            "History and Undo/Redo:\n"
            "  • Most actions are recorded. Use Edit → Undo/Redo or open History → Show History Panel to jump to previous states. Text edits and transform operations are included in the history.\n\n"
            "Miscellaneous:\n"
            "  • Toggle dark/light theme and configure autosave from the Settings menu.\n"
            "  • Use drag‑and‑drop (if supported by your platform) to load image files onto the canvas.\n"
            "  • Export images using common social media sizes from the Export menu.\n"
        )
        win = tk.Toplevel(self)
        win.title("User Guide")
        win.geometry("700x600")
        text = scrolledtext.ScrolledText(win, wrap=tk.WORD)
        text.insert(tk.END, help_text)
        text.configure(state="disabled")
        text.pack(fill=tk.BOTH, expand=True)

    def _on_drop(self, event) -> None:
        """Handle files dropped onto the canvas as new layers.

        This handler is used when drag-and-drop support is available (tkdnd).  The
        event contains the dropped file paths as a Tcl list.  Each file is
        opened and added to the document as a new layer.  If no layers exist
        prior to dropping, the canvas size is updated to match the first
        image.  Errors during loading are silently ignored to avoid
        interrupting the user.  A history snapshot is saved once for the
        whole operation.
        """
        # Extract file paths from event; on Windows, braces may surround paths
        try:
            paths = self.tk.splitlist(event.data)
        except Exception:
            paths = [event.data]
        if not paths:
            return
        # Save history at the start of a drop operation
        self._current_action_desc = "Drag & Drop Image"
        self._save_history()
        first_image_added = len(self.layers) == 0
        for item in paths:
            path = str(item).strip('{}')
            if not os.path.isfile(path):
                continue
            try:
                img = Image.open(path).convert("RGBA")
            except Exception:
                continue
            layer_name = os.path.basename(path)
            layer = Layer(img, layer_name)
            self.layers.append(layer)
            self.current_layer_index = len(self.layers) - 1
            if first_image_added:
                # Set canvas size to match first image
                self.canvas.config(width=img.width, height=img.height)
                first_image_added = False
        # Refresh UI after adding all layers
        self._refresh_layer_list()
        self._update_composite()

    def _paint_line_advanced(self, x0: int, y0: int, x1: int, y1: int):
        """Paint a line using advanced brush dynamics."""
        if self.current_layer_index is None:
            return

        layer = self.layers[self.current_layer_index]
        if not isinstance(layer, Layer):
            return

        # Calculate distance and steps
        stroke_start = time.perf_counter()
        distance = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        if distance == 0:
            return

        steps = max(1, int(distance))
        
        for i in range(steps + 1):
            t = i / steps
            x = int(x0 + (x1 - x0) * t)
            y = int(y0 + (y1 - y0) * t)
            
            # Calculate brush properties using brush engine
            speed = self.brush_engine.calculate_speed(x, y)
            size, opacity, spacing = self.brush_engine.calculate_brush_properties(x, y, pressure=1.0, speed=speed)
            
            # Check if we should paint at this position
            if self.brush_engine.should_paint_dab(x, y):
                # Add jitter
                jitter_x, jitter_y = self.brush_engine.add_jitter(x, y)
                
                # Paint dab
                self._paint_dab_advanced(jitter_x, jitter_y, size, opacity)
                
                # Update brush engine state
                self.brush_engine.last_position = (x, y)
                self.brush_engine.last_time = time.time()
        self.performance_metrics.register_stroke((time.perf_counter() - stroke_start) * 1000.0, False)
        self._update_status_bar()

    def _paint_dab_advanced(self, x: int, y: int, size: float, opacity: float):
        """Paint a single dab with advanced brush dynamics."""
        if self.current_layer_index is None:
            return
            
        layer = self.layers[self.current_layer_index]
        if not isinstance(layer, Layer):
            return
            
        # Create brush dab
        brush_size = int(size)
        if brush_size <= 0:
            return
            
        # Create brush mask based on hardness
        hardness = self.brush_settings.hardness
        brush_mask = Image.new("L", (brush_size * 2 + 1, brush_size * 2 + 1), 0)
        draw = ImageDraw.Draw(brush_mask)
        
        # Draw circle with hardness-based falloff
        center = brush_size
        for r in range(brush_size + 1):
            if hardness >= 1.0:
                # Hard brush - solid circle
                if r <= brush_size:
                    draw.ellipse([center - r, center - r, center + r, center + r], fill=255)
            else:
                # Soft brush - gradient falloff
                alpha = int(255 * (1.0 - (r / brush_size) ** (1.0 / hardness)))
                if alpha > 0:
                    draw.ellipse([center - r, center - r, center + r, center + r], fill=alpha)
        
        # Apply brush texture
        if self.brush_settings.texture == "spray":
            brush_mask = self._apply_spray_texture(brush_mask)
        elif self.brush_settings.texture == "chalk":
            brush_mask = self._apply_chalk_texture(brush_mask)
        elif self.brush_settings.texture == "calligraphy":
            brush_mask = self._apply_calligraphy_texture(brush_mask)
        
        # Apply opacity
        if opacity < 1.0:
            brush_mask = brush_mask.point(lambda p: int(p * opacity))
        
        # Create brush color
        color = self.brush_color
        if color.startswith('#'):
            color = color[1:]
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        
        # Create brush image
        brush_img = Image.new("RGBA", brush_mask.size, (r, g, b, 0))
        brush_img.putalpha(brush_mask)
        
        # Paste brush onto layer
        layer_x = x - brush_size
        layer_y = y - brush_size
        
        # Ensure we're within layer bounds
        if (layer_x < 0 or layer_y < 0 or 
            layer_x + brush_img.width > layer.original.width or 
            layer_y + brush_img.height > layer.original.height):
            return
            
        # Paste brush onto original image
        layer.original.paste(brush_img, (layer_x, layer_y), brush_img)
        layer.apply_adjustments()
        self._update_composite()

    def _apply_spray_texture(self, mask: Image.Image) -> Image.Image:
        """Apply spray texture to brush mask."""
        # Add random noise to create spray effect
        mask_array = np.array(mask)
        noise = np.random.random(mask_array.shape) * 0.3
        mask_array = mask_array * (1 - noise)
        return Image.fromarray(mask_array.astype(np.uint8))

    def _apply_chalk_texture(self, mask: Image.Image) -> Image.Image:
        """Apply chalk texture to brush mask."""
        # Add grain and irregular edges
        mask_array = np.array(mask, dtype=np.float32)
        # Add noise
        noise = np.random.random(mask_array.shape) * 0.2
        mask_array = mask_array * (1 - noise)
        # Add some irregularity
        mask_array = mask_array * (0.8 + 0.4 * np.random.random(mask_array.shape))
        return Image.fromarray(np.clip(mask_array, 0, 255).astype(np.uint8))

    def _apply_calligraphy_texture(self, mask: Image.Image) -> Image.Image:
        """Apply calligraphy texture to brush mask."""
        # Create pressure-sensitive brush with varying width
        mask_array = np.array(mask, dtype=np.float32)
        h, w = mask_array.shape
        center_x, center_y = w // 2, h // 2
        
        # Create pressure variation
        for y in range(h):
            for x in range(w):
                dist_from_center = abs(x - center_x)
                pressure_factor = 1.0 - (dist_from_center / (w // 2))
                pressure_factor = max(0, pressure_factor)
                mask_array[y, x] *= pressure_factor
                
        return Image.fromarray(np.clip(mask_array, 0, 255).astype(np.uint8))

    def _create_vector_object(self, x0: int, y0: int, x1: int, y1: int):
        """Create a vector object based on current tool."""
        if self.current_layer_index is None:
            return
            
        layer = self.layers[self.current_layer_index]
        if not isinstance(layer, VectorLayer):
            messagebox.showwarning("Invalid Layer", "Vector tools can only be used on vector layers.")
            return
            
        # Ask for colors
        fill_color = colorchooser.askcolor(title="Choose fill color", initialcolor=self.brush_color)
        if not fill_color or not fill_color[1]:
            fill_color = ("#000000", "#000000")
            
        stroke_color = colorchooser.askcolor(title="Choose stroke color", initialcolor="#000000")
        if not stroke_color or not stroke_color[1]:
            stroke_color = ("#000000", "#000000")
            
        # Ask for stroke width
        stroke_width = simpledialog.askfloat("Stroke Width", "Enter stroke width:", initialvalue=2.0, minvalue=0.0)
        if stroke_width is None:
            stroke_width = 2.0
            
        # Create vector object based on tool
        if self.current_tool == "vector_rectangle":
            width = abs(x1 - x0)
            height = abs(y1 - y0)
            rect = VectorRectangle(
                x=min(x0, x1), y=min(y0, y1),
                width=width, height=height,
                fill_color=fill_color[1], stroke_color=stroke_color[1],
                stroke_width=stroke_width
            )
            layer.add_object(rect)
            
        elif self.current_tool == "vector_circle":
            radius = math.sqrt((x1 - x0)**2 + (y1 - y0)**2) / 2
            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2
            circle = VectorCircle(
                x=center_x, y=center_y, radius=radius,
                fill_color=fill_color[1], stroke_color=stroke_color[1],
                stroke_width=stroke_width
            )
            layer.add_object(circle)
            
        elif self.current_tool == "vector_ellipse":
            width = abs(x1 - x0)
            height = abs(y1 - y0)
            ellipse = VectorEllipse(
                x=min(x0, x1), y=min(y0, y1),
                width=width, height=height,
                fill_color=fill_color[1], stroke_color=stroke_color[1],
                stroke_width=stroke_width
            )
            layer.add_object(ellipse)
            
        elif self.current_tool == "vector_line":
            line = VectorLine(
                x=x0, y=y0, x2=x1, y2=y1,
                fill_color=fill_color[1], stroke_color=stroke_color[1],
                stroke_width=stroke_width
            )
            layer.add_object(line)
            
        self._update_composite()

    def _add_vector_text(self, x: int, y: int, text: str):
        """Add vector text to the current vector layer."""
        if self.current_layer_index is None:
            return
            
        layer = self.layers[self.current_layer_index]
        if not isinstance(layer, VectorLayer):
            messagebox.showwarning("Invalid Layer", "Vector text can only be used on vector layers.")
            return
            
        # Ask for font properties
        font_size = simpledialog.askfloat("Font Size", "Enter font size:", initialvalue=12.0, minvalue=1.0)
        if font_size is None:
            font_size = 12.0
            
        font_family = simpledialog.askstring("Font Family", "Enter font family:", initialvalue="Arial")
        if not font_family:
            font_family = "Arial"
            
        # Ask for color
        color = colorchooser.askcolor(title="Choose text color", initialcolor=self.brush_color)
        if not color or not color[1]:
            color = ("#000000", "#000000")
            
        # Create vector text object
        vector_text = VectorText(
            x=x, y=y, text=text,
            font_size=font_size, font_family=font_family,
            fill_color=color[1]
        )
        layer.add_object(vector_text)
        self._update_composite()


def main():
    app = ImageEditor()
    app.mainloop()


if __name__ == "__main__":
    main()
