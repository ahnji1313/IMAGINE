                listbox.delete(0, tk.END)
                for c in items:
                    star = "★ " if c["id"] in pinned else "  "
                    listbox.insert(tk.END, f"{star}{c['label']}")
                if listbox.size() > 0:
                    listbox.selection_clear(0, tk.END)
                    listbox.selection_set(0)

            def run_selected(event=None):
                try:
                    sel = listbox.curselection()
                    if not sel:
                        return
                    items = current_items()
                    cmd = items[sel[0]]
                    win.destroy()
                    cmd["run"]()
                except Exception:
                    pass

            def toggle_pin(event=None):
                try:
                    sel = listbox.curselection()
                    if not sel:
                        return
                    items = current_items()
                    cmd = items[sel[0]]
                    cid = cmd["id"]
                    if cid in pinned:
                        pinned.remove(cid)
                        self.show_toast(f"Unpinned: {cmd['label']}")
                    else:
                        pinned.insert(0, cid)
                        self.show_toast(f"Pinned: {cmd['label']}")
                    self._save_pinned_commands(pinned)
                    refresh_list()
                except Exception:
                    pass

            def move_selection(delta: int):
                try:
                    size = listbox.size()
                    if size == 0:
                        return
                    cur = listbox.curselection()
                    idx = cur[0] if cur else 0
                    idx = max(0, min(size - 1, idx + delta))
                    listbox.selection_clear(0, tk.END)
                    listbox.selection_set(idx)
                    listbox.see(idx)
                except Exception:
                    pass

            entry.bind("<KeyRelease>", lambda _e: refresh_list())
            entry.bind("<Return>", run_selected)
            entry.bind("<Escape>", lambda _e: win.destroy())
            entry.bind("<Control-p>", toggle_pin)
            listbox.bind("<Return>", run_selected)
            listbox.bind("<Double-Button-1>", run_selected)
            listbox.bind("<Up>", lambda _e: move_selection(-1))
            listbox.bind("<Down>", lambda _e: move_selection(1))
            win.bind("<Up>", lambda _e: move_selection(-1))
            win.bind("<Down>", lambda _e: move_selection(1))

            entry.focus_set()
            refresh_list()
        except Exception:
            pass

    def _set_autosave_interval(self, seconds: Optional[int] = None) -> None:
        """Configure autosave interval either interactively or via a preset."""

        interval_seconds: Optional[int]
        if seconds is None:
            minutes = simpledialog.askinteger(
                "Autosave Interval",
                "Enter autosave interval in minutes (0 to disable):",
                initialvalue=5,
                minvalue=0,
            )
            if minutes is None:
                return
            interval_seconds = minutes * 60
        else:
            interval_seconds = max(0, seconds)

        if not interval_seconds:
            self.autosave_interval_ms = None
            if self.autosave_after_id:
                try:
                    self.after_cancel(self.autosave_after_id)
                except Exception:
                    pass
                self.autosave_after_id = None
            self._set_status("Autosave disabled")
            return

        self.autosave_interval_ms = int(interval_seconds * 1000)
        self._schedule_autosave()
        minutes = interval_seconds / 60 if interval_seconds >= 60 else None
        if minutes and minutes.is_integer():
            self._set_status(f"Autosave every {int(minutes)} minute(s)")
        else:
            self._set_status(f"Autosave every {interval_seconds} seconds")

    def _schedule_autosave(self) -> None:
        """Schedule the next autosave call."""

        if self.autosave_after_id:
            try:
                self.after_cancel(self.autosave_after_id)
            except Exception:
                pass
            self.autosave_after_id = None
        if self.autosave_interval_ms:
            self.autosave_after_id = self.after(self.autosave_interval_ms, self._perform_autosave)

    def _perform_autosave(self) -> None:
        """Persist the current composite image and history snapshot in the background."""

        with self._autosave_lock:
            try:
                self._save_history()
            except Exception:
                pass
            composite = self._create_composite_image()
            if composite is None:
                self._schedule_autosave()
                return
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            png_path = self._autosave_dir / f"autosave-{timestamp}.png"
            try:
                composite.save(png_path)
                # Maintain a small rolling window of autosave files
                existing = sorted(self._autosave_dir.glob("autosave-*.png"))
                if len(existing) > 10:
                    for old in existing[:-10]:
                        try:
                            old.unlink()
                        except Exception:
                            pass
                # Persist a lightweight history snapshot alongside the image for recovery
                if self.history_index >= 0:
                    snapshot = self.history[self.history_index]
                    import pickle
                    with open(self._autosave_dir / "autosave.pkl", "wb") as handle:
                        pickle.dump(snapshot, handle)
                self._log_action(f"Autosaved to {png_path}")
                self._update_status_bar(message=f"Autosaved to {png_path.name}")
            except Exception as exc:
                self._log_action(f"Autosave failed: {exc}")
                self._update_status_bar(message="Autosave failed")
        self._schedule_autosave()

    def _on_layer_select(self, event):
        """Callback when a different layer is selected in the listbox."""
        selection = self.layer_listbox.curselection()
        if selection:
            self.current_layer_index = selection[0]
            layer = self.layers[self.current_layer_index]
            # Update sliders to reflect selected layer's properties
            self.alpha_slider.set(layer.alpha)
            self.brightness_slider.set(layer.brightness)
            if hasattr(self, 'contrast_slider'):
                self.contrast_slider.set(layer.contrast)
            if hasattr(self, 'color_slider'):
                self.color_slider.set(layer.color)
            # Update gamma slider if present
            if hasattr(self, 'gamma_slider'):
                try:
                    self.gamma_slider.set(layer.gamma)
                except Exception:
                    pass

    def _rename_layer(self, event) -> None:
        """Prompt the user to rename the double‑clicked layer."""
        index = self.layer_listbox.nearest(event.y)
        if index < 0 or index >= len(self.layers):
            return
        self.current_layer_index = index
        current_name = self.layers[index].name
        new_name = simpledialog.askstring("Rename Layer", "Enter new layer name:", initialvalue=current_name)
        if new_name and new_name.strip():
            self._save_history()
            self.layers[index].name = new_name.strip()
            self._refresh_layer_list()
    # ------------------------------------------------------------------
    # Composite management
    # ------------------------------------------------------------------
    def _create_composite_image(self, include_hidden: bool = True, preview: bool = False) -> Image.Image:
        """Return a new PIL image that composites all layers.

        :param include_hidden: if False, skip layers whose ``visible``
            attribute is False.
        :returns: an ``Image`` object representing the composite.
        """
        if not self.layers:
            return None
        
        # Find the base size from the first raster layer
        base_size = None
        for layer in self.layers:
            if isinstance(layer, Layer):
                base_size = layer.image.size
                break
            elif isinstance(layer, VectorLayer):
                base_size = (layer.width, layer.height)
                break
        
        if base_size is None:
            return None

        # Fast path: if every contributing layer is a raster layer using normal blending
        # we can composite purely with NumPy for a substantial speed boost.
        fast_layers: List[Tuple[Image.Image, Tuple[int, int], float]] = []
        fast_path = True
        for layer in self.layers:
            if not (include_hidden or layer.visible):
                continue
            if isinstance(layer, Layer) and layer.blend_mode == 'normal':
                if preview and getattr(layer, 'preview_image', None) is not None:
                    layer_image = layer.preview_image
                else:
                    layer_image = layer.image
                fast_layers.append((layer_image, layer.offset, 1.0))
            else:
                fast_path = False
                break
        if fast_path and fast_layers:
            try:
                return fast_normal_composite(fast_layers, base_size)
            except Exception:
                # Fallback to standard path on any failure
                pass

        composite = Image.new("RGBA", base_size, (0, 0, 0, 0))

        # Build composite by blending layers according to their blend_mode
        for layer in self.layers:
            if include_hidden or layer.visible:
                # Get layer image based on type
                if isinstance(layer, Layer):
                    # Regular raster layer
                    if preview and getattr(layer, 'preview_image', None) is not None:
                        layer_image = layer.preview_image
                    elif preview and getattr(layer, 'preview_image', None) is None:
                        # generate a reasonable preview size based on composite target
                        try:
                            layer.generate_preview(max_size=(800, 600))
                            layer_image = layer.preview_image if layer.preview_image is not None else layer.image
                        except Exception:
                            layer_image = layer.image
                    else:
                        layer_image = layer.image
                    ox, oy = layer.offset
                elif isinstance(layer, VectorLayer):
                    # Vector layer - rasterize it
                    layer_image = layer.rasterize(base_size)
                    ox, oy = layer.offset
                elif isinstance(layer, AdjustmentLayer):
                    # Adjustment layer - apply to current composite
                    composite = layer.apply_to_image(composite)
                    continue
                else:
                    continue
                
                # Create an overlay image the size of the composite with layer placed at its offset
                overlay = Image.new("RGBA", base_size, (0, 0, 0, 0))
                overlay.paste(layer_image, (int(ox), int(oy)), layer_image)
                
                if layer.blend_mode == 'normal':
                    composite = Image.alpha_composite(composite, overlay)
                elif layer.blend_mode in ('multiply', 'screen', 'overlay', 'softlight'):
                    # Custom blend modes with alpha compositing.  Split images into channels.
                    b_r, b_g, b_b, b_a = composite.split()
                    o_r, o_g, o_b, o_a = overlay.split()
                    # Compute inverted overlay alpha once
                    inv_o_a = ImageChops.invert(o_a)
                    # Compute new alpha: o_a + b_a * (1 - o_a/255)
                    new_a = ImageChops.add(o_a, ImageChops.multiply(b_a, inv_o_a))
                    # Determine blended colour channels based on blend mode
                    if layer.blend_mode == 'multiply':
                        # Multiply colour channels
                        blend_r = ImageChops.multiply(b_r, o_r)
                        blend_g = ImageChops.multiply(b_g, o_g)
                        blend_b = ImageChops.multiply(b_b, o_b)
                    elif layer.blend_mode == 'screen':
                        # Screen: 255 - (255 - base)*(255 - overlay)/255
                        blend_r = ImageChops.screen(b_r, o_r)
                        blend_g = ImageChops.screen(b_g, o_g)
                        blend_b = ImageChops.screen(b_b, o_b)
                    elif layer.blend_mode == 'overlay':
                        # Overlay: if base < 128 then multiply, else screen the inverted values
                        try:
                            blend_r = ImageChops.overlay(b_r, o_r)
                            blend_g = ImageChops.overlay(b_g, o_g)
                            blend_b = ImageChops.overlay(b_b, o_b)
                        except Exception:
                            # Fallback: approximate overlay by combining multiply and screen
                            blend_r = ImageChops.add(ImageChops.multiply(b_r, o_r), ImageChops.screen(b_r, o_r))
                            blend_g = ImageChops.add(ImageChops.multiply(b_g, o_g), ImageChops.screen(b_g, o_g))
                            blend_b = ImageChops.add(ImageChops.multiply(b_b, o_b), ImageChops.screen(b_b, o_b))
                    elif layer.blend_mode == 'softlight':
                        # Soft light blending: uses PIL's built‑in soft_light if available
                        try:
                            blend_r = ImageChops.soft_light(b_r, o_r)
                            blend_g = ImageChops.soft_light(b_g, o_g)
                            blend_b = ImageChops.soft_light(b_b, o_b)
                        except Exception:
                            # Fallback: use overlay
                            blend_r = ImageChops.overlay(b_r, o_r)
                            blend_g = ImageChops.overlay(b_g, o_g)
                            blend_b = ImageChops.overlay(b_b, o_b)
                    else:
                        # Should not reach here, fallback to normal
                        composite = Image.alpha_composite(composite, overlay)
                        continue
                    # Combine base and blended colours according to alpha
                    out_r = ImageChops.add(ImageChops.multiply(b_r, inv_o_a), ImageChops.multiply(blend_r, o_a))
                    out_g = ImageChops.add(ImageChops.multiply(b_g, inv_o_a), ImageChops.multiply(blend_g, o_a))
                    out_b = ImageChops.add(ImageChops.multiply(b_b, inv_o_a), ImageChops.multiply(blend_b, o_a))
                    composite = Image.merge("RGBA", (out_r, out_g, out_b, new_a))
                else:
                    # Fallback to normal blending
                    composite = Image.alpha_composite(composite, overlay)
        return composite

    def _update_composite(self):
        """Redraw the main canvas with the current composite image."""
        if not self.layers:
            self.canvas.delete("all")
            return
        with ScopedTimer(lambda ms: self.performance_metrics.register_compose(ms)):
            composite = self._create_composite_image()
        if composite is None:
            self.canvas.delete("all")
            self._update_status_bar(message="No layers")
            return
        self._last_composite_image = composite.copy()
        # Apply zoom scaling if necessary
        if getattr(self, 'zoom', 1.0) != 1.0:
            # Compute new size based on zoom factor
            w, h = composite.size
            new_w = max(1, int(w * self.zoom))
            new_h = max(1, int(h * self.zoom))
            # Use NEAREST resampling for performance; other filters possible
            composite = composite.resize((new_w, new_h), Image.NEAREST)
            # Resize canvas to match new dimensions
            self.canvas.config(width=new_w, height=new_h)
        else:
            # Ensure canvas matches image size when not zoomed
            self.canvas.config(width=composite.width, height=composite.height)
        # Convert to PhotoImage for Tkinter display
        self.tk_composite = ImageTk.PhotoImage(composite)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_composite)
        self._update_status_bar()

    def _show_layer_context_menu(self, event):
        """Show context menu for layers on right click."""
        # Select the clicked item
        try:
            index = self.layer_listbox.nearest(event.y)
            self.layer_listbox.select_clear(0, tk.END)
            self.layer_listbox.select_set(index)
            self.current_layer_index = index
        except Exception:
            return
        self.layer_menu.tk_popup(event.x_root, event.y_root)

    def _update_status(self, event):
        """Update the status bar with current tool and mouse coordinates."""
        self._update_status_bar(mouse=(event.x, event.y))

    def _update_status_bar(self, message: Optional[str] = None, mouse: Optional[Tuple[int, int]] = None) -> None:
        """Render a professional status readout with performance metrics."""

        parts: List[str] = []
        if mouse is not None:
            parts.append(f"Tool: {self.current_tool or 'None'}  Pos: ({mouse[0]}, {mouse[1]})")
        elif message:
            parts.append(message)
        else:
            parts.append(f"Tool: {self.current_tool or 'None'}")

        metrics = self.performance_metrics.summary()
        if metrics:
            parts.append(metrics)

        parts.append(f"Zoom {self.zoom:.2f}×")
        self.status_var.set("    ".join(parts))

    def _set_status(self, message: str) -> None:
        """Display *message* in the status bar and log it."""

        self._update_status_bar(message=message)
        self._log_action(message)

    def _log_action(self, message: str) -> None:
        """Record a timestamped entry in the session log."""

        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"{timestamp}  {message}"
        self.session_log.append(entry)

    def _highlight_tool(self) -> None:
        """Update the appearance of tool buttons to indicate the active tool."""
        for name, btn in getattr(self, 'tool_buttons', {}).items():
            if self.current_tool == name:
                # Highlight active tool with a slightly darker shade and sunken relief
                # Slightly darker highlight when active
                btn.config(relief=tk.SUNKEN, bg="#d0d0d0")
            else:
                # Reset to normal appearance
                btn.config(relief=tk.RAISED, bg="#e6e6e6")

    def _set_blend_mode(self, mode: str) -> None:
        """Set the blending mode of the currently selected layer and update the composite."""
        if self.current_layer_index is None:
            return
        if mode not in ('normal', 'multiply', 'screen', 'overlay', 'softlight'):
            messagebox.showinfo("Unsupported blend mode", f"Blend mode '{mode}' is not supported.")
            return
        self._save_history()
        self.layers[self.current_layer_index].blend_mode = mode
        self._update_composite()

    def _nudge_selection_event(self, event, dx: int, dy: int) -> None:
        """Nudge the active selection or floating layer by dx,dy pixels.

        Arrow keys call this with small deltas; Shift+arrow calls with larger deltas.
        If the select tool is active and there's an immediate floating selection
        (a layer created by selection transform), this will move that layer's
        offset. Otherwise, if there's a current layer, adjust its offset.
        """
        # Only act if select tool active or there's an active selection
        if getattr(self, 'current_tool', None) != 'select' and not hasattr(self, '_last_selection_mask'):
            return
        # Determine target layer: use current layer or last added floating selection
        tgt_idx = self.current_layer_index
        # Prefer last layer if it's named like a selection
        if self.layers:
            last = self.layers[-1]
            if isinstance(last, Layer) and last.name.startswith('Selection'):
                tgt_idx = len(self.layers) - 1
        if tgt_idx is None:
            return
        # Save history and adjust offset
        try:
            self._current_action_desc = f"Nudge selection {dx},{dy}"
            self._save_history()
            layer = self.layers[tgt_idx]
            ox, oy = getattr(layer, 'offset', (0, 0))
            layer.offset = (ox + dx, oy + dy)
            self._update_composite()
        except Exception:
            pass

    def _on_zoom(self, event) -> None:
        """Handle mouse wheel events with control held to zoom in/out.

        Adjust the zoom factor based on the scroll direction and
        redraw the composite image at the new scale.  The zoom range
        is clamped between 0.1× and 10× to prevent extreme values.
        """
        # event.delta is positive for scroll up (Windows).  On Linux
        # this may be multiples of 120.  Multiply by sign to determine
        # direction.
        delta = event.delta
        if delta > 0:
            self.zoom *= 1.1
        else:
            self.zoom /= 1.1
        # Clamp zoom to a reasonable range
        if self.zoom < 0.1:
            self.zoom = 0.1
        if self.zoom > 10.0:
            self.zoom = 10.0
        # Redraw composite at new zoom factor
        self._update_composite()
    # ------------------------------------------------------------------
    # History operations
    # ------------------------------------------------------------------
    def _save_history(self):
        """Save a snapshot of the current editor state for undo/redo."""
        # Discard any redo states beyond the current index
        if self.history_index < len(self.history) - 1:
            self.history = self.history[: self.history_index + 1]
        # Create snapshot of layers
        snapshot_layers = []
        for layer in self.layers:
            # Copy original image to preserve pixel data
            layer_data = {
                "image": layer.original.copy(),
                "mask": layer.mask.copy(),
                "offset": layer.offset,
                "name": layer.name,
                "visible": layer.visible,
                "alpha": layer.alpha,
                "brightness": layer.brightness,
                "contrast": layer.contrast,
                "color": layer.color,
                "gamma": layer.gamma,
                "blend_mode": layer.blend_mode,
                "red": layer.red,
                "green": layer.green,
                "blue": layer.blue,
                "type": "raster",
            }
            # Include TextLayer specific fields
            if isinstance(layer, TextLayer):
                layer_data['type'] = 'text'
                layer_data['text'] = layer.text
                layer_data['font_spec'] = layer.font_spec
                layer_data['font_size'] = layer.font_size
                layer_data['text_color'] = layer.color
                layer_data['effects'] = layer.effects
                layer_data['position'] = layer.position
            snapshot_layers.append(layer_data)
        snapshot = {
            "layers": snapshot_layers,
            "current_index": self.current_layer_index,
        }
        self.history.append(snapshot)
        # Save a corresponding history description
        desc = getattr(self, '_current_action_desc', None)
        if not desc:
            desc = f"State {len(self.history)}"
        self.history_descriptions.append(desc)
        # Reset current action description
        if hasattr(self, '_current_action_desc'):
            delattr(self, '_current_action_desc')
        # Enforce history size limit
        if len(self.history) > self.max_history:
            # remove oldest state and adjust index accordingly
            self.history.pop(0)
            # also remove oldest description
            if self.history_descriptions:
                self.history_descriptions.pop(0)
            if self.history_index > 0:
                self.history_index -= 1
        # Move history index to the end
        self.history_index = len(self.history) - 1

    def _restore_history_state(self, snapshot: dict) -> None:
        """Restore layers and selection from a snapshot."""
        self.layers = []
        for item in snapshot["layers"]:
            if item.get('type') == 'text':
                # Recreate TextLayer
                canvas_size = item["image"].size
                tl = TextLayer(canvas_size, text=item.get('text', ''), name=item.get('name', 'Text'), font_spec=item.get('font_spec'), font_size=item.get('font_size', 32), color=item.get('text_color', '#000000'), position=item.get('position', (0,0)), effects=item.get('effects', {}))
                # Restore raster image (in case it's been rasterized previously)
                tl.original = item["image"].copy()
                layer = tl
            else:
                layer = Layer(item["image"], item["name"])
            # Restore mask and offset
            if "mask" in item:
                layer.mask = item["mask"].copy()
            if "offset" in item:
                layer.offset = item["offset"]
            layer.visible = item["visible"]
            layer.alpha = item["alpha"]
            layer.brightness = item["brightness"]
            layer.contrast = item["contrast"]
            layer.color = item["color"]
            # Restore gamma and blend_mode if present
            if "gamma" in item:
                layer.gamma = item["gamma"]
            if "blend_mode" in item:
                layer.blend_mode = item["blend_mode"]
            # Restore selective colour adjustments
            if "red" in item:
                layer.red = item["red"]
            if "green" in item:
                layer.green = item["green"]
            if "blue" in item:
                layer.blue = item["blue"]
            layer.apply_adjustments()
            self.layers.append(layer)
        self.current_layer_index = snapshot.get("current_index")
        self._refresh_layer_list()
        self._update_composite()

    def _undo(self):
        """Revert to the previous state in history."""
        if self.history_index <= 0:
            return
        self.history_index -= 1
        snapshot = self.history[self.history_index]
        self._restore_history_state(snapshot)

    def _redo(self):
        """Advance to the next state in history if available."""
        if self.history_index >= len(self.history) - 1:
            return
        self.history_index += 1
        snapshot = self.history[self.history_index]
        self._restore_history_state(snapshot)

    # ------------------------------------------------------------------
    # History panel
    # ------------------------------------------------------------------
    def _show_history_panel(self) -> None:
        """Display a panel with a list of history states for quick navigation."""
        # Create a new window
        panel = tk.Toplevel(self)
        panel.title("History Panel")
        panel.geometry("300x400")
        # Listbox to display history descriptions
        lb = tk.Listbox(panel, selectmode=tk.SINGLE)
        lb.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # Populate with descriptions
        for idx, desc in enumerate(self.history_descriptions):
            lb.insert(tk.END, f"{idx}: {desc}")
        # Highlight current history index
        if self.history_index >= 0:
            lb.selection_set(self.history_index)
            lb.see(self.history_index)
        # Button to apply selected history state
        def apply_selected():
            sel = lb.curselection()
            if sel:
                idx = sel[0]
                if 0 <= idx < len(self.history):
                    self.history_index = idx
                    self._restore_history_state(self.history[idx])
                    panel.destroy()
        apply_btn = tk.Button(panel, text="Apply", command=apply_selected)
        apply_btn.pack(side=tk.BOTTOM, fill=tk.X)

    # ------------------------------------------------------------------
    # Layer operations
    # ------------------------------------------------------------------
    def _duplicate_layer(self):
        """Create a duplicate of the currently selected layer."""
        if self.current_layer_index is None:
            return
        self._save_history()
        src = self.layers[self.current_layer_index]
        new_img = src.original.copy()
        name = src.name + " copy"
        dup_layer = Layer(new_img, name)
        dup_layer.visible = src.visible
        dup_layer.alpha = src.alpha
        dup_layer.brightness = src.brightness
        dup_layer.contrast = src.contrast
        dup_layer.color = src.color
        # Copy mask and offset
        dup_layer.mask = src.mask.copy()
        dup_layer.offset = src.offset
        dup_layer.apply_adjustments()
        # Insert above the original layer
        self.layers.insert(self.current_layer_index + 1, dup_layer)
        self.current_layer_index += 1
        self._refresh_layer_list()
        self._update_composite()

    def _move_layer(self, offset: int) -> None:
        """Move the selected layer up or down by one position.

        :param offset: -1 to move up, +1 to move down.
        """
        if self.current_layer_index is None:
            return
        new_index = self.current_layer_index + offset
        if new_index < 0 or new_index >= len(self.layers):
            return
        self._save_history()
        # Swap layers
        self.layers[self.current_layer_index], self.layers[new_index] = self.layers[new_index], self.layers[self.current_layer_index]
        self.current_layer_index = new_index
        self._refresh_layer_list()
        self._update_composite()

    # ------------------------------------------------------------------
    # Transform operations
    # ------------------------------------------------------------------
    def _rotate_layer(self, degrees: int) -> None:
        """Rotate the currently selected layer clockwise by the given degrees."""
        if self.current_layer_index is None:
            return
        self._save_history()
        layer = self.layers[self.current_layer_index]
        # rotate original image around centre, expand to fit
        # Pillow rotates counter-clockwise by default; pass negative for clockwise
        try:
            rotated = layer.original.rotate(-degrees, expand=True)
        except Exception:
            # Fallback: if rotate fails, do nothing
            return
        # Resize rotated image to canvas size by pasting onto transparent canvas
        # Use the first layer as the canonical canvas size
        base_size = self.layers[0].image.size
        new_img = Image.new("RGBA", base_size, (0, 0, 0, 0))
        # centre rotated image on base
        x = (base_size[0] - rotated.width) // 2
        y = (base_size[1] - rotated.height) // 2
        new_img.paste(rotated, (x, y), rotated)

        # Also rotate and re-align the mask so sizes match. Preserve mask content.
        try:
            old_mask = layer.mask if getattr(layer, 'mask', None) is not None else Image.new('L', layer.original.size, 255)
            rotated_mask = old_mask.rotate(-degrees, expand=True)
            new_mask = Image.new('L', base_size, 0)
            new_mask.paste(rotated_mask, (x, y), rotated_mask)
        except Exception:
            # If mask operations fail, create an all-opaque mask matching base
            new_mask = Image.new('L', base_size, 255)

        # Assign rotated results back to the layer but preserve other properties
        layer.original = new_img
        layer.mask = new_mask
        # Keep existing offset but ensure it's numeric and doesn't break composites
        if not hasattr(layer, 'offset') or layer.offset is None:
            layer.offset = (0, 0)

        # Recalculate working image with adjustments (alpha, brightness, etc.)
        try:
            layer.apply_adjustments()
        except Exception:
            # If adjustments fail (mismatched sizes etc.) ensure image has correct alpha channel
            try:
                if layer.image.size != layer.original.size:
                    layer.image = layer.original.copy()
                # Ensure mask size matches
                if layer.mask.size != layer.image.size:
                    layer.mask = Image.new('L', layer.image.size, 255)
                layer.apply_adjustments()
            except Exception:
                pass

        self._update_composite()

    def _flip_layer(self, axis: str) -> None:
        """Flip the currently selected layer horizontally or vertically."""
        if self.current_layer_index is None:
            return
        self._save_history()
        layer = self.layers[self.current_layer_index]
        if axis == "horizontal":
            flipped = layer.original.transpose(Image.FLIP_LEFT_RIGHT)
            try:
                layer.mask = layer.mask.transpose(Image.FLIP_LEFT_RIGHT)
            except Exception:
                pass
        elif axis == "vertical":
            flipped = layer.original.transpose(Image.FLIP_TOP_BOTTOM)
            try:
                layer.mask = layer.mask.transpose(Image.FLIP_TOP_BOTTOM)
            except Exception:
                pass
        else:
            return
        # Paste onto same sized canvas to maintain alignment
        base_size = layer.original.size
        new_img = Image.new("RGBA", base_size, (0, 0, 0, 0))
        new_img.paste(flipped, (0, 0), flipped)
        layer.original = new_img
        layer.apply_adjustments()
        self._update_composite()

    def _resize_canvas(self) -> None:
        """Prompt user for new dimensions and resize all layers accordingly."""
        if not self.layers:
            return
        # Ask for new width and height
        current_w = self.layers[0].image.width
        current_h = self.layers[0].image.height
        # Use simpledialog to ask for width and height
        new_w = simpledialog.askinteger("Resize", "Enter new width", initialvalue=current_w, minvalue=1)
        if new_w is None:
            return
        new_h = simpledialog.askinteger("Resize", "Enter new height", initialvalue=current_h, minvalue=1)
        if new_h is None:
            return
        # Save history before resizing
        self._save_history()
        scale_x = new_w / current_w
        scale_y = new_h / current_h
        for layer in self.layers:
            # Resize original image
            w, h = layer.original.size
            resized = layer.original.resize((int(w * scale_x), int(h * scale_y)), resample=Image.BICUBIC)
            layer.original = resized
            # Resize mask accordingly
            layer.mask = layer.mask.resize((int(w * scale_x), int(h * scale_y)), resample=Image.BICUBIC)
            # Scale offset
            ox, oy = layer.offset
            layer.offset = (ox * scale_x, oy * scale_y)
            # Update adjustments
            layer.apply_adjustments()
        # Update canvas size
        self.canvas.config(width=new_w, height=new_h)
        self._refresh_layer_list()
        self._update_composite()

    def _resize_canvas_no_scale(self) -> None:
        """Change the canvas size without scaling the existing layers.

        This operation behaves similarly to the canvas size command in
        professional editors: the dimensions of the drawing surface
        change, but each layer's pixel data remains unscaled.  When
        enlarging, empty space is added around the existing content.  If
        the new size is smaller than the current canvas, the image is
        cropped at the top‑left corner.  All layer offsets are preserved.
        """
        if not self.layers:
            return
        # Current canvas dimensions are taken from the first layer
        current_w = self.layers[0].image.width
        current_h = self.layers[0].image.height
        # Ask for new width and height
        new_w = simpledialog.askinteger(
            "Resize Canvas (No Scaling)",
            "Enter new width",
            initialvalue=current_w,
            minvalue=1,
        )
        if new_w is None:
            return
        new_h = simpledialog.askinteger(
            "Resize Canvas (No Scaling)",
            "Enter new height",
            initialvalue=current_h,
            minvalue=1,
        )
        if new_h is None:
            return
        # If dimensions are unchanged, nothing to do
        if new_w == current_w and new_h == current_h:
            return
        # Save history before making changes
        self._save_history()
        # Enlarge canvas: if new dimensions are greater than current
        if new_w >= current_w and new_h >= current_h:
            for layer in self.layers:
                # Create new transparent canvas for original image
                new_img = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
                # Paste existing original at its offset; offset is relative to
                # the top‑left, so paste at (0,0) and later offset will shift
                new_img.paste(layer.original, (0, 0), layer.original)
                layer.original = new_img
                # Expand mask similarly; fill new areas with 0 (fully hidden) so
                # transparency is preserved
                new_mask = Image.new("L", (new_w, new_h), 0)
                new_mask.paste(layer.mask, (0, 0))
                layer.mask = new_mask
                # Offsets remain unchanged because we paste at (0,0)
                layer.apply_adjustments()
            # Update canvas size
            self.canvas.config(width=new_w, height=new_h)
            self._refresh_layer_list()
            self._update_composite()
            return
        # Otherwise, we are shrinking the canvas; crop at top‑left (0,0)
        # Use _perform_crop helper to crop all layers and update offsets
        # Compose bounding coordinates from (0,0) to (new_w, new_h)
        # We call _perform_crop directly using canvas coordinates
        self._perform_crop(0, 0, new_w, new_h)
        # After _perform_crop, canvas size will be updated

    def _reset_history_flag(self):
        """Reset the history saved flag after a slider or drawing operation."""
        self._history_saved_for_stroke = False

    # ------------------------------------------------------------------
    # Mask refinement operations
    # ------------------------------------------------------------------
    def _feather_mask(self) -> None:
        """Blur the mask edges of the currently selected layer to soften transitions.

        Prompts the user for a blur radius and applies a Gaussian blur
        to the layer's mask.  A larger radius results in a softer edge.
        """
        if self.current_layer_index is None:
            return
        radius = simpledialog.askinteger("Feather Mask", "Enter blur radius (1-50)", initialvalue=5, minvalue=1, maxvalue=50)
        if radius is None:
            return
        self._save_history()
        layer = self.layers[self.current_layer_index]
        # Apply Gaussian blur to the mask
        layer.mask = layer.mask.filter(ImageFilter.GaussianBlur(radius=radius))
        layer.apply_adjustments()
        self._update_composite()

    def _invert_mask(self) -> None:
        """Invert the mask of the currently selected layer.

        Pixels that were hidden become visible and vice versa.  Useful
        when refining selection areas or quickly toggling the mask.
        """
        if self.current_layer_index is None:
            return
        self._save_history()
        layer = self.layers[self.current_layer_index]
        layer.mask = ImageChops.invert(layer.mask)
        layer.apply_adjustments()
        self._update_composite()

    # ------------------------------------------------------------------
    # Adjustments and filters
    # ------------------------------------------------------------------
    def _on_alpha_change(self, value):
        if self.current_layer_index is None:
            return
        layer = self.layers[self.current_layer_index]
        # Save history only on first adjustment within a slider movement
        if not self._history_saved_for_stroke:
            self._save_history()
            self._history_saved_for_stroke = True
        layer.alpha = float(value)
        layer.apply_adjustments()
        self._update_composite()

    def _on_brightness_change(self, value):
        if self.current_layer_index is None:
            return
        layer = self.layers[self.current_layer_index]
        if not self._history_saved_for_stroke:
            self._save_history()
            self._history_saved_for_stroke = True
        layer.brightness = float(value)
        layer.apply_adjustments()
        self._update_composite()

    def _on_contrast_change(self, value):
        """Callback when the contrast slider is moved."""
        if self.current_layer_index is None:
            return
        layer = self.layers[self.current_layer_index]
        if not self._history_saved_for_stroke:
            self._save_history()
            self._history_saved_for_stroke = True
        layer.contrast = float(value)
        layer.apply_adjustments()
        self._update_composite()

    def _on_color_change(self, value):
        """Callback when the colour (saturation) slider is moved."""
        if self.current_layer_index is None:
            return
        layer = self.layers[self.current_layer_index]
        if not self._history_saved_for_stroke:
            self._save_history()
            self._history_saved_for_stroke = True
        layer.color = float(value)
        layer.apply_adjustments()
        self._update_composite()

    def _on_gamma_change(self, value):
        """Callback when the gamma (exposure) slider is moved."""
        if self.current_layer_index is None:
            return
        layer = self.layers[self.current_layer_index]
        if not self._history_saved_for_stroke:
            self._save_history()
            self._history_saved_for_stroke = True
        layer.gamma = float(value)
        layer.apply_adjustments()
        self._update_composite()

    def _apply_filter(self, filter_name: str):
        if self.current_layer_index is None:
            return
        # Run filter in background thread and show progress dialog
        layer = self.layers[self.current_layer_index]
        self._save_history()

        prog = ProgressDialog(self, title="Applying filter", initial_text=f"Applying {filter_name}...")

        # create a queue to receive final result or exception
        result_q = queue.Queue()

        def worker():
            try:
                # register progress callback
                def cb(pct):
                    try:
                        prog.set(pct)
                    except Exception:
                        pass
                register_progress_callback(cb)
                layer.apply_filter(filter_name)
                register_progress_callback(None)
                result_q.put((True, None))
            except Exception as e:
                register_progress_callback(None)
                result_q.put((False, e))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        # Poll for completion
        def poll():
            try:
                done, payload = result_q.get_nowait()
            except queue.Empty:
                # still running; schedule another poll
                self.after(80, poll)
                return
            prog.close()
            if done:
                # success
                try:
                    self._update_composite()
                except Exception:
                    pass
            else:
                e = payload
                messagebox.showerror("Filter error", str(e))

        self.after(80, poll)

    def _apply_high_pass_sharpen(self) -> None:
        """Apply a configurable high-pass sharpen to the active raster layer."""

        if self.current_layer_index is None:
            messagebox.showinfo("High Pass Sharpen", "Please select a layer first.")
            return
        layer = self.layers[self.current_layer_index]
        if not isinstance(layer, Layer):
            messagebox.showinfo("High Pass Sharpen", "This tool works on raster layers only.")
            return
        radius = simpledialog.askfloat("High Pass Radius", "Blur radius (suggest 3-12):", minvalue=0.1, initialvalue=4.0)
        if radius is None:
            return
        amount = simpledialog.askfloat("High Pass Amount", "Sharpen amount (1.0 = subtle, 3.0 = strong):", minvalue=0.1, initialvalue=1.6)
        if amount is None:
            return
        self._save_history()
        layer.original = apply_high_pass_detail(layer.original, radius, amount)
        layer.apply_adjustments()
        self._update_composite()
        self._set_status(f"High-pass sharpen applied (r={radius:.1f}, amt={amount:.1f})")

    def _apply_unblur(self) -> None:
        """Quickly reduce blur using an aggressive sharpening preset."""

        if self.current_layer_index is None:
            messagebox.showinfo("Unblur", "Please select a raster layer first.")
            return
        layer = self.layers[self.current_layer_index]
        if not isinstance(layer, Layer):
            messagebox.showinfo("Unblur", "This feature works on raster layers only.")
            return
        radius = simpledialog.askfloat("Unblur Radius", "Blur radius to compensate (suggest 2-6):", minvalue=0.5, initialvalue=3.0)
        if radius is None:
            return
        strength = simpledialog.askfloat("Unblur Strength", "Sharpen intensity (1.0-4.0):", minvalue=0.5, maxvalue=6.0, initialvalue=2.5)
        if strength is None:
            return
        self._save_history()
        layer.original = apply_high_pass_detail(layer.original, radius, strength)
        layer.apply_adjustments()
        self._update_composite()
        self._set_status(f"Unblur applied (radius {radius:.1f}, strength {strength:.1f})")

    def _apply_liquify_dialog(self, mode: str) -> None:
        """Run a lightweight liquify deformation on the current layer."""

        if self.current_layer_index is None:
            messagebox.showinfo("Liquify", "Select a raster layer first.")
            return
        layer = self.layers[self.current_layer_index]
        if not isinstance(layer, Layer):
            messagebox.showinfo("Liquify", "Liquify tools require a raster layer.")
            return
        radius = simpledialog.askfloat("Liquify Radius", "Effect radius (pixels):", minvalue=10.0, initialvalue=80.0)
        if radius is None:
            return
        strength = simpledialog.askfloat("Liquify Strength", "Strength (0.1-1.0):", minvalue=0.1, maxvalue=1.0, initialvalue=0.45)
        if strength is None:
            return
        angle = 0.0
        if mode in ("push", "pull"):
            angle_val = simpledialog.askfloat("Liquify Angle", "Direction angle (degrees):", initialvalue=self.gradient_settings.get('base_angle', 0.0))
            if angle_val is None:
                return
            angle = angle_val
        center = (layer.original.width / 2, layer.original.height / 2)
        self._save_history()
        result = liquify_deform(layer.original, mode=mode, center=center, radius=radius, strength=strength, angle_degrees=angle)
        layer.original = result
        layer.apply_adjustments()
        self._update_composite()
        self._set_status(f"Liquify {mode} applied (radius {radius:.0f}, strength {strength:.2f})")

    def _open_precise_lighting_panel(self) -> None:
        """Open an interactive panel for precise lighting and colour control."""

        if self.current_layer_index is None:
            messagebox.showinfo("Precise Adjustments", "Select a raster layer first.")
            return
        layer = self.layers[self.current_layer_index]
        if not isinstance(layer, Layer):
            messagebox.showinfo("Precise Adjustments", "Only raster layers support these adjustments.")
            return

        source_full = layer.original.copy()
        preview_src = source_full.copy()
        max_dim = 520
        scale = min(max_dim / preview_src.width, max_dim / preview_src.height, 1.0)
        if scale < 1.0:
            try:
                resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
            except Exception:
                resample = Image.ANTIALIAS
            preview_src = preview_src.resize((int(preview_src.width * scale), int(preview_src.height * scale)), resample=resample)

        win = tk.Toplevel(self)
        win.title("Precise Lighting & Color")
        win.configure(bg="#2f2f2f")

        params = {
            "exposure": tk.DoubleVar(value=0.0),
            "highlights": tk.IntVar(value=0),
            "shadows": tk.IntVar(value=0),
            "whites": tk.IntVar(value=0),
            "blacks": tk.IntVar(value=0),
            "temperature": tk.IntVar(value=0),
            "tint": tk.IntVar(value=0),
            "vibrance": tk.IntVar(value=0),
            "saturation": tk.IntVar(value=0),
            "clarity": tk.IntVar(value=0),
        }

        preview_label = tk.Label(win, bg="#2f2f2f")
        preview_label.grid(row=0, column=0, rowspan=11, padx=12, pady=12)

        sliders = tk.Frame(win, bg="#2f2f2f")
        sliders.grid(row=0, column=1, sticky="n", padx=12, pady=12)

        slider_specs = [
            ("Exposure", "exposure", -2.0, 2.0, 0.1),
            ("Highlights", "highlights", -100, 100, 1),
            ("Shadows", "shadows", -100, 100, 1),
            ("Whites", "whites", -100, 100, 1),
            ("Blacks", "blacks", -100, 100, 1),
            ("Temperature", "temperature", -100, 100, 1),
            ("Tint", "tint", -100, 100, 1),
            ("Vibrance", "vibrance", -100, 100, 1),
            ("Saturation", "saturation", -100, 100, 1),
            ("Clarity", "clarity", -100, 100, 1),
        ]

        for idx, (label, key, mn, mx, step) in enumerate(slider_specs):
            tk.Label(sliders, text=label, fg="white", bg="#2f2f2f").grid(row=idx, column=0, sticky="w", pady=2)
            scale_widget = tk.Scale(
                sliders,
                from_=mn,
                to=mx,
                orient=tk.HORIZONTAL,
                resolution=step,
                variable=params[key],
                length=220,
                bg="#3a3a3a",
                fg="white",
                troughcolor="#1f1f1f",
                highlightthickness=0,
                command=lambda _=None: update_preview(),
            )
            scale_widget.grid(row=idx, column=1, padx=(6, 0), pady=2)

        preview_img = None

        def current_params() -> dict:
            return {name: var.get() for name, var in params.items()}

        def update_preview() -> None:
            nonlocal preview_img
            try:
                preview_img = apply_precise_lighting_color_adjustments(
                    preview_src,
                    exposure=params["exposure"].get(),
                    highlights=params["highlights"].get(),
                    shadows=params["shadows"].get(),
                    whites=params["whites"].get(),
                    blacks=params["blacks"].get(),
                    temperature=params["temperature"].get(),
                    tint=params["tint"].get(),
                    vibrance=params["vibrance"].get(),
                    saturation=params["saturation"].get(),
                    clarity=params["clarity"].get(),
                )
            except Exception as exc:
                messagebox.showerror("Preview", f"Failed to render preview: {exc}")
                return
            photo = ImageTk.PhotoImage(preview_img)
            preview_label.configure(image=photo)
            preview_label.image = photo

        def apply_changes() -> None:
            self._save_history()
            try:
                layer.original = apply_precise_lighting_color_adjustments(
                    source_full,
                    **current_params(),
                )
                layer.apply_adjustments()
                self._update_composite()
                self._set_status("Applied precise lighting and colour adjustments")
            except Exception as exc:
                messagebox.showerror("Precise Adjustments", f"Failed to apply adjustments: {exc}")
            finally:
                win.destroy()

        def reset_sliders() -> None:
            for key, var in params.items():
                if isinstance(var, tk.DoubleVar):
                    var.set(0.0)
                else:
                    var.set(0)
            update_preview()

        btn_frame = tk.Frame(win, bg="#2f2f2f")
        btn_frame.grid(row=len(slider_specs), column=0, columnspan=2, pady=(8, 12))

        tk.Button(btn_frame, text="Apply", command=apply_changes, bg="#4a4a4a", fg="white", width=10).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_frame, text="Reset", command=reset_sliders, bg="#4a4a4a", fg="white", width=10).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_frame, text="Cancel", command=win.destroy, bg="#4a4a4a", fg="white", width=10).pack(side=tk.LEFT, padx=6)

        update_preview()


    def _auto_enhance(self) -> None:
        """Automatically adjust exposure, contrast and colour, then apply mild sharpening.

        This simple implementation enhances the current layer by boosting
        brightness, contrast and saturation slightly and applying an
        unsharp mask to improve clarity.  It saves the operation to
        history so it can be undone.
        """
        if self.current_layer_index is None:
            return
        self._save_history()
        layer = self.layers[self.current_layer_index]
        img = layer.original
        # Boost brightness, contrast and colour
        enh = ImageEnhance.Brightness(img)
        img = enh.enhance(1.1)
        enh = ImageEnhance.Contrast(img)
        img = enh.enhance(1.15)
        enh = ImageEnhance.Color(img)
        img = enh.enhance(1.1)
        # Apply mild sharpen
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=125, threshold=3))
        layer.original = img
        layer.apply_adjustments()
        self._update_composite()

    def _focus_peaking_overlay(self) -> None:
        """Generate a coloured focus-peaking overlay above the active layer."""

        if self.current_layer_index is None:
            messagebox.showinfo("Focus Peaking", "Select a raster layer first.")
            return
        layer = self.layers[self.current_layer_index]
        if not isinstance(layer, Layer):
            messagebox.showinfo("Focus Peaking", "Focus peaking works on raster layers only.")
            return
        overlay = generate_focus_peaking_overlay(layer.image)
        if overlay.getbbox() is None:
            messagebox.showinfo("Focus Peaking", "No prominent edges detected.")
            return
        focus_layer = Layer(overlay, f"{layer.name} Focus")
        focus_layer.alpha = 0.75
        focus_layer.apply_adjustments()
        insert_index = self.current_layer_index + 1
        self.layers.insert(insert_index, focus_layer)
        self.current_layer_index = insert_index
        self._refresh_layer_list()
        self._update_composite()
        self._set_status("Focus peaking overlay added")

    def _frequency_separation_layer(self) -> None:
        """Split the active layer into low/high frequency components for retouching."""

        if self.current_layer_index is None:
            messagebox.showinfo("Frequency Separation", "Select a raster layer first.")
            return
        layer = self.layers[self.current_layer_index]
        if not isinstance(layer, Layer):
            messagebox.showinfo("Frequency Separation", "Only raster layers can be separated.")
            return
        radius = simpledialog.askfloat("Frequency Separation", "Blur radius (higher = softer skin):", minvalue=1.0, initialvalue=12.0)
        if radius is None:
            return
        self._save_history()
        low, high = frequency_separation_layers(layer.original, radius)
        original_name = layer.name
        layer.original = low
        layer.name = f"{original_name} (Low Freq)"
        layer.apply_adjustments()
        high_layer = Layer(high, f"{original_name} (High Freq)")
        high_layer.blend_mode = 'overlay'
        high_layer.alpha = 0.7
        high_layer.apply_adjustments()
        self.layers.insert(self.current_layer_index + 1, high_layer)
        self._refresh_layer_list()
        self._update_composite()
        self._set_status(f"Frequency separation created (radius {radius:.1f})")

    def _open_frequency_separation_panel(self) -> None:
        """Interactive panel for frequency separation with preview and export."""

        if self.current_layer_index is None:
            messagebox.showinfo("Frequency Separation", "Select a raster layer first.")
            return
        layer = self.layers[self.current_layer_index]
        if not isinstance(layer, Layer):
            messagebox.showinfo("Frequency Separation", "Only raster layers can be separated.")
            return
        radius = simpledialog.askfloat("Frequency Separation", "Blur radius (larger = softer skin):", minvalue=1.0, initialvalue=10.0)
        if radius is None:
            return
        try:
            low, high = frequency_separation_layers(layer.original, radius)
        except Exception as exc:
            messagebox.showerror("Frequency Separation", f"Failed to compute separation: {exc}")
            return

        top = tk.Toplevel(self)
        top.title("Frequency Separation Preview")
        top.configure(bg="#2b2b2b")

        try:
            resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
        except Exception:
            resample = Image.ANTIALIAS
        preview_size = 320
        def make_preview(img: Image.Image) -> ImageTk.PhotoImage:
            copy = img.copy()
            scale = min(preview_size / copy.width, preview_size / copy.height, 1.0)
            if scale < 1.0:
                copy = copy.resize((int(copy.width * scale), int(copy.height * scale)), resample=resample)
            return ImageTk.PhotoImage(copy)

        low_photo = make_preview(low)
        high_photo = make_preview(high)

        low_label = tk.Label(top, image=low_photo, bg="#2b2b2b")
        low_label.image = low_photo
        low_label.grid(row=0, column=0, padx=12, pady=12)
        tk.Label(top, text="Low Frequency", fg="#f0f0f0", bg="#2b2b2b").grid(row=1, column=0)

        high_label = tk.Label(top, image=high_photo, bg="#2b2b2b")
        high_label.image = high_photo
        high_label.grid(row=0, column=1, padx=12, pady=12)
        tk.Label(top, text="High Frequency", fg="#f0f0f0", bg="#2b2b2b").grid(row=1, column=1)

        button_frame = tk.Frame(top, bg="#2b2b2b")
        button_frame.grid(row=2, column=0, columnspan=2, pady=(8, 12))

        def apply_layers():
            self._save_history()
            original_name = layer.name
            layer.original = low
            layer.name = f"{original_name} (Low Freq)"
            layer.apply_adjustments()
            high_layer = Layer(high, f"{original_name} (High Freq)")
            high_layer.blend_mode = 'overlay'
            high_layer.alpha = 0.7
            high_layer.apply_adjustments()
            self.layers.insert(self.current_layer_index + 1, high_layer)
            self._refresh_layer_list()
            self._update_composite()
            self._set_status(f"Frequency separation applied (radius {radius:.1f})")
            top.destroy()

        def export_image(img: Image.Image, label: str):
            path = filedialog.asksaveasfilename(title=f"Save {label}", defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("All files", "*.*")])
            if not path:
                return
            try:
                img.save(path)
                self._set_status(f"Saved {label} to {path}")
            except Exception as exc:
                messagebox.showerror("Export", f"Failed to save {label}: {exc}")

        tk.Button(button_frame, text="Apply as Layers", command=apply_layers).grid(row=0, column=0, padx=8)
        tk.Button(button_frame, text="Save Low…", command=lambda: export_image(low, "low frequency")).grid(row=0, column=1, padx=8)
        tk.Button(button_frame, text="Save High…", command=lambda: export_image(high, "high frequency")).grid(row=0, column=2, padx=8)
        tk.Button(button_frame, text="Close", command=top.destroy).grid(row=0, column=3, padx=8)

    def _content_aware_fill(self) -> None:
        """Fill transparent pixels using nearby colours."""

        if self.current_layer_index is None:
            messagebox.showinfo("Content-Aware Fill", "Select a raster layer first.")
            return
        layer = self.layers[self.current_layer_index]
        if not isinstance(layer, Layer):
            messagebox.showinfo("Content-Aware Fill", "This tool works on raster layers only.")
            return
        self._save_history()
        layer.original = content_aware_fill(layer.original)
        layer.apply_adjustments()
        self._update_composite()
        self._set_status("Content-aware fill applied to transparent regions")

    def _replace_background(self) -> None:
        """Replace the background of the current layer outside the mask.

        This function uses the layer's mask to identify the subject. The
        user chooses a replacement colour; all pixels where the mask is
        mostly transparent (value < 128) are filled with that colour.
        """
        if self.current_layer_index is None:
            return
        # Ask the user for a replacement colour
        color = colorchooser.askcolor(title="Choose background colour")
        if not color or not color[0]:
            return
        rgb = tuple(int(c) for c in color[0])
        self._save_history()
        layer = self.layers[self.current_layer_index]
        orig = layer.original.copy()
        # Invert mask: areas with value < 128 will be replaced
        mask_inv = layer.mask.point(lambda v: 255 if v < 128 else 0)
        # Create solid colour image
        bg = Image.new("RGBA", orig.size, rgb + (255,))
        # Composite: fill transparent areas with bg using mask_inv
        orig.paste(bg, (0, 0), mask_inv)
        layer.original = orig
        layer.apply_adjustments()
        self._update_composite()

    def _export_preset(self, target_w: int, target_h: int) -> None:
        """Export the current composite image to a predefined size.

        The resulting image is resized to fill the target dimensions,
        cropping to preserve aspect ratio.  A save dialog prompts for
        the filename.  Use this to quickly generate social media
        friendly images.

        :param target_w: target width in pixels
        :param target_h: target height in pixels
        """
        if not self.layers:
            messagebox.showinfo("No image", "There is nothing to export.")
            return
        composite = self._create_composite_image()
        # Fit image to target size with cropping while preserving aspect ratio
        try:
            from PIL import ImageOps
            export_img = ImageOps.fit(composite, (target_w, target_h), method=Image.BICUBIC)
        except Exception:
            export_img = composite.resize((target_w, target_h), resample=Image.BICUBIC)
        filetypes = [("PNG", "*.png"), ("JPEG", "*.jpg"), ("All files", "*.*")]
        filepath = filedialog.asksaveasfilename(title="Export Image", defaultextension=".png", filetypes=filetypes)
        if not filepath:
            return
        try:
            export_img.save(filepath)
            messagebox.showinfo("Exported", f"Image exported to {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save exported image: {e}")

    def _preview_and_apply_filter(self, filter_name: str) -> None:
        """Preview a filter on the current layer before applying it.

        A downscaled version of the selected layer is shown with the filter
        applied.  The user can decide whether to commit the change or
        cancel it.  If cancelled, no modification is made.

        :param filter_name: name of the filter to apply (grayscale, blur, etc.)
        """
        if self.current_layer_index is None:
            return
        layer = self.layers[self.current_layer_index]
        # Make a copy of the original layer image for preview
        preview_img = layer.original.copy()
        temp_layer = Layer(preview_img.copy(), "preview")
        try:
            # Support custom filter implemented outside Layer.apply_filter
            if filter_name == 'poster_edges':
                preview_img = poster_edges_filter(preview_img)
            else:
                temp_layer.apply_filter(filter_name)
                preview_img = temp_layer.original
        except Exception as e:
            messagebox.showerror("Filter error", str(e))
            return
        # Downscale preview to fit a small window while preserving aspect ratio
        max_preview_size = 300
        w, h = preview_img.size
        scale = min(max_preview_size / w, max_preview_size / h, 1.0)
        prev = preview_img.copy()
        if scale < 1.0:
            prev = prev.resize((int(w * scale), int(h * scale)), resample=Image.BICUBIC)
        # Create preview window
        win = tk.Toplevel(self)
        win.title(f"Preview: {filter_name}")
        win.configure(bg="#3a3a3a")
        # Display image
        photo = ImageTk.PhotoImage(prev)
        img_label = tk.Label(win, image=photo)
        img_label.image = photo  # keep reference
        img_label.pack(padx=10, pady=10)
        # Add Apply / Cancel buttons
        def _apply():
            self._save_history()
            try:
                if filter_name == 'poster_edges':
                    target = getattr(self.layers[self.current_layer_index], 'original', None) or getattr(self.layers[self.current_layer_index], 'image', None)
                    if target is not None:
                        self.layers[self.current_layer_index].original = poster_edges_filter(target)
                else:
                    self.layers[self.current_layer_index].apply_filter(filter_name)
            except Exception:
                pass
            self._update_composite()
            win.destroy()

        def _cancel():
            win.destroy()

        btn_frame = tk.Frame(win, bg="#3a3a3a")
        btn_frame.pack(pady=(0,10))
        apply_btn = tk.Button(btn_frame, text="Apply", command=_apply)
        apply_btn.pack(side=tk.LEFT, padx=8)
        cancel_btn = tk.Button(btn_frame, text="Cancel", command=_cancel)
        cancel_btn.pack(side=tk.LEFT, padx=8)
        # Buttons
        btn_frame = tk.Frame(win, bg="#3a3a3a")
        btn_frame.pack(pady=5)
        def apply_change():
            # Apply filter for real and update
            self._apply_filter(filter_name)
            win.destroy()
        def cancel():
            win.destroy()
        apply_btn = tk.Button(btn_frame, text="Apply", command=apply_change, bg="#5c5c5c", fg="white")
        apply_btn.pack(side=tk.LEFT, padx=5)
        cancel_btn = tk.Button(btn_frame, text="Cancel", command=cancel, bg="#5c5c5c", fg="white")
        cancel_btn.pack(side=tk.LEFT, padx=5)

    def _adjustable_blur(self) -> None:
        """Apply adjustable blur filter with radius slider."""
        if self.current_layer_index is None:
            return
        self._show_adjustable_filter_dialog("blur", "Blur Radius", 0, 20, 2, "radius")

    def _adjustable_sharpen(self) -> None:
        """Apply adjustable sharpen filter with strength slider."""
        if self.current_layer_index is None:
            return
        self._show_adjustable_filter_dialog("sharpen", "Sharpen Strength", 0, 500, 100, "percent")

    def _adjustable_sepia(self) -> None:
        """Apply adjustable sepia filter with intensity slider."""
        if self.current_layer_index is None:
            return
        self._show_adjustable_filter_dialog("sepia", "Sepia Intensity", 0, 200, 100, "intensity")

    def _apply_canva_preset(self, preset: str) -> None:
        """Preview and apply a Canva-style preset to the current raster layer."""

        if self.current_layer_index is None:
            messagebox.showinfo("Canva Preset", "Please select a layer first.")
            return
        layer = self.layers[self.current_layer_index]
        if not isinstance(layer, Layer):
            messagebox.showinfo("Canva Preset", "Presets can only be applied to raster layers.")
            return
        try:
            preview_img = apply_canva_preset(layer.original, preset)
        except Exception as exc:
            messagebox.showerror("Canva Preset", f"Unable to apply preset: {exc}")
            return

        display = preview_img.copy()
        max_dim = 360
        scale = min(max_dim / display.width, max_dim / display.height, 1.0)
        if scale < 1.0:
            try:
                resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
            except Exception:
                resample = Image.ANTIALIAS
            display = display.resize((int(display.width * scale), int(display.height * scale)), resample=resample)

        win = tk.Toplevel(self)
        win.title(f"Canva Preset: {preset.replace('_', ' ').title()}")
        win.configure(bg="#3a3a3a")

        photo = ImageTk.PhotoImage(display)
        img_label = tk.Label(win, image=photo, bg="#3a3a3a")
        img_label.image = photo
        img_label.pack(padx=12, pady=12)

        btn_frame = tk.Frame(win, bg="#3a3a3a")
        btn_frame.pack(pady=(0, 12))

        def apply_and_close() -> None:
            self._save_history()
            try:
                layer.original = apply_canva_preset(layer.original, preset)
                layer.apply_adjustments()
                self._update_composite()
                self._set_status(f"Applied Canva preset: {preset.replace('_', ' ').title()}")
            except Exception as exc2:
                messagebox.showerror("Canva Preset", f"Failed to apply preset: {exc2}")
            finally:
                win.destroy()

        def cancel() -> None:
            win.destroy()

        tk.Button(btn_frame, text="Apply", command=apply_and_close, bg="#5c5c5c", fg="white").pack(side=tk.LEFT, padx=6)
        tk.Button(btn_frame, text="Cancel", command=cancel, bg="#5c5c5c", fg="white").pack(side=tk.LEFT, padx=6)

    def _show_adjustable_filter_dialog(self, filter_name: str, param_name: str, min_val: int, max_val: int, default_val: int, param_key: str) -> None:
        """Show a dialog with slider for adjustable filter parameters."""
        layer = self.layers[self.current_layer_index]
        
        # Create dialog window
        win = tk.Toplevel(self)
        win.title(f"Adjustable {filter_name.title()}")
        win.configure(bg="#3a3a3a")
        win.geometry("400x300")
        
        # Make a copy for preview
        preview_img = layer.original.copy()
        temp_layer = Layer(preview_img.copy(), "preview")
        
        # Parameter value
        param_value = tk.IntVar(value=default_val)
        
        # Preview image
        photo = None
        img_label = None
        
        def update_preview():
            nonlocal photo, img_label
            try:
                # Apply filter with current parameter
                temp_layer.original = layer.original.copy()
                temp_layer.apply_filter_with_param(filter_name, {param_key: param_value.get()})
                
                # Downscale for preview
                max_preview_size = 200
                w, h = temp_layer.original.size
                scale = min(max_preview_size / w, max_preview_size / h, 1.0)
                prev = temp_layer.original.copy()
                if scale < 1.0:
                    prev = prev.resize((int(w * scale), int(h * scale)), resample=Image.BICUBIC)
                
                # Update image
                photo = ImageTk.PhotoImage(prev)
                if img_label:
                    img_label.configure(image=photo)
                    img_label.image = photo
                else:
                    img_label = tk.Label(win, image=photo, bg="#3a3a3a")
                    img_label.image = photo
                    img_label.pack(pady=10)
            except Exception as e:
                messagebox.showerror("Filter error", str(e))
        
        # Parameter slider
        param_frame = tk.Frame(win, bg="#3a3a3a")
        param_frame.pack(pady=10)
        
        tk.Label(param_frame, text=f"{param_name}:", bg="#3a3a3a", fg="white", font=("Arial", 12)).pack()
        
        slider = tk.Scale(
            param_frame,
            from_=min_val,
            to=max_val,
            orient=tk.HORIZONTAL,
            variable=param_value,
            command=lambda x: update_preview(),
            bg="#5c5c5c",
            fg="white",
            highlightthickness=0,
            length=300
        )
        slider.pack(pady=5)
        
        # Value label
        value_label = tk.Label(param_frame, text=f"Value: {param_value.get()}", bg="#3a3a3a", fg="white")
        value_label.pack()
        
        def update_value_label(*args):
            value_label.configure(text=f"Value: {param_value.get()}")
        
        param_value.trace('w', update_value_label)
        
        # Buttons
        btn_frame = tk.Frame(win, bg="#3a3a3a")
        btn_frame.pack(pady=10)
        
        def apply_filter():
            try:
                self._save_history()
                layer.apply_filter_with_param(filter_name, {param_key: param_value.get()})
                self._update_composite()
                win.destroy()
            except Exception as e:
                messagebox.showerror("Filter error", str(e))
        
        def cancel():
            win.destroy()
        
        apply_btn = tk.Button(btn_frame, text="Apply", command=apply_filter, bg="#5c5c5c", fg="white")
        apply_btn.pack(side=tk.LEFT, padx=5)
        cancel_btn = tk.Button(btn_frame, text="Cancel", command=cancel, bg="#5c5c5c", fg="white")
        cancel_btn.pack(side=tk.LEFT, padx=5)
        
        # Initial preview
        update_preview()

    # ------------------------------------------------------------------
    # Palette helpers and smart tools
    # ------------------------------------------------------------------
    def _normalise_hex(self, value: str) -> str:
        if not value:
            return "#000000"
        value = value.strip()
        if not value.startswith('#'):
            value = '#' + value
        if len(value) == 4:
            # short form #rgb -> #rrggbb
            value = '#' + ''.join(ch * 2 for ch in value[1:4])
        return value[:7]

    def _hex_to_rgb_text(self, value: str) -> str:
        value = self._normalise_hex(value)
        try:
            r = int(value[1:3], 16)
            g = int(value[3:5], 16)
            b = int(value[5:7], 16)
        except Exception:
            r = g = b = 0
        return f"RGB({r},{g},{b})"

    def _hex_to_rgb_tuple(self, value: str) -> Tuple[int, int, int]:
        value = self._normalise_hex(value)
        try:
            return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]
        except Exception:
            return (0, 0, 0)

    def _update_color_palette(self) -> None:
        self.color1_value.set(self._hex_to_rgb_text(self.primary_color))
        self.color2_value.set(self._hex_to_rgb_text(self.secondary_color))
        if hasattr(self, 'color1_swatch'):
            self.color1_swatch.config(bg=self.primary_color)
        if hasattr(self, 'color2_swatch'):
            self.color2_swatch.config(bg=self.secondary_color)
        if hasattr(self, 'color1_radio'):
            self.color1_radio.config(selectcolor=self.primary_color)
        if hasattr(self, 'color2_radio'):
            self.color2_radio.config(selectcolor=self.secondary_color)
        if hasattr(self, 'active_color_var'):
            self.active_color_var.set(self.active_color_slot)

    def _set_color(self, slot: str, hex_color: str) -> None:
        colour = self._normalise_hex(hex_color)
        if slot == 'primary':
            self.primary_color = colour
            if self.active_color_slot == 'primary':
                self.brush_color = colour
            stops = list(self.gradient_settings.get('stops', []))
            if stops:
                stops[0] = (stops[0][0], colour)
                self.gradient_settings['stops'] = stops
        else:
            self.secondary_color = colour
            if self.active_color_slot == 'secondary':
                self.brush_color = colour
            stops = list(self.gradient_settings.get('stops', []))
            if stops:
                stops[-1] = (stops[-1][0], colour)
                self.gradient_settings['stops'] = stops
        self._update_color_palette()

    def _swap_colors(self) -> None:
        self.primary_color, self.secondary_color = self.secondary_color, self.primary_color
        if self.active_color_slot == 'primary':
            self.brush_color = self.primary_color
        elif self.active_color_slot == 'secondary':
            self.brush_color = self.secondary_color
        stops = list(self.gradient_settings.get('stops', []))
        if stops and len(stops) >= 2:
            stops[0] = (stops[0][0], self.primary_color)
            stops[-1] = (stops[-1][0], self.secondary_color)
            self.gradient_settings['stops'] = stops
        self._update_color_palette()

    def _apply_sampled_color(self, rgb: Tuple[int, int, int]) -> None:
        hex_color = '#{:02x}{:02x}{:02x}'.format(*[max(0, min(255, int(v))) for v in rgb])
        self._set_color(self.active_color_slot, hex_color)

    def _sync_pattern_var(self) -> None:
        self.pattern_settings['type'] = self.pattern_type_var.get()

