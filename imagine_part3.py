        macro_menu.add_command(label="Stop Recording", command=self._stop_macro_recording)
        macro_menu.add_separator()
        macro_menu.add_command(label="Play Macro", command=self._play_macro)
        macro_menu.add_separator()
        macro_menu.add_command(label="Save Macro", command=self._save_macro)
        macro_menu.add_command(label="Load Macro", command=self._load_macro)
        menubar.add_cascade(label="Macro", menu=macro_menu)

        # Transform menu
        transform_menu = tk.Menu(menubar, tearoff=0)
        transform_menu.add_command(label="Rotate 90° CW", command=lambda: self._rotate_layer(90))
        transform_menu.add_command(label="Rotate 180°", command=lambda: self._rotate_layer(180))
        transform_menu.add_command(label="Rotate 270° CW", command=lambda: self._rotate_layer(270))
        transform_menu.add_separator()
        transform_menu.add_command(label="Flip Horizontal", command=lambda: self._flip_layer("horizontal"))
        transform_menu.add_command(label="Flip Vertical", command=lambda: self._flip_layer("vertical"))
        transform_menu.add_separator()
        transform_menu.add_command(label="Resize Canvas (Scale Images)", command=self._resize_canvas)
        transform_menu.add_command(label="Resize Canvas (No Scaling)", command=self._resize_canvas_no_scale)
        transform_menu.add_separator()
        transform_menu.add_command(label="Perspective Transform", command=self._perspective_transform_layer)
        # Additional advanced transforms
        transform_menu.add_command(label="Warp (Sine)", command=self._warp_layer)
        transform_menu.add_command(label="Face Ratio Adjust", command=self._face_ratio_adjust_layer)
        transform_menu.add_command(label="Free Distort", command=self._free_distort_layer)
        transform_menu.add_separator()
        transform_menu.add_command(label="Upscale Layer (2×)", command=lambda: self._scale_layer(2.0))
        transform_menu.add_command(label="Downscale Layer (0.5×)", command=lambda: self._scale_layer(0.5))

        # Resize current raster layer to specified width/height (non-proportional allowed)
        def _resize_layer_custom():
            if self.current_layer_index is None:
                messagebox.showinfo("Resize Layer", "No layer selected.")
                return
            layer = self.layers[self.current_layer_index]
            # Only apply to raster layers (Layer instances), not VectorLayer or AdjustmentLayer
            if not isinstance(layer, Layer):
                messagebox.showinfo("Resize Layer", "Resize only supported for raster layers.")
                return
            w = simpledialog.askinteger("Resize Layer", "Enter new width (px):", parent=self, minvalue=1)
            if w is None:
                return
            h = simpledialog.askinteger("Resize Layer", "Enter new height (px):", parent=self, minvalue=1)
            if h is None:
                return
            try:
                # Save history for undo
                self._current_action_desc = f"Resize Layer {layer.name} to {w}×{h}"
                self._save_history()
                # Use LANCZOS resampling when available
                try:
                    resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
                except Exception:
                    resample = Image.ANTIALIAS
                # Resize the original image and mask
                layer.original = layer.original.resize((w, h), resample)
                try:
                    layer.mask = layer.mask.resize((w, h), resample)
                except Exception:
                    # If mask resizing fails, recreate a full selection mask
                    layer.mask = Image.new("L", (w, h), 255)
                # Reset offset to keep the layer anchored (optionally could scale offset)
                layer.offset = (0, 0)
                layer.apply_adjustments()
                self._update_composite()
            except Exception as e:
                messagebox.showerror("Resize Layer", f"Resize failed: {e}")

        transform_menu.add_separator()
        transform_menu.add_command(label="Resize Layer (Width/Height)…", command=_resize_layer_custom)

        # Resize selection dialog
        def _resize_selection_dialog():
            if not hasattr(self, '_last_selection_mask') or self._last_selection_mask is None:
                messagebox.showinfo('Resize Selection', 'No active selection to resize.')
                return
            # Ask new width/height for selection area
            sel_bbox = self._last_selection_mask.getbbox()
            if not sel_bbox:
                messagebox.showinfo('Resize Selection', 'Selection is empty.')
                return
            sw = simpledialog.askinteger('Selection Width', 'Enter new width (px):', parent=self, minvalue=1, initialvalue=sel_bbox[2]-sel_bbox[0])
            if sw is None:
                return
            sh = simpledialog.askinteger('Selection Height', 'Enter new height (px):', parent=self, minvalue=1, initialvalue=sel_bbox[3]-sel_bbox[1])
            if sh is None:
                return
            keep_aspect = messagebox.askyesno('Keep Aspect', 'Keep aspect ratio for selection?')
            try:
                self._current_action_desc = f"Resize Selection to {sw}×{sh}"
                self._save_history()
                # Extract selection region from active layer image
                layer = self.layers[self.current_layer_index] if self.current_layer_index is not None else None
                if layer is None:
                    messagebox.showinfo('Resize Selection', 'No active layer for selection.')
                    return
                src_img = layer.original
                bbox = sel_bbox
                region = src_img.crop(bbox)
                # Compute target size
                if keep_aspect:
                    # Preserve aspect by scaling to fit within sw/sh
                    rw, rh = region.size
                    scale = min(sw / rw, sh / rh)
                    tw = max(1, int(rw * scale))
                    th = max(1, int(rh * scale))
                else:
                    tw, th = sw, sh
                try:
                    resample = Image.Resampling.LANCZOS
                except Exception:
                    resample = Image.ANTIALIAS
                resized_region = region.resize((tw, th), resample)
                # Paste back as a floating selection (create new layer)
                float_img = Image.new('RGBA', src_img.size, (0,0,0,0))
                # place at original bbox top-left
                float_img.paste(resized_region, bbox[:2], resized_region)
                # Add as a new layer so user can move/commit
                float_layer = Layer(float_img, f"Selection {len(self.layers)}")
                self.layers.append(float_layer)
                self.current_layer_index = len(self.layers) - 1
                self._refresh_layer_list()
                self._update_composite()
            except Exception as e:
                messagebox.showerror('Resize Selection', f'Failed: {e}')

        transform_menu.add_command(label="Resize Selection…", command=_resize_selection_dialog)

        # Custom rotate
        def _rotate_custom():
            angle = simpledialog.askstring("Rotate Custom", "Enter angle in degrees (integer):", parent=self)
            if angle is None:
                return
            try:
                degrees = int(angle)
            except Exception:
                messagebox.showerror("Rotate Custom", "Please enter a valid integer angle.")
                return
            self._current_action_desc = f"Rotate {degrees}°"
            self._rotate_layer(degrees)
            self._update_composite()

        transform_menu.add_separator()
        transform_menu.add_command(label="Rotate Custom…", command=_rotate_custom)

        # Custom scale
        def _scale_custom():
            val = simpledialog.askstring("Scale Custom", "Enter scale factor (e.g. 0.7 for 70%, 1.2 for 120%):", parent=self)
            if val is None:
                return
            try:
                factor = float(val)
            except Exception:
                messagebox.showerror("Scale Custom", "Please enter a valid number for scale factor.")
                return
            if factor <= 0:
                messagebox.showerror("Scale Custom", "Scale factor must be greater than zero.")
                return
            self._current_action_desc = f"Scale {factor}×"
            self._scale_layer(factor)
            self._update_composite()

        transform_menu.add_command(label="Scale Custom…", command=_scale_custom)
        menubar.add_cascade(label="Transform", menu=transform_menu)
        # Professional feature set menu
        pro_menu = tk.Menu(menubar, tearoff=0)
        pro_menu.add_command(label="High Pass Sharpen…", command=self._apply_high_pass_sharpen)
        pro_menu.add_command(label="Unblur / Deblur…", command=self._apply_unblur)
        pro_menu.add_command(label="Add Focus Peaking Overlay", command=self._focus_peaking_overlay)
        pro_menu.add_command(label="Frequency Separation Panel…", command=self._open_frequency_separation_panel)
        pro_menu.add_command(label="Frequency Separation Layers…", command=self._frequency_separation_layer)
        pro_menu.add_command(label="Content-Aware Fill", command=self._content_aware_fill)
        pro_menu.add_separator()
        pro_menu.add_command(label="Precise Lighting & Color…", command=self._open_precise_lighting_panel)
        pro_menu.add_command(label="Show Histogram", command=self._show_histogram)
        pro_menu.add_command(label="Generate Contact Sheet…", command=self._generate_contact_sheet)
        menubar.add_cascade(label="Pro", menu=pro_menu)

        # Selection menu with refinement tools
        select_menu = tk.Menu(menubar, tearoff=0)
        select_menu.add_command(label="Select Subject", command=self._select_subject)
        select_menu.add_command(label="Expand Selection…", command=self._expand_selection_cmd)
        select_menu.add_command(label="Refine Edge…", command=self._refine_edge_tool)
        select_menu.add_command(label="Make Cutline…", command=self._make_cutline_cmd)
        menubar.add_cascade(label="Select", menu=select_menu)
        # Blend mode menu
        blend_menu = tk.Menu(menubar, tearoff=0)
        blend_menu.add_command(label="Normal", command=lambda: self._set_blend_mode('normal'))
        blend_menu.add_command(label="Multiply", command=lambda: self._set_blend_mode('multiply'))
        blend_menu.add_command(label="Screen", command=lambda: self._set_blend_mode('screen'))
        blend_menu.add_command(label="Overlay", command=lambda: self._set_blend_mode('overlay'))
        blend_menu.add_command(label="Soft Light", command=lambda: self._set_blend_mode('softlight'))
        menubar.add_cascade(label="Blend Mode", menu=blend_menu)

        # View menu to toggle layer visibility
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Toggle Layer Visibility", command=self._toggle_visibility)
        menubar.add_cascade(label="View", menu=view_menu)

        # History menu to open history panel
        history_menu = tk.Menu(menubar, tearoff=0)
        history_menu.add_command(label="Show History Panel", command=self._show_history_panel)
        menubar.add_cascade(label="History", menu=history_menu)

        # Settings menu for theme and autosave
        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Toggle Dark/Light Theme", command=self._toggle_theme)
        settings_menu.add_command(label="Set Autosave Interval", command=self._set_autosave_interval)
        menubar.add_cascade(label="Settings", menu=settings_menu)

        # Help menu provides an in-app user guide
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="User Guide", command=self._show_help)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.extensions_menu = tk.Menu(menubar, tearoff=0)
        self.extensions_menu.add_command(label="Load Extension…", command=self._prompt_extension_upload)
        menubar.add_cascade(label="Extensions", menu=self.extensions_menu)

        self.config(menu=menubar)
        # Apply theme after creating all menus and widgets
        self._apply_theme()

        # Export menu for common social media presets
        export_menu = tk.Menu(menubar, tearoff=0)
        export_menu.add_command(label="Instagram (1080×1080)", command=lambda: self._export_preset(1080, 1080))
        export_menu.add_command(label="Twitter 16:9 (1920×1080)", command=lambda: self._export_preset(1920, 1080))
        export_menu.add_command(label="Facebook Cover (820×312)", command=lambda: self._export_preset(820, 312))
        menubar.add_cascade(label="Export", menu=export_menu)

        # Collage menu for creating collages and auto layout
        collage_menu = tk.Menu(menubar, tearoff=0)
        collage_menu.add_command(label="Create Collage from Files", command=self._create_collage_from_files)
        collage_menu.add_command(label="Layout Visible Layers as Collage", command=self._layout_visible_layers)
        # New advanced collage creation command
        collage_menu.add_command(label="Create Collage (Advanced)", command=self._create_collage_advanced)
        # Auto balance layers for cohesive composition
        collage_menu.add_command(label="Auto Balance Layers", command=self._auto_balance_layers)
        menubar.add_cascade(label="Collage", menu=collage_menu)

        # Templates menu for quick layouts
        templates_menu = tk.Menu(menubar, tearoff=0)
        templates_menu.add_command(label="2×2 Grid", command=lambda: self._create_template_layout("2x2"))
        templates_menu.add_command(label="3×3 Grid", command=lambda: self._create_template_layout("3x3"))
        templates_menu.add_command(label="1×3 Horizontal", command=lambda: self._create_template_layout("1x3h"))
        templates_menu.add_command(label="3×1 Vertical", command=lambda: self._create_template_layout("3x1v"))
        templates_menu.add_command(label="Random Mosaic", command=lambda: self._create_template_layout("random"))
        menubar.add_cascade(label="Templates", menu=templates_menu)

        # Drafts menu for saving/loading temporary projects
        drafts_menu = tk.Menu(menubar, tearoff=0)
        drafts_menu.add_command(label="Save Draft", command=self._save_draft)
        drafts_menu.add_command(label="Load Draft", command=self._load_draft)
        drafts_menu.add_command(label="Delete All Drafts", command=self._delete_all_drafts)
        menubar.add_cascade(label="Drafts", menu=drafts_menu)

    # ------------------------------------------------------------------
    # Layer management
    # ------------------------------------------------------------------

    def _load_extensions(self) -> None:
        """Scan the extensions directory and load user extensions."""

        try:
            self.extensions_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        if not hasattr(self, 'extensions_menu'):
            return

        self.extensions_menu.delete(0, tk.END)
        self.extensions_menu.add_command(label="Load Extension…", command=self._prompt_extension_upload)
        self.extensions_menu.add_separator()
        self.loaded_extensions.clear()

        found = False
        for path in sorted(self.extensions_dir.glob('*.py')):
            found = True
            try:
                spec = importlib.util.spec_from_file_location(f"imagine_ext_{uuid.uuid4().hex}", path)
                if spec is None or spec.loader is None:
                    raise RuntimeError("Unable to load extension spec")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as exc:
                err = str(exc)
                self.extensions_menu.add_command(
                    label=f"{path.stem} (error)",
                    command=lambda e=err: messagebox.showerror("Extension Error", e)
                )
                continue
            self.loaded_extensions[path.stem] = module
            if hasattr(module, 'register_extension'):
                try:
                    entries = module.register_extension(self)
                except Exception as exc:
                    self.extensions_menu.add_command(
                        label=f"{path.stem} (error)",
                        command=lambda e=str(exc): messagebox.showerror("Extension Error", e)
                    )
                    continue
                if entries:
                    submenu = tk.Menu(self.extensions_menu, tearoff=0)
                    for entry in entries:
                        if isinstance(entry, (tuple, list)) and len(entry) == 2:
                            label, callback = entry
                            submenu.add_command(label=label, command=callback)
                    if submenu.index('end') is not None:
                        self.extensions_menu.add_cascade(label=path.stem, menu=submenu)
                        continue
            self.extensions_menu.add_command(
                label=path.stem,
                command=lambda name=path.stem: messagebox.showinfo("Extension", f"Extension '{name}' loaded.")
            )

        if not found:
            self.extensions_menu.add_command(label="No extensions installed", state=tk.DISABLED)

    def _prompt_extension_upload(self) -> None:
        """Prompt the user to load a new extension file."""

        path = filedialog.askopenfilename(title="Select Extension", filetypes=[("Python", "*.py"), ("All files", "*.*")])
        if not path:
            return
        try:
            dest = self.extensions_dir / Path(path).name
            shutil.copy(Path(path), dest)
        except Exception as exc:
            messagebox.showerror("Extensions", f"Could not install extension: {exc}")
            return
        self._set_status(f"Extension '{dest.name}' installed")
        self._load_extensions()

    def _new_blank_layer(self):
        """Deprecated wrapper for backwards compatibility. Calls _new_layer()."""
        self._new_layer()

    def _new_layer(self):
        """Create a new layer of various types based on user selection.

        Options include a blank transparent layer, a solid colour fill,
        a gradient fill or a simple pattern.  The new layer uses the size
        of the first existing layer as its canvas.  If no layers exist
        yet, the user is prompted to open an image first.
        """
        # Determine canvas/base size.  If no layers exist yet, prompt user
        # for initial canvas dimensions and (optionally) background colour.
        if not self.layers:
            # Ask width and height for new document
            width = simpledialog.askinteger(
                "Canvas Width",
                "Enter new canvas width (pixels):",
                initialvalue=800,
                minvalue=1,
            )
            if width is None:
                return
            height = simpledialog.askinteger(
                "Canvas Height",
                "Enter new canvas height (pixels):",
                initialvalue=600,
                minvalue=1,
            )
            if height is None:
                return
            base_size = (width, height)
            # Ask whether to fill the base with a colour
            fill_choice = messagebox.askyesno("Background", "Fill background with colour?")
            if fill_choice:
                colour = colorchooser.askcolor(title="Choose background colour")
                rgb = None
                if colour and colour[0]:
                    rgb = tuple(int(c) for c in colour[0])
                # If no colour chosen, default to white
                if rgb is None:
                    rgb = (255, 255, 255)
                base_img = Image.new("RGBA", base_size, rgb + (255,))
            else:
                base_img = Image.new("RGBA", base_size, (0, 0, 0, 0))
            # Create base layer
            base_layer_name = simpledialog.askstring("Layer Name", "Enter name for first layer:", initialvalue="Background")
            if not base_layer_name:
                base_layer_name = "Background"
            base_layer = Layer(base_img, base_layer_name)
            # Save history and add
            self._save_history()
            self.layers.append(base_layer)
            self.current_layer_index = 0
            # Set canvas size accordingly
            self.canvas.config(width=width, height=height)
            self._refresh_layer_list()
            self._update_composite()
            # If user only wanted a single base layer, exit now
            # Additional layers can be created by calling New Layer again.
            return
        else:
            base_size = self.layers[0].image.size
        # Ask for type of layer with expanded options
        layer_type = simpledialog.askstring(
            "New Layer",
            "Layer type (blank, color, gradient, pattern, noise, circle, ring):",
            initialvalue="blank",
        )
        if not layer_type:
            return
        layer_type = layer_type.strip().lower()
        new_img = None
        layer_name = simpledialog.askstring("Layer Name", "Enter layer name:", initialvalue=f"Layer {len(self.layers)}")
        if not layer_name:
            layer_name = f"Layer {len(self.layers)}"
        if layer_type == "blank":
            new_img = Image.new("RGBA", base_size, (0, 0, 0, 0))
        elif layer_type in ("color", "solid", "colour"):
            # Ask colour
            colour = colorchooser.askcolor(title="Choose fill colour")
            if not colour or not colour[0]:
                return
            rgb = tuple(int(c) for c in colour[0])
            new_img = Image.new("RGBA", base_size, rgb + (255,))
        elif layer_type == "gradient":
            # Ask orientation and colours
            orientation = simpledialog.askstring(
                "Gradient Orientation", "Enter orientation (horizontal or vertical):", initialvalue="horizontal"
            )
            if not orientation:
                return
            orientation = orientation.strip().lower()
            c1 = colorchooser.askcolor(title="Gradient start colour")
            if not c1 or not c1[0]:
                return
            c2 = colorchooser.askcolor(title="Gradient end colour")
            if not c2 or not c2[0]:
                return
            r1, g1, b1 = [int(v) for v in c1[0]]
            r2, g2, b2 = [int(v) for v in c2[0]]
            new_img = Image.new("RGBA", base_size)
            w, h = base_size
            if orientation.startswith("h"):
                # horizontal gradient left to right
                for x in range(w):
                    t = x / (w - 1) if w > 1 else 0
                    r = int(r1 + (r2 - r1) * t)
                    g = int(g1 + (g2 - g1) * t)
                    b = int(b1 + (b2 - b1) * t)
                    for y in range(h):
                        new_img.putpixel((x, y), (r, g, b, 255))
            else:
                # vertical gradient top to bottom
                for y in range(h):
                    t = y / (h - 1) if h > 1 else 0
                    r = int(r1 + (r2 - r1) * t)
                    g = int(g1 + (g2 - g1) * t)
                    b = int(b1 + (b2 - b1) * t)
                    for x in range(w):
                        new_img.putpixel((x, y), (r, g, b, 255))
        elif layer_type == "pattern":
            # Create a simple striped or checker pattern
            ptype = simpledialog.askstring(
                "Pattern Type", "Enter pattern (stripes or checker):", initialvalue="stripes"
            )
            if ptype is None:
                return
            ptype = ptype.strip().lower()
            stripe_width = simpledialog.askinteger(
                "Pattern Size",
                "Enter pattern size in pixels (default 20)",
                initialvalue=20,
                minvalue=1,
                maxvalue=200,
            )
            if not stripe_width:
                stripe_width = 20
            c = colorchooser.askcolor(title="Pattern colour")
            if not c or not c[0]:
                return
            pattern_color = tuple(int(v) for v in c[0])
            new_img = Image.new("RGBA", base_size, (0, 0, 0, 0))
            w, h = base_size
            for y in range(h):
                for x in range(w):
                    if ptype.startswith("str"):
                        band = ((x // stripe_width) % 2)
                    else:
                        band = (((x // stripe_width) + (y // stripe_width)) % 2)
                    if band == 0:
                        new_img.putpixel((x, y), pattern_color + (255,))
        elif layer_type == "noise":
            # Create random noise across the layer; ask for density (0-100)
            density = simpledialog.askinteger(
                "Noise Density", "Enter noise density percentage (0-100)", initialvalue=50, minvalue=0, maxvalue=100
            )
            if density is None:
                density = 50
            # Ask for foreground colour of noise
            fg_colour = colorchooser.askcolor(title="Noise colour (foreground)", initialcolor="#ffffff")
            if not fg_colour or not fg_colour[0]:
                return
            fr, fg, fb = [int(v) for v in fg_colour[0]]
            # Ask for background colour (for non-noise pixels)
            bg_colour = colorchooser.askcolor(title="Background colour (behind noise)", initialcolor="#000000")
            if not bg_colour or not bg_colour[0]:
                return
            br, bg_, bb = [int(v) for v in bg_colour[0]]
            new_img = Image.new("RGBA", base_size, (br, bg_, bb, 255))
            w, h = base_size
            for y in range(h):
                for x in range(w):
                    if random.randint(0, 100) < density:
                        new_img.putpixel((x, y), (fr, fg, fb, 255))
        elif layer_type == "circle":
            # Ask circle radius fraction and colour
            radius_ratio = simpledialog.askfloat(
                "Circle Size",
                "Enter circle radius as fraction of canvas (0-1)",
                initialvalue=0.3,
                minvalue=0.01,
                maxvalue=1.0,
            )
            if radius_ratio is None:
                return
            ccol = colorchooser.askcolor(title="Circle colour")
            if not ccol or not ccol[0]:
                return
            rc, gc, bc = [int(v) for v in ccol[0]]
            new_img = Image.new("RGBA", base_size, (0, 0, 0, 0))
            w, h = base_size
            # Determine radius
            rad = int(min(w, h) * radius_ratio)
            cx = w // 2
            cy = h // 2
            for y in range(h):
                for x in range(w):
                    if (x - cx) ** 2 + (y - cy) ** 2 <= rad ** 2:
                        new_img.putpixel((x, y), (rc, gc, bc, 255))
        elif layer_type == "ring":
            # Ask inner and outer radius fractions and colour
            outer_ratio = simpledialog.askfloat(
                "Outer Radius",
                "Enter outer radius fraction (0-1)",
                initialvalue=0.45,
                minvalue=0.01,
                maxvalue=1.0,
            )
            if outer_ratio is None:
                return
            inner_ratio = simpledialog.askfloat(
                "Inner Radius",
                "Enter inner radius fraction (0-1, less than outer)",
                initialvalue=0.3,
                minvalue=0.0,
                maxvalue=outer_ratio,
            )
            if inner_ratio is None:
                inner_ratio = 0.0
            ring_colour = colorchooser.askcolor(title="Ring colour")
            if not ring_colour or not ring_colour[0]:
                return
            rr, rg, rb = [int(v) for v in ring_colour[0]]
            new_img = Image.new("RGBA", base_size, (0, 0, 0, 0))
            w, h = base_size
            max_rad = int(min(w, h) * outer_ratio)
            min_rad = int(min(w, h) * inner_ratio)
            cx = w // 2
            cy = h // 2
            for y in range(h):
                for x in range(w):
                    dist2 = (x - cx) ** 2 + (y - cy) ** 2
                    if min_rad ** 2 < dist2 <= max_rad ** 2:
                        new_img.putpixel((x, y), (rr, rg, rb, 255))
        else:
            # Unknown type: create blank
            new_img = Image.new("RGBA", base_size, (0, 0, 0, 0))
        # Create and add layer
        layer = Layer(new_img, layer_name)
        self._save_history()
        self.layers.append(layer)
        self.current_layer_index = len(self.layers) - 1
        self._refresh_layer_list()
        self._update_composite()
    def _on_canvas_click_zoom(self, event) -> None:
        """Zoom into the area the user clicked (Ctrl+Click to zoom in, Shift+Click to zoom out).

        This centers the zoom on the click position rather than using mouse wheel.
        """
        # Only act when modifier keys are used to avoid interfering with drawing
        ctrl = (event.state & 0x4) != 0
        shift = (event.state & 0x1) != 0
        if ctrl and not shift:
            # Zoom in and center
            self.zoom = min(10.0, self.zoom * 1.5)
            # Optionally record focus point for advanced panning (not implemented fully)
            self._zoom_focus = (event.x, event.y)
            self._update_composite()
        elif shift and not ctrl:
            self.zoom = max(0.1, self.zoom / 1.5)
            self._update_composite()

    def _select_subject(self) -> None:
        """Auto-detect and create a selection mask for the main subject.

        Places mask into self._last_selection_mask and shows a preview.
        """
        if self.current_layer_index is None:
            messagebox.showinfo("Select Subject", "No layer selected.")
            return
        layer = self.layers[self.current_layer_index]
        img = getattr(layer, 'original', None) or getattr(layer, 'image', None)
        if img is None:
            messagebox.showinfo("Select Subject", "Selected layer has no image.")
            return
        mask = None
        try:
            import cv2
            arr = np.array(img.convert('RGB'))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            rects, weights = hog.detectMultiScale(gray, winStride=(8,8))
            m = np.zeros((img.height, img.width), dtype=np.uint8)
            for (x, y, w, h) in rects:
                cv2.rectangle(m, (x, y), (x+w, y+h), 255, -1)
            mask = Image.fromarray(m, mode='L')
        except Exception:
            # fallback saliency-based heuristic
            gray = img.convert('L')
            blurred = gray.filter(ImageFilter.GaussianBlur(radius=5))
            detail = ImageChops.difference(gray, blurred)
            detail = detail.point(lambda p: p * 2)
            mask = detail.point(lambda p: 255 if p > 30 else 0)
            mask = expand_selection(mask, pixels=8)
        self._last_selection_mask = mask
        # Preview overlay
        try:
            preview = img.convert('RGBA').copy()
            overlay = Image.new('RGBA', img.size, (255, 0, 0, 120))
            preview.paste(overlay, mask=mask)
            win = tk.Toplevel(self)
            win.title('Selection Preview')
            tk_img = ImageTk.PhotoImage(preview.resize((min(800, preview.width), int(preview.height * (min(800, preview.width)/preview.width)))))
            lbl = tk.Label(win, image=tk_img)
            lbl.image = tk_img
            lbl.pack()
        except Exception:
            pass

    def _expand_selection_cmd(self) -> None:
        if not hasattr(self, '_last_selection_mask') or self._last_selection_mask is None:
            messagebox.showinfo('Expand Selection', 'No active selection to expand.')
            return
        val = simpledialog.askinteger('Expand Selection', 'Expand by pixels:', initialvalue=8, minvalue=1, maxvalue=500)
        if val is None:
            return
        new_mask = expand_selection(self._last_selection_mask, pixels=val)
        self._last_selection_mask = new_mask
        messagebox.showinfo('Expand Selection', 'Selection expanded.')

    def _refine_edge_tool(self) -> None:
        """Interactive refine-edge dialog for the current selection mask."""

        if not hasattr(self, '_last_selection_mask') or self._last_selection_mask is None:
            messagebox.showinfo('Refine Edge', 'No active selection to refine.')
            return

        base_mask = self._last_selection_mask.convert('L')
        refined_mask = base_mask.copy()

        if self.current_layer_index is not None and isinstance(self.layers[self.current_layer_index], Layer):
            layer = self.layers[self.current_layer_index]
            preview_base = layer.image.copy()
        else:
            preview_base = Image.new('RGBA', base_mask.size, (200, 200, 200, 255))

        display_img = preview_base.copy()
        max_dim = 480
        scale = min(max_dim / display_img.width, max_dim / display_img.height, 1.0)
        if scale < 1.0:
            try:
                resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
            except Exception:
                resample = Image.ANTIALIAS
            display_img = display_img.resize((int(display_img.width * scale), int(display_img.height * scale)), resample=resample)

        win = tk.Toplevel(self)
        win.title('Refine Edge')
        win.configure(bg='#2a2a2a')

        preview_label = tk.Label(win, bg='#2a2a2a')
        preview_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        smooth_var = tk.IntVar(value=2)
        feather_var = tk.IntVar(value=8)
        contrast_var = tk.DoubleVar(value=0.0)
        shift_var = tk.IntVar(value=0)

        def update_preview(*_args) -> None:
            nonlocal refined_mask
            refined_mask = refine_selection_edges(
                base_mask,
                smooth=max(0, smooth_var.get()),
                feather=max(0, feather_var.get()),
                contrast=contrast_var.get(),
                shift=shift_var.get(),
            )
            mask_for_preview = refined_mask
            if mask_for_preview.size != display_img.size:
                try:
                    resample = Image.Resampling.BILINEAR  # type: ignore[attr-defined]
                except Exception:
                    resample = Image.NEAREST
                mask_for_preview = mask_for_preview.resize(display_img.size, resample=resample)
            overlay = Image.new('RGBA', display_img.size, (0, 180, 255, 120))
            overlay.putalpha(mask_for_preview)
            comp = Image.alpha_composite(display_img.convert('RGBA'), overlay)
            photo = ImageTk.PhotoImage(comp)
            preview_label.configure(image=photo)
            preview_label.image = photo

        controls = tk.Frame(win, bg='#2a2a2a')
        controls.grid(row=1, column=0, padx=10, pady=(0, 10))

        tk.Label(controls, text='Smooth', fg='white', bg='#2a2a2a').grid(row=0, column=0, sticky='w')
        tk.Scale(controls, from_=0, to=20, orient=tk.HORIZONTAL, variable=smooth_var, command=update_preview,
                 length=200, bg='#3a3a3a', fg='white', troughcolor='#1a1a1a', highlightthickness=0).grid(row=0, column=1, padx=6)

        tk.Label(controls, text='Feather', fg='white', bg='#2a2a2a').grid(row=1, column=0, sticky='w')
        tk.Scale(controls, from_=0, to=60, orient=tk.HORIZONTAL, variable=feather_var, command=update_preview,
                 length=200, bg='#3a3a3a', fg='white', troughcolor='#1a1a1a', highlightthickness=0).grid(row=1, column=1, padx=6)

        tk.Label(controls, text='Edge Contrast', fg='white', bg='#2a2a2a').grid(row=2, column=0, sticky='w')
        tk.Scale(controls, from_=-100, to=100, orient=tk.HORIZONTAL, resolution=1, variable=contrast_var, command=update_preview,
                 length=200, bg='#3a3a3a', fg='white', troughcolor='#1a1a1a', highlightthickness=0).grid(row=2, column=1, padx=6)

        tk.Label(controls, text='Shift Edge', fg='white', bg='#2a2a2a').grid(row=3, column=0, sticky='w')
        tk.Scale(controls, from_=-40, to=40, orient=tk.HORIZONTAL, variable=shift_var, command=update_preview,
                 length=200, bg='#3a3a3a', fg='white', troughcolor='#1a1a1a', highlightthickness=0).grid(row=3, column=1, padx=6)

        btns = tk.Frame(win, bg='#2a2a2a')
        btns.grid(row=1, column=1, padx=10, pady=(0, 10))

        def commit_selection() -> None:
            self._last_selection_mask = refined_mask.copy()
            self._set_status('Selection refined')
            win.destroy()

        def apply_to_layer() -> None:
            if self.current_layer_index is None or not isinstance(self.layers[self.current_layer_index], Layer):
                messagebox.showinfo('Apply Mask', 'Select a raster layer to update its mask.')
                return
            self._save_history()
            layer = self.layers[self.current_layer_index]
            layer.mask = refined_mask.copy()
            layer.apply_adjustments()
            self._update_composite()
            self._last_selection_mask = refined_mask.copy()
            self._set_status('Refined edge mask applied to layer')
            win.destroy()

        tk.Button(btns, text='Apply Selection', command=commit_selection, bg='#4a4a4a', fg='white').pack(fill=tk.X, pady=4)
        tk.Button(btns, text='Apply to Layer Mask', command=apply_to_layer, bg='#4a4a4a', fg='white').pack(fill=tk.X, pady=4)
        tk.Button(btns, text='Cancel', command=win.destroy, bg='#4a4a4a', fg='white').pack(fill=tk.X, pady=4)

        update_preview()

    def _make_cutline_cmd(self) -> None:
        if not hasattr(self, '_last_selection_mask') or self._last_selection_mask is None:
            messagebox.showinfo('Make Cutline', 'No selection available.')
            return
        paths = make_cutline(self._last_selection_mask)
        if not paths:
            messagebox.showinfo('Make Cutline', 'No contours found.')
            return
        save = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON','*.json'), ('Text','*.txt')], title='Save cutline as...')
        if save:
            try:
                with open(save, 'w', encoding='utf-8') as fh:
                    json.dump(paths, fh)
                messagebox.showinfo('Make Cutline', f'Cutline saved to {save}')
            except Exception as e:
                messagebox.showerror('Make Cutline', f'Failed to save: {e}')

    def _new_vector_layer(self):
        """Create a new vector layer."""
        if not self.layers:
            messagebox.showwarning("No Base Layer", "Please create a base layer first.")
            return
            
        # Get base size from existing layers
        base_size = self.layers[0].image.size if isinstance(self.layers[0], Layer) else (800, 600)
        
        layer_name = simpledialog.askstring("Vector Layer Name", "Enter layer name:", initialvalue=f"Vector Layer {len(self.layers)}")
        if not layer_name:
            layer_name = f"Vector Layer {len(self.layers)}"
            
        vector_layer = VectorLayer(layer_name, base_size[0], base_size[1])
        
        self._save_history()
        self.layers.append(vector_layer)
        self.current_layer_index = len(self.layers) - 1
        self._refresh_layer_list()
        self._update_composite()

    def _new_adjustment_layer(self):
        """Create a new adjustment layer."""
        if not self.layers:
            messagebox.showwarning("No Base Layer", "Please create a base layer first.")
            return
            
        # Show adjustment type selection dialog
        adjustment_types = [
            "brightness", "contrast", "saturation", "hue_shift", 
            "levels", "color_balance", "vibrance"
        ]
        
        # Create a simple dialog for adjustment type selection
        dialog = tk.Toplevel(self)
        dialog.title("New Adjustment Layer")
        dialog.geometry("300x200")
        dialog.transient(self)
        dialog.grab_set()
        
        tk.Label(dialog, text="Select adjustment type:", font=("Arial", 12, "bold")).pack(pady=10)
        
        selected_type = tk.StringVar(value=adjustment_types[0])
        for adj_type in adjustment_types:
            tk.Radiobutton(dialog, text=adj_type.replace("_", " ").title(), 
                          variable=selected_type, value=adj_type).pack(anchor=tk.W, padx=20)
        
        layer_name = tk.StringVar(value="Adjustment Layer")
        tk.Label(dialog, text="Layer name:").pack(pady=(10, 0))
        tk.Entry(dialog, textvariable=layer_name, width=20).pack(pady=5)
        
        def create_adjustment():
            adj_type = selected_type.get()
            name = layer_name.get() or f"Adjustment Layer {len(self.layers)}"
            
            # Create default parameters based on adjustment type
            params = {}
            if adj_type == "brightness":
                params = {"brightness": 1.0}
            elif adj_type == "contrast":
                params = {"contrast": 1.0}
            elif adj_type == "saturation":
                params = {"saturation": 1.0}
            elif adj_type == "hue_shift":
                params = {"hue_shift": 0.0}
            elif adj_type == "levels":
                params = {"black_point": 0, "white_point": 255, "gamma": 1.0}
            elif adj_type == "color_balance":
                params = {"red_shift": 0.0, "green_shift": 0.0, "blue_shift": 0.0}
            elif adj_type == "vibrance":
                params = {"vibrance": 0.0}
            
            adjustment_layer = AdjustmentLayer(name, adj_type, params)
            
            self._save_history()
            self.layers.append(adjustment_layer)
            self.current_layer_index = len(self.layers) - 1
            self._refresh_layer_list()
            self._update_composite()
            
            dialog.destroy()
        
        def cancel():
            dialog.destroy()
        
        tk.Button(dialog, text="Create", command=create_adjustment).pack(side=tk.LEFT, padx=20, pady=20)
        tk.Button(dialog, text="Cancel", command=cancel).pack(side=tk.RIGHT, padx=20, pady=20)

    def _open_image(self):
        """Load an image from disk and create a new layer with it.

        The selected file will be opened using Pillow.  If this is the
        first layer being added, the canvas size will be set to the
        image's dimensions.  Additional images can be loaded as
        separate layers on top.
        """
        filetypes = [("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files", "*.*")]
        filepath = filedialog.askopenfilename(title="Open Image", filetypes=filetypes)
        if not filepath:
            return
        try:
            image = Image.open(filepath)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image: {e}")
            return
        # Convert to RGBA to support transparency
        image = image.convert("RGBA")
        layer_name = os.path.basename(filepath)
        layer = Layer(image, layer_name)
        self._save_history()
        self.layers.append(layer)
        self.current_layer_index = len(self.layers) - 1
        # If this is the first layer, set canvas size
        if len(self.layers) == 1:
            self.canvas.config(width=image.width, height=image.height)
        self._refresh_layer_list()
        self._update_composite()
        self._set_status(f"Opened {os.path.basename(filepath)}")

    def _save_image(self):
        """Save the current composite image to disk."""
        if not self.layers:
            messagebox.showinfo("Nothing to save", "There is no image to save.")
            return
        filetypes = [("PNG", "*.png"), ("JPEG", "*.jpg"), ("All files", "*.*")]
        filepath = filedialog.asksaveasfilename(title="Save Image", defaultextension=".png", filetypes=filetypes)
        if not filepath:
            return
        composite = self._create_composite_image(preview=getattr(self, 'preview_mode', False))
        # Offer DPI metadata change
        try:
            change = messagebox.askyesno('DPI', 'Change output DPI metadata?')
        except Exception:
            change = False

        try:
            if change:
                cur_dpi = get_image_ppi_from_dpi(composite)
                dx = simpledialog.askinteger('DPI - Horizontal', 'Horizontal DPI (pixels per inch):', initialvalue=cur_dpi[0], minvalue=1)
                if dx is None:
                    return
                dy = simpledialog.askinteger('DPI - Vertical', 'Vertical DPI (pixels per inch):', initialvalue=cur_dpi[1], minvalue=1)
                if dy is None:
                    return
                try:
                    composite.save(filepath, dpi=(int(dx), int(dy)))
                except Exception:
                    composite = set_image_dpi(composite, (int(dx), int(dy)))
                    composite.save(filepath)
            else:
                composite.save(filepath)
            messagebox.showinfo("Saved", f"Image saved to {filepath}")
            self._set_status(f"Saved composite to {os.path.basename(filepath)}")
        except Exception as e:
            # final fallback
            try:
                composite.save(filepath)
                messagebox.showinfo("Saved", f"Image saved to {filepath}")
                self._set_status(f"Saved composite to {os.path.basename(filepath)}")
            except Exception as e2:
                messagebox.showerror("Error", f"Could not save image: {e2}")

    def _quick_export(self) -> None:
        """Export the composite image to multiple resolution presets in one step."""

        if not self.layers:
            messagebox.showinfo("Quick Export", "There is no image to export.")
            return
        filepath = filedialog.asksaveasfilename(
            title="Quick Export Base Filename",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All files", "*.*")],
        )
        if not filepath:
            return
        composite = self._create_composite_image(preview=getattr(self, 'preview_mode', False))
        if composite is None:
            messagebox.showerror("Quick Export", "Unable to generate composite image.")
            return

        base_path = Path(filepath)
        try:
            resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
        except Exception:
            resample = Image.ANTIALIAS

        exports = [("full", 1.0), ("medium", 0.5), ("small", 0.25)]
        try:
            for idx, (suffix, scale) in enumerate(exports):
                if scale == 1.0:
                    image = composite
                else:
                    new_size = (max(1, int(composite.width * scale)), max(1, int(composite.height * scale)))
                    image = composite.resize(new_size, resample=resample)
                if idx == 0:
                    out_path = base_path
                else:
                    out_path = base_path.with_name(f"{base_path.stem}-{suffix}{base_path.suffix}")
                image.save(out_path)
        except Exception as exc:
            messagebox.showerror("Quick Export", f"Export failed: {exc}")
            return

        self._set_status(f"Quick export variants saved next to {base_path.name}")

    def _delete_layer(self):
        """Delete the currently selected layer."""
        if self.current_layer_index is None:
            return
        self._save_history()
        del self.layers[self.current_layer_index]
        # Adjust selected index
        if not self.layers:
            self.current_layer_index = None
            self.canvas.delete("all")
        else:
            self.current_layer_index = max(0, self.current_layer_index - 1)
        self._refresh_layer_list()
        self._update_composite()

    def _show_session_log(self) -> None:
        """Display a read-only window with the current session log."""

        if not self.session_log:
            messagebox.showinfo("Session Log", "No actions have been logged yet.")
            return
        top = tk.Toplevel(self)
        top.title("Session Log")
        top.geometry("600x360")
        text = scrolledtext.ScrolledText(top, wrap=tk.WORD, state="normal")
        text.pack(fill=tk.BOTH, expand=True)
        text.insert("end", "\n".join(self.session_log))
        text.config(state="disabled")
        ttk.Button(top, text="Close", command=top.destroy).pack(pady=6)

    def _show_histogram(self) -> None:
        """Display an RGB histogram of the current composite image."""

        composite = self._create_composite_image(preview=getattr(self, 'preview_mode', False))
        if composite is None:
            messagebox.showinfo("Histogram", "No image available for histogram analysis.")
            return
        hist_img = render_histogram_image(composite)
        top = tk.Toplevel(self)
        top.title("Histogram")
        photo = ImageTk.PhotoImage(hist_img)
        label = ttk.Label(top, image=photo)
        label.image = photo  # keep reference
        label.pack(padx=12, pady=12)
        ttk.Button(top, text="Close", command=top.destroy).pack(pady=(0, 12))
        self._set_status("Histogram generated")

    def _generate_contact_sheet(self) -> None:
        """Create a contact sheet image from all visible raster layers."""

        raster_images = [layer.image for layer in self.layers if isinstance(layer, Layer) and layer.visible]
        if not raster_images:
            messagebox.showinfo("Contact Sheet", "No visible raster layers to include.")
            return
        try:
            sheet = build_contact_sheet(raster_images)
        except Exception as exc:
            messagebox.showerror("Contact Sheet", f"Failed to create contact sheet: {exc}")
            return
        filepath = filedialog.asksaveasfilename(
            title="Save Contact Sheet",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All files", "*.*")],
        )
        if not filepath:
            return
        try:
            sheet.save(filepath)
            self._set_status(f"Contact sheet saved to {os.path.basename(filepath)}")
        except Exception as exc:
            messagebox.showerror("Contact Sheet", f"Could not save contact sheet: {exc}")

    def _merge_visible_layers(self):
        """Merge all visible layers into a single layer.

        Hidden layers are left untouched.  This is useful to flatten
        completed portions of your composition so that you can continue
        editing with a simpler layer stack.
        """
        if not self.layers:
            return
        # Create composite of visible layers
        composite = self._create_composite_image(include_hidden=False)
        # Remove all visible layers and insert merged one
        new_layers = []
        for layer in self.layers:
            if not layer.visible:
                new_layers.append(layer)
        self._save_history()
        merged_layer = Layer(composite, "Merged")
        new_layers.append(merged_layer)
        self.layers = new_layers
        self.current_layer_index = len(self.layers) - 1
        self._refresh_layer_list()
        self._update_composite()

    def _toggle_visibility(self):
        """Toggle the visibility of the currently selected layer."""
        if self.current_layer_index is None:
            return
        layer = self.layers[self.current_layer_index]
        self._save_history()
        layer.visible = not layer.visible
        self._refresh_layer_list()
        self._update_composite()

    def _refresh_layer_list(self):
        """Update the layer list UI to reflect current layers."""
        self.layer_listbox.delete(0, tk.END)
        for idx, layer in enumerate(self.layers):
            visibility = "👁️ " if layer.visible else "🚫 "
            name = visibility + layer.name
            self.layer_listbox.insert(tk.END, name)
        if self.current_layer_index is not None:
            self.layer_listbox.select_set(self.current_layer_index)
        self.layer_listbox.update_idletasks()

    # ------------------------------------------------------------------
    # Layer grouping
    # ------------------------------------------------------------------
    def _group_selected_layers(self) -> None:
        """Group selected layers by merging them into a single composite layer."""
        # Get selected indices from listbox
        indices = list(self.layer_listbox.curselection())
        if len(indices) < 2:
            messagebox.showinfo("Group Layers", "Select two or more layers to group.")
            return
        # Sort ascending
        indices.sort()
        # Save history
        self._current_action_desc = "Group Layers"
        self._save_history()
        # Determine base size
        if not self.layers:
            return
        base_w, base_h = self.layers[0].original.size
        composite = Image.new("RGBA", (base_w, base_h), (0, 0, 0, 0))
        # Composite selected layers from bottom to top
        for idx in indices:
            layer = self.layers[idx]
            layer.apply_adjustments()
            composite.paste(layer.image, layer.offset, layer.image)
        # Ask group name
        name = simpledialog.askstring("Group Name", "Enter name for the new group:", initialvalue="Group")
        if not name:
            name = "Group"
        new_layer = Layer(composite, name)
        new_layer.offset = (0, 0)
        # Remove selected layers (from highest index down)
        for idx in reversed(indices):
            del self.layers[idx]
        # Append new layer and update index
        self.layers.append(new_layer)
        self.current_layer_index = len(self.layers) - 1
        self._refresh_layer_list()
        self._update_composite()

    # ------------------------------------------------------------------
    # Layer drag-and-drop reordering
    # ------------------------------------------------------------------
    def _on_layer_drag_start(self, event) -> None:
        """Record the initial index of a layer for drag-and-drop."""
        self._dragging_layer_index = self.layer_listbox.nearest(event.y)
        # Save history once at start of drag
        self._current_action_desc = "Reorder Layers"
        self._save_history()

    def _on_layer_drag_motion(self, event) -> None:
        """Handle drag motion to reorder layers."""
        if not hasattr(self, '_dragging_layer_index'):
            return
        new_index = self.layer_listbox.nearest(event.y)
        old_index = getattr(self, '_dragging_layer_index')
        if new_index == old_index:
            return
        # Move layer in list
        layer = self.layers.pop(old_index)
        self.layers.insert(new_index, layer)
        # Update drag index and current selection
        self._dragging_layer_index = new_index
        self.current_layer_index = new_index
        # Refresh list and composite
        self._refresh_layer_list()
        self._update_composite()
        # Update listbox selection
        self.layer_listbox.selection_clear(0, tk.END)
        self.layer_listbox.selection_set(new_index)

    # ------------------------------------------------------------------
    # Auto adjustments
    # ------------------------------------------------------------------
    def _auto_enhance_layer(self) -> None:
        """Automatically enhance brightness, contrast, color and sharpness of the current layer."""
        if self.current_layer_index is None:
            return
        layer = self.layers[self.current_layer_index]
        # Save history with description
        self._current_action_desc = "Auto Enhance"
        self._save_history()
        img = layer.original
        # Use Pillow's ImageEnhance modules
        try:
            from PIL import ImageEnhance
        except ImportError:
            messagebox.showerror("Error", "ImageEnhance module is not available.")
            return
        # Sequential enhancements
        img = ImageEnhance.Brightness(img).enhance(1.1)
        img = ImageEnhance.Contrast(img).enhance(1.15)
        img = ImageEnhance.Color(img).enhance(1.1)
        img = ImageEnhance.Sharpness(img).enhance(1.05)
        layer.original = img
        layer.apply_adjustments()
        self._update_composite()

    def _one_click_beauty(self) -> None:
        """Apply a skin smoothing filter and mild enhancements for a beauty effect."""
        if self.current_layer_index is None:
            return
        layer = self.layers[self.current_layer_index]
        self._current_action_desc = "One Click Beauty"
        self._save_history()
        # Apply skin smoothing filter
        try:
            layer.apply_filter("skin smooth")
        except Exception:
            pass
        # Enhance brightness and contrast and color slightly
        try:
            from PIL import ImageEnhance
        except ImportError:
            pass
        img = layer.original
        img = ImageEnhance.Brightness(img).enhance(1.05)
        img = ImageEnhance.Contrast(img).enhance(1.05)
        img = ImageEnhance.Color(img).enhance(1.02)
        layer.original = img
        layer.apply_adjustments()
        self._update_composite()

    def _remove_background(self) -> None:
        """Automatically remove a uniform background from the current layer using colour similarity.

        This function samples the border of the image to estimate the background colour and
        removes pixels within the specified tolerance by making them transparent.  Useful
        for quickly isolating a subject against a plain background.  Large tolerance values
        will remove more colours.
        """
        if self.current_layer_index is None:
            return
        layer = self.layers[self.current_layer_index]
        tol = simpledialog.askinteger(
            "Remove Background",
            "Enter colour tolerance (0-255):",
            initialvalue=30,
            minvalue=0,
            maxvalue=255,
        )
        if tol is None:
            return
        # Save history
        self._current_action_desc = "Remove Background"
        self._save_history()
        # Convert to numpy array
        img = layer.original
        arr = np.array(img)
        # Estimate background colour as average of border pixels (top, bottom, left, right)
        top = arr[0, :, :3]
        bottom = arr[-1, :, :3]
        left = arr[:, 0, :3]
        right = arr[:, -1, :3]
        border = np.concatenate((top, bottom, left, right), axis=0)
        base_color = border.mean(axis=0)
        # Compute difference from base colour
        diff = np.abs(arr[:, :, :3].astype(float) - base_color)
        diff_sum = diff.sum(axis=2)
        mask = (diff_sum <= tol).astype(np.uint8)
        # Set alpha channel to zero where mask is 1
        alpha = arr[:, :, 3]
        alpha[mask == 1] = 0
        arr[:, :, 3] = alpha
        new_img = Image.fromarray(arr)
        layer.original = new_img
        layer.apply_adjustments()
        self._update_composite()

    # ------------------------------------------------------------------
    # Perspective transform
    # ------------------------------------------------------------------
    def _perspective_transform_layer(self) -> None:
        """Apply a perspective warp to the current layer based on user-defined destination corners."""
        if self.current_layer_index is None:
            return
        layer = self.layers[self.current_layer_index]
        w, h = layer.original.size
        prompt = (
            "Enter destination coordinates for the four corners of the layer as 8 values:\n"
            "Format: x0,y0,x1,y1,x2,y2,x3,y3 where:\n"
            "(x0,y0) = top-left, (x1,y1) = top-right, (x2,y2) = bottom-right, (x3,y3) = bottom-left.\n"
            "You can specify values between 0 and 1 to indicate a fraction of width/height."
        )
        s = simpledialog.askstring("Perspective Transform", prompt, initialvalue="0,0,1,0,1,1,0,1")
        if not s:
            return
        try:
            parts = [float(x.strip()) for x in s.replace(';', ',').split(',')]
        except Exception:
            messagebox.showerror("Error", "Invalid coordinates format.")
            return
        if len(parts) != 8:
            messagebox.showerror("Error", "You must provide exactly 8 numbers.")
            return
        dest = []
        for i in range(0, 8, 2):
            dx = parts[i]
            dy = parts[i + 1]
            if -1.0 <= dx <= 1.0:
                x_val = dx * w
            else:
                x_val = dx
            if -1.0 <= dy <= 1.0:
                y_val = dy * h
            else:
                y_val = dy
            dest.append((x_val, y_val))
        src = [(0, 0), (w, 0), (w, h), (0, h)]
        # Compute coefficients
        try:
            def _find_coeffs(pa, pb):
                matrix = []
                for p1, p2 in zip(pa, pb):
                    matrix.append([p2[0], p2[1], 1, 0, 0, 0, -p1[0] * p2[0], -p1[0] * p2[1]])
                    matrix.append([0, 0, 0, p2[0], p2[1], 1, -p1[1] * p2[0], -p1[1] * p2[1]])
                A = np.array(matrix)
                B = np.array(pa).reshape(8)
                res = np.linalg.solve(A, B)
                return res
            coeffs = _find_coeffs(src, dest)
        except Exception as e:
            messagebox.showerror("Error", f"Could not compute transform: {e}")
            return
        # Save history and apply transform
        self._current_action_desc = "Perspective Transform"
        self._save_history()
        try:
            new_img = layer.original.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
        except Exception as e:
            messagebox.showerror("Error", f"Transform failed: {e}")
            return
        layer.original = new_img
        layer.apply_adjustments()
        self._update_composite()

    def _warp_layer(self) -> None:
        """Apply a simple sine-based warp to the current layer.

        This transformation shifts each row of the image horizontally by
        an amount proportional to a sine wave.  The user specifies the
        amplitude (maximum shift in pixels) and the period (vertical
        wavelength) of the sine.  Positive amplitude shifts rows to the
        right; negative amplitude shifts to the left.
        """
        if self.current_layer_index is None:
            return
        layer = self.layers[self.current_layer_index]
        w, h = layer.original.size
        amp = simpledialog.askfloat(
            "Warp", "Enter warp amplitude in pixels (e.g. 20):", initialvalue=20.0
        )
        if amp is None:
            return
        period = simpledialog.askfloat(
            "Warp", "Enter warp period (vertical wavelength in pixels):", initialvalue=100.0
        )
        if period is None or period == 0:
            return
        # Save history
        self._current_action_desc = "Warp"
        self._save_history()
        # Create a new blank image
        new_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        src_pixels = layer.original.load()
        dst_pixels = new_img.load()
        for y in range(h):
            # Compute horizontal offset for this row
            shift = int(amp * math.sin(2 * math.pi * y / period))
            for x in range(w):
                src_x = x - shift
                if 0 <= src_x < w:
                    dst_pixels[x, y] = src_pixels[src_x, y]
                else:
                    dst_pixels[x, y] = (0, 0, 0, 0)
        layer.original = new_img
        layer.apply_adjustments()
        self._update_composite()

    def _face_ratio_adjust_layer(self) -> None:
        """Adjust the horizontal and vertical scaling of the current layer.

        This can be used to change the aspect ratio of a face or object,
        stretching or squashing it in the x and y directions.
        """
        if self.current_layer_index is None:
            return
        layer = self.layers[self.current_layer_index]
        w, h = layer.original.size
        # Ask for horizontal and vertical scale factors
        hs = simpledialog.askfloat(
            "Face Ratio Adjust", "Enter horizontal scale factor (e.g. 1.2 means widen):", initialvalue=1.0
        )
        if hs is None or hs <= 0:
            return
        vs = simpledialog.askfloat(
            "Face Ratio Adjust", "Enter vertical scale factor (e.g. 0.9 means shorten):", initialvalue=1.0
        )
        if vs is None or vs <= 0:
            return
        # Save history
        self._current_action_desc = "Face Ratio Adjust"
        self._save_history()
        new_w = max(1, int(w * hs))
        new_h = max(1, int(h * vs))
        try:
            resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
        except Exception:
            resample = Image.ANTIALIAS
        new_img = layer.original.resize((new_w, new_h), resample)
        layer.original = new_img
        layer.apply_adjustments()
        self._update_composite()

    def _free_distort_layer(self) -> None:
        """Apply a free distort transform to the current layer.

        This is similar to a perspective transform but allows arbitrary
        destination coordinates for each corner.  The user provides
        eight numbers specifying the new positions of the top-left,
        top-right, bottom-right and bottom-left corners.  Values
        between -1 and 1 are interpreted as fractions of the layer's
        width and height, while values outside that range are treated
        as absolute pixel coordinates.
        """
        if self.current_layer_index is None:
            return
        layer = self.layers[self.current_layer_index]
        w, h = layer.original.size
        prompt = (
            "Enter destination coordinates for the four corners of the layer as 8 values:\n"
            "Format: x0,y0,x1,y1,x2,y2,x3,y3 where:\n"
            "(x0,y0) = top-left, (x1,y1) = top-right, (x2,y2) = bottom-right, (x3,y3) = bottom-left.\n"
            "Values between -1 and 1 are treated as fractions of width/height; others as pixels."
        )
        s = simpledialog.askstring("Free Distort", prompt, initialvalue="0,0,1,0,1,1,0,1")
        if not s:
            return
        try:
            parts = [float(x.strip()) for x in s.replace(';', ',').split(',')]
        except Exception:
            messagebox.showerror("Error", "Invalid coordinates format.")
            return
        if len(parts) != 8:
            messagebox.showerror("Error", "You must provide exactly 8 numbers.")
            return
        dest = []
        for i in range(0, 8, 2):
            dx = parts[i]
            dy = parts[i + 1]
            # Interpret fractional values as proportion of size
            if -1.0 <= dx <= 1.0:
                x_val = dx * w
            else:
                x_val = dx
            if -1.0 <= dy <= 1.0:
                y_val = dy * h
            else:
                y_val = dy
            dest.append((x_val, y_val))
        src = [(0, 0), (w, 0), (w, h), (0, h)]
        # Compute transformation coefficients using numpy
        try:
            import numpy as np
            def _find_coeffs(pa, pb):
                matrix = []
                for p1, p2 in zip(pa, pb):
                    matrix.append([p2[0], p2[1], 1, 0, 0, 0, -p1[0] * p2[0], -p1[0] * p2[1]])
                    matrix.append([0, 0, 0, p2[0], p2[1], 1, -p1[1] * p2[0], -p1[1] * p2[1]])
                A = np.array(matrix)
                B = np.array(pa).reshape(8)
                res = np.linalg.solve(A, B)
                return res
            coeffs = _find_coeffs(src, dest)
        except Exception as e:
            messagebox.showerror("Error", f"Could not compute transform: {e}")
            return
        # Save history and apply transform
        self._current_action_desc = "Free Distort"
        self._save_history()
        try:
            new_img = layer.original.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
        except Exception as e:
            messagebox.showerror("Error", f"Transform failed: {e}")
            return
        layer.original = new_img
        layer.apply_adjustments()
        self._update_composite()

    def _scale_layer(self, factor: float) -> None:
        """Scale the current layer up or down by the given factor.

        A factor greater than 1 enlarges the layer; less than 1 shrinks it.  The layer's
        offset is scaled accordingly so that relative positioning remains consistent.
        """
        if self.current_layer_index is None:
            return
        layer = self.layers[self.current_layer_index]
        if factor <= 0:
            messagebox.showerror("Scale", "Scale factor must be positive.")
            return
        w, h = layer.original.size
        new_w = max(1, int(w * factor))
        new_h = max(1, int(h * factor))
        # Save history
        self._current_action_desc = f"Scale {factor}×"
        self._save_history()
        # If this is a TextLayer, keep it editable: scale font size and re-render
        try:
            resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
        except Exception:
            resample = Image.ANTIALIAS
        if isinstance(layer, TextLayer):
            # Scale font size and position
            new_font_size = max(1, int(layer.font_size * factor))
            layer.font_size = new_font_size
            ox, oy = layer.offset
            layer.offset = (int(ox * factor), int(oy * factor))
            # Scale text position
            px, py = layer.position
            layer.position = (int(px * factor), int(py * factor))
            # Re-render text
            try:
                layer.render_text()
            except Exception:
                # fallback to resizing rasterized original
                layer.original = layer.original.resize((new_w, new_h), resample)
                try:
                    layer.mask = layer.mask.resize((new_w, new_h), resample)
                except Exception:
                    pass
                layer.apply_adjustments()
        else:
            # Resize original raster image
            layer.original = layer.original.resize((new_w, new_h), resample)
            # Scale mask as well
            try:
                layer.mask = layer.mask.resize((new_w, new_h), resample)
            except Exception:
                pass
            # Scale offset proportionally
            ox, oy = layer.offset
            layer.offset = (int(ox * factor), int(oy * factor))
            # Apply adjustments and refresh
            layer.apply_adjustments()
        self._update_composite()

    # ------------------------------------------------------------------
    # Theme and autosave
    # ------------------------------------------------------------------
    def _apply_theme(self) -> None:
        """Apply the current theme (dark or light) to key interface elements."""
        # Configure ttk styles
        style = ttk.Style(self)
        try:
            # Use a modern theme
            available_themes = style.theme_names()
            if 'vista' in available_themes:
                style.theme_use('vista')
            elif 'clam' in available_themes:
                style.theme_use('clam')
        except Exception:
            pass

        # Configure ttk styles for dark/light mode
        if self.dark_mode:
            style.configure('TFrame', background='#2e2e2e')
            style.configure('TLabel', background='#2e2e2e', foreground='#e0e0e0')
            style.configure('TButton', background='#4c4c4c', foreground='#e0e0e0')
            style.map('TButton', background=[('active', '#5c5c5c'), ('pressed', '#3c3c3c')])
            style.configure('TEntry', fieldbackground='#3a3a3a', foreground='#e0e0e0', insertcolor='#e0e0e0')
            style.configure('TListbox', background='#4c4c4c', foreground='#e0e0e0')
            style.configure('TScale', background='#2e2e2e', troughcolor='#4c4c4c')
        else:
            style.configure('TFrame', background='#fafafa')
            style.configure('TLabel', background='#fafafa', foreground='#333333')
            style.configure('TButton', background='#e6e6e6', foreground='#333333')
            style.map('TButton', background=[('active', '#d6d6d6'), ('pressed', '#c6c6c6')])
            style.configure('TEntry', fieldbackground='#ffffff', foreground='#333333', insertcolor='#333333')
            style.configure('TListbox', background='#ffffff', foreground='#333333')
            style.configure('TScale', background='#fafafa', troughcolor='#f5f5f5')

        # Define colour schemes for light and dark modes
        if self.dark_mode:
            bg_root = "#2e2e2e"
            panel_bg = "#3a3a3a"
            toolbar_bg = "#3a3a3a"
            btn_bg = "#4c4c4c"
            btn_fg = "#e0e0e0"
            slider_bg = "#4c4c4c"
            slider_fg = "#e0e0e0"
            label_bg = panel_bg
            label_fg = "#e0e0e0"
        else:
            bg_root = "#fafafa"
            panel_bg = "#f9f9f9"
            toolbar_bg = "#f1f1f1"
            btn_bg = "#e6e6e6"
            btn_fg = "#333333"
            slider_bg = "#f5f5f5"
            slider_fg = "#333333"
            label_bg = panel_bg
            label_fg = "#333333"
        # Apply root background
        self.configure(bg=bg_root)
        # Update major frames if they exist
        try:
            self.left_frame.config(bg=panel_bg)
            self.canvas.config(bg=toolbar_bg)
        except Exception:
            pass
        # Update existing tool buttons and slider backgrounds
        for name, btn in getattr(self, 'tool_buttons', {}).items():
            btn.config(bg=btn_bg, fg=btn_fg, activebackground=btn_bg)
        # Update layer listbox if it exists
        if hasattr(self, 'layer_listbox') and self.layer_listbox is not None:
            try:
                self.layer_listbox.config(bg="#ffffff" if not self.dark_mode else "#4c4c4c", fg=btn_fg)
            except Exception:
                pass
        # Update sliders and labels if they exist
        slider_names = ['alpha_slider', 'brightness_slider', 'contrast_slider', 'color_slider', 'gamma_slider', 'red_slider', 'green_slider', 'blue_slider']
        for name in slider_names:
            widget = getattr(self, name, None)
            if widget:
                try:
                    widget.config(bg=slider_bg, fg=slider_fg, troughcolor=slider_bg, highlightthickness=0)
                except Exception:
                    pass
        # Force a refresh
        try:
            self.update_idletasks()
        except Exception:
            pass

        # Save preference
        self._save_theme_preference()

    def _toggle_theme(self) -> None:
        """Toggle between dark and light modes."""
        self.dark_mode = not self.dark_mode
        self._apply_theme()

    # ------------------------------------------------------------------
    # Theme persistence helpers
    # ------------------------------------------------------------------
    def _theme_pref_file(self) -> str:
        try:
            base = os.path.join(os.path.expanduser("~"), ".image_editor_prefs")
            os.makedirs(base, exist_ok=True)
            return os.path.join(base, "prefs.json")
        except Exception:
            return os.path.join(os.getcwd(), "prefs.json")

    def _load_prefs(self) -> dict:
        try:
            path = self._theme_pref_file()
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
        except Exception:
            pass
        return {}

    def _save_prefs(self, prefs: dict) -> None:
        try:
            path = self._theme_pref_file()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(prefs, f)
        except Exception:
            pass

    def _load_theme_preference(self) -> bool:
        data = self._load_prefs()
        return bool(data.get("dark_mode", False))

    def _save_theme_preference(self) -> None:
        data = self._load_prefs()
        data["dark_mode"] = self.dark_mode
        self._save_prefs(data)

    def _load_pinned_commands(self) -> list:
        data = self._load_prefs()
        pins = data.get("pinned_commands", [])
        return pins if isinstance(pins, list) else []

    def _save_pinned_commands(self, pins: list) -> None:
        data = self._load_prefs()
        data["pinned_commands"] = list(dict.fromkeys(pins))
        self._save_prefs(data)

    # ------------------------------------------------------------------
    # Toast notifications
    # ------------------------------------------------------------------
    def show_toast(self, message: str, duration_ms: int = 2200) -> None:
        try:
            x = self.winfo_rootx() + self.winfo_width() - 360
            y = self.winfo_rooty() + self.winfo_height() - 120 - (len(self._active_toasts) * 56)
            toast = tk.Toplevel(self)
            toast.wm_overrideredirect(True)
            toast.attributes("-topmost", True)
            toast.wm_geometry(f"+{max(x, 20)}+{max(y, 20)}")
            bg = "#222222" if self.dark_mode else "#333333"
            fg = "#ffffff"
            frame = tk.Frame(toast, bg=bg, bd=0, highlightthickness=0)
            frame.pack(fill=tk.BOTH, expand=True)
            label = tk.Label(frame, text=message, bg=bg, fg=fg, padx=14, pady=10, wraplength=320, justify=tk.LEFT)
            label.pack()
            self._active_toasts.append(toast)
            def _close():
                try:
                    if toast in self._active_toasts:
                        self._active_toasts.remove(toast)
                    toast.destroy()
                except Exception:
                    pass
            toast.after(duration_ms, _close)
        except Exception:
            # Fallback to console
            print(message)

    # ------------------------------------------------------------------
    # Command Palette
    # ------------------------------------------------------------------
    def _get_command_registry(self) -> list:
        return [
            {"id": "new_layer", "label": "New Layer", "run": self._new_layer},
            {"id": "open_image", "label": "Open Image", "run": self._open_image},
            {"id": "save_image", "label": "Save Image", "run": self._save_image},
            {"id": "toggle_theme", "label": "Toggle Theme", "run": self._toggle_theme},
            {"id": "merge_visible", "label": "Merge Visible Layers", "run": self._merge_visible_layers},
            {"id": "group_layers", "label": "Group Selected Layers", "run": self._group_selected_layers},
            {"id": "auto_enhance", "label": "Auto Enhance Layer", "run": self._auto_enhance_layer},
            {"id": "one_click_beauty", "label": "One-Click Beauty", "run": self._one_click_beauty},
            {"id": "remove_bg", "label": "Remove Background", "run": self._remove_background},
            {"id": "history_panel", "label": "Show History Panel", "run": self._show_history_panel},
        ]

    def _open_command_palette(self, _event=None):
        try:
            registry = self._get_command_registry()
            id_to_cmd = {c["id"]: c for c in registry}
            pinned = self._load_pinned_commands()

            win = tk.Toplevel(self)
            win.title("Command Palette")
            win.geometry("560x400")
            win.transient(self)
            win.grab_set()
            bg = "#2e2e2e" if self.dark_mode else "#ffffff"
            fg = "#f0f0f0" if self.dark_mode else "#111111"
            hint = "#8a8a8a" if self.dark_mode else "#777777"
            win.configure(bg=bg)

            query_var = tk.StringVar()
            entry = tk.Entry(win, textvariable=query_var, bg=bg, fg=fg, insertbackground=fg, relief=tk.FLAT, font=("Segoe UI", 12))
            entry.pack(fill=tk.X, padx=12, pady=(12,6))

            info = tk.Label(win, text="Enter to run • Ctrl+P pin/unpin • Esc close", bg=bg, fg=hint)
            info.pack(fill=tk.X, padx=12)

            listbox = tk.Listbox(win, activestyle='none')
            listbox.pack(fill=tk.BOTH, expand=True, padx=12, pady=(6,12))

            def normalize(s: str) -> str:
                return s.lower()

            def fuzzy_match(query: str, label: str) -> bool:
                q = normalize(query)
                L = normalize(label)
                if not q:
                    return True
                # simple subsequence match
                i = 0
                for ch in L:
                    if i < len(q) and q[i] == ch:
                        i += 1
                return i == len(q)

            def current_items():
                q = query_var.get().strip()
                # Build list: pinned first if no query, then others
                all_items = registry
                if not q:
                    ordered = [id_to_cmd[pid] for pid in pinned if pid in id_to_cmd]
                    remaining = [c for c in all_items if c["id"] not in set(pinned)]
                    return ordered + remaining
                # filter with fuzzy
                filtered = [c for c in all_items if fuzzy_match(q, c["label"])]
                # light sort by label length to bias tighter matches
                filtered.sort(key=lambda c: (c["label"].lower().find(q.lower()[:1]) if q else 0, len(c["label"])) )
                return filtered

            def refresh_list():
                items = current_items()
