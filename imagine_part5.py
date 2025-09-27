    def _sample_layer_color(self, x: int, y: int) -> Optional[Tuple[int, int, int]]:
        if self.current_layer_index is None:
            return None
        layer = self.layers[self.current_layer_index]
        ox, oy = layer.offset
        lx = int(x - ox)
        ly = int(y - oy)
        if lx < 0 or ly < 0 or lx >= layer.original.width or ly >= layer.original.height:
            return None
        try:
            pixel = layer.original.convert('RGBA').getpixel((lx, ly))
        except Exception:
            pixel = layer.original.getpixel((lx, ly))
        return tuple(pixel[:3])

    def _sample_color_at(self, x: int, y: int, layer_only: bool = False) -> Optional[Tuple[int, int, int]]:
        if layer_only:
            colour = self._sample_layer_color(x, y)
            if colour:
                return colour
        if self._last_composite_image is None:
            return None
        if x < 0 or y < 0 or x >= self._last_composite_image.width or y >= self._last_composite_image.height:
            return None
        try:
            return self._last_composite_image.convert('RGB').getpixel((int(x), int(y)))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Tools (brush and text)
    # ------------------------------------------------------------------
    def _select_brush(self):
        """Activate the brush tool and prompt user for colour, size, and texture."""
        self.current_tool = "brush"
        # Ask colour
        color = colorchooser.askcolor(title="Choose brush colour", initialcolor=self.brush_color)
        if color and color[1]:
            self.active_color_slot = 'primary'
            self._set_color('primary', color[1])
        # Ask size
        size = simpledialog.askinteger("Brush Size", "Enter brush size (1-100)", initialvalue=self.brush_size, minvalue=1, maxvalue=100)
        if size:
            self.brush_size = size
        # Ask texture
        texture = simpledialog.askstring(
            "Brush Texture",
            "Enter brush texture (solid, spray, chalk, calligraphy):",
            initialvalue="solid"
        )
        if texture is None:
            return
        texture = texture.strip().lower()
        
        if texture not in ("solid", "spray", "chalk", "calligraphy"):
            messagebox.showinfo("Invalid Texture", "Brush texture must be 'solid', 'spray', 'chalk', or 'calligraphy'")
            return
            
        self.brush_texture = texture
        # Highlight selected tool
        self._highlight_tool()

    def _select_eraser(self):
        """Activate the eraser tool and prompt user for size."""
        self.current_tool = "eraser"
        size = simpledialog.askinteger("Eraser Size", "Enter eraser size (1-100)", initialvalue=self.brush_size, minvalue=1, maxvalue=100)
        if size:
            self.brush_size = size
        self._highlight_tool()

    def _select_eyedropper(self) -> None:
        """Activate the eyedropper for sampling colours from the canvas."""
        self.current_tool = "eyedropper"
        self._highlight_tool()

    def _configure_stamp_tool(self) -> None:
        choice = simpledialog.askstring("Stamp Type", "Enter stamp type (text or image):", initialvalue="text")
        if not choice:
            return
        choice = choice.strip().lower()
        if choice.startswith('image'):
            path = filedialog.askopenfilename(title="Choose stamp image", filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
            if not path:
                return
            try:
                img = Image.open(path).convert('RGBA')
            except Exception as exc:
                messagebox.showerror("Stamp", f"Unable to load image: {exc}")
                return
            self.stamp_image = img
            self._stamp_cache.clear()
        else:
            text = simpledialog.askstring("Text Seal", "Enter text for the stamp:", initialvalue="IMAGINE")
            if not text:
                return
            font_size = simpledialog.askinteger("Font Size", "Font size: (12-256)", initialvalue=96, minvalue=12, maxvalue=256)
            if font_size is None:
                font_size = 96
            outline = simpledialog.askstring("Outline", "Outline (circle, rectangle, none):", initialvalue="circle") or "circle"
            outline = outline.strip().lower()
            size = max(128, font_size * max(2, len(text) // 2))
            img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except Exception:
                tw, th = draw.textsize(text, font=font)
            x = (size - tw) // 2
            y = (size - th) // 2
            draw.text((x, y), text, font=font, fill=self.primary_color)
            outline_color = self.secondary_color
            if outline == 'circle':
                margin = font_size // 4
                draw.ellipse((margin, margin, size - margin, size - margin), outline=outline_color, width=max(3, font_size // 12))
            elif outline == 'rectangle':
                margin = font_size // 4
                draw.rectangle((margin, margin, size - margin, size - margin), outline=outline_color, width=max(3, font_size // 12))
            self.stamp_image = img
            self._stamp_cache.clear()
        self._set_status("Stamp configured")

    def _select_stamp_tool(self) -> None:
        if self.stamp_image is None:
            self._configure_stamp_tool()
            if self.stamp_image is None:
                return
        self.current_tool = "stamp"
        self._highlight_tool()

    def _configure_pattern_brush(self) -> None:
        size = simpledialog.askinteger("Pattern Size", "Tile size (pixels)", initialvalue=self.pattern_settings.get('size', 32), minvalue=4, maxvalue=256)
        if size:
            self.pattern_settings['size'] = int(size)
        color_a = colorchooser.askcolor(title="Pattern colour A", initialcolor=self.pattern_settings.get('colors', [self.primary_color, self.secondary_color])[0])
        if color_a and color_a[1]:
            self.pattern_settings.setdefault('colors', [self.primary_color, self.secondary_color])
            self.pattern_settings['colors'][0] = self._normalise_hex(color_a[1])
        color_b = colorchooser.askcolor(title="Pattern colour B", initialcolor=self.pattern_settings.get('colors', [self.primary_color, self.secondary_color])[1])
        if color_b and color_b[1]:
            self.pattern_settings['colors'][1] = self._normalise_hex(color_b[1])
        pattern_type = simpledialog.askstring("Pattern Style", "Style (checker, stripes, diagonal, dots):", initialvalue=self.pattern_settings.get('type', 'checker'))
        if pattern_type:
            self.pattern_settings['type'] = pattern_type.strip().lower()
            self.pattern_type_var.set(self.pattern_settings['type'])

    def _open_gradient_settings(self) -> None:
        gtype = simpledialog.askstring("Gradient Type", "Type (linear, radial, diamond):", initialvalue=self.gradient_settings.get('type', 'linear'))
        if gtype:
            self.gradient_settings['type'] = gtype.strip().lower()
        try:
            snap = messagebox.askyesno("Angle Snap", "Snap gradient angle to 15° increments?")
        except Exception:
            snap = True
        self.gradient_settings['snap'] = bool(snap)
        base_angle = simpledialog.askfloat("Base Angle", "Default angle (degrees)", initialvalue=self.gradient_settings.get('base_angle', 0.0))
        if base_angle is not None:
            self.gradient_settings['base_angle'] = float(base_angle)
        stops_str = ", ".join(f"{color}@{pos:.2f}" for pos, color in self.gradient_settings.get('stops', []))
        new_stops = simpledialog.askstring("Gradient Stops", "Stops as color@position (0-1), comma separated:", initialvalue=stops_str)
        if new_stops:
            parsed: List[Tuple[float, str]] = []
            for token in new_stops.split(','):
                token = token.strip()
                if not token:
                    continue
                if '@' in token:
                    colour, pos = token.split('@', 1)
                    try:
                        parsed.append((max(0.0, min(1.0, float(pos))), self._normalise_hex(colour)))
                    except Exception:
                        continue
                else:
                    parsed.append((len(parsed) / max(1, len(new_stops.split(',')) - 1), self._normalise_hex(token)))
            if parsed:
                parsed.sort(key=lambda item: item[0])
                self.gradient_settings['stops'] = parsed
                if parsed:
                    self.primary_color = parsed[0][1]
                    self.secondary_color = parsed[-1][1]
                    self._update_color_palette()

    def _select_pattern_brush(self) -> None:
        self.pattern_settings['type'] = self.pattern_type_var.get()
        self.current_tool = "pattern"
        self._highlight_tool()

    def _select_smart_eraser(self) -> None:
        self.current_tool = "smart_eraser"
        self._highlight_tool()

    def _select_move(self):
        """Activate the move tool to reposition the selected layer."""
        self.current_tool = "move"
        self._drag_prev = None
        self._highlight_tool()

    def _select_mask(self):
        """Activate the mask editing tool or create special masks.

        Users can choose between painting the mask manually (as before) or
        generating a linear, radial or patterned mask.  Painting requires
        selecting hide/reveal mode and brush size.  Generated masks are
        applied immediately and the tool is reset.
        """
        # Choose mask type
        mtype = simpledialog.askstring(
            "Mask Type",
            "Choose mask type (paint, linear, radial, pattern, ellipse, ring, diagonal, noise, triangle, adjust, filter):",
            initialvalue="paint",
        )
        if mtype is None:
            return
        mtype = mtype.strip().lower()
        if mtype == "paint":
            # Ask hide or reveal
            choice = simpledialog.askstring(
                "Mask Mode",
                "Enter mask mode: hide or reveal",
                initialvalue="hide",
            )
            if choice is None:
                return
            mode = choice.strip().lower()
            if mode not in ("hide", "reveal"):
                messagebox.showinfo("Invalid mode", "Mask mode must be 'hide' or 'reveal'")
                return
            self.mask_mode = mode
            # Ask brush size
            size = simpledialog.askinteger(
                "Mask Brush Size",
                "Enter mask brush size (1-200)",
                initialvalue=self.brush_size,
                minvalue=1,
                maxvalue=200,
            )
            if size:
                self.brush_size = size
            self.current_tool = "mask"
            self._highlight_tool()
        else:
            if self.current_layer_index is None:
                return
            layer = self.layers[self.current_layer_index]
            # Some special operations adjust existing mask rather than creating new
            if mtype in ("adjust", "filter"):
                # Save history
                self._save_history()
                w, h = layer.mask.size
                mask = layer.mask.copy()
                if mtype == "adjust":
                    # Ask which adjustment: lighten, darken, invert, threshold
                    adj = simpledialog.askstring(
                        "Adjust Mask",
                        "Adjustment (lighten, darken, invert, contrast):",
                        initialvalue="lighten",
                    )
                    if adj is None:
                        return
                    adj = adj.strip().lower()
                    if adj == "lighten":
                        factor = simpledialog.askfloat(
                            "Lighten Amount",
                            "Enter lighten amount (0-1, where 0 no change, 1 full white):",
                            initialvalue=0.2,
                            minvalue=0.0,
                            maxvalue=1.0,
                        )
                        if factor is None:
                            return
                        # lighten by factor: mask + factor*(255-mask)
                        mask = mask.point(lambda i, f=factor: int(i + f * (255 - i)))
                    elif adj == "darken":
                        factor = simpledialog.askfloat(
                            "Darken Amount",
                            "Enter darken amount (0-1, where 0 no change, 1 full black):",
                            initialvalue=0.2,
                            minvalue=0.0,
                            maxvalue=1.0,
                        )
                        if factor is None:
                            return
                        mask = mask.point(lambda i, f=factor: int(i * (1 - f)))
                    elif adj == "invert":
                        mask = ImageChops.invert(mask)
                    elif adj == "contrast":
                        # Enhance contrast: scale difference from mid grey
                        factor = simpledialog.askfloat(
                            "Contrast Factor",
                            "Enter contrast factor (>1 increases contrast, <1 decreases)",
                            initialvalue=1.5,
                            minvalue=0.1,
                            maxvalue=5.0,
                        )
                        if factor is None:
                            return
                        def adjust_contrast(p, f):
                            # map 0-255 to -1 to 1
                            np = p / 255.0
                            np = (np - 0.5) * f + 0.5
                            return int(max(0, min(255, np * 255)))
                        mask = mask.point(lambda i, f=factor: adjust_contrast(i, f))
                    else:
                        messagebox.showinfo("Adjustment", "Unsupported adjustment type.")
                        return
                    layer.mask = mask
                    layer.apply_adjustments()
                    self._update_composite()
                    return
                elif mtype == "filter":
                    # Ask filter name for mask: blur, sharpen, smooth, emboss
                    filt = simpledialog.askstring(
                        "Mask Filter",
                        "Mask filter (blur, sharpen, smooth, emboss):",
                        initialvalue="blur",
                    )
                    if filt is None:
                        return
                    filt = filt.strip().lower()
                    if filt == "blur":
                        mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
                    elif filt == "sharpen":
                        mask = mask.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3))
                    elif filt == "smooth":
                        mask = mask.filter(ImageFilter.SMOOTH_MORE)
                    elif filt == "emboss":
                        mask = mask.filter(ImageFilter.EMBOSS)
                    else:
                        messagebox.showinfo("Mask Filter", "Unsupported mask filter")
                        return
                    layer.mask = mask
                    layer.apply_adjustments()
                    self._update_composite()
                    return
            # For generated masks, ask hide or reveal
            choice = simpledialog.askstring(
                "Mask Mode",
                "Enter mask mode: hide or reveal",
                initialvalue="hide",
            )
            if choice is None:
                return
            hide = choice.strip().lower() == "hide"
            # Save history
            self._save_history()
            w, h = layer.mask.size
            mask = Image.new("L", (w, h), 0)
            if mtype == "linear":
                # Ask orientation
                orientation = simpledialog.askstring(
                    "Gradient Orientation",
                    "Enter orientation (horizontal or vertical):",
                    initialvalue="horizontal",
                )
                if orientation is None:
                    return
                orientation = orientation.strip().lower()
                if orientation.startswith("h"):
                    for x in range(w):
                        t = x / (w - 1) if w > 1 else 0
                        val = int(255 * t)
                        if hide:
                            val = 255 - val
                        for y in range(h):
                            mask.putpixel((x, y), val)
                else:
                    for y in range(h):
                        t = y / (h - 1) if h > 1 else 0
                        val = int(255 * t)
                        if hide:
                            val = 255 - val
                        for x in range(w):
                            mask.putpixel((x, y), val)
            elif mtype == "radial":
                cx = w / 2
                cy = h / 2
                max_r = math.hypot(cx, cy)
                for y in range(h):
                    for x in range(w):
                        dx = x - cx
                        dy = y - cy
                        r = math.hypot(dx, dy)
                        t = r / max_r if max_r > 0 else 0
                        if hide:
                            val = int(255 * (1 - t))
                        else:
                            val = int(255 * t)
                        val = max(0, min(255, val))
                        mask.putpixel((x, y), val)
            elif mtype == "pattern":
                # Ask pattern type
                ptype = simpledialog.askstring(
                    "Pattern Type",
                    "Enter pattern (stripes, checker, diagonal):",
                    initialvalue="stripes",
                )
                if ptype is None:
                    return
                ptype = ptype.strip().lower()
                stripe_width = simpledialog.askinteger(
                    "Pattern Size",
                    "Enter pattern size (pixels):",
                    initialvalue=20,
                    minvalue=1,
                    maxvalue=200,
                )
                if not stripe_width:
                    stripe_width = 20
                for y in range(h):
                    for x in range(w):
                        if ptype.startswith("diag"):
                            band = ((x + y) // stripe_width) % 2
                        elif ptype.startswith("str"):
                            band = (x // stripe_width) % 2
                        else:
                            band = ((x // stripe_width) + (y // stripe_width)) % 2
                        val = 0 if (band == 0) ^ hide else 255
                        mask.putpixel((x, y), val)
            elif mtype == "ellipse":
                # Create elliptical gradient or solid; ask solid or gradient
                solid = messagebox.askyesno("Ellipse Mask", "Solid ellipse? (Yes=solid, No=gradient)")
                for y in range(h):
                    for x in range(w):
                        # normalised distances relative to centre
                        nx = (x - w / 2) / (w / 2)
                        ny = (y - h / 2) / (h / 2)
                        dist = nx * nx + ny * ny
                        if solid:
                            inside = dist <= 1.0
                            val = 0 if inside ^ hide else 255
                        else:
                            # gradient: distance squared -> alpha
                            # inside centre (dist=0) -> full effect; at edge (dist=1) -> 0
                            t = min(1.0, dist)
                            val = int(255 * (1 - t)) if hide else int(255 * t)
                        mask.putpixel((x, y), val)
            elif mtype == "ring":
                # Ring shaped mask; ask inner and outer radius fractions
                inner_ratio = simpledialog.askfloat(
                    "Inner Radius",
                    "Enter inner radius fraction (0-1)",
                    initialvalue=0.3,
                    minvalue=0.0,
                    maxvalue=1.0,
                )
                if inner_ratio is None:
                    return
                outer_ratio = simpledialog.askfloat(
                    "Outer Radius",
                    "Enter outer radius fraction (0-1, > inner)",
                    initialvalue=0.6,
                    minvalue=inner_ratio,
                    maxvalue=1.0,
                )
                if outer_ratio is None:
                    return
                cx = w / 2
                cy = h / 2
                max_rad = (min(w, h) / 2) * outer_ratio
                min_rad = (min(w, h) / 2) * inner_ratio
                for y in range(h):
                    for x in range(w):
                        dx = x - cx
                        dy = y - cy
                        dist = math.hypot(dx, dy)
                        if min_rad <= dist <= max_rad:
                            val = 0 if hide else 255
                        else:
                            val = 255 if hide else 0
                        mask.putpixel((x, y), val)
            elif mtype == "diagonal":
                # Diagonal gradient from top-left to bottom-right
                for y in range(h):
                    for x in range(w):
                        t = (x + y) / (w + h - 2) if (w + h) > 2 else 0
                        val = int(255 * t)
                        if hide:
                            val = 255 - val
                        mask.putpixel((x, y), val)
            elif mtype == "noise":
                # Random noise mask; ask density or threshold
                density = simpledialog.askinteger(
                    "Noise Mask",
                    "Enter density percentage for noise (0-100)",
                    initialvalue=50,
                    minvalue=0,
                    maxvalue=100,
                )
                if density is None:
                    density = 50
                import random
                for y in range(h):
                    for x in range(w):
                        r = random.randint(0, 100)
                        if r < density:
                            val = 0 if hide else 255
                        else:
                            val = 255 if hide else 0
                        mask.putpixel((x, y), val)
            elif mtype == "triangle":
                # Triangular gradient; ask orientation (up, down, left, right)
                orient = simpledialog.askstring(
                    "Triangle Orientation",
                    "Enter orientation (up, down, left, right)",
                    initialvalue="up",
                )
                if orient is None:
                    return
                orient = orient.strip().lower()
                if orient == "up":
                    # Gradient from bottom to top (0 at bottom, 255 at top if hide)
                    for y in range(h):
                        t = 1 - (y / (h - 1) if h > 1 else 0)
                        val_row = int(255 * t)
                        if not hide:
                            val_row = 255 - val_row
                        for x in range(w):
                            mask.putpixel((x, y), val_row)
                elif orient == "down":
                    for y in range(h):
                        t = y / (h - 1) if h > 1 else 0
                        val_row = int(255 * t)
                        if hide:
                            val_row = 255 - val_row
                        for x in range(w):
                            mask.putpixel((x, y), val_row)
                elif orient == "left":
                    for x in range(w):
                        t = 1 - (x / (w - 1) if w > 1 else 0)
                        val_col = int(255 * t)
                        if not hide:
                            val_col = 255 - val_col
                        for y in range(h):
                            mask.putpixel((x, y), val_col)
                else:
                    for x in range(w):
                        t = x / (w - 1) if w > 1 else 0
                        val_col = int(255 * t)
                        if hide:
                            val_col = 255 - val_col
                        for y in range(h):
                            mask.putpixel((x, y), val_col)
            else:
                messagebox.showinfo("Unknown mask type", f"Mask type '{mtype}' is not supported.")
                return
            # Apply mask to layer
            layer.mask = mask
            layer.apply_adjustments()
            self._update_composite()
            # Reset tool
            self.current_tool = None
            self._highlight_tool()

    def _select_crop(self):
        """Activate the crop tool to select an area to keep."""
        self.current_tool = "crop"
        # Remove any existing rectangle
        self._crop_rect_id = None
        self._crop_start = None
        self._highlight_tool()

    def _select_select_tool(self):
        """Activate the select tool for editing layer properties via clicking."""
        self.current_tool = "select"
        self._highlight_tool()

    def _select_text_tool(self):
        """Activate the text tool and prompt user for content, font and colour."""
        self.current_tool = "text"
        # Ask for text content
        text = simpledialog.askstring("Text", "Enter text to add:")
        if text is None or text == "":
            self.current_tool = None
            return
        self.pending_text = text
        # Ask font size
        size = simpledialog.askinteger("Font Size", "Enter font size (8-200)", initialvalue=32, minvalue=8, maxvalue=200)
        if size:
            self.pending_font_size = size
        else:
            self.pending_font_size = 32
        # Ask font name or path
        font_name = simpledialog.askstring("Font", "Enter font family or path (leave blank for default):")
        if font_name:
            self.pending_font_name = font_name.strip()
        else:
            self.pending_font_name = None
        # Ask colour
        color = colorchooser.askcolor(title="Choose text colour", initialcolor="#ffffff")
        if color and color[1]:
            self.pending_text_color = color[1]
        else:
            self.pending_text_color = "#ffffff"
        # Wait for click on canvas to place text
        messagebox.showinfo("Place Text", "Click on the image to place the text.")
        self._highlight_tool()

    def _select_vector_rectangle(self):
        """Activate the vector rectangle tool."""
        self.current_tool = "vector_rectangle"
        self.vector_tool_state["drawing"] = False
        self.vector_tool_state["start_point"] = None
        self._highlight_tool()

    def _select_vector_circle(self):
        """Activate the vector circle tool."""
        self.current_tool = "vector_circle"
        self.vector_tool_state["drawing"] = False
        self.vector_tool_state["start_point"] = None
        self._highlight_tool()

    def _select_vector_ellipse(self):
        """Activate the vector ellipse tool."""
        self.current_tool = "vector_ellipse"
        self.vector_tool_state["drawing"] = False
        self.vector_tool_state["start_point"] = None
        self._highlight_tool()

    def _select_vector_line(self):
        """Activate the vector line tool."""
        self.current_tool = "vector_line"
        self.vector_tool_state["drawing"] = False
        self.vector_tool_state["start_point"] = None
        self._highlight_tool()

    def _select_vector_text(self):
        """Activate the vector text tool."""
        self.current_tool = "vector_text"
        self.vector_tool_state["drawing"] = False
        self.vector_tool_state["start_point"] = None
        self._highlight_tool()

    def _show_brush_settings(self):
        """Show advanced brush settings dialog."""
        dialog = tk.Toplevel(self)
        dialog.title("Brush Settings")
        dialog.geometry("400x300")
        dialog.transient(self)
        dialog.grab_set()
        
        # Brush size
        tk.Label(dialog, text="Size:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=5)
        size_var = tk.DoubleVar(value=self.brush_settings.size)
        size_scale = tk.Scale(dialog, from_=1, to=100, orient=tk.HORIZONTAL, variable=size_var)
        size_scale.pack(fill=tk.X, padx=10)
        
        # Hardness
        tk.Label(dialog, text="Hardness:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 5))
        hardness_var = tk.DoubleVar(value=self.brush_settings.hardness)
        hardness_scale = tk.Scale(dialog, from_=0, to=1, resolution=0.1, orient=tk.HORIZONTAL, variable=hardness_var)
        hardness_scale.pack(fill=tk.X, padx=10)
        
        # Spacing
        tk.Label(dialog, text="Spacing:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 5))
        spacing_var = tk.DoubleVar(value=self.brush_settings.spacing)
        spacing_scale = tk.Scale(dialog, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL, variable=spacing_var)
        spacing_scale.pack(fill=tk.X, padx=10)
        
        # Flow
        tk.Label(dialog, text="Flow:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 5))
        flow_var = tk.DoubleVar(value=self.brush_settings.flow)
        flow_scale = tk.Scale(dialog, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL, variable=flow_var)
        flow_scale.pack(fill=tk.X, padx=10)
        
        # Jitter
        tk.Label(dialog, text="Jitter:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 5))
        jitter_var = tk.DoubleVar(value=self.brush_settings.jitter)
        jitter_scale = tk.Scale(dialog, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL, variable=jitter_var)
        jitter_scale.pack(fill=tk.X, padx=10)
        
        # Checkboxes
        pressure_var = tk.BooleanVar(value=self.brush_settings.pressure_sensitivity)
        tk.Checkbutton(dialog, text="Pressure Sensitivity", variable=pressure_var).pack(anchor=tk.W, padx=10, pady=5)
        
        speed_var = tk.BooleanVar(value=self.brush_settings.speed_sensitivity)
        tk.Checkbutton(dialog, text="Speed Sensitivity", variable=speed_var).pack(anchor=tk.W, padx=10, pady=5)
        
        def apply_settings():
            self.brush_settings.size = size_var.get()
            self.brush_settings.hardness = hardness_var.get()
            self.brush_settings.spacing = spacing_var.get()
            self.brush_settings.flow = flow_var.get()
            self.brush_settings.jitter = jitter_var.get()
            self.brush_settings.pressure_sensitivity = pressure_var.get()
            self.brush_settings.speed_sensitivity = speed_var.get()
            self.brush_engine.update_settings(self.brush_settings)
            dialog.destroy()
        
        def cancel():
            dialog.destroy()
        
        tk.Button(dialog, text="Apply", command=apply_settings).pack(side=tk.LEFT, padx=20, pady=20)
        tk.Button(dialog, text="Cancel", command=cancel).pack(side=tk.RIGHT, padx=20, pady=20)

    def _select_filter_region(self):
        """Activate tool to select an area for applying a filter.

        In addition to the default rectangular region, the user can choose
        alternative selection shapes such as an ellipse or freeform polygon.
        Once a shape is chosen, subsequent mouse actions will collect the
        appropriate coordinates for that shape.  On release, the selected
        region will be processed and the chosen filter applied only within
        that region.
        """
        # Prompt for selection shape
        shape = simpledialog.askstring(
            "Filter Selection Shape",
            "Choose selection shape for filter (rect, ellipse, polygon)",
            initialvalue="rect",
        )
        if shape is None:
            return
        shape = shape.strip().lower()
        if shape not in ("rect", "ellipse", "polygon"):
            messagebox.showinfo("Selection", "Unsupported selection shape.")
            return
        # Store the shape and reset region tracking variables
        self.filter_selection_shape = shape
        self.current_tool = 'filter_region'
        # For rectangle or ellipse we record starting point and overlay id
        self._filter_region_start = None
        self._filter_rect_id = None
        # For polygon we record list of points and line ids
        self.filter_lasso_points = []
        self._filter_line_ids = []
        self._highlight_tool()

    def _select_heal(self):
        """Activate spot heal tool. Prompt for brush size."""
        self.current_tool = 'heal'
        # Ask user for heal brush size; default to previous heal size or current brush size
        size = simpledialog.askinteger(
            "Heal Brush Size",
            "Enter heal brush size (1-100)",
            initialvalue=getattr(self, 'heal_size', self.brush_size),
            minvalue=1,
            maxvalue=100,
        )
        if size:
            self.heal_size = size
        # Highlight selected tool
        self._highlight_tool()

    def _select_dodge(self):
        """Activate dodge tool. Prompt for brush size and strength."""
        self.current_tool = 'dodge'
        # Ask for brush size
        size = simpledialog.askinteger(
            "Dodge Brush Size",
            "Enter dodge brush size (1-100)",
            initialvalue=getattr(self, 'dodge_size', self.brush_size),
            minvalue=1,
            maxvalue=100,
        )
        if size:
            self.dodge_size = size
        # Ask for strength factor (>1 to lighten)
        strength = simpledialog.askfloat(
            "Dodge Strength",
            "Enter dodge strength (>1.0 to lighten)",
            initialvalue=getattr(self, 'dodge_strength', 1.2),
            minvalue=1.0,
            maxvalue=5.0,
        )
        if strength:
            self.dodge_strength = strength
        self._highlight_tool()

    def _select_burn(self):
        """Activate burn tool. Prompt for brush size and strength."""
        self.current_tool = 'burn'
        # Ask for brush size
        size = simpledialog.askinteger(
            "Burn Brush Size",
            "Enter burn brush size (1-100)",
            initialvalue=getattr(self, 'burn_size', self.brush_size),
            minvalue=1,
            maxvalue=100,
        )
        if size:
            self.burn_size = size
        # Ask for strength (<1 to darken)
        strength = simpledialog.askfloat(
            "Burn Strength",
            "Enter burn strength (<1.0 to darken)",
            initialvalue=getattr(self, 'burn_strength', 0.8),
            minvalue=0.1,
            maxvalue=1.0,
        )
        if strength:
            self.burn_strength = strength
        self._highlight_tool()

    def _select_extract(self):
        """Activate a tool to select an area and extract it into a new layer."""
        # Prompt for selection shape: rectangle, ellipse, polygon
        shape = simpledialog.askstring(
            "Selection Shape",
            "Choose selection shape (rect, ellipse, polygon)",
            initialvalue="rect",
        )
        if shape is None:
            return
        shape = shape.strip().lower()
        if shape not in ("rect", "ellipse", "polygon"):
            messagebox.showinfo("Selection", "Unsupported shape type.")
            return
        self.selection_shape = shape
        self.current_tool = 'extract'
        # Initialize variables for selection
        self.lasso_points = []
        self._extract_overlay_id = None
        self._extract_line_ids = []
        self._extract_start = None
        self._highlight_tool()

    def _select_magic_wand(self) -> None:
        """Activate the magic wand tool for contiguous region selection and filtering."""
        # Ask for tolerance threshold
        tol = simpledialog.askinteger(
            "Magic Wand Tolerance",
            "Enter color difference tolerance (0-255):",
            initialvalue=30,
            minvalue=0,
            maxvalue=255,
        )
        if tol is None:
            return
        self.magic_tolerance = tol
        self.current_tool = 'magicwand'
        self._highlight_tool()

    def _select_quick_select(self) -> None:
        """Activate the quick select tool that selects similar colour/texture while dragging.

        The user will be prompted for a colour tolerance (similar to magic wand) and a filter to apply.
        As the user drags the mouse, contiguous regions similar to the initial colour will be
        progressively selected and filtered.
        """
        tol = simpledialog.askinteger(
            "Quick Select Tolerance",
            "Enter colour difference tolerance (0-255):",
            initialvalue=30,
            minvalue=0,
            maxvalue=255,
        )
        if tol is None:
            return
        # Choose filter to apply
        filter_options = [
            "grayscale", "invert", "blur", "sharpen", "emboss", "edge", "contour", "detail", "smooth", "liquify", "sepia",
            "skin smooth", "frequency separation", "teeth whitening", "red eye removal", "warm", "cool", "vintage",
            "anime", "oil", "cyberpunk", "portrait optimize", "posterize", "solarize"
        ]
        fname = simpledialog.askstring("Quick Select Filter", f"Enter filter to apply {filter_options}")
        if not fname:
            return
        fname = fname.strip().lower()
        if fname not in [opt.lower() for opt in filter_options]:
            messagebox.showinfo("Filter", "Unsupported filter name.")
            return
        self.quick_tolerance = tol
        self.quick_filter = fname
        self.current_tool = 'quickselect'
        self._highlight_tool()
        # Reset history flag so we save only once on press
        self._history_saved_for_stroke = False

    def _select_gradient(self) -> None:
        """Activate the gradient fill tool for creating smooth color gradients.
        
        Users can drag across the canvas to create linear gradients between two colors.
        The tool will prompt for start and end colors, and gradient type.
        """
        if self.current_layer_index is None:
            messagebox.showinfo("No Layer", "Please select a layer first.")
            return
            
        self.current_tool = 'gradient'
        self._highlight_tool()
        # Reset history flag so we save only once on press
        self._history_saved_for_stroke = False
        self._set_status("Gradient tool active - drag to paint. Use 'Gradient Settings' for multi-stop control.")

    def _select_rectangle(self) -> None:
        """Activate the rectangle shape tool."""
        if self.current_layer_index is None:
            messagebox.showinfo("No Layer", "Please select a layer first.")
            return
        self._setup_shape_tool("rectangle")

    def _select_circle(self) -> None:
        """Activate the circle shape tool."""
        if self.current_layer_index is None:
            messagebox.showinfo("No Layer", "Please select a layer first.")
            return
        self._setup_shape_tool("circle")

    def _select_arrow(self) -> None:
        """Activate the arrow shape tool."""
        if self.current_layer_index is None:
            messagebox.showinfo("No Layer", "Please select a layer first.")
            return
        self._setup_shape_tool("arrow")

    def _select_polygon(self) -> None:
        """Activate the polygon shape tool."""
        if self.current_layer_index is None:
            messagebox.showinfo("No Layer", "Please select a layer first.")
            return
        self._setup_shape_tool("polygon")

    def _setup_shape_tool(self, shape_type: str) -> None:
        """Setup common parameters for shape tools."""
        # Ask for fill color
        fill_color = colorchooser.askcolor(
            title="Choose fill color", 
            initialcolor=self.brush_color
        )
        if not fill_color or not fill_color[1]:
            return
            
        # Ask for stroke color
        stroke_color = colorchooser.askcolor(
            title="Choose stroke color", 
            initialcolor="#000000"
        )
        if not stroke_color or not stroke_color[1]:
            return
            
        # Ask for stroke width
        stroke_width = simpledialog.askinteger(
            "Stroke Width",
            "Enter stroke width (0 for no stroke):",
            initialvalue=2,
            minvalue=0,
            maxvalue=50
        )
        if stroke_width is None:
            return
            
        # Ask for fill style
        fill_style = simpledialog.askstring(
            "Fill Style",
            "Enter fill style (filled, outline, both):",
            initialvalue="filled"
        )
        if fill_style is None:
            return
        fill_style = fill_style.strip().lower()
        
        if fill_style not in ("filled", "outline", "both"):
            messagebox.showinfo("Invalid Style", "Fill style must be 'filled', 'outline', or 'both'")
            return
            
        self.shape_type = shape_type
        self.shape_fill_color = fill_color[1]
        self.shape_stroke_color = stroke_color[1]
        self.shape_stroke_width = stroke_width
        self.shape_fill_style = fill_style
        self.current_tool = 'shape'
        self._highlight_tool()
        # Reset history flag so we save only once on press
        self._history_saved_for_stroke = False

    def _quick_select_wand_at(self, ux: int, uy: int) -> None:
        """Apply the quick select filter at the given unscaled coordinates using stored tolerance and filter.
        This method reuses the magic wand algorithm but uses the predetermined tolerance and filter
        without prompting the user.  It updates the layer in place.
        """
        if self.current_layer_index is None:
            return
        if not hasattr(self, 'quick_tolerance') or not hasattr(self, 'quick_filter'):
            return
        layer = self.layers[self.current_layer_index]
        ox, oy = layer.offset
        lx = ux - int(ox)
        ly = uy - int(oy)
        if lx < 0 or ly < 0 or lx >= layer.original.width or ly >= layer.original.height:
            return
        tol = self.quick_tolerance
        fname = self.quick_filter
        # For the first invocation in a stroke, save history
        if not getattr(self, '_history_saved_for_stroke', False):
            self._current_action_desc = "Quick Select"
            self._save_history()
            self._history_saved_for_stroke = True
        # Convert image to numpy array
        np_img = np.array(layer.original.convert('RGB'))
        base_color = np_img[ly, lx].astype(int)
        diff = np.abs(np_img.astype(int) - base_color)
        diff_sum = diff.sum(axis=2)
        mask = (diff_sum <= tol).astype(np.uint8)
        ys, xs = np.where(mask == 1)
        if len(xs) == 0 or len(ys) == 0:
            return
        x0 = int(xs.min())
        x1 = int(xs.max()) + 1
        y0 = int(ys.min())
        y1 = int(ys.max()) + 1
        region = layer.original.crop((x0, y0, x1, y1)).copy()
        # Apply filter on temporary layer
        temp_layer = Layer(region, "temp")
        try:
            temp_layer.apply_filter(fname)
        except Exception:
            pass
        filtered = temp_layer.original
        submask = (mask[y0:y1, x0:x1] * 255).astype('uint8')
        mask_img = Image.fromarray(submask)
        # Smooth edges with Gaussian blur to make selection behave like a brush
        try:
            mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=3))
            mask_img = mask_img.point(lambda p: 255 if p > 64 else 0)
        except Exception:
            pass
        layer.original.paste(filtered, (x0, y0), mask_img)
        layer.apply_adjustments()
        self._update_composite()

    # ------------------------------------------------------------------
    # Canvas events
    # ------------------------------------------------------------------
    def _on_canvas_press(self, event):
        """Handle mouse press events on the canvas."""
        if self.current_layer_index is None:
            return
        # Compute unscaled coordinates for editing based on zoom factor
        ux = int(event.x / max(self.zoom, 1e-6))
        uy = int(event.y / max(self.zoom, 1e-6))
        if self.current_tool == "eyedropper":
            colour = self._sample_color_at(ux, uy)
            if colour:
                self._apply_sampled_color(colour)
                self._set_status(f"Sampled colour RGB{colour}")
            self.current_tool = None
            self._highlight_tool()
            return
        # If drawing, erasing or retouching, save state at start of stroke
        if self.current_tool in ("brush", "eraser", "heal", "dodge", "burn", "stamp", "pattern", "smart_eraser"):
            self._save_history()
            self._history_saved_for_stroke = True
            # Store previous unscaled coordinate for stroke interpolation
            self._drag_prev = (ux, uy)
            if self.current_tool == "brush":
                self._paint_at(ux, uy)
            elif self.current_tool == "eraser":
                self._erase_at(ux, uy)
            elif self.current_tool == "heal":
                self._heal_at(ux, uy)
            elif self.current_tool == "dodge":
                self._dodge_at(ux, uy)
            elif self.current_tool == "burn":
                self._burn_at(ux, uy)
            elif self.current_tool == "stamp":
                self._apply_stamp_at(ux, uy)
            elif self.current_tool == "pattern":
                self._apply_pattern_at(ux, uy)
            elif self.current_tool == "smart_eraser":
                self._smart_eraser_color = self._sample_layer_color(ux, uy)
                self._smart_erase_at(ux, uy)
        elif self.current_tool == "text":
            # Save history before adding text
            self._save_history()
            self._add_text(event.x, event.y)
            # After placing text, reset tool
            self.current_tool = None
        elif self.current_tool == "move":
            # Save history at beginning of move
            self._save_history()
            self._history_saved_for_stroke = True
            # Store previous unscaled coordinate
            self._drag_prev = (ux, uy)
            # Record initial offset for the layer
            self._move_start_offset = self.layers[self.current_layer_index].offset
        elif self.current_tool == "mask":
            # Save history at beginning of mask painting
            self._save_history()
            self._history_saved_for_stroke = True
            self._drag_prev = (ux, uy)
            # Paint initial dot on mask using unscaled coordinates
            self._mask_at(ux, uy)
        elif self.current_tool == "crop":
            # Save history (crop will crop on release)
            self._save_history()
            self._history_saved_for_stroke = True
            # Store unscaled starting point
            self._crop_start = (ux, uy)
            # Create rectangle overlay on scaled coordinates
            self._crop_rect_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="yellow", dash=(4, 2))
        elif self.current_tool == "select":
            # Select the topmost layer at this point and open a property window
            # Use unscaled coordinates for hit detection
            self._select_layer_at(ux, uy)
            self._open_properties_window()
        elif self.current_tool == "filter_region":
            # Start selecting a region for filtering
            self._save_history()
            self._history_saved_for_stroke = True
            shape = getattr(self, 'filter_selection_shape', 'rect')
            if shape in ("rect", "ellipse"):
                # Record starting point in unscaled coords and create overlay at scaled coords
                self._filter_region_start = (ux, uy)
                if shape == "rect":
                    self._filter_rect_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#00ff00", dash=(4, 2))
                else:
                    self._filter_rect_id = self.canvas.create_oval(event.x, event.y, event.x, event.y, outline="#00ff00", dash=(4, 2))
            else:
                # Polygon selection: store first point in unscaled coordinates and create line overlays list
                self.filter_lasso_points = [(ux, uy)]
                self._filter_line_ids = []
        elif self.current_tool == "magicwand":
            # Use magic wand selection at the clicked position
            self._magic_wand_at(ux, uy)
            # Reset tool after applying
            self.current_tool = None
        elif self.current_tool == "quickselect":
            # Quick select: apply filter at click and start stroke
            self._quick_select_wand_at(ux, uy)
        elif self.current_tool == "gradient":
            # Save history at beginning of gradient creation
            self._save_history()
            self._history_saved_for_stroke = True
            # Store unscaled starting point
            self._gradient_start = (ux, uy)
            # Create line overlay to show gradient direction
            self._gradient_line_id = self.canvas.create_line(event.x, event.y, event.x, event.y, fill="red", width=2, dash=(4, 2))
        elif self.current_tool == "shape":
            # Save history at beginning of shape creation
            self._save_history()
            self._history_saved_for_stroke = True
            # Store unscaled starting point
            self._shape_start = (ux, uy)
            # Create overlay based on shape type
            if self.shape_type == "polygon":
                # For polygon, start collecting points
                self._shape_points = [(ux, uy)]
                self._shape_line_ids = []
            else:
                # For other shapes, create preview overlay
                if self.shape_type == "rectangle":
                    self._shape_overlay_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline=self.shape_stroke_color, width=self.shape_stroke_width, dash=(4, 2))
                elif self.shape_type == "circle":
                    self._shape_overlay_id = self.canvas.create_oval(event.x, event.y, event.x, event.y, outline=self.shape_stroke_color, width=self.shape_stroke_width, dash=(4, 2))
                elif self.shape_type == "arrow":
                    self._shape_overlay_id = self.canvas.create_line(event.x, event.y, event.x, event.y, fill=self.shape_stroke_color, width=self.shape_stroke_width, arrow=tk.LAST, dash=(4, 2))
        elif self.current_tool == "extract":
            # Save history at beginning of extraction selection
            self._save_history()
            self._history_saved_for_stroke = True
            # Determine the selection shape: rectangle, ellipse, or polygon
            sel_shape = getattr(self, 'selection_shape', None)
            if sel_shape in ("rect", "ellipse"):
                # For rectangular/elliptical selection, record the unscaled starting point
                # and create the overlay at the corresponding screen coordinate
                self._extract_start = (ux, uy)
                # Convert unscaled start to screen coordinates for overlay
                sx0 = ux * self.zoom
                sy0 = uy * self.zoom
                if sel_shape == "rect":
                    self._extract_overlay_id = self.canvas.create_rectangle(sx0, sy0, sx0, sy0, outline="#ff00ff", dash=(4, 2))
                else:
                    self._extract_overlay_id = self.canvas.create_oval(sx0, sy0, sx0, sy0, outline="#ff00ff", dash=(4, 2))
            elif sel_shape == "polygon":
                # Start collecting polygon points in unscaled coordinates
                self.lasso_points = [(ux, uy)]
                self._extract_line_ids = []
            # Do not set _drag_prev since extraction uses its own state
        elif self.current_tool in ["vector_rectangle", "vector_circle", "vector_ellipse", "vector_line"]:
            # Vector tools - start drawing
            self.vector_tool_state["drawing"] = True
            self.vector_tool_state["start_point"] = (ux, uy)
            self._save_history()
            self._history_saved_for_stroke = True
        elif self.current_tool == "vector_text":
            # Vector text - prompt for text and add it
            text = simpledialog.askstring("Vector Text", "Enter text:")
            if text:
                self._add_vector_text(ux, uy, text)
        else:
            # For future tools (e.g., selection) we could handle here
            pass

    def _on_canvas_drag(self, event):
        if self.current_layer_index is None:
            return
        # Compute unscaled coordinates
        ux = int(event.x / max(self.zoom, 1e-6))
        uy = int(event.y / max(self.zoom, 1e-6))
        if self.current_tool == "brush" and self._drag_prev:
            # Draw line from previous point to current in unscaled coords
            x0, y0 = self._drag_prev
            x1, y1 = ux, uy
            self._paint_line_advanced(x0, y0, x1, y1)
            self._drag_prev = (x1, y1)
        elif self.current_tool == "eraser" and self._drag_prev:
            x0, y0 = self._drag_prev
            x1, y1 = ux, uy
            self._erase_line(x0, y0, x1, y1)
            self._drag_prev = (x1, y1)
        elif self.current_tool == "stamp" and self._drag_prev:
            x0, y0 = self._drag_prev
            x1, y1 = ux, uy
            self._stamp_line(x0, y0, x1, y1)
            self._drag_prev = (x1, y1)
        elif self.current_tool == "pattern" and self._drag_prev:
            x0, y0 = self._drag_prev
            x1, y1 = ux, uy
            self._pattern_line(x0, y0, x1, y1)
            self._drag_prev = (x1, y1)
        elif self.current_tool == "smart_eraser" and self._drag_prev:
            x0, y0 = self._drag_prev
            x1, y1 = ux, uy
            self._smart_erase_line(x0, y0, x1, y1)
            self._drag_prev = (x1, y1)
        elif self.current_tool == "move" and self._drag_prev:
            # Move layer by delta from starting position (unscaled)
            x0, y0 = self._drag_prev
            dx = ux - x0
            dy = uy - y0
            ox0, oy0 = self._move_start_offset
            self.layers[self.current_layer_index].offset = (ox0 + dx, oy0 + dy)
            self._update_composite()
        elif self.current_tool == "mask" and self._drag_prev:
            x0, y0 = self._drag_prev
            x1, y1 = ux, uy
            self._mask_line(x0, y0, x1, y1)
            self._drag_prev = (x1, y1)
        elif self.current_tool == "crop" and self._crop_rect_id is not None and self._crop_start:
            # Update cropping rectangle overlay using scaled coords for overlay and unscaled start
            x0, y0 = self._crop_start
            # Convert unscaled start back to screen coordinates for overlay
            sx0 = x0 * self.zoom
            sy0 = y0 * self.zoom
            self.canvas.coords(self._crop_rect_id, sx0, sy0, event.x, event.y)
        elif self.current_tool == "filter_region":
            shape = getattr(self, 'filter_selection_shape', 'rect')
            if shape in ("rect", "ellipse") and self._filter_rect_id is not None and self._filter_region_start:
                x0, y0 = self._filter_region_start
                # Convert unscaled start to screen coords
                sx0 = x0 * self.zoom
                sy0 = y0 * self.zoom
                self.canvas.coords(self._filter_rect_id, sx0, sy0, event.x, event.y)
            elif shape == "polygon" and getattr(self, 'filter_lasso_points', None):
                # For polygon, draw line segments as user drags
                # Append new line from last point to current position on screen
                last_x, last_y = self.filter_lasso_points[-1]
                # Convert last point to screen coords
                sx_last = last_x * self.zoom
                sy_last = last_y * self.zoom
                line_id = self.canvas.create_line(sx_last, sy_last, event.x, event.y, fill="#00ff00", dash=(4, 2))
                self._filter_line_ids.append(line_id)
                # Append new unscaled point
                self.filter_lasso_points.append((ux, uy))
        elif self.current_tool == "extract":
            # During extract selection, update overlay shapes for rectangle/ellipse or polygon
            shape = getattr(self, 'selection_shape', 'rect')
            if shape in ("rect", "ellipse") and getattr(self, '_extract_overlay_id', None) is not None and getattr(self, '_extract_start', None):
                # Use unscaled start point stored in _extract_start and convert to screen coordinates
                start_ux, start_uy = self._extract_start
                sx0 = start_ux * self.zoom
                sy0 = start_uy * self.zoom
                # Update overlay rectangle/oval using current mouse screen coords
                self.canvas.coords(self._extract_overlay_id, sx0, sy0, event.x, event.y)
            elif shape == "polygon" and getattr(self, 'lasso_points', None):
                # Draw lasso lines as polygon selection
                last_ux, last_uy = self.lasso_points[-1]
                # Convert last unscaled point to screen coords
                sx_last = last_ux * self.zoom
                sy_last = last_uy * self.zoom
                # Draw line from last screen position to current cursor
                line_id = self.canvas.create_line(sx_last, sy_last, event.x, event.y, fill="#ff00ff", dash=(4, 2))
                if not hasattr(self, '_extract_line_ids'):
                    self._extract_line_ids = []
                self._extract_line_ids.append(line_id)
                # Append new unscaled coordinate to polygon list
                self.lasso_points.append((ux, uy))
        elif self.current_tool == "quickselect":
            # Apply quick select filter continuously while dragging
            self._quick_select_wand_at(ux, uy)
        elif self.current_tool == "gradient" and self._gradient_line_id is not None and self._gradient_start:
            # Update gradient line overlay to show direction
            x0, y0 = self._gradient_start
            # Convert unscaled start to screen coords
            sx0 = x0 * self.zoom
            sy0 = y0 * self.zoom
            self.canvas.coords(self._gradient_line_id, sx0, sy0, event.x, event.y)
        elif self.current_tool == "shape" and self._shape_start:
            if self.shape_type == "polygon":
                # For polygon, draw line segments as user drags
                if hasattr(self, '_shape_points') and self._shape_points:
                    last_x, last_y = self._shape_points[-1]
                    # Convert last point to screen coords
                    sx_last = last_x * self.zoom
                    sy_last = last_y * self.zoom
                    line_id = self.canvas.create_line(sx_last, sy_last, event.x, event.y, fill=self.shape_stroke_color, width=self.shape_stroke_width, dash=(4, 2))
                    if not hasattr(self, '_shape_line_ids'):
                        self._shape_line_ids = []
                    self._shape_line_ids.append(line_id)
                    # Append new unscaled point
                    self._shape_points.append((ux, uy))
            else:
                # Update shape overlay for rectangle, circle, arrow
                if hasattr(self, '_shape_overlay_id') and self._shape_overlay_id is not None:
                    x0, y0 = self._shape_start
                    # Convert unscaled start to screen coords
                    sx0 = x0 * self.zoom
                    sy0 = y0 * self.zoom
                    self.canvas.coords(self._shape_overlay_id, sx0, sy0, event.x, event.y)
        elif self.current_tool in ["vector_rectangle", "vector_circle", "vector_ellipse", "vector_line"] and self.vector_tool_state["drawing"]:
            # Vector tools - update preview
            start_x, start_y = self.vector_tool_state["start_point"]
            # Convert unscaled start to screen coords
            sx0 = start_x * self.zoom
            sy0 = start_y * self.zoom
            # Create or update preview overlay
            if not hasattr(self, '_vector_preview_id'):
                if self.current_tool == "vector_rectangle":
                    self._vector_preview_id = self.canvas.create_rectangle(sx0, sy0, event.x, event.y, outline="#00ff00", dash=(4, 2))
                elif self.current_tool == "vector_circle":
                    self._vector_preview_id = self.canvas.create_oval(sx0, sy0, event.x, event.y, outline="#00ff00", dash=(4, 2))
                elif self.current_tool == "vector_ellipse":
                    self._vector_preview_id = self.canvas.create_oval(sx0, sy0, event.x, event.y, outline="#00ff00", dash=(4, 2))
                elif self.current_tool == "vector_line":
                    self._vector_preview_id = self.canvas.create_line(sx0, sy0, event.x, event.y, fill="#00ff00", width=2, dash=(4, 2))
            else:
                # Update existing preview
                if self.current_tool in ["vector_rectangle", "vector_circle", "vector_ellipse"]:
                    self.canvas.coords(self._vector_preview_id, sx0, sy0, event.x, event.y)
                elif self.current_tool == "vector_line":
                    self.canvas.coords(self._vector_preview_id, sx0, sy0, event.x, event.y)

    def _on_canvas_release(self, event):
        # End drawing or other stroke
        # Compute unscaled coordinates for operations
        ux = int(event.x / max(self.zoom, 1e-6))
        uy = int(event.y / max(self.zoom, 1e-6))
        if self.current_tool == "move":
            # After move, nothing else to do (history saved at press)
            pass
        elif self.current_tool == "mask":
            # mask painting done
            pass
        elif self.current_tool == "crop":
            # Perform crop based on rectangle using unscaled coordinates
            if self._crop_rect_id is not None and self._crop_start:
                x0, y0 = self._crop_start  # unscaled start
                x1, y1 = ux, uy  # unscaled end
                # Remove overlay rectangle
                self.canvas.delete(self._crop_rect_id)
                self._crop_rect_id = None
                self._perform_crop(x0, y0, x1, y1)
            self._crop_start = None
            # Reset tool after cropping
            self.current_tool = None
        elif self.current_tool == "filter_region":
            # On release, finalise region selection and apply filter accordingly
            shape = getattr(self, 'filter_selection_shape', 'rect')
            # Prepare to ask for filter name
            # Provide a list including extended filters
            filter_options = [
                "grayscale", "invert", "blur", "sharpen", "emboss", "edge", "contour", "detail", "smooth", "liquify", "sepia",
                "skin smooth", "frequency separation", "teeth whitening", "red eye removal", "warm", "cool", "vintage",
                "anime", "oil", "cyberpunk", "portrait optimize", "posterize", "solarize"
            ]
            filter_name = simpledialog.askstring("Filter", f"Enter filter to apply {filter_options}")
            if filter_name:
                fname = filter_name.strip().lower()
                if fname not in [opt.lower() for opt in filter_options]:
                    messagebox.showinfo("Filter", "Unsupported filter name.")
                else:
                    # Determine selection coordinates in canvas and layer space
                    if shape in ("rect", "ellipse") and self._filter_region_start is not None and self._filter_rect_id is not None:
                        x0, y0 = self._filter_region_start  # unscaled start
                        x1, y1 = ux, uy  # unscaled end
                        # Remove overlay shape
                        self.canvas.delete(self._filter_rect_id)
                        self._filter_rect_id = None
                        left = int(min(x0, x1))
                        upper = int(min(y0, y1))
                        right = int(max(x0, x1))
                        lower = int(max(y0, y1))
                        if right > left and lower > upper and self.current_layer_index is not None:
                            layer = self.layers[self.current_layer_index]
                            ox, oy = layer.offset
                            # Region in layer coordinates
                            lx0 = left - int(ox)
                            ly0 = upper - int(oy)
                            lx1 = right - int(ox)
                            ly1 = lower - int(oy)
                            # Clamp to layer bounds
                            lx0 = max(0, lx0)
                            ly0 = max(0, ly0)
                            lx1 = min(layer.original.width, lx1)
                            ly1 = min(layer.original.height, ly1)
                            if lx1 > lx0 and ly1 > ly0:
                                # Crop region from original
                                region = layer.original.crop((lx0, ly0, lx1, ly1)).copy()
                                # Apply filter on temporary layer
                                temp_layer = Layer(region, "temp")
                                try:
                                    temp_layer.apply_filter(fname)
                                except Exception:
                                    pass
                                filtered_region = temp_layer.original
                                if shape == "rect":
                                    # Replace rectangular area directly
                                    layer.original.paste(filtered_region, (lx0, ly0))
                                else:
                                    # For ellipse, create an elliptical mask and paste filtered region
                                    mask = Image.new("L", filtered_region.size, 0)
                                    draw = ImageDraw.Draw(mask)
                                    draw.ellipse([(0, 0), (filtered_region.width, filtered_region.height)], fill=255)
                                    layer.original.paste(filtered_region, (lx0, ly0), mask)
                                # Reapply adjustments and update composite
                                layer.apply_adjustments()
                                self._update_composite()
                        # Reset start coordinates
                        self._filter_region_start = None
                    elif shape == "polygon" and getattr(self, 'filter_lasso_points', None):
                        # Remove overlay lines
                        for lid in getattr(self, '_filter_line_ids', []):
                            self.canvas.delete(lid)
                        self._filter_line_ids = []
                        # Compute bounding box of polygon points
                        xs = [p[0] for p in self.filter_lasso_points]
                        ys = [p[1] for p in self.filter_lasso_points]
                        left = int(min(xs))
                        upper = int(min(ys))
                        right = int(max(xs))
                        lower = int(max(ys))
                        if right > left and lower > upper and self.current_layer_index is not None:
                            layer = self.layers[self.current_layer_index]
                            ox, oy = layer.offset
                            # Convert polygon points to layer coordinates
                            l_points = []
                            for (px, py) in self.filter_lasso_points:
                                l_points.append((px - ox, py - oy))
                            # Bounding box in layer coords
                            lx0 = left - int(ox)
                            ly0 = upper - int(oy)
                            lx1 = right - int(ox)
                            ly1 = lower - int(oy)
                            # Clamp box
                            lx0 = max(0, lx0)
                            ly0 = max(0, ly0)
                            lx1 = min(layer.original.width, lx1)
                            ly1 = min(layer.original.height, ly1)
                            if lx1 > lx0 and ly1 > ly0:
                                # Crop region and apply filter
                                region = layer.original.crop((lx0, ly0, lx1, ly1)).copy()
                                temp_layer = Layer(region, "temp")
                                try:
                                    temp_layer.apply_filter(fname)
                                except Exception:
                                    pass
                                filtered_region = temp_layer.original
                                # Build mask for polygon relative to bounding box
                                mask = Image.new("L", filtered_region.size, 0)
                                draw = ImageDraw.Draw(mask)
                                # Shift lasso points relative to bounding box
                                rel_points = []
                                for (px, py) in l_points:
                                    rel_points.append((px - lx0, py - ly0))
                                draw.polygon(rel_points, fill=255)
                                # Paste filtered region back with mask
                                layer.original.paste(filtered_region, (lx0, ly0), mask)
                                layer.apply_adjustments()
                                self._update_composite()
                        # Reset lasso points
                        self.filter_lasso_points = []
            # Clean up region selection variables
            if hasattr(self, '_filter_region_start'):
                self._filter_region_start = None
            # Ensure overlay is removed
            if hasattr(self, '_filter_rect_id') and self._filter_rect_id is not None:
                self.canvas.delete(self._filter_rect_id)
                self._filter_rect_id = None
            # Reset tool
            self.current_tool = None
        elif self.current_tool == "gradient":
            # Apply gradient on release
            if self._gradient_line_id is not None and self._gradient_start:
                x0, y0 = self._gradient_start
                x1, y1 = ux, uy
                dx = x1 - x0
                dy = y1 - y0
                if dx == 0 and dy == 0:
                    base_angle = self.gradient_settings.get('base_angle', 0.0)
                    length = max(1.0, float(self.brush_size))
                else:
                    base_angle = math.degrees(math.atan2(dy, dx))
                    length = math.hypot(dx, dy)
                if self.gradient_settings.get('snap', True):
                    snap_angle = round(base_angle / 15.0) * 15.0
                else:
                    snap_angle = base_angle
                rad = math.radians(snap_angle)
                x1 = int(x0 + math.cos(rad) * length)
                y1 = int(y0 + math.sin(rad) * length)
                self.gradient_settings['base_angle'] = snap_angle
                # Remove overlay line
                self.canvas.delete(self._gradient_line_id)
                self._gradient_line_id = None
                # Apply gradient
                self._apply_gradient(x0, y0, x1, y1, snap_angle)
            self._gradient_start = None
            # Reset tool after applying gradient
            self.current_tool = None
        elif self.current_tool == "shape":
            # Apply shape on release
            if self._shape_start:
                if self.shape_type == "polygon":
                    # For polygon, check if we have enough points
                    if hasattr(self, '_shape_points') and len(self._shape_points) >= 3:
                        # Close polygon by connecting last point to first
                        if len(self._shape_points) > 2:
                            first_x, first_y = self._shape_points[0]
                            last_x, last_y = self._shape_points[-1]
                            sx_first = first_x * self.zoom
                            sy_first = first_y * self.zoom
                            sx_last = last_x * self.zoom
                            sy_last = last_y * self.zoom
                            line_id = self.canvas.create_line(sx_last, sy_last, sx_first, sy_first, fill=self.shape_stroke_color, width=self.shape_stroke_width, dash=(4, 2))
                            if not hasattr(self, '_shape_line_ids'):
                                self._shape_line_ids = []
                            self._shape_line_ids.append(line_id)
                        # Apply polygon
                        self._apply_polygon(self._shape_points)
                        # Clean up overlays
                        if hasattr(self, '_shape_line_ids'):
                            for line_id in self._shape_line_ids:
                                self.canvas.delete(line_id)
                            self._shape_line_ids = []
                    else:
                        messagebox.showinfo("Polygon", "Polygon needs at least 3 points. Click to add more points.")
                        return  # Don't reset tool, allow more points
                else:
                    # For other shapes, apply based on start and end points
                    x0, y0 = self._shape_start
                    x1, y1 = ux, uy
                    # Remove overlay
                    if hasattr(self, '_shape_overlay_id') and self._shape_overlay_id is not None:
                        self.canvas.delete(self._shape_overlay_id)
                        self._shape_overlay_id = None
                    # Apply shape
                    self._apply_shape(x0, y0, x1, y1)
                self._shape_start = None
                # Reset tool after applying shape (except for polygon with < 3 points)
                if self.shape_type != "polygon" or (hasattr(self, '_shape_points') and len(self._shape_points) >= 3):
                    self.current_tool = None
        elif self.current_tool in ["vector_rectangle", "vector_circle", "vector_ellipse", "vector_line"] and self.vector_tool_state["drawing"]:
            # Apply vector shape on release
            start_x, start_y = self.vector_tool_state["start_point"]
            end_x, end_y = ux, uy
            
            # Remove preview overlay
            if hasattr(self, '_vector_preview_id'):
                self.canvas.delete(self._vector_preview_id)
                self._vector_preview_id = None
            
            # Create vector object
            self._create_vector_object(start_x, start_y, end_x, end_y)
            
            # Reset tool state
            self.vector_tool_state["drawing"] = False
            self.vector_tool_state["start_point"] = None
            self.current_tool = None
        elif self.current_tool == "extract":
            # Finalise extraction selection and create new layer
            shape = getattr(self, 'selection_shape', 'rect')
            layer_idx = self.current_layer_index
            if layer_idx is not None:
                layer = self.layers[layer_idx]
                ox, oy = layer.offset
                if shape in ("rect", "ellipse") and getattr(self, '_extract_start', None) is not None and getattr(self, '_extract_overlay_id', None) is not None:
                    # For rectangle/ellipse selection, convert start and end to unscaled coordinates
                    start_ux, start_uy = self._extract_start
                    end_ux, end_uy = ux, uy
                    # Remove overlay shape
                    self.canvas.delete(self._extract_overlay_id)
                    self._extract_overlay_id = None
                    # Determine bounding box in unscaled coords
                    left = int(min(start_ux, end_ux))
                    upper = int(min(start_uy, end_uy))
                    right = int(max(start_ux, end_ux))
                    lower = int(max(start_uy, end_uy))
                    if right > left and lower > upper:
                        # Convert to layer coords by subtracting layer offset
                        lx0 = left - int(ox)
                        ly0 = upper - int(oy)
                        lx1 = right - int(ox)
                        ly1 = lower - int(oy)
                        # Clamp to layer bounds
                        lx0 = max(0, lx0)
                        ly0 = max(0, ly0)
                        lx1 = min(layer.original.width, lx1)
                        ly1 = min(layer.original.height, ly1)
                        if lx1 > lx0 and ly1 > ly0:
                            # Ask for new layer name
                            new_name = simpledialog.askstring("Extract", "Name for extracted layer:", initialvalue="Extracted")
                            if not new_name:
                                new_name = "Extracted"
                            # Ask whether to remove selection from original
                            remove_original = messagebox.askyesno("Extract", "Remove selection from original layer?")
                            # Crop region from original
                            region = layer.original.crop((lx0, ly0, lx1, ly1)).copy()
                            if shape == "ellipse":
                                # For ellipse, create mask and blank out outside region
                                mask = Image.new("L", region.size, 0)
                                draw = ImageDraw.Draw(mask)
                                draw.ellipse([(0, 0), (region.width, region.height)], fill=255)
                                # Compose region onto transparent background using mask
                                new_rgba = Image.new("RGBA", region.size, (0, 0, 0, 0))
                                new_rgba.paste(region, (0, 0), mask)
                                extracted_img = new_rgba
                            else:
                                extracted_img = region
                            # Create new layer
                            new_layer = Layer(extracted_img, new_name)
                            # Set offset relative to document (ox+lx0)
                            new_layer.offset = (ox + lx0, oy + ly0)
                            # Append layer and select it
                            self.layers.append(new_layer)
                            self.current_layer_index = len(self.layers) - 1
                            # Remove region from original if requested
                            if remove_original:
                                # Blank out region in original
                                blank = Image.new("RGBA", region.size, (0, 0, 0, 0))
                                if shape == "ellipse":
                                    layer.original.paste(blank, (lx0, ly0), mask)
                                else:
                                    layer.original.paste(blank, (lx0, ly0))
                                layer.apply_adjustments()
                            # Refresh list and composite
                            self._refresh_layer_list()
                            self._update_composite()
                    # Reset start variable for next extraction
                    self._extract_start = None
                elif shape == "polygon" and getattr(self, 'lasso_points', None):
                    # Polygon extraction using unscaled lasso_points
                    # Remove overlay line IDs
                    for lid in getattr(self, '_extract_line_ids', []):
                        self.canvas.delete(lid)
                    self._extract_line_ids = []
                    # Compute bounding box of polygon in unscaled coords
                    xs = [p[0] for p in self.lasso_points]
                    ys = [p[1] for p in self.lasso_points]
                    if xs and ys:
                        left = int(min(xs))
                        upper = int(min(ys))
                        right = int(max(xs))
                        lower = int(max(ys))
                        if right > left and lower > upper:
                            # Convert bounding box to layer coordinates
                            lx0 = left - int(ox)
                            ly0 = upper - int(oy)
                            lx1 = right - int(ox)
                            ly1 = lower - int(oy)
                            # Clamp to layer bounds
                            lx0 = max(0, lx0)
                            ly0 = max(0, ly0)
                            lx1 = min(layer.original.width, lx1)
                            ly1 = min(layer.original.height, ly1)
                            if lx1 > lx0 and ly1 > ly0:
                                # Ask for new layer name
                                new_name = simpledialog.askstring("Extract", "Name for extracted layer:", initialvalue="Extracted")
                                if not new_name:
                                    new_name = "Extracted"
                                remove_original = messagebox.askyesno("Extract", "Remove selection from original layer?")
                                # Crop region from original
                                region = layer.original.crop((lx0, ly0, lx1, ly1)).copy()
                                # Build mask for polygon relative to bounding box
                                mask = Image.new("L", (lx1 - lx0, ly1 - ly0), 0)
                                draw = ImageDraw.Draw(mask)
                                # Convert lasso points to layer coords and relative to bounding box
                                rel_points = []
                                for (px, py) in self.lasso_points:
                                    rel_points.append((px - ox - lx0, py - oy - ly0))
                                draw.polygon(rel_points, fill=255)
                                # Extract image onto transparent background
                                new_rgba = Image.new("RGBA", mask.size, (0, 0, 0, 0))
                                new_rgba.paste(region, (0, 0), mask)
                                # Create new layer
                                new_layer = Layer(new_rgba, new_name)
                                new_layer.offset = (ox + lx0, oy + ly0)
                                # Append and select new layer
                                self.layers.append(new_layer)
                                self.current_layer_index = len(self.layers) - 1
                                if remove_original:
                                    # Remove polygon from original using mask
                                    blank = Image.new("RGBA", region.size, (0, 0, 0, 0))
                                    layer.original.paste(blank, (lx0, ly0), mask)
                                    layer.apply_adjustments()
                                # Refresh list and composite
                                self._refresh_layer_list()
                                self._update_composite()
                    # Reset lasso points list for next extraction
                    self.lasso_points = []
            # Reset tool state regardless of outcome
            self.current_tool = None
        elif self.current_tool == "quickselect":
            # Finish quick select operation and reset tool
            # Tool state is reset after stroke
            self.current_tool = None
        if self.current_tool == "smart_eraser":
            self._smart_eraser_color = None
        self._drag_prev = None
        # Reset history flag after stroke
        self._reset_history_flag()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def _paint_at(self, x: int, y: int):
        """Draw a dot at the given coordinates on the current layer with the selected brush texture."""
        layer = self.layers[self.current_layer_index]
        draw = ImageDraw.Draw(layer.original)
        
        # Get brush texture
        texture = getattr(self, 'brush_texture', 'solid')
        
        if texture == "solid":
            # Original solid brush
            r = self.brush_size / 2
            draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=self.brush_color)
        elif texture == "spray":
            # Spray texture - multiple small dots with random distribution
            self._paint_spray(draw, x, y)
        elif texture == "chalk":
            # Chalk texture - irregular, textured strokes
            self._paint_chalk(draw, x, y)
        elif texture == "calligraphy":
            # Calligraphy texture - pressure-sensitive strokes
            self._paint_calligraphy(draw, x, y)
        
        # Apply brightness and alpha to update layer.image
        layer.apply_adjustments()
        self._update_composite()

    def _paint_spray(self, draw, x: int, y: int):
        """Paint with spray texture - multiple small dots with random distribution."""
        import random
        r = self.brush_size / 2
        num_dots = max(1, int(r * 2))  # More dots for larger brushes
        
        for _ in range(num_dots):
            # Random offset within brush radius
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, r)
            offset_x = int(distance * math.cos(angle))
            offset_y = int(distance * math.sin(angle))
            
            # Random size for each dot
            dot_size = random.uniform(0.5, 2.0)
            
            # Draw small dot
            draw.ellipse([
                (x + offset_x - dot_size, y + offset_y - dot_size),
                (x + offset_x + dot_size, y + offset_y + dot_size)
            ], fill=self.brush_color)

    def _paint_chalk(self, draw, x: int, y: int):
        """Paint with chalk texture - irregular, textured strokes."""
        import random
        r = self.brush_size / 2
        
        # Create irregular chalk stroke with multiple overlapping circles
        for i in range(int(r)):
            # Vary the radius and position slightly
            offset_x = random.uniform(-r/3, r/3)
            offset_y = random.uniform(-r/3, r/3)
            current_r = r - i + random.uniform(-1, 1)
            current_r = max(1, current_r)
            
            # Vary opacity slightly for chalk effect
            alpha = random.uniform(200, 255)
            color_with_alpha = self.brush_color + f"{int(alpha):02x}"
            
            draw.ellipse([
                (x + offset_x - current_r, y + offset_y - current_r),
                (x + offset_x + current_r, y + offset_y + current_r)
            ], fill=color_with_alpha)

    def _paint_calligraphy(self, draw, x: int, y: int):
        """Paint with calligraphy texture - pressure-sensitive strokes."""
        import random
        r = self.brush_size / 2

        # Create calligraphy stroke with varying width
        # Simulate pressure by varying the stroke width
        pressure = random.uniform(0.3, 1.0)
        current_r = r * pressure
        
        # Add some randomness to simulate hand movement
