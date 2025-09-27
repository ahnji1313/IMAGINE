        offset_x = random.uniform(-r/4, r/4)
        offset_y = random.uniform(-r/4, r/4)
        
        # Draw main stroke
        draw.ellipse([
            (x + offset_x - current_r, y + offset_y - current_r),
            (x + offset_x + current_r, y + offset_y + current_r)
        ], fill=self.brush_color)
        
        # Add some texture with smaller dots
        for _ in range(int(r/2)):
            tex_x = x + random.uniform(-r, r)
            tex_y = y + random.uniform(-r, r)
            tex_r = random.uniform(0.5, 1.5)
            draw.ellipse([
                (tex_x - tex_r, tex_y - tex_r),
                (tex_x + tex_r, tex_y + tex_r)
            ], fill=self.brush_color)

    def _get_stamp_for_size(self, size: int) -> Optional[Image.Image]:
        if self.stamp_image is None:
            return None
        scale = max(16, int(size * 4))
        if scale not in self._stamp_cache:
            try:
                resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
            except Exception:
                resample = Image.ANTIALIAS
            self._stamp_cache[scale] = self.stamp_image.resize((scale, scale), resample=resample)
        return self._stamp_cache.get(scale)

    def _apply_stamp_at(self, x: int, y: int) -> None:
        if self.current_layer_index is None:
            return
        stamp = self._get_stamp_for_size(self.brush_size)
        if stamp is None:
            return
        layer = self.layers[self.current_layer_index]
        ox, oy = layer.offset
        lx = int(x - ox)
        ly = int(y - oy)
        width, height = layer.original.size
        sw, sh = stamp.size
        x0 = lx - sw // 2
        y0 = ly - sh // 2
        sx0 = 0
        sy0 = 0
        x1 = x0 + sw
        y1 = y0 + sh
        if x0 < 0:
            sx0 = -x0
            x0 = 0
        if y0 < 0:
            sy0 = -y0
            y0 = 0
        if x1 > width:
            sw -= x1 - width
        if y1 > height:
            sh -= y1 - height
        if sw <= 0 or sh <= 0:
            return
        stamp_region = stamp.crop((sx0, sy0, sx0 + sw, sy0 + sh)) if (sx0 or sy0 or sw != stamp.width or sh != stamp.height) else stamp
        layer.original.paste(stamp_region, (x0, y0), stamp_region)
        layer.apply_adjustments()
        self._update_composite()

    def _stamp_line(self, x0: int, y0: int, x1: int, y1: int) -> None:
        dist = math.hypot(x1 - x0, y1 - y0)
        step = max(4, self.brush_size / 2)
        steps = int(dist / step) + 1
        for i in range(steps + 1):
            t = i / max(1, steps)
            x = int(x0 + (x1 - x0) * t)
            y = int(y0 + (y1 - y0) * t)
            self._apply_stamp_at(x, y)

    def _get_pattern_stamp(self, size: int) -> Image.Image:
        pattern_type = self.pattern_settings.get('type', 'checker').lower()
        colors = self.pattern_settings.get('colors', [self.primary_color, self.secondary_color])
        tile_size = max(4, int(self.pattern_settings.get('size', 32)))
        scale = max(16, int(size * 4))
        key = (scale, pattern_type, tuple(colors), tile_size)
        if key in self._pattern_cache:
            return self._pattern_cache[key]

        tile = Image.new('RGBA', (tile_size, tile_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(tile)
        col_a = self._hex_to_rgb_tuple(colors[0]) + (255,)
        col_b = self._hex_to_rgb_tuple(colors[1]) + (255,)
        if pattern_type.startswith('stripe'):
            stripe = max(1, tile_size // 4)
            for x in range(tile_size):
                fill = col_a if (x // stripe) % 2 == 0 else col_b
                draw.line([(x, 0), (x, tile_size)], fill=fill)
        elif pattern_type.startswith('diag'):
            stripe = max(1, tile_size // 4)
            for x in range(tile_size * 2):
                fill = col_a if (x // stripe) % 2 == 0 else col_b
                draw.line([(x, 0), (x - tile_size, tile_size)], fill=fill)
        elif pattern_type.startswith('dot'):
            draw.rectangle((0, 0, tile_size, tile_size), fill=col_b)
            radius = max(1, tile_size // 6)
            draw.ellipse((tile_size // 2 - radius, tile_size // 2 - radius,
                          tile_size // 2 + radius, tile_size // 2 + radius), fill=col_a)
        else:  # checker
            half = tile_size // 2
            draw.rectangle((0, 0, half, half), fill=col_a)
            draw.rectangle((half, 0, tile_size, half), fill=col_b)
            draw.rectangle((0, half, half, tile_size), fill=col_b)
            draw.rectangle((half, half, tile_size, tile_size), fill=col_a)

        stamp = Image.new('RGBA', (scale, scale), (0, 0, 0, 0))
        for y in range(0, scale, tile_size):
            for x in range(0, scale, tile_size):
                stamp.paste(tile, (x, y))
        mask = Image.new('L', (scale, scale), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, scale, scale), fill=255)
        stamp.putalpha(mask)
        self._pattern_cache[key] = stamp
        return stamp

    def _apply_pattern_at(self, x: int, y: int) -> None:
        if self.current_layer_index is None:
            return
        stamp = self._get_pattern_stamp(self.brush_size)
        layer = self.layers[self.current_layer_index]
        ox, oy = layer.offset
        lx = int(x - ox)
        ly = int(y - oy)
        width, height = layer.original.size
        sw, sh = stamp.size
        x0 = lx - sw // 2
        y0 = ly - sh // 2
        sx0 = 0
        sy0 = 0
        x1 = x0 + sw
        y1 = y0 + sh
        if x0 < 0:
            sx0 = -x0
            x0 = 0
        if y0 < 0:
            sy0 = -y0
            y0 = 0
        if x1 > width:
            sw -= x1 - width
        if y1 > height:
            sh -= y1 - height
        if sw <= 0 or sh <= 0:
            return
        region = stamp.crop((sx0, sy0, sx0 + sw, sy0 + sh)) if (sx0 or sy0 or sw != stamp.width or sh != stamp.height) else stamp
        layer.original.paste(region, (x0, y0), region)
        layer.apply_adjustments()
        self._update_composite()

    def _pattern_line(self, x0: int, y0: int, x1: int, y1: int) -> None:
        dist = math.hypot(x1 - x0, y1 - y0)
        step = max(4, self.brush_size / 2)
        steps = int(dist / step) + 1
        for i in range(steps + 1):
            t = i / max(1, steps)
            x = int(x0 + (x1 - x0) * t)
            y = int(y0 + (y1 - y0) * t)
            self._apply_pattern_at(x, y)

    def _smart_erase_at(self, x: int, y: int) -> None:
        if self.current_layer_index is None or self._smart_eraser_color is None:
            return
        layer = self.layers[self.current_layer_index]
        arr = np.array(layer.original.convert('RGBA'))
        ox, oy = layer.offset
        lx = int(x - ox)
        ly = int(y - oy)
        radius = max(4, int(self.brush_size))
        x0 = max(0, lx - radius)
        y0 = max(0, ly - radius)
        x1 = min(arr.shape[1], lx + radius)
        y1 = min(arr.shape[0], ly + radius)
        if x1 <= x0 or y1 <= y0:
            return
        region = arr[y0:y1, x0:x1]
        rgb = region[..., :3].astype(np.float32)
        target = np.array(self._smart_eraser_color, dtype=np.float32)
        diff = np.sqrt(((rgb - target) ** 2).sum(axis=2))
        tol_var = getattr(self, 'smart_eraser_tol_var', None)
        tol = float(tol_var.get()) if tol_var is not None else 35.0
        mask = diff <= tol
        region[..., 3][mask] = 0
        arr[y0:y1, x0:x1] = region
        layer.original = Image.fromarray(arr, 'RGBA')
        layer.apply_adjustments()
        self._update_composite()

    def _smart_erase_line(self, x0: int, y0: int, x1: int, y1: int) -> None:
        dist = math.hypot(x1 - x0, y1 - y0)
        step = max(4, self.brush_size / 2)
        steps = int(dist / step) + 1
        for i in range(steps + 1):
            t = i / max(1, steps)
            x = int(x0 + (x1 - x0) * t)
            y = int(y0 + (y1 - y0) * t)
            self._smart_erase_at(x, y)

    def _apply_gradient(self, x0: int, y0: int, x1: int, y1: int, angle: Optional[float] = None):
        """Apply a gradient between two points on the current layer."""
        if self.current_layer_index is None:
            return

        layer = self.layers[self.current_layer_index]
        ox, oy = layer.offset
        
        # Convert canvas coordinates to layer coordinates
        lx0 = x0 - int(ox)
        ly0 = y0 - int(oy)
        lx1 = x1 - int(ox)
        ly1 = y1 - int(oy)
        
        width, height = layer.original.size

        gradient_img = self._create_gradient_image(
            width, height, lx0, ly0, lx1, ly1,
            self.gradient_settings, angle
        )
        
        if gradient_img:
            # Paste gradient onto layer
            layer.original.paste(gradient_img, (0, 0), gradient_img)
            layer.apply_adjustments()
            self._update_composite()

    def _create_gradient_image(self, width: int, height: int, x0: int, y0: int, x1: int, y1: int,
                              settings: dict, angle: Optional[float] = None) -> Image.Image:
        """Create a gradient image with the specified parameters."""

        stops = settings.get('stops') or [(0.0, self.primary_color), (1.0, self.secondary_color)]
        processed: List[Tuple[float, Tuple[int, int, int]]] = []
        for pos, colour in stops:
            processed.append((max(0.0, min(1.0, float(pos))), self._hex_to_rgb_tuple(colour)))
        processed.sort(key=lambda item: item[0])
        if not processed:
            processed = [(0.0, self._hex_to_rgb_tuple(self.primary_color)), (1.0, self._hex_to_rgb_tuple(self.secondary_color))]

        def interpolate(t: float) -> Tuple[int, int, int]:
            if t <= processed[0][0]:
                return processed[0][1]
            if t >= processed[-1][0]:
                return processed[-1][1]
            for (p0, c0), (p1, c1) in zip(processed[:-1], processed[1:]):
                if p0 <= t <= p1:
                    span = max(1e-6, p1 - p0)
                    local = (t - p0) / span
                    return tuple(int(c0[i] + (c1[i] - c0[i]) * local) for i in range(3))
            return processed[-1][1]

        gradient = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        gtype = settings.get('type', 'linear')

        if gtype == 'linear':
            dx = x1 - x0
            dy = y1 - y0
            if angle is not None:
                length = math.hypot(dx, dy)
                if length == 0:
                    length = math.hypot(width, height)
                rad = math.radians(angle)
                dx = math.cos(rad) * length
                dy = math.sin(rad) * length
            length_sq = dx * dx + dy * dy
            if length_sq == 0:
                length_sq = 1.0
            for y in range(height):
                for x in range(width):
                    t = ((x - x0) * dx + (y - y0) * dy) / length_sq
                    t = max(0.0, min(1.0, t))
                    r, g, b = interpolate(t)
                    gradient.putpixel((x, y), (r, g, b, 255))
        elif gtype == 'radial':
            radius = math.hypot(x1 - x0, y1 - y0)
            if radius == 0:
                radius = math.hypot(width, height) / 2.0
            for y in range(height):
                for x in range(width):
                    dist = math.hypot(x - x0, y - y0)
                    t = max(0.0, min(1.0, dist / radius))
                    r, g, b = interpolate(t)
                    gradient.putpixel((x, y), (r, g, b, 255))
        elif gtype == 'diamond':
            max_dist = max(1.0, max(x0, width - x0, y0, height - y0))
            for y in range(height):
                for x in range(width):
                    dist = abs(x - x0) + abs(y - y0)
                    t = max(0.0, min(1.0, dist / max_dist))
                    r, g, b = interpolate(t)
                    gradient.putpixel((x, y), (r, g, b, 255))
        else:
            return self._create_gradient_image(width, height, x0, y0, x1, y1, {"type": "linear", "stops": processed}, angle)

        return gradient

    def _apply_shape(self, x0: int, y0: int, x1: int, y1: int):
        """Apply a shape between two points on the current layer."""
        if self.current_layer_index is None:
            return
            
        layer = self.layers[self.current_layer_index]
        ox, oy = layer.offset
        
        # Convert canvas coordinates to layer coordinates
        lx0 = x0 - int(ox)
        ly0 = y0 - int(oy)
        lx1 = x1 - int(ox)
        ly1 = y1 - int(oy)
        
        # Get layer dimensions
        width, height = layer.original.size
        
        # Create shape image
        shape_img = self._create_shape_image(
            width, height, lx0, ly0, lx1, ly1,
            self.shape_fill_color, self.shape_stroke_color,
            self.shape_stroke_width, self.shape_fill_style,
            self.shape_type
        )
        
        if shape_img:
            # Paste shape onto layer
            layer.original.paste(shape_img, (0, 0), shape_img)
            layer.apply_adjustments()
            self._update_composite()

    def _apply_polygon(self, points: list):
        """Apply a polygon shape on the current layer."""
        if self.current_layer_index is None:
            return
            
        layer = self.layers[self.current_layer_index]
        ox, oy = layer.offset
        
        # Convert canvas coordinates to layer coordinates
        l_points = []
        for x, y in points:
            lx = x - int(ox)
            ly = y - int(oy)
            l_points.append((lx, ly))
        
        # Get layer dimensions
        width, height = layer.original.size
        
        # Create polygon image
        polygon_img = self._create_polygon_image(
            width, height, l_points,
            self.shape_fill_color, self.shape_stroke_color,
            self.shape_stroke_width, self.shape_fill_style
        )
        
        if polygon_img:
            # Paste polygon onto layer
            layer.original.paste(polygon_img, (0, 0), polygon_img)
            layer.apply_adjustments()
            self._update_composite()

    def _create_shape_image(self, width: int, height: int, x0: int, y0: int, x1: int, y1: int,
                           fill_color: str, stroke_color: str, stroke_width: int, 
                           fill_style: str, shape_type: str) -> Image.Image:
        """Create a shape image with the specified parameters."""
        # Parse colors
        def hex_to_rgba(hex_color):
            hex_color = hex_color.lstrip('#')
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (r, g, b, 255)
        
        fill_rgba = hex_to_rgba(fill_color)
        stroke_rgba = hex_to_rgba(stroke_color)
        
        # Create shape image
        shape = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(shape)
        
        # Ensure coordinates are within bounds
        x0 = max(0, min(width-1, x0))
        y0 = max(0, min(height-1, y0))
        x1 = max(0, min(width-1, x1))
        y1 = max(0, min(height-1, y1))
        
        if shape_type == "rectangle":
            if fill_style in ("filled", "both"):
                draw.rectangle([x0, y0, x1, y1], fill=fill_rgba)
            if fill_style in ("outline", "both") and stroke_width > 0:
                # Draw outline by drawing multiple rectangles
                for i in range(stroke_width):
                    draw.rectangle([x0-i, y0-i, x1+i, y1+i], outline=stroke_rgba)
                    
        elif shape_type == "circle":
            if fill_style in ("filled", "both"):
                draw.ellipse([x0, y0, x1, y1], fill=fill_rgba)
            if fill_style in ("outline", "both") and stroke_width > 0:
                # Draw outline by drawing multiple ellipses
                for i in range(stroke_width):
                    draw.ellipse([x0-i, y0-i, x1+i, y1+i], outline=stroke_rgba)
                    
        elif shape_type == "arrow":
            # Create arrow shape
            if fill_style in ("filled", "both"):
                self._draw_arrow_filled(draw, x0, y0, x1, y1, fill_rgba)
            if fill_style in ("outline", "both") and stroke_width > 0:
                self._draw_arrow_outline(draw, x0, y0, x1, y1, stroke_rgba, stroke_width)
        
        return shape

    def _create_polygon_image(self, width: int, height: int, points: list,
                             fill_color: str, stroke_color: str, stroke_width: int, 
                             fill_style: str) -> Image.Image:
        """Create a polygon image with the specified parameters."""
        # Parse colors
        def hex_to_rgba(hex_color):
            hex_color = hex_color.lstrip('#')
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (r, g, b, 255)
        
        fill_rgba = hex_to_rgba(fill_color)
        stroke_rgba = hex_to_rgba(stroke_color)
        
        # Create polygon image
        polygon = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(polygon)
        
        # Ensure points are within bounds
        bounded_points = []
        for x, y in points:
            x = max(0, min(width-1, x))
            y = max(0, min(height-1, y))
            bounded_points.append((x, y))
        
        if len(bounded_points) >= 3:
            if fill_style in ("filled", "both"):
                draw.polygon(bounded_points, fill=fill_rgba)
            if fill_style in ("outline", "both") and stroke_width > 0:
                # Draw outline by drawing multiple polygons
                for i in range(stroke_width):
                    offset_points = []
                    for x, y in bounded_points:
                        # Simple offset - could be improved with proper polygon offsetting
                        offset_points.append((x-i, y-i))
                    draw.polygon(offset_points, outline=stroke_rgba)
        
        return polygon

    def _draw_arrow_filled(self, draw, x0: int, y0: int, x1: int, y1: int, fill_rgba):
        """Draw a filled arrow shape."""
        # Calculate arrow direction and length
        dx = x1 - x0
        dy = y1 - y0
        length = (dx*dx + dy*dy)**0.5
        
        if length == 0:
            return
            
        # Normalize direction
        dx_norm = dx / length
        dy_norm = dy / length
        
        # Arrow parameters
        arrow_length = min(length * 0.3, 20)  # Arrow head length
        arrow_width = min(length * 0.1, 8)    # Arrow head width
        
        # Calculate arrow head points
        # Main arrow line
        draw.line([x0, y0, x1, y1], fill=fill_rgba, width=3)
        
        # Arrow head
        head_x = x1 - arrow_length * dx_norm
        head_y = y1 - arrow_length * dy_norm
        
        # Perpendicular direction for arrow head
        perp_x = -dy_norm * arrow_width
        perp_y = dx_norm * arrow_width
        
        # Arrow head points
        p1_x = int(head_x + perp_x)
        p1_y = int(head_y + perp_y)
        p2_x = int(head_x - perp_x)
        p2_y = int(head_y - perp_y)
        
        # Draw arrow head
        draw.polygon([(x1, y1), (p1_x, p1_y), (p2_x, p2_y)], fill=fill_rgba)

    def _draw_arrow_outline(self, draw, x0: int, y0: int, x1: int, y1: int, stroke_rgba, stroke_width: int):
        """Draw an outlined arrow shape."""
        # Draw main line
        draw.line([x0, y0, x1, y1], fill=stroke_rgba, width=stroke_width)
        
        # Calculate arrow head
        dx = x1 - x0
        dy = y1 - y0
        length = (dx*dx + dy*dy)**0.5
        
        if length == 0:
            return
            
        # Normalize direction
        dx_norm = dx / length
        dy_norm = dy / length
        
        # Arrow parameters
        arrow_length = min(length * 0.3, 20)
        arrow_width = min(length * 0.1, 8)
        
        # Calculate arrow head points
        head_x = x1 - arrow_length * dx_norm
        head_y = y1 - arrow_length * dy_norm
        
        # Perpendicular direction
        perp_x = -dy_norm * arrow_width
        perp_y = dx_norm * arrow_width
        
        # Arrow head points
        p1_x = int(head_x + perp_x)
        p1_y = int(head_y + perp_y)
        p2_x = int(head_x - perp_x)
        p2_y = int(head_y - perp_y)
        
        # Draw arrow head lines
        draw.line([x1, y1, p1_x, p1_y], fill=stroke_rgba, width=stroke_width)
        draw.line([x1, y1, p2_x, p2_y], fill=stroke_rgba, width=stroke_width)

    def _magic_wand_at(self, ux: int, uy: int) -> None:
        """Perform magic wand selection at the given unscaled coordinates and apply a filter to that area."""
        # Ensure a layer is selected
        if self.current_layer_index is None:
            return
        layer = self.layers[self.current_layer_index]
        ox, oy = layer.offset
        # Convert to layer coordinate
        lx = ux - int(ox)
        ly = uy - int(oy)
        if lx < 0 or ly < 0 or lx >= layer.original.width or ly >= layer.original.height:
            return
        # Ask for filter name
        filter_options = [
            "grayscale", "invert", "blur", "sharpen", "emboss", "edge", "contour", "detail", "smooth", "liquify", "sepia",
            "skin smooth", "frequency separation", "teeth whitening", "red eye removal", "warm", "cool", "vintage",
            "anime", "oil", "cyberpunk", "portrait optimize", "posterize", "solarize"
        ]
        fname = simpledialog.askstring("Magic Wand Filter", f"Enter filter to apply {filter_options}")
        if not fname:
            return
        fname = fname.strip().lower()
        if fname not in [opt.lower() for opt in filter_options]:
            messagebox.showinfo("Filter", "Unsupported filter name.")
            return
        # Save history
        self._current_action_desc = "Magic Wand"
        self._save_history()
        # Extract region mask based on tolerance
        tol = getattr(self, 'magic_tolerance', 30)
        # Convert image to numpy array for computation
        np_img = np.array(layer.original.convert('RGB'))
        base_color = np_img[ly, lx].astype(int)
        diff = np.abs(np_img.astype(int) - base_color)
        diff_sum = diff.sum(axis=2)
        mask = (diff_sum <= tol).astype(np.uint8)
        # Determine bounding box of mask
        ys, xs = np.where(mask == 1)
        if len(xs) == 0 or len(ys) == 0:
            return
        x0 = int(xs.min())
        x1 = int(xs.max()) + 1
        y0 = int(ys.min())
        y1 = int(ys.max()) + 1
        # Crop region
        region = layer.original.crop((x0, y0, x1, y1)).copy()
        # Apply filter on temporary layer
        temp_layer = Layer(region, "temp")
        try:
            temp_layer.apply_filter(fname)
        except Exception:
            pass
        filtered = temp_layer.original
        # Create mask image for the region
        submask = (mask[y0:y1, x0:x1] * 255).astype('uint8')
        mask_img = Image.fromarray(submask)
        # Smooth mask edges with Gaussian blur to create a soft brush effect
        # Blur radius proportional to tolerance for natural feathering
        try:
            mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=3))
            # Threshold blurred mask back to binary (soft edges remain through alpha blending)
            mask_img = mask_img.point(lambda p: 255 if p > 64 else 0)
        except Exception:
            pass
        # Paste filtered region back using smoothed mask
        layer.original.paste(filtered, (x0, y0), mask_img)
        # Reapply adjustments and update composite
        layer.apply_adjustments()
        self._update_composite()

    def _paint_line(self, x0: int, y0: int, x1: int, y1: int):
        """Draw a thick line between two points on the current layer."""
        layer = self.layers[self.current_layer_index]
        draw = ImageDraw.Draw(layer.original)
        draw.line([(x0, y0), (x1, y1)], fill=self.brush_color, width=self.brush_size)
        # If brush size > 1, draw circles at endpoints to avoid gaps
        r = self.brush_size / 2
        draw.ellipse([(x0 - r, y0 - r), (x0 + r, y0 + r)], fill=self.brush_color)
        draw.ellipse([(x1 - r, y1 - r), (x1 + r, y1 + r)], fill=self.brush_color)
        layer.apply_adjustments()
        self._update_composite()

    def _add_text(self, x: int, y: int):
        """Render pending text at the specified location on the current layer."""
        # Create a new TextLayer placed above the current layer
        canvas_size = self.layers[0].image.size if self.layers else (800, 600)
        text = getattr(self, 'pending_text', '')
        font_spec = getattr(self, 'pending_font_name', None)
        font_size = getattr(self, 'pending_font_size', 32)
        color = getattr(self, 'pending_text_color', '#000000')
        effects = getattr(self, 'pending_text_effects', {})
        # Convert canvas click coords (scaled) to unscaled coordinates
        ux = int(x / max(self.zoom, 1e-6))
        uy = int(y / max(self.zoom, 1e-6))
        tl = TextLayer(canvas_size, text=text, name=f"Text {len(self.layers)}", font_spec=font_spec, font_size=font_size, color=color, position=(ux, uy), effects=effects)
        tl.visible = True
        # Render text into its original image and apply adjustments
        try:
            tl.render_text()
        except Exception:
            try:
                tl.render_text()
            except Exception:
                pass
        # Insert above current layer or at top
        insert_index = (self.current_layer_index + 1) if self.current_layer_index is not None else len(self.layers)
        self.layers.insert(insert_index, tl)
        self.current_layer_index = insert_index
        self._refresh_layer_list()
        self._update_composite()

    # ------------------------------------------------------------------
    # Mask painting helpers
    # ------------------------------------------------------------------
    def _mask_at(self, x: int, y: int) -> None:
        """Paint on the current layer's mask at a single point.

        When mask_mode is 'hide', this draws black (0) to hide pixels.
        When mask_mode is 'reveal', it draws white (255) to reveal.
        """
        if self.current_layer_index is None:
            return
        layer = self.layers[self.current_layer_index]
        # Determine value to paint
        value = 0 if getattr(self, 'mask_mode', 'hide') == 'hide' else 255
        r = self.brush_size / 2
        draw = ImageDraw.Draw(layer.mask)
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=value)
        layer.apply_adjustments()
        self._update_composite()

    def _mask_line(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Paint on the mask along a line by interpolating points."""
        dx = x1 - x0
        dy = y1 - y0
        dist = (dx * dx + dy * dy) ** 0.5
        steps = int(dist / (self.brush_size / 2)) + 1
        for i in range(steps + 1):
            t = i / steps
            x = int(x0 + dx * t)
            y = int(y0 + dy * t)
            self._mask_at(x, y)

    # ------------------------------------------------------------------
    # Cropping helper
    # ------------------------------------------------------------------
    def _perform_crop(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Crop all layers to the rectangular region defined by two points."""
        if not self.layers:
            return
        # Determine bounding box and clamp within image bounds
        x0_clamped = max(0, min(self.canvas.winfo_width(), x0))
        y0_clamped = max(0, min(self.canvas.winfo_height(), y0))
        x1_clamped = max(0, min(self.canvas.winfo_width(), x1))
        y1_clamped = max(0, min(self.canvas.winfo_height(), y1))
        left = int(min(x0_clamped, x1_clamped))
        upper = int(min(y0_clamped, y1_clamped))
        right = int(max(x0_clamped, x1_clamped))
        lower = int(max(y0_clamped, y1_clamped))
        if right - left <= 0 or lower - upper <= 0:
            return
        box = (left, upper, right, lower)
        # Crop each layer's original and mask, adjusting offset
        for layer in self.layers:
            # Adjust offset: remove region from left/top
            ox, oy = layer.offset
            new_offset = (ox - left, oy - upper)
            # Crop original and mask to bounding box considering offset
            # Create a copy of the image with offset applied
            # For cropping, we need to align the crop box relative to the layer image
            # The displayed position of layer pixel (img_x, img_y) on canvas is (img_x + ox, img_y + oy)
            # So pixel corresponds to original coordinate (img_x) = canvas_x - ox.
            # Therefore cropping region for original is box shifted by (-ox, -oy)
            shift_box = (box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy)
            # Crop original and mask
            layer.original = layer.original.crop(shift_box)
            layer.mask = layer.mask.crop(shift_box)
            layer.offset = new_offset
            layer.apply_adjustments()
        # Update canvas size
        new_width = right - left
        new_height = lower - upper
        self.canvas.config(width=new_width, height=new_height)
        self._refresh_layer_list()
        self._update_composite()

    # ------------------------------------------------------------------
    # Selection tool helpers
    # ------------------------------------------------------------------
    def _select_layer_at(self, canvas_x: int, canvas_y: int) -> None:
        """Set current_layer_index to the topmost visible layer at the given canvas coordinates."""
        # Iterate top to bottom
        for idx in reversed(range(len(self.layers))):
            layer = self.layers[idx]
            if not layer.visible:
                continue
            # Calculate relative coordinates inside the layer
            ox, oy = layer.offset
            rel_x = canvas_x - int(ox)
            rel_y = canvas_y - int(oy)
            if rel_x < 0 or rel_y < 0 or rel_x >= layer.image.width or rel_y >= layer.image.height:
                continue
            # Check alpha at this pixel
            try:
                pixel = layer.image.getpixel((int(rel_x), int(rel_y)))
            except Exception:
                continue
            if len(pixel) == 4 and pixel[3] > 0:
                self.current_layer_index = idx
                # Update sliders to selected layer
                self.alpha_slider.set(layer.alpha)
                self.brightness_slider.set(layer.brightness)
                if hasattr(self, 'contrast_slider'):
                    self.contrast_slider.set(layer.contrast)
                if hasattr(self, 'color_slider'):
                    self.color_slider.set(layer.color)
                self._refresh_layer_list()
                return

    def _on_canvas_double_click(self, event):
        """Handle double-clicks on the canvas: if a text layer is clicked, open edit dialog."""
        ux = int(event.x / max(self.zoom, 1e-6))
        uy = int(event.y / max(self.zoom, 1e-6))
        # Select topmost layer under cursor
        self._select_layer_at(ux, uy)
        if self.current_layer_index is None:
            return
        layer = self.layers[self.current_layer_index]
        if isinstance(layer, TextLayer):
            # Open editor
            self._edit_text_layer(self.current_layer_index)

    def _edit_text_layer(self, index: int):
        """Open a dialog to edit a TextLayer's properties."""
        if index is None or index < 0 or index >= len(self.layers):
            return
        layer = self.layers[index]
        if not isinstance(layer, TextLayer):
            return
        # Simple dialog sequence: text, size, font, color, outline thickness+color, shadow
        new_text = simpledialog.askstring("Edit Text", "Text:", initialvalue=layer.text, parent=self)
        if new_text is None:
            return
        new_size = simpledialog.askinteger("Font Size", "Font size:", initialvalue=layer.font_size, minvalue=6, maxvalue=500, parent=self)
        if new_size is None:
            new_size = layer.font_size
        new_font = simpledialog.askstring("Font", "Font family or path:", initialvalue=layer.font_spec or '', parent=self)
        color = colorchooser.askcolor(title="Text colour", initialcolor=layer.color)
        if color and color[1]:
            new_color = color[1]
        else:
            new_color = layer.color
        # Outline
        outline_thickness = simpledialog.askinteger("Outline", "Outline thickness (0 for none):", initialvalue=layer.effects.get('outline', (None,0))[1] if layer.effects.get('outline') else 0, minvalue=0, parent=self)
        outline_color = None
        if outline_thickness and outline_thickness > 0:
            oc = colorchooser.askcolor(title="Outline colour", initialcolor=layer.effects.get('outline', ('#000000',0))[0])
            outline_color = oc[1] if oc and oc[1] else '#000000'
        # Shadow
        shadow_enabled = messagebox.askyesno("Shadow", "Add drop shadow?", parent=self)
        shadow = None
        if shadow_enabled:
            sdx = simpledialog.askinteger("Shadow X", "Shadow X offset:", initialvalue=layer.effects.get('shadow', (2,2,0.6,'#000000'))[0], parent=self)
            sdy = simpledialog.askinteger("Shadow Y", "Shadow Y offset:", initialvalue=layer.effects.get('shadow', (2,2,0.6,'#000000'))[1], parent=self)
            sop = simpledialog.askfloat("Shadow Opacity", "Shadow opacity (0.0-1.0):", initialvalue=layer.effects.get('shadow', (2,2,0.6,'#000000'))[2], minvalue=0.0, maxvalue=1.0, parent=self)
            sc = colorchooser.askcolor(title="Shadow colour", initialcolor=layer.effects.get('shadow', (2,2,0.6,'#000000'))[3])
            sc = sc[1] if sc and sc[1] else '#000000'
            shadow = (sdx or 2, sdy or 2, sop or 0.6, sc)

        # Save changes
        self._save_history()
        layer.text = new_text
        layer.font_spec = new_font or layer.font_spec
        layer.font_size = new_size
        layer.color = new_color
        effects = {}
        if outline_thickness and outline_thickness > 0:
            effects['outline'] = (outline_color or '#000000', outline_thickness)
        if shadow:
            effects['shadow'] = shadow
        layer.effects = effects
        # Re-render
        try:
            layer.render_text()
        except Exception:
            pass
        self._refresh_layer_list()
        self._update_composite()

    def _open_properties_window(self) -> None:
        """Open a small window with sliders to edit current layer's properties."""
        if self.current_layer_index is None:
            return
        layer = self.layers[self.current_layer_index]
        # Create new top-level window
        prop_win = tk.Toplevel(self)
        prop_win.title(f"Properties - {layer.name}")
        prop_win.configure(bg="#3a3a3a")
        # Sliders for alpha, brightness, contrast, colour
        def on_prop_change(val, prop):
            # Save history only first time any property is changed
            if not hasattr(on_prop_change, 'saved'):
                self._save_history()
                on_prop_change.saved = True
            if prop == 'alpha':
                layer.alpha = float(alpha_scale.get())
            elif prop == 'brightness':
                layer.brightness = float(bright_scale.get())
            elif prop == 'contrast':
                layer.contrast = float(contrast_scale.get())
            elif prop == 'color':
                layer.color = float(color_scale.get())
            elif prop == 'gamma':
                layer.gamma = float(gamma_scale.get())
            elif prop == 'red':
                layer.red = float(red_scale.get())
            elif prop == 'green':
                layer.green = float(green_scale.get())
            elif prop == 'blue':
                layer.blue = float(blue_scale.get())
            layer.apply_adjustments()
            self._update_composite()
        # Create scales
        tk.Label(prop_win, text="Opacity", bg="#3a3a3a", fg="white").pack(pady=2)
        alpha_scale = tk.Scale(prop_win, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, bg="#3a3a3a", fg="white", length=200)
        alpha_scale.set(layer.alpha)
        alpha_scale.config(command=lambda v: on_prop_change(v, 'alpha'))
        alpha_scale.pack(pady=2)
        tk.Label(prop_win, text="Brightness", bg="#3a3a3a", fg="white").pack(pady=2)
        bright_scale = tk.Scale(prop_win, from_=0.1, to=2, resolution=0.05, orient=tk.HORIZONTAL, bg="#3a3a3a", fg="white", length=200)
        bright_scale.set(layer.brightness)
        bright_scale.config(command=lambda v: on_prop_change(v, 'brightness'))
        bright_scale.pack(pady=2)
        tk.Label(prop_win, text="Contrast", bg="#3a3a3a", fg="white").pack(pady=2)
        contrast_scale = tk.Scale(prop_win, from_=0.1, to=2, resolution=0.05, orient=tk.HORIZONTAL, bg="#3a3a3a", fg="white", length=200)
        contrast_scale.set(layer.contrast)
        contrast_scale.config(command=lambda v: on_prop_change(v, 'contrast'))
        contrast_scale.pack(pady=2)
        tk.Label(prop_win, text="Color", bg="#3a3a3a", fg="white").pack(pady=2)
        color_scale = tk.Scale(prop_win, from_=0.1, to=2, resolution=0.05, orient=tk.HORIZONTAL, bg="#3a3a3a", fg="white", length=200)
        color_scale.set(layer.color)
        color_scale.config(command=lambda v: on_prop_change(v, 'color'))
        color_scale.pack(pady=2)
        # Gamma slider
        tk.Label(prop_win, text="Gamma", bg="#3a3a3a", fg="white").pack(pady=2)
        gamma_scale = tk.Scale(prop_win, from_=0.2, to=3.0, resolution=0.05, orient=tk.HORIZONTAL, bg="#3a3a3a", fg="white", length=200)
        gamma_scale.set(layer.gamma)
        gamma_scale.config(command=lambda v: on_prop_change(v, 'gamma'))
        gamma_scale.pack(pady=2)
        # Red, Green, Blue channel sliders for selective colour adjustments
        tk.Label(prop_win, text="Red", bg="#3a3a3a", fg="white").pack(pady=2)
        red_scale = tk.Scale(prop_win, from_=0.0, to=3.0, resolution=0.05, orient=tk.HORIZONTAL, bg="#3a3a3a", fg="white", length=200)
        red_scale.set(layer.red)
        red_scale.config(command=lambda v: on_prop_change(v, 'red'))
        red_scale.pack(pady=2)
        tk.Label(prop_win, text="Green", bg="#3a3a3a", fg="white").pack(pady=2)
        green_scale = tk.Scale(prop_win, from_=0.0, to=3.0, resolution=0.05, orient=tk.HORIZONTAL, bg="#3a3a3a", fg="white", length=200)
        green_scale.set(layer.green)
        green_scale.config(command=lambda v: on_prop_change(v, 'green'))
        green_scale.pack(pady=2)
        tk.Label(prop_win, text="Blue", bg="#3a3a3a", fg="white").pack(pady=2)
        blue_scale = tk.Scale(prop_win, from_=0.0, to=3.0, resolution=0.05, orient=tk.HORIZONTAL, bg="#3a3a3a", fg="white", length=200)
        blue_scale.set(layer.blue)
        blue_scale.config(command=lambda v: on_prop_change(v, 'blue'))
        blue_scale.pack(pady=2)
        # Close button
        def close_window():
            prop_win.destroy()
            # Reset property change flag for next opening
            if hasattr(on_prop_change, 'saved'):
                del on_prop_change.saved
        tk.Button(prop_win, text="Close", command=close_window, bg="#5c5c5c", fg="white").pack(pady=5)

    def _erase_at(self, x: int, y: int) -> None:
        """Erase a circular region at the given coordinates by setting alpha to 0."""
        layer = self.layers[self.current_layer_index]
        # Optimize by creating a small transparent patch and pasting only in the affected bbox
        w, h = layer.original.size
        r = int(max(1, self.brush_size / 2))
        left = max(0, int(x - r))
        upper = max(0, int(y - r))
        right = min(w, int(x + r) + 1)
        lower = min(h, int(y + r) + 1)
        if right <= left or lower <= upper:
            return
        patch_w = right - left
        patch_h = lower - upper
        # Create small mask and transparent patch
        mask = Image.new("L", (patch_w, patch_h), 0)
        draw_mask = ImageDraw.Draw(mask)
        # draw circle relative to mask coords
        cx = x - left
        cy = y - upper
        draw_mask.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=255)
        transparent_patch = Image.new("RGBA", (patch_w, patch_h), (0, 0, 0, 0))
        # Paste transparent patch into layer using small mask
        layer.original.paste(transparent_patch, (left, upper), mask)
        # Reapply adjustments for the layer and update composite once
        layer.apply_adjustments()
        self._update_composite()

    def _erase_line(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Erase along a line between two points using multiple circles to avoid gaps."""
        # Draw along line by interpolating points at small intervals
        layer = self.layers[self.current_layer_index]
        # approximate number of steps based on distance and brush size
        dx = x1 - x0
        dy = y1 - y0
        dist = (dx * dx + dy * dy) ** 0.5
        steps = int(dist / (self.brush_size / 2)) + 1
        # Instead of calling _erase_at repeatedly (which reapplies adjustments each time),
        # build a temporary small mask that covers the whole line bbox and paste once.
        w, h = layer.original.size
        # bounding box of the line strokes
        min_x = max(0, int(min(x0, x1) - self.brush_size))
        max_x = min(w, int(max(x0, x1) + self.brush_size) + 1)
        min_y = max(0, int(min(y0, y1) - self.brush_size))
        max_y = min(h, int(max(y0, y1) + self.brush_size) + 1)
        if max_x <= min_x or max_y <= min_y:
            return
        mask_w = max_x - min_x
        mask_h = max_y - min_y
        line_mask = Image.new("L", (mask_w, mask_h), 0)
        draw = ImageDraw.Draw(line_mask)
        # Draw circles along the line onto line_mask
        for i in range(steps + 1):
            t = i / steps
            x = int(x0 + dx * t)
            y = int(y0 + dy * t)
            rx = x - min_x
            ry = y - min_y
            r = int(max(1, self.brush_size / 2))
            draw.ellipse([(rx - r, ry - r), (rx + r, ry + r)], fill=255)
        transparent_patch = Image.new("RGBA", (mask_w, mask_h), (0, 0, 0, 0))
        layer.original.paste(transparent_patch, (min_x, min_y), line_mask)
        layer.apply_adjustments()
        self._update_composite()

    # ------------------------------------------------------------------
    # Spot heal, dodge and burn helpers
    # ------------------------------------------------------------------
    def _heal_at(self, x: int, y: int) -> None:
        """Perform a spot heal at the given canvas coordinates on the current layer.

        This uses a median filter over a small neighbourhood to replace the target
        region with a locally averaged texture, approximating a spot heal tool."""
        if self.current_layer_index is None:
            return
        layer = self.layers[self.current_layer_index]
        # Convert canvas coords to layer coords by subtracting offset
        ox, oy = layer.offset
        lx = int(x - ox)
        ly = int(y - oy)
        # Determine brush radius
        r = getattr(self, 'heal_size', self.brush_size) / 2
        radius = int(r)
        # Define bounding box around heal area within layer bounds
        left = max(0, lx - radius)
        upper = max(0, ly - radius)
        right = min(layer.original.width, lx + radius)
        lower = min(layer.original.height, ly + radius)
        if right <= left or lower <= upper:
            return
        box = (left, upper, right, lower)
        region = layer.original.crop(box)
        # Apply median filter to region to smooth blemishes
        try:
            filtered = region.filter(ImageFilter.MedianFilter(size=3))
        except Exception:
            # Fallback to gaussian blur
            filtered = region.filter(ImageFilter.GaussianBlur(radius=2))
        # Paste filtered region back
        layer.original.paste(filtered, box)
        # Reapply adjustments to update working copy and composite
        layer.apply_adjustments()
        self._update_composite()

    def _heal_line(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Heal along a line between two canvas points by interpolating heal calls."""
        dx = x1 - x0
        dy = y1 - y0
        dist = (dx * dx + dy * dy) ** 0.5
        brush_size = getattr(self, 'heal_size', self.brush_size)
        # Step length depends on brush radius; we sample at half radius to avoid gaps
        step = max(1, brush_size / 2)
        steps = int(dist / step) + 1
        for i in range(steps + 1):
            t = i / steps
            x = int(x0 + dx * t)
            y = int(y0 + dy * t)
            self._heal_at(x, y)

    def _dodge_at(self, x: int, y: int) -> None:
        """Lighten pixels within a radius around the canvas coordinate on the current layer."""
        if self.current_layer_index is None:
            return
        layer = self.layers[self.current_layer_index]
        ox, oy = layer.offset
        lx = int(x - ox)
        ly = int(y - oy)
        r = getattr(self, 'dodge_size', self.brush_size) / 2
        radius = int(r)
        left = max(0, lx - radius)
        upper = max(0, ly - radius)
        right = min(layer.original.width, lx + radius)
        lower = min(layer.original.height, ly + radius)
        if right <= left or lower <= upper:
            return
        box = (left, upper, right, lower)
        region = layer.original.crop(box)
        # Lighten region by applying brightness enhancement
        factor = getattr(self, 'dodge_strength', 1.2)
        enhancer = ImageEnhance.Brightness(region)
        brightened = enhancer.enhance(factor)
        # Create soft elliptical mask for brush smoothing
        mask = Image.new('L', region.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse([(0, 0), (region.size[0], region.size[1])], fill=255)
        # Blur mask to create gradual falloff; radius proportional to brush size
        try:
            blur_radius = max(1, int(radius * 0.5))
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        except Exception:
            pass
        # Blend original and brightened using mask as alpha
        result = Image.composite(brightened, region, mask)
        layer.original.paste(result, box)
        layer.apply_adjustments()
        self._update_composite()

    def _dodge_line(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Apply dodge along a line by sampling points."""
        dx = x1 - x0
        dy = y1 - y0
        dist = (dx * dx + dy * dy) ** 0.5
        brush_size = getattr(self, 'dodge_size', self.brush_size)
        step = max(1, brush_size / 2)
        steps = int(dist / step) + 1
        for i in range(steps + 1):
            t = i / steps
            x = int(x0 + dx * t)
            y = int(y0 + dy * t)
            self._dodge_at(x, y)

    def _burn_at(self, x: int, y: int) -> None:
        """Darken pixels within a radius around the canvas coordinate on the current layer."""
        if self.current_layer_index is None:
            return
        layer = self.layers[self.current_layer_index]
        ox, oy = layer.offset
        lx = int(x - ox)
        ly = int(y - oy)
        r = getattr(self, 'burn_size', self.brush_size) / 2
        radius = int(r)
        left = max(0, lx - radius)
        upper = max(0, ly - radius)
        right = min(layer.original.width, lx + radius)
        lower = min(layer.original.height, ly + radius)
        if right <= left or lower <= upper:
            return
        box = (left, upper, right, lower)
        region = layer.original.crop(box)
        # Darken region by applying brightness enhancement factor < 1
        factor = getattr(self, 'burn_strength', 0.8)
        enhancer = ImageEnhance.Brightness(region)
        darkened = enhancer.enhance(factor)
        # Create soft elliptical mask to make burn behave like a brush
        mask = Image.new('L', region.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse([(0, 0), (region.size[0], region.size[1])], fill=255)
        try:
            blur_radius = max(1, int(radius * 0.5))
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        except Exception:
            pass
        result = Image.composite(darkened, region, mask)
        layer.original.paste(result, box)
        layer.apply_adjustments()
        self._update_composite()

    def _burn_line(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Apply burn along a line by sampling points."""
        dx = x1 - x0
        dy = y1 - y0
        dist = (dx * dx + dy * dy) ** 0.5
        brush_size = getattr(self, 'burn_size', self.brush_size)
        step = max(1, brush_size / 2)
        steps = int(dist / step) + 1
        for i in range(steps + 1):
            t = i / steps
            x = int(x0 + dx * t)
            y = int(y0 + dy * t)
            self._burn_at(x, y)

    # ------------------------------------------------------------------
    # Collage and composition helpers
    # ------------------------------------------------------------------
    def _create_collage_from_files(self) -> None:
        """Prompt user to select multiple images and arrange them into a collage.

        The user is asked to choose one or more image files.  They can then
        specify the number of columns, cell size and background colour.  A new
        document is created (optionally replacing the current one) and each
        selected image becomes a layer positioned within a grid.  Images may
        be scaled to fit the cell dimensions.
        """
        # Ask for image files
        filetypes = [("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files", "*.*")]
        paths = filedialog.askopenfilenames(title="Select Images for Collage", filetypes=filetypes)
        if not paths:
            return
        num_images = len(paths)
        # Ask number of columns
        cols = simpledialog.askinteger(
            "Columns",
            "Enter number of columns for collage (1-10)",
            initialvalue=min(3, num_images),
            minvalue=1,
            maxvalue=max(10, num_images),
        )
        if cols is None or cols <= 0:
            return
        rows = (num_images + cols - 1) // cols
        # Load first image to determine default cell size
        try:
            first_img = Image.open(paths[0]).convert("RGBA")
        except Exception as e:
            messagebox.showerror("Error", f"Unable to open image: {e}")
            return
        default_w, default_h = first_img.size
        # Ask cell width and height
        cell_w = simpledialog.askinteger(
            "Cell Width",
            "Enter width for each collage cell (pixels)",
            initialvalue=default_w,
            minvalue=1,
        )
        if cell_w is None:
            return
        cell_h = simpledialog.askinteger(
            "Cell Height",
            "Enter height for each collage cell (pixels)",
            initialvalue=default_h,
            minvalue=1,
        )
        if cell_h is None:
            return
        # Ask whether to start a new project (clearing existing layers)
        replace = messagebox.askyesno(
            "New Document?",
            "Create collage in a new document?\nThis will discard current layers.",
        )
        # Ask background colour for collage
        bg_colour = colorchooser.askcolor(title="Choose background colour for collage", initialcolor="#ffffff")
        if not bg_colour or not bg_colour[0]:
            # default white
            bg_rgb = (255, 255, 255)
        else:
            bg_rgb = tuple(int(v) for v in bg_colour[0])
        # Determine canvas size
        canvas_w = cell_w * cols
        canvas_h = cell_h * rows
        # Save history before modifications
        self._save_history()
        if replace:
            # Clear layers and history for new document
            self.layers = []
            self.history = []
            self.history_index = -1
            # Set new canvas size
            self.canvas.config(width=canvas_w, height=canvas_h)
        # Create background base layer if replacing
        if replace:
            bg_img = Image.new("RGBA", (canvas_w, canvas_h), bg_rgb + (255,))
            bg_layer = Layer(bg_img, "Collage Background")
            self.layers.append(bg_layer)
        # Add each selected image as a new layer
        for idx, p in enumerate(paths):
            try:
                img = Image.open(p).convert("RGBA")
            except Exception as e:
                messagebox.showerror("Error", f"Could not open {p}: {e}")
                continue
            # Scale image to cell size preserving aspect ratio
            # Compute scale factor
            scale_x = cell_w / img.width
            scale_y = cell_h / img.height
            scale = min(scale_x, scale_y)
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)
            resized = img.resize((new_w, new_h), resample=Image.LANCZOS)
            # Create a blank cell and paste resized image centered
            cell_img = Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))
            paste_x = (cell_w - new_w) // 2
            paste_y = (cell_h - new_h) // 2
            cell_img.paste(resized, (paste_x, paste_y), resized)
            layer_name = f"Collage {idx}" if replace else f"Collage {len(self.layers)}"
            layer = Layer(cell_img, layer_name)
            # Determine offset for cell position
            row = idx // cols
            col = idx % cols
            layer.offset = (col * cell_w, row * cell_h)
            self.layers.append(layer)
        # Set current layer to last added layer
        if self.layers:
            self.current_layer_index = len(self.layers) - 1
        # Refresh and composite
        self._refresh_layer_list()
        self._update_composite()

    def _layout_visible_layers(self) -> None:
        """Arrange all visible layers into a grid collage within the current document.

        The user specifies the number of columns and whether to scale layers.  Layers
        are repositioned (and optionally resized) to fit into a collage grid.
        """
        if not self.layers:
            return
        # Collect visible layers to arrange
        visible_layers = [layer for layer in self.layers if layer.visible]
        if not visible_layers:
            messagebox.showinfo("No Visible Layers", "There are no visible layers to layout.")
            return
        n = len(visible_layers)
        cols = simpledialog.askinteger(
            "Columns",
            "Enter number of columns (1-10)",
            initialvalue=min(3, n),
            minvalue=1,
            maxvalue=max(10, n),
        )
        if cols is None or cols <= 0:
            return
        rows = (n + cols - 1) // cols
        # Ask if scale to uniform cell size
        do_scale = messagebox.askyesno(
            "Scale Layers",
            "Scale layers to fit cells?\nYes: images scaled uniformly to cell size\nNo: images keep original size and may overflow",
        )
        # Determine cell size
        # Use max width and height among visible layers as default cell size
        max_w = max(layer.original.width for layer in visible_layers)
        max_h = max(layer.original.height for layer in visible_layers)
        # Ask optional cell width/height
        if do_scale:
            cell_w = simpledialog.askinteger(
                "Cell Width",
                "Enter width for each cell (pixels)",
                initialvalue=max_w,
                minvalue=1,
            )
            if cell_w is None:
                return
            cell_h = simpledialog.askinteger(
                "Cell Height",
                "Enter height for each cell (pixels)",
                initialvalue=max_h,
                minvalue=1,
            )
            if cell_h is None:
                return
        else:
            cell_w, cell_h = max_w, max_h
        # Compute new canvas size
        new_w = cell_w * cols
        new_h = cell_h * rows
        # Save history
        self._save_history()
        # Optionally scale each visible layer
        for layer in visible_layers:
            # Determine cell index for this layer (preserve order of visible_layers)
            idx = visible_layers.index(layer)
            r = idx // cols
            c = idx % cols
            if do_scale:
                # Scale original and mask
                w, h = layer.original.size
                scale_x = cell_w / w
                scale_y = cell_h / h
                scale = min(scale_x, scale_y)
                new_size = (int(w * scale), int(h * scale))
                resized = layer.original.resize(new_size, resample=Image.LANCZOS)
                # Resize mask accordingly
                layer.original = resized
                layer.mask = layer.mask.resize(new_size, resample=Image.LANCZOS)
                # Reset selective colour factors? Keep same
                layer.apply_adjustments()
                # Create a cell image to center
                # but easier: set offset to position so that image appears at top-left of cell; we keep original size but we center if cell bigger than image width/height
            # Set layer offset to position within collage
            offset_x = c * cell_w
            offset_y = r * cell_h
            # If not scaling, we may want to center smaller images within cell
            if do_scale:
                # After scaling, layer.original size may be smaller than cell; compute centre offset
                dw = layer.original.width
                dh = layer.original.height
                offset_x += (cell_w - dw) // 2
                offset_y += (cell_h - dh) // 2
            layer.offset = (offset_x, offset_y)
        # Optionally adjust canvas size
        self.canvas.config(width=new_w, height=new_h)
        self._refresh_layer_list()
        self._update_composite()

    # ------------------------------------------------------------------
    # Advanced collage creation
    # ------------------------------------------------------------------
    def _create_collage_advanced(self) -> None:
        """Create a customised collage from multiple files with advanced layout options.

        Users can choose a layout type (grid, horizontal, vertical, random),
        specify spacing and margins, cell sizes and whether to start a new
        document.  Images are scaled to fit within their cells and arranged
        accordingly.  Background colour is also configurable.
        """
        # Prompt for files
        filetypes = [("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files", "*.*")]
        paths = filedialog.askopenfilenames(title="Select Images for Collage", filetypes=filetypes)
        if not paths:
            return
        num_images = len(paths)
        # Ask layout type
        layout = simpledialog.askstring(
            "Collage Layout",
            "Enter layout type (grid, horizontal, vertical, random):",
            initialvalue="grid",
        )
        if not layout:
            return
        layout = layout.strip().lower()
        if layout not in ("grid", "horizontal", "vertical", "random"):
            messagebox.showinfo("Unsupported Layout", f"Layout '{layout}' is not supported.")
            return
        # Ask whether to start a new document
        replace = messagebox.askyesno(
            "New Document?",
            "Create collage in a new document?\nThis will discard current layers.",
        )
        # Ask background colour
        bg_colour = colorchooser.askcolor(title="Choose background colour for collage", initialcolor="#ffffff")
        if not bg_colour or not bg_colour[0]:
            bg_rgb = (255, 255, 255)
        else:
            bg_rgb = tuple(int(v) for v in bg_colour[0])
        # Load first image for default size
        try:
            first_img = Image.open(paths[0]).convert("RGBA")
        except Exception as e:
            messagebox.showerror("Error", f"Unable to open image: {e}")
            return
        default_w, default_h = first_img.size
        # Ask cell size
        cell_w = simpledialog.askinteger(
            "Cell Width",
            "Enter width for each cell (pixels)",
            initialvalue=default_w,
            minvalue=1,
        )
        if cell_w is None:
            return
        cell_h = simpledialog.askinteger(
            "Cell Height",
            "Enter height for each cell (pixels)",
            initialvalue=default_h,
            minvalue=1,
        )
        if cell_h is None:
            return
        # Ask for spacing and margin
        spacing = simpledialog.askinteger(
            "Spacing",
            "Enter spacing between cells (pixels)",
            initialvalue=10,
            minvalue=0,
        )
        if spacing is None:
            return
        margin = simpledialog.askinteger(
            "Margin",
            "Enter margin around collage (pixels)",
            initialvalue=20,
            minvalue=0,
        )
        if margin is None:
            return
        # Determine grid dimensions
        if layout == "grid":
            cols = simpledialog.askinteger(
                "Columns",
                "Enter number of columns",
                initialvalue=min(3, num_images),
                minvalue=1,
                maxvalue=max(10, num_images),
            )
            if not cols or cols <= 0:
                return
            rows = (num_images + cols - 1) // cols
        elif layout == "horizontal":
            cols = num_images
            rows = 1
        elif layout == "vertical":
            cols = 1
            rows = num_images
        else:
            cols = rows = None
            canvas_w = simpledialog.askinteger(
                "Canvas Width",
                "Enter width of collage canvas (pixels)",
                initialvalue=default_w * num_images,
                minvalue=1,
            )
            if canvas_w is None:
                return
            canvas_h = simpledialog.askinteger(
                "Canvas Height",
                "Enter height of collage canvas (pixels)",
                initialvalue=default_h * num_images,
                minvalue=1,
            )
            if canvas_h is None:
                return
        # Compute canvas size
        if layout == "grid":
            canvas_w = margin * 2 + cols * cell_w + (cols - 1) * spacing
            canvas_h = margin * 2 + rows * cell_h + (rows - 1) * spacing
        elif layout == "horizontal":
            canvas_w = margin * 2 + num_images * cell_w + (num_images - 1) * spacing
            canvas_h = margin * 2 + cell_h
        elif layout == "vertical":
            canvas_w = margin * 2 + cell_w
            canvas_h = margin * 2 + num_images * cell_h + (num_images - 1) * spacing
        # Save history
        self._save_history()
        if replace:
            # Clear layers and history
            self.layers = []
            self.history = []
            self.history_index = -1
        # Configure canvas size
        self.canvas.config(width=canvas_w, height=canvas_h)
        # Create background layer if replacing
        if replace:
            bg_img = Image.new("RGBA", (canvas_w, canvas_h), bg_rgb + (255,))
            self.layers.append(Layer(bg_img, "Collage Background"))
        # Add each image as a layer
        import random
        for idx, path in enumerate(paths):
            try:
                img = Image.open(path).convert("RGBA")
            except Exception as e:
                messagebox.showerror("Error", f"Could not open {path}: {e}")
                continue
            # Resize to fit cell
            scale_x = cell_w / img.width
            scale_y = cell_h / img.height
            scale = min(scale_x, scale_y)
            new_w = max(1, int(img.width * scale))
            new_h = max(1, int(img.height * scale))
            resized = img.resize((new_w, new_h), resample=Image.LANCZOS)
            # Create cell
            cell_img = Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))
            paste_x = (cell_w - new_w) // 2
            paste_y = (cell_h - new_h) // 2
            cell_img.paste(resized, (paste_x, paste_y), resized)
            layer_name = f"Collage {idx}" if replace else f"Collage {len(self.layers)}"
            layer = Layer(cell_img, layer_name)
            # Determine offset
            if layout == "grid":
                r = idx // cols
                c = idx % cols
                offset_x = margin + c * (cell_w + spacing)
                offset_y = margin + r * (cell_h + spacing)
            elif layout == "horizontal":
                offset_x = margin + idx * (cell_w + spacing)
                offset_y = margin
            elif layout == "vertical":
                offset_x = margin
                offset_y = margin + idx * (cell_h + spacing)
            else:
                # random placement within margins
                max_x = max(margin, canvas_w - margin - cell_w)
                max_y = max(margin, canvas_h - margin - cell_h)
                offset_x = random.randint(margin, max_x)
                offset_y = random.randint(margin, max_y)
            layer.offset = (offset_x, offset_y)
            self.layers.append(layer)
        # Set selection to last added layer
        if self.layers:
            self.current_layer_index = len(self.layers) - 1
        self._refresh_layer_list()
        self._update_composite()

    # ------------------------------------------------------------------
    # Auto balance layers for composition
    # ------------------------------------------------------------------
    def _auto_balance_layers(self) -> None:
        """Automatically adjust brightness of all visible layers for consistent exposure.

        This helper computes the average luminance of each visible layer and
        scales its brightness so that all layers tend toward a common mean.
        It's useful when combining photos with different exposures.
        """
        if not self.layers:
            return
        # Collect visible layers
        visible_layers = [layer for layer in self.layers if layer.visible]
        if not visible_layers:
            return
        # Compute mean brightness for each visible layer
        means = []
        for layer in visible_layers:
            gray = layer.original.convert("L")
            hist = gray.histogram()
            total = gray.width * gray.height
            s = 0
            for i, count in enumerate(hist):
                s += i * count
            mean_val = s / total if total > 0 else 0
            means.append(mean_val)
        # Determine target mean brightness
        target = sum(means) / len(means)
        # Save history once before adjusting
        self._save_history()
        for i, layer in enumerate(visible_layers):
            current_mean = means[i]
            if current_mean <= 0:
                factor = 1.0
            else:
                factor = target / current_mean
            # Clamp factor to [0.5, 2.0] to avoid extremes
            factor = max(0.5, min(2.0, factor))
            layer.brightness *= factor
            # Apply adjustments to update working copy
            layer.apply_adjustments()
        # Update sliders to reflect current layer's properties
        if self.current_layer_index is not None:
            curr = self.layers[self.current_layer_index]
            self.brightness_slider.set(curr.brightness)
            self.contrast_slider.set(curr.contrast)
        # Refresh composite
        self._update_composite()

    # ------------------------------------------------------------------
    # Template collage creation
    # ------------------------------------------------------------------
    def _create_template_layout(self, template: str) -> None:
        """Quickly create a collage based on a predefined template.

        Templates:
        - "2x2": 2 rows x 2 columns grid
        - "3x3": 3 rows x 3 columns grid
        - "1x3h": 1 row x 3 columns (horizontal strip)
        - "3x1v": 3 rows x 1 column (vertical strip)
        - "random": random placement in specified canvas

        When invoked, the user is prompted to select images and can customise
        cell size, spacing, margin and background colour. The layout is
        then created on a new canvas (with option to retain current layers).
        """
        # Map template identifiers to layout type and grid dimensions
        layout = "grid"
        rows = cols = None
        if template == "2x2":
            rows, cols = 2, 2
        elif template == "3x3":
            rows, cols = 3, 3
        elif template == "1x3h":
            rows, cols = 1, 3
            layout = "horizontal"
        elif template == "3x1v":
            rows, cols = 3, 1
            layout = "vertical"
        elif template == "random":
            layout = "random"
        else:
            messagebox.showinfo("Unknown Template", f"Template '{template}' is not recognised.")
            return
        # Ask for images
        filetypes = [("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files", "*.*")]
        paths = filedialog.askopenfilenames(title="Select Images for Template", filetypes=filetypes)
        if not paths:
            return
        num_images = len(paths)
        # Determine grid dims if not random
        if layout == "grid":
            # Use provided rows/cols; if more images, extend rows automatically
            if rows is None or cols is None:
                # Fallback to square grid
                cols = int(math.ceil(math.sqrt(num_images)))
                rows = int(math.ceil(num_images / cols))
            else:
                # Adjust rows if needed
                needed_rows = (num_images + cols - 1) // cols
                if needed_rows > rows:
                    rows = needed_rows
        elif layout == "horizontal":
            # 1 row, columns equal to number of images
            rows, cols = 1, num_images
        elif layout == "vertical":
            rows, cols = num_images, 1
        # Ask cell size (default from first image)
        try:
            img0 = Image.open(paths[0]).convert("RGBA")
        except Exception as e:
            messagebox.showerror("Error", f"Unable to open image: {e}")
            return
        default_w, default_h = img0.size
        cell_w = simpledialog.askinteger(
            "Cell Width",
            "Enter width for each cell (pixels)",
            initialvalue=default_w,
            minvalue=1,
        )
        if cell_w is None:
            return
        cell_h = simpledialog.askinteger(
            "Cell Height",
            "Enter height for each cell (pixels)",
            initialvalue=default_h,
            minvalue=1,
        )
        if cell_h is None:
            return
        # Ask spacing and margin with defaults
        spacing = simpledialog.askinteger(
            "Spacing",
            "Enter spacing between cells (pixels)",
            initialvalue=10,
            minvalue=0,
        )
        if spacing is None:
            return
        margin = simpledialog.askinteger(
            "Margin",
            "Enter margin around collage (pixels)",
            initialvalue=20,
            minvalue=0,
        )
        if margin is None:
            return
        # Ask background colour
        bg_colour = colorchooser.askcolor(title="Choose background colour", initialcolor="#ffffff")
        if not bg_colour or not bg_colour[0]:
            bg_rgb = (255, 255, 255)
        else:
            bg_rgb = tuple(int(v) for v in bg_colour[0])
        # Ask whether to start new doc
        replace = messagebox.askyesno(
            "New Document?",
            "Create this collage in a new document?\nThis will discard current layers.",
        )
        # Determine canvas size
        if layout == "grid":
            canvas_w = margin * 2 + cols * cell_w + (cols - 1) * spacing
            canvas_h = margin * 2 + rows * cell_h + (rows - 1) * spacing
        elif layout == "horizontal":
            canvas_w = margin * 2 + num_images * cell_w + (num_images - 1) * spacing
            canvas_h = margin * 2 + cell_h
        elif layout == "vertical":
            canvas_w = margin * 2 + cell_w
            canvas_h = margin * 2 + num_images * cell_h + (num_images - 1) * spacing
        else:  # random
            # Ask canvas size for random mosaic
            canvas_w = simpledialog.askinteger(
                "Canvas Width",
                "Enter width of the random mosaic canvas (pixels)",
                initialvalue=default_w * max(1, min(num_images, 3)),
                minvalue=1,
            )
            if canvas_w is None:
                return
            canvas_h = simpledialog.askinteger(
                "Canvas Height",
                "Enter height of the random mosaic canvas (pixels)",
                initialvalue=default_h * max(1, min(num_images, 3)),
                minvalue=1,
            )
            if canvas_h is None:
                return
        # Save history and optionally clear
        self._save_history()
        if replace:
            self.layers = []
            self.history = []
            self.history_index = -1
        # Set canvas size
        self.canvas.config(width=canvas_w, height=canvas_h)
        # Create background layer if replacing
        if replace:
            bg_img = Image.new("RGBA", (canvas_w, canvas_h), bg_rgb + (255,))
            self.layers.append(Layer(bg_img, "Template Background"))
        import random
        # Add each image
        for idx, path in enumerate(paths):
            try:
                img = Image.open(path).convert("RGBA")
            except Exception as e:
                messagebox.showerror("Error", f"Could not open {path}: {e}")
                continue
            # Resize to fit cell
            scale_x = cell_w / img.width
            scale_y = cell_h / img.height
            scale = min(scale_x, scale_y)
            new_w = max(1, int(img.width * scale))
            new_h = max(1, int(img.height * scale))
            resized = img.resize((new_w, new_h), resample=Image.LANCZOS)
            # Create cell image
            cell_img = Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))
            paste_x = (cell_w - new_w) // 2
            paste_y = (cell_h - new_h) // 2
            cell_img.paste(resized, (paste_x, paste_y), resized)
            layer_name = f"Template {idx}" if replace else f"Template {len(self.layers)}"
            layer = Layer(cell_img, layer_name)
            # Determine offset
            if layout == "grid":
                r = idx // cols
                c = idx % cols
                offset_x = margin + c * (cell_w + spacing)
                offset_y = margin + r * (cell_h + spacing)
            elif layout == "horizontal":
                offset_x = margin + idx * (cell_w + spacing)
                offset_y = margin
            elif layout == "vertical":
                offset_x = margin
                offset_y = margin + idx * (cell_h + spacing)
            else:  # random
                max_x = max(margin, canvas_w - margin - cell_w)
                max_y = max(margin, canvas_h - margin - cell_h)
                offset_x = random.randint(margin, max_x)
                offset_y = random.randint(margin, max_y)
            layer.offset = (offset_x, offset_y)
            self.layers.append(layer)
        # Update selection and composite
        if self.layers:
            self.current_layer_index = len(self.layers) - 1
        self._refresh_layer_list()
        self._update_composite()

    # ------------------------------------------------------------------
    # Draft management (local storage)
    # ------------------------------------------------------------------
    def _save_draft(self) -> None:
        """Save the current editing session to a draft file in the draft directory.

        The user is prompted for a name, and the current history snapshot
        (including all layers and their properties) is stored using pickle.
        """
        if not self.layers:
            messagebox.showinfo("Nothing to Save", "There is no layer to save as a draft.")
            return
        # Ask for draft name
        name = simpledialog.askstring("Save Draft", "Enter a name for this draft:")
        if not name:
            return
        # Save current history snapshot
        # Ensure the latest state is saved
        self._save_history()
        if self.history_index < 0:
            messagebox.showerror("Error", "No state available to save.")
            return
        snapshot = self.history[self.history_index]
        import pickle
        filename = os.path.join(self.draft_dir, f"{name}.pkl")
        try:
            with open(filename, "wb") as f:
                pickle.dump(snapshot, f)
            messagebox.showinfo("Draft Saved", f"Draft '{name}' saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save draft: {e}")

    def _load_draft(self) -> None:
