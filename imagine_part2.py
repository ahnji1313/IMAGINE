from PySide6 import QtCore, QtGui, QtWidgets


class _QtStringVar:
    """Simple replacement for tkinter.StringVar while migrating to Qt."""

    def __init__(self, value: str = ""):
        self._value = value

    def set(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value

    def __str__(self) -> str:
        return self._value
                 visible: bool = True, font_size: float = 12.0, 
                 font_family: str = "Arial", font_weight: str = "normal", 
                 font_style: str = "normal"):
        super().__init__(x, y, fill_color, stroke_color, stroke_width, opacity, visible)
        self.text = text
        self.font_size = font_size
        self.font_family = font_family
        self.font_weight = font_weight
        self.font_style = font_style

class VectorLayer:
    """Represents a vector layer containing vector objects."""
    
    def __init__(self, name: str, width: int = 800, height: int = 600):
        self.name = name
        self.width = width
        self.height = height
        self.visible = True
        self.opacity = 1.0
        self.offset = (0, 0)
        self.blend_mode = 'normal'
        self.vector_objects: List[VectorObject] = []
        
    def add_object(self, obj: VectorObject) -> None:
        """Add a vector object to the layer."""
        self.vector_objects.append(obj)
        
    def remove_object(self, obj: VectorObject) -> None:
        """Remove a vector object from the layer."""
        if obj in self.vector_objects:
            self.vector_objects.remove(obj)
            
    def rasterize(self, target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """Rasterize the vector layer to a PIL Image."""
        if target_size:
            width, height = target_size
        else:
            width, height = self.width, self.height
            
        # Create transparent image
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Apply offset
        offset_x, offset_y = self.offset
        
        for obj in self.vector_objects:
            if not obj.visible:
                continue
                
            # Calculate final position with offset
            x = obj.x + offset_x
            y = obj.y + offset_y
            
            # Parse colors
            def hex_to_rgba(hex_color: str, opacity: float = 1.0) -> Tuple[int, int, int, int]:
                hex_color = hex_color.lstrip('#')
                if len(hex_color) == 3:
                    hex_color = ''.join([c*2 for c in hex_color])
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                a = int(255 * opacity * obj.opacity)
                return (r, g, b, a)
            
            fill_rgba = hex_to_rgba(obj.fill_color, self.opacity)
            stroke_rgba = hex_to_rgba(obj.stroke_color, self.opacity)
            
            if isinstance(obj, VectorRectangle):
                if obj.corner_radius > 0:
                    # Rounded rectangle
                    draw.rounded_rectangle(
                        [x, y, x + obj.width, y + obj.height],
                        radius=obj.corner_radius,
                        fill=fill_rgba,
                        outline=stroke_rgba,
                        width=int(obj.stroke_width)
                    )
                else:
                    # Regular rectangle
                    draw.rectangle(
                        [x, y, x + obj.width, y + obj.height],
                        fill=fill_rgba,
                        outline=stroke_rgba,
                        width=int(obj.stroke_width)
                    )
                    
            elif isinstance(obj, VectorCircle):
                draw.ellipse(
                    [x - obj.radius, y - obj.radius, x + obj.radius, y + obj.radius],
                    fill=fill_rgba,
                    outline=stroke_rgba,
                    width=int(obj.stroke_width)
                )
                
            elif isinstance(obj, VectorEllipse):
                draw.ellipse(
                    [x, y, x + obj.width, y + obj.height],
                    fill=fill_rgba,
                    outline=stroke_rgba,
                    width=int(obj.stroke_width)
                )
                
            elif isinstance(obj, VectorLine):
                draw.line(
                    [x, y, obj.x2 + offset_x, obj.y2 + offset_y],
                    fill=stroke_rgba,
                    width=int(obj.stroke_width)
                )
                
            elif isinstance(obj, VectorPath):
                if len(obj.points) >= 2:
                    # Convert points to absolute coordinates
                    abs_points = [(p[0] + offset_x, p[1] + offset_y) for p in obj.points]
                    if obj.closed:
                        abs_points.append(abs_points[0])  # Close the path
                    draw.line(abs_points, fill=stroke_rgba, width=int(obj.stroke_width))
                    
            elif isinstance(obj, VectorText):
                try:
                    # Try to load font
                    font = ImageFont.truetype(f"{obj.font_family.lower()}.ttf", int(obj.font_size))
                except:
                    try:
                        font = ImageFont.load_default()
                    except:
                        font = None
                
                if font:
                    draw.text(
                        (x, y),
                        obj.text,
                        font=font,
                        fill=fill_rgba
                    )
                else:
                    # Fallback to default text rendering
                    draw.text(
                        (x, y),
                        obj.text,
                        fill=fill_rgba
                    )
        
        return img


# ----------------------------------------------------------------------
# Adjustment Layer Support
# ----------------------------------------------------------------------

class AdjustmentLayer:
    """Represents a non-destructive adjustment layer."""
    
    def __init__(self, name: str, adjustment_type: str, params: Dict[str, Any]):
        self.name = name
        self.adjustment_type = adjustment_type
        self.params = params
        self.visible = True
        self.opacity = 1.0
        
    def apply_to_image(self, image: Image.Image) -> Image.Image:
        """Apply the adjustment to an image."""
        if not self.visible:
            return image
            
        result = image.copy()
        
        if self.adjustment_type == "brightness":
            brightness = self.params.get("brightness", 1.0)
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(result)
                result = enhancer.enhance(brightness)
                
        elif self.adjustment_type == "contrast":
            contrast = self.params.get("contrast", 1.0)
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(result)
                result = enhancer.enhance(contrast)
                
        elif self.adjustment_type == "saturation":
            saturation = self.params.get("saturation", 1.0)
            if saturation != 1.0:
                enhancer = ImageEnhance.Color(result)
                result = enhancer.enhance(saturation)
                
        elif self.adjustment_type == "hue_shift":
            hue_shift = self.params.get("hue_shift", 0.0)
            if hue_shift != 0.0:
                # Convert to HSV, shift hue, convert back
                hsv = result.convert("HSV")
                h, s, v = hsv.split()
                h_array = np.array(h)
                h_array = (h_array + int(hue_shift * 255 / 360)) % 255
                h = Image.fromarray(h_array.astype(np.uint8))
                result = Image.merge("HSV", (h, s, v)).convert("RGBA")
                
        elif self.adjustment_type == "levels":
            # Simple levels adjustment
            black_point = self.params.get("black_point", 0)
            white_point = self.params.get("white_point", 255)
            gamma = self.params.get("gamma", 1.0)
            
            # Apply levels adjustment
            r, g, b, a = result.split()
            
            def apply_levels(channel, black, white, gamma_val):
                # Normalize to 0-1
                channel_array = np.array(channel, dtype=np.float32) / 255.0
                # Apply black and white points
                channel_array = np.clip((channel_array - black/255.0) / (white/255.0 - black/255.0), 0, 1)
                # Apply gamma
                channel_array = np.power(channel_array, 1.0 / gamma_val)
                return Image.fromarray((channel_array * 255).astype(np.uint8))
            
            r = apply_levels(r, black_point, white_point, gamma)
            g = apply_levels(g, black_point, white_point, gamma)
            b = apply_levels(b, black_point, white_point, gamma)
            result = Image.merge("RGBA", (r, g, b, a))
            
        elif self.adjustment_type == "color_balance":
            # Color balance adjustment
            red_shift = self.params.get("red_shift", 0.0)
            green_shift = self.params.get("green_shift", 0.0)
            blue_shift = self.params.get("blue_shift", 0.0)
            
            if red_shift != 0.0 or green_shift != 0.0 or blue_shift != 0.0:
                r, g, b, a = result.split()
                
                def apply_shift(channel, shift):
                    channel_array = np.array(channel, dtype=np.float32)
                    channel_array = np.clip(channel_array + shift * 255, 0, 255)
                    return Image.fromarray(channel_array.astype(np.uint8))
                
                r = apply_shift(r, red_shift)
                g = apply_shift(g, green_shift)
                b = apply_shift(b, blue_shift)
                result = Image.merge("RGBA", (r, g, b, a))
                
        elif self.adjustment_type == "vibrance":
            vibrance = self.params.get("vibrance", 0.0)
            if vibrance != 0.0:
                # Convert to HSV, adjust saturation, convert back
                hsv = result.convert("HSV")
                h, s, v = hsv.split()
                s_array = np.array(s, dtype=np.float32)
                # Apply vibrance (affects less saturated colors more)
                saturation_factor = 1.0 + vibrance * (1.0 - s_array / 255.0)
                s_array = np.clip(s_array * saturation_factor, 0, 255)
                s = Image.fromarray(s_array.astype(np.uint8))
                result = Image.merge("HSV", (h, s, v)).convert("RGBA")
        
        # Apply opacity
        if self.opacity < 1.0:
            r, g, b, a = result.split()
            a_array = np.array(a, dtype=np.float32)
            a_array = a_array * self.opacity
            a = Image.fromarray(a_array.astype(np.uint8))
            result.putalpha(a)
            
        return result


# ----------------------------------------------------------------------
# Brush Dynamics Support
# ----------------------------------------------------------------------

@dataclass
class BrushSettings:
    """Advanced brush settings for dynamic painting."""
    size: float = 5.0
    hardness: float = 1.0  # 0.0 = soft, 1.0 = hard
    spacing: float = 0.25  # 0.0 = continuous, 1.0 = maximum spacing
    flow: float = 1.0  # 0.0 = no paint, 1.0 = full paint
    jitter: float = 0.0  # 0.0 = no jitter, 1.0 = maximum jitter
    pressure_sensitivity: bool = True
    speed_sensitivity: bool = True
    texture: str = "solid"  # "solid", "spray", "chalk", "calligraphy"

class BrushEngine:
    """Advanced brush engine with dynamics support."""
    
    def __init__(self):
        self.settings = BrushSettings()
        self.last_position = None
        self.last_time = None
        self.stroke_points = []
        
    def update_settings(self, settings: BrushSettings):
        """Update brush settings."""
        self.settings = settings
        
    def calculate_brush_properties(self, x: int, y: int, pressure: float = 1.0, 
                                 speed: float = 1.0) -> Tuple[float, float, float]:
        """Calculate brush size, opacity, and spacing based on dynamics."""
        # Base size
        size = self.settings.size
        
        # Pressure sensitivity
        if self.settings.pressure_sensitivity:
            size *= pressure
            
        # Speed sensitivity (faster = smaller brush)
        if self.settings.speed_sensitivity and speed > 0:
            size *= (1.0 - min(speed * 0.3, 0.5))  # Cap speed effect at 50%
            
        # Calculate opacity based on flow and pressure
        opacity = self.settings.flow * pressure
        
        # Calculate spacing
        spacing = self.settings.spacing * size
        
        return size, opacity, spacing
        
    def should_paint_dab(self, x: int, y: int) -> bool:
        """Determine if we should paint a dab at this position based on spacing."""
        if self.last_position is None:
            return True
            
        last_x, last_y = self.last_position
        distance = math.sqrt((x - last_x)**2 + (y - last_y)**2)
        
        # Calculate required spacing
        _, _, spacing = self.calculate_brush_properties(x, y)
        
        return distance >= spacing
        
    def add_jitter(self, x: int, y: int) -> Tuple[int, int]:
        """Add random jitter to position."""
        if self.settings.jitter <= 0:
            return x, y
            
        jitter_amount = self.settings.jitter * self.settings.size
        jitter_x = random.uniform(-jitter_amount, jitter_amount)
        jitter_y = random.uniform(-jitter_amount, jitter_amount)
        
        return int(x + jitter_x), int(y + jitter_y)
        
    def calculate_speed(self, x: int, y: int) -> float:
        """Calculate cursor speed for speed sensitivity."""
        if self.last_position is None or self.last_time is None:
            return 1.0
            
        last_x, last_y = self.last_position
        current_time = time.time()
        time_delta = current_time - self.last_time
        
        if time_delta <= 0:
            return 1.0
            
        distance = math.sqrt((x - last_x)**2 + (y - last_y)**2)
        speed = distance / time_delta
        
        # Normalize speed (adjust this based on your needs)
        return min(speed / 100.0, 5.0)  # Cap at 5x normal speed


# ----------------------------------------------------------------------
# Macro Recording Support
# ----------------------------------------------------------------------

class MacroRecorder:
    """Records and plays back macro sequences."""
    
    def __init__(self):
        self.is_recording = False
        self.actions = []
        self.current_macro = None
        
    def start_recording(self, macro_name: str):
        """Start recording a new macro."""
        self.is_recording = True
        self.actions = []
        self.current_macro = macro_name
        
    def stop_recording(self):
        """Stop recording the current macro."""
        self.is_recording = False
        return self.actions.copy()
        
    def record_action(self, action_type: str, params: Dict[str, Any]):
        """Record an action during macro recording."""
        if not self.is_recording:
            return
            
        action = {
            "type": action_type,
            "params": params,
            "timestamp": time.time()
        }
        self.actions.append(action)
        
    def save_macro(self, filename: str):
        """Save macro to file."""
        macro_data = {
            "name": self.current_macro,
            "actions": self.actions,
            "created": time.time()
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(macro_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving macro: {e}")
            return False
            
    def load_macro(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load macro from file."""
        try:
            with open(filename, 'r') as f:
                macro_data = json.load(f)
            return macro_data
        except Exception as e:
            print(f"Error loading macro: {e}")
            return None
            
    def play_macro(self, macro_data: Dict[str, Any], editor_instance):
        """Play back a macro on the editor instance."""
        actions = macro_data.get("actions", [])
        
        for action in actions:
            action_type = action.get("type")
            params = action.get("params", {})
            
            try:
                if action_type == "apply_filter":
                    filter_name = params.get("filter_name")
                    if filter_name:
                        editor_instance._apply_filter(filter_name)
                        
                elif action_type == "adjust_brightness":
                    value = params.get("value", 1.0)
                    if editor_instance.current_layer_index is not None:
                        layer = editor_instance.layers[editor_instance.current_layer_index]
                        layer.brightness = value
                        layer.apply_adjustments()
                        editor_instance._update_composite()
                        
                elif action_type == "adjust_contrast":
                    value = params.get("value", 1.0)
                    if editor_instance.current_layer_index is not None:
                        layer = editor_instance.layers[editor_instance.current_layer_index]
                        layer.contrast = value
                        layer.apply_adjustments()
                        editor_instance._update_composite()
                        
                elif action_type == "rotate_layer":
                    degrees = params.get("degrees", 0)
                    editor_instance._rotate_layer(degrees)
                    
                elif action_type == "flip_layer":
                    axis = params.get("axis", "horizontal")
                    editor_instance._flip_layer(axis)
                    
                elif action_type == "scale_layer":
                    factor = params.get("factor", 1.0)
                    editor_instance._scale_layer(factor)
                    
                # Add more action types as needed
                
            except Exception as e:
                print(f"Error executing macro action {action_type}: {e}")
                continue


class Layer:
    """Represents a single editable image layer.

    Each layer stores both its original unmodified image and a working
    image.  Filters and adjustments (brightness, contrast, colour and
    transparency) are applied to the working copy so that changes are
    non‑destructive — the original can be used as the base for
    reapplying operations if needed.
    """

    def __init__(self, image: Image.Image, name: str):
        # Always store images in RGBA mode to support transparency
        self.original = image.convert("RGBA")
        self.image = self.original.copy()
        self.name = name
        self.visible = True
        # adjustment factors
        self.alpha: float = 1.0
        self.brightness: float = 1.0
        self.contrast: float = 1.0
        self.color: float = 1.0  # colour (saturation) factor
        # gamma (exposure) adjustment; 1.0 means no change
        self.gamma: float = 1.0
        # positional offset (dx, dy) for moving the layer
        self.offset = (0, 0)
        # mask for non‑destructive hiding/revealing of pixels
        self.mask = Image.new("L", self.original.size, 255)
        # blending mode: 'normal' or 'multiply'
        self.blend_mode: str = 'normal'
        # Channel factors for selective colour adjustments
        # Multipliers for red, green and blue channels respectively. 1.0 means no change.
        self.red: float = 1.0
        self.green: float = 1.0
        self.blue: float = 1.0
        # Preview cache for low-resolution fast rendering
        self.preview_image: Optional[Image.Image] = None

    def apply_adjustments(self) -> None:
        """Reapply brightness, contrast, colour and alpha adjustments.

        The working copy is reset to the original and then brightness,
        contrast and colour enhancements are applied in sequence.  Finally
        the alpha channel is scaled.  Calling this after any change
        ensures the displayed image reflects the current adjustment
        settings.
        """
        # Start from original
        self.image = self.original.copy()
        # Apply brightness
        if self.brightness != 1.0:
            enhancer = ImageEnhance.Brightness(self.image)
            self.image = enhancer.enhance(self.brightness)
        # Apply contrast
        if self.contrast != 1.0:
            enhancer = ImageEnhance.Contrast(self.image)
            self.image = enhancer.enhance(self.contrast)
        # Apply colour (saturation)
        if self.color != 1.0:
            enhancer = ImageEnhance.Color(self.image)
            self.image = enhancer.enhance(self.color)
        # Apply gamma (exposure)
        if self.gamma != 1.0:
            r, g, b, a = self.image.split()
            def apply_gamma(channel: Image.Image, gamma: float) -> Image.Image:
                return channel.point(lambda i: int((i / 255.0) ** (1.0 / gamma) * 255))
            r = apply_gamma(r, self.gamma)
            g = apply_gamma(g, self.gamma)
            b = apply_gamma(b, self.gamma)
            self.image = Image.merge("RGBA", (r, g, b, a))
        # Apply per channel (selective colour) adjustments
        if self.red != 1.0 or self.green != 1.0 or self.blue != 1.0:
            r, g, b, a = self.image.split()
            # Scale each channel by its factor and clip
            if self.red != 1.0:
                r = r.point(lambda i: int(min(255, i * self.red)))
            if self.green != 1.0:
                g = g.point(lambda i: int(min(255, i * self.green)))
            if self.blue != 1.0:
                b = b.point(lambda i: int(min(255, i * self.blue)))
            self.image = Image.merge("RGBA", (r, g, b, a))
        # Apply alpha transparency
        r, g, b, a = self.image.split()
        new_alpha = a.point(lambda i: int(i * self.alpha))
        # Combine with mask so that painted mask regions hide pixels
        combined_alpha = ImageChops.multiply(new_alpha, self.mask)
        self.image.putalpha(combined_alpha)

    def generate_preview(self, max_size: Tuple[int, int] = (800, 600)) -> None:
        """Generate a downscaled preview image for fast canvas rendering.

        This stores a downsampled version of the adjusted image in
        `self.preview_image`. Use this in the main compositor when
        preview mode is enabled.
        """
        try:
            # Use the current working image (post adjustments) as source
            src = self.image
            w, h = src.size
            max_w, max_h = max_size
            # Compute scale keeping aspect ratio
            scale = min(1.0, max_w / max(1, w), max_h / max(1, h))
            if scale >= 1.0:
                self.preview_image = src.copy()
            else:
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                # Use bilinear downscale for speed
                self.preview_image = src.resize(new_size, resample=Image.BILINEAR)
        except Exception:
            self.preview_image = None

    def apply_filter(self, filter_name: str) -> None:
        """Apply a named filter to the original image and update working copy.

        Supported filters include: grayscale, invert, blur, sharpen,
        emboss, edge, contour, detail and smooth.  After applying
        the filter to the original image, adjustments are reapplied to
        update the working copy.
        """
        fname = filter_name.lower()
        if fname == "grayscale":
            gray = self.original.convert("L")
            self.original = Image.merge("RGBA", (gray, gray, gray, self.original.split()[3]))
        elif fname == "invert":
            r, g, b, a = self.original.split()
            inv_r = r.point(lambda i: 255 - i)
            inv_g = g.point(lambda i: 255 - i)
            inv_b = b.point(lambda i: 255 - i)
            self.original = Image.merge("RGBA", (inv_r, inv_g, inv_b, a))
        elif fname == "blur":
            self.original = self.original.filter(ImageFilter.GaussianBlur(radius=2))
        elif fname == "sharpen":
            self.original = self.original.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        elif fname == "emboss":
            self.original = self.original.filter(ImageFilter.EMBOSS)
        elif fname == "edge":
            self.original = self.original.filter(ImageFilter.EDGE_ENHANCE)
        elif fname == "contour":
            self.original = self.original.filter(ImageFilter.CONTOUR)
        elif fname == "detail":
            self.original = self.original.filter(ImageFilter.DETAIL)
        elif fname == "smooth":
            self.original = self.original.filter(ImageFilter.SMOOTH_MORE)
        elif fname in ("mirror", "mirror_horizontal", "flip_horizontal", "mirror horizontal", "flip horizontal"):
            self.original = self.original.transpose(Image.FLIP_LEFT_RIGHT)
        elif fname in ("mirror_vertical", "flip_vertical", "mirror vertical", "flip vertical"):
            self.original = self.original.transpose(Image.FLIP_TOP_BOTTOM)
        elif fname.startswith("canva_"):
            preset = fname.split("canva_", 1)[1]
            self.original = apply_canva_preset(self.original, preset)
        elif fname in ("liquify", "swirl"):
            # Apply swirl liquify effect around the centre
            self.original = swirl_image(self.original)
        elif fname == "sepia":
            # Apply a sepia tone to the image. This gives a warm, vintage look
            # by converting to grayscale and then mapping the channels to a
            # sepia palette. See: https://en.wikipedia.org/wiki/Sepia_(color)
            r, g, b, a = self.original.split()
            # Convert to grayscale
            gray = Image.merge("RGB", (r, g, b)).convert("L")
            # Create a sepia tinted image by applying custom matrix
            # Try vectorized implementation using numpy for speed on large images
            try:
                arr = np.array(self.original)
                # arr shape (H, W, 4) RGBA
                rgba = arr.astype(np.float32)
                gray_vals = (rgba[:, :, 0] * 0.299 + rgba[:, :, 1] * 0.587 + rgba[:, :, 2] * 0.114)
                sr = np.clip(gray_vals * 1.07, 0, 255).astype(np.uint8)
                sg = np.clip(gray_vals * 0.74, 0, 255).astype(np.uint8)
                sb = np.clip(gray_vals * 0.43, 0, 255).astype(np.uint8)
                a_chan = rgba[:, :, 3].astype(np.uint8)
                out = np.stack([sr, sg, sb, a_chan], axis=2)
                self.original = Image.fromarray(out, mode="RGBA")
                # Report completion
                try:
                    if _progress_callback is not None:
                        _progress_callback(100)
                except Exception:
                    pass
            except Exception:
                # Fallback to per-row loop with progress updates
                sepia = Image.new("RGBA", self.original.size)
                pixels = sepia.load()
                gray_pixels = gray.load()
                for j in range(self.original.height):
                    for i in range(self.original.width):
                        val = gray_pixels[i, j]
                        # Apply sepia transformation: red=val*1.07, green=val*0.74, blue=val*0.43
                        # Clip to 255
                        sr = min(255, int(val * 1.07))
                        sg = min(255, int(val * 0.74))
                        sb = min(255, int(val * 0.43))
                        pixels[i, j] = (sr, sg, sb, a.getpixel((i, j)))
                    try:
                        if _progress_callback is not None:
                            _progress_callback(int((j + 1) / self.original.height * 100))
                    except Exception:
                        pass
                self.original = sepia
        elif fname == "skin smooth" or fname == "skin_smooth":
            # Apply median blur to smooth skin while preserving edges to some degree.
            # This approximates frequency separation by reducing high frequency noise.
            self.original = self.original.filter(ImageFilter.MedianFilter(size=5))
        elif fname in ("frequency separation", "frequency"):  # New advanced skin smoothing
            # Perform simple frequency separation: split image into low and high frequency components.
            # Low frequency: Gaussian blur to smooth colour transitions.
            # High frequency: original minus blurred version to retain fine details.
            # Recombine by adding low and attenuated high frequency components.
            # Convert to RGB and split channels
            blurred = self.original.filter(ImageFilter.GaussianBlur(radius=5))
            # Create high frequency by subtracting blurred from original
            # Convert to float arrays for blending
            try:
                import numpy as np  # local import to avoid global dependency if not used
                orig_arr = np.array(self.original).astype(np.float32)
                blur_arr = np.array(blurred).astype(np.float32)
                high = orig_arr - blur_arr
                # Attenuate high frequency slightly to avoid over sharpening
                high = high * 0.5
                # Recombine
                recombined = blur_arr + high
                recombined = np.clip(recombined, 0, 255).astype(np.uint8)
                self.original = Image.fromarray(recombined, mode="RGBA")
            except Exception:
                # Fallback: just apply a median filter as a simple smooth if numpy unavailable
                self.original = self.original.filter(ImageFilter.MedianFilter(size=5))
        elif fname == "teeth whitening" or fname == "teeth_whitening":
            # Lighten selected areas by increasing brightness and decreasing colour saturation slightly
            # Convert to RGB for manipulation
            r, g, b, a = self.original.split()
            import numpy as np  # local import
            arr = np.array(Image.merge("RGB", (r, g, b))).astype(np.float32)
            # Increase brightness on all channels
            arr = arr * 1.2
            # Slightly desaturate by mixing with grayscale
            gray = arr.mean(axis=2, keepdims=True)
            arr = arr * 0.8 + gray * 0.2
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            new_rgb = Image.fromarray(arr, mode="RGB")
            self.original = Image.merge("RGBA", (*new_rgb.split(), a))
        elif fname == "red eye removal" or fname == "red_eye_removal":
            # Replace red channel in areas where red dominates (common red-eye) with average of G and B channels
            import numpy as np
            r, g, b, a = self.original.split()
            arr_r = np.array(r).astype(np.float32)
            arr_g = np.array(g).astype(np.float32)
            arr_b = np.array(b).astype(np.float32)
            # Find pixels where red channel significantly exceeds green and blue
            mask = (arr_r > arr_g * 1.3) & (arr_r > arr_b * 1.3)
            avg = ((arr_g + arr_b) / 2.0)
            arr_r[mask] = avg[mask]
            # Merge back
            r2 = Image.fromarray(arr_r.clip(0, 255).astype(np.uint8))
            self.original = Image.merge("RGBA", (r2, g, b, a))
        elif fname == "warm":
            # Apply a warm tone by boosting reds and greens and reducing blues
            r, g, b, a = self.original.split()
            r = r.point(lambda i: min(255, int(i * 1.15)))
            g = g.point(lambda i: min(255, int(i * 1.1)))
            b = b.point(lambda i: int(i * 0.9))
            self.original = Image.merge("RGBA", (r, g, b, a))
        elif fname == "cool":
            # Apply a cool tone by boosting blues and reducing reds
            r, g, b, a = self.original.split()
            r = r.point(lambda i: int(i * 0.9))
            g = g.point(lambda i: min(255, int(i * 1.05)))
            b = b.point(lambda i: min(255, int(i * 1.15)))
            self.original = Image.merge("RGBA", (r, g, b, a))
        elif fname == "vintage":
            # Apply a faded vintage effect: decrease contrast, add warm tint and slight sepia-like tone
            # Reduce contrast
            enhancer = ImageEnhance.Contrast(self.original)
            temp = enhancer.enhance(0.85)
            # Apply sepia tone matrix similar to sepia filter but less intense
            r, g, b, a = temp.split()
            gray = Image.merge("RGB", (r, g, b)).convert("L")
            sepia = Image.new("RGBA", temp.size)
            pixels = sepia.load()
            gray_pixels = gray.load()
            for j in range(temp.height):
                for i in range(temp.width):
                    val = gray_pixels[i, j]
                    sr = min(255, int(val * 1.1))
                    sg = min(255, int(val * 0.85))
                    sb = min(255, int(val * 0.6))
                    pixels[i, j] = (sr, sg, sb, a.getpixel((i, j)))
            self.original = sepia
        elif fname == "anime":
            # Approximate an anime style by increasing saturation, boosting brightness and adding a slight edge effect
            enhancer = ImageEnhance.Color(self.original)
            temp = enhancer.enhance(1.5)  # boost colour
            enhancer = ImageEnhance.Brightness(temp)
            temp = enhancer.enhance(1.1)  # slightly brighten
            # Overlay edge to emphasise lines
            edge = temp.filter(ImageFilter.FIND_EDGES)
            # Merge edge and temp with reduced opacity
            base = temp.convert("RGBA")
            edge_rgba = edge.convert("RGBA")
            # reduce edge opacity
            r, g, b, a = edge_rgba.split()
            a = a.point(lambda i: int(i * 0.5))
            edge_rgba.putalpha(a)
            self.original = Image.alpha_composite(base, edge_rgba)
        elif fname == "oil" or fname == "oil painting":
            # Approximate oil painting by applying strong smoothing and edge enhancement
            temp = self.original.filter(ImageFilter.SMOOTH_MORE)
            temp = temp.filter(ImageFilter.SMOOTH_MORE)
            temp = temp.filter(ImageFilter.EDGE_ENHANCE_MORE)
            self.original = temp
        elif fname == "cyberpunk":
            # Apply a neon/cyberpunk tone by boosting blues and magentas and increasing contrast
            enhancer = ImageEnhance.Contrast(self.original)
            temp = enhancer.enhance(1.4)
            r, g, b, a = temp.split()
            # Increase blue and magenta channels
            r = r.point(lambda i: min(255, int(i * 1.1)))
            g = g.point(lambda i: int(i * 0.9))
            b = b.point(lambda i: min(255, int(i * 1.4)))
            self.original = Image.merge("RGBA", (r, g, b, a))
        elif fname == "portrait optimize" or fname == "portrait":
            # Automatically smooth skin, whiten teeth and balance brightness/contrast for portraits
            # Skin smoothing
            self.original = self.original.filter(ImageFilter.MedianFilter(size=5))
            # Brighten and increase contrast slightly
            enhancer = ImageEnhance.Brightness(self.original)
            temp = enhancer.enhance(1.05)
            enhancer = ImageEnhance.Contrast(temp)
            temp = enhancer.enhance(1.05)
            # Slightly increase colour
            enhancer = ImageEnhance.Color(temp)
            temp = enhancer.enhance(1.05)
            self.original = temp
        elif fname == "posterize":
            # Reduce colour depth to give a posterised effect. Convert to RGB before posterising.
            # Use 4 bits per channel (16 levels) for dramatic effect.
            rgb = self.original.convert("RGB")
            poster = ImageOps.posterize(rgb, bits=4)
            self.original = poster.convert("RGBA")
        elif fname == "solarize":
            # Invert all pixels above a threshold to create a dramatic solarise effect
            rgb = self.original.convert("RGB")
            solar = ImageOps.solarize(rgb, threshold=128)
            self.original = solar.convert("RGBA")
        else:
            raise ValueError(f"Unsupported filter: {filter_name}")
        # After modifying the original image, reapply adjustments so that
        # the working copy reflects changes. Without this, the filtered
        # result may not appear on the canvas.
        self.apply_adjustments()

    def apply_filter_with_param(self, filter_name: str, params: dict) -> None:
        """Apply a named filter with parameters to the original image and update working copy.
        
        This method extends apply_filter to support parameterized filters.
        """
        fname = filter_name.lower()
        if fname == "blur":
            radius = params.get("radius", 2)
            self.original = self.original.filter(ImageFilter.GaussianBlur(radius=radius))
        elif fname == "sharpen":
            percent = params.get("percent", 150)
            radius = params.get("radius", 2)
            threshold = params.get("threshold", 3)
            self.original = self.original.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
        elif fname == "sepia":
            intensity = params.get("intensity", 100) / 100.0  # Convert to 0-1 range
            r, g, b, a = self.original.split()
            # Convert to grayscale
            gray = Image.merge("RGB", (r, g, b)).convert("L")
            # Create a sepia tinted image by applying custom matrix with intensity
            try:
                arr = np.array(self.original).astype(np.float32)
                gray_vals = (arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114)
                r_val = np.clip(gray_vals * (0.393 + 0.607 * intensity), 0, 255).astype(np.uint8)
                g_val = np.clip(gray_vals * (0.769 + 0.231 * intensity), 0, 255).astype(np.uint8)
                b_val = np.clip(gray_vals * (0.189 + 0.811 * intensity), 0, 255).astype(np.uint8)
                a_chan = arr[:, :, 3].astype(np.uint8)
                out = np.stack([r_val, g_val, b_val, a_chan], axis=2)
                self.original = Image.fromarray(out, mode="RGBA")
                try:
                    if _progress_callback is not None:
                        _progress_callback(100)
                except Exception:
                    pass
            except Exception:
                sepia = Image.new("RGBA", self.original.size)
                pixels = sepia.load()
                gray_pixels = gray.load()
                for j in range(self.original.height):
                    for i in range(self.original.width):
                        val = gray_pixels[i, j]
                        # Apply sepia transformation with intensity control
                        r_v = int(val * (0.393 + 0.607 * intensity))
                        g_v = int(val * (0.769 + 0.231 * intensity))
                        b_v = int(val * (0.189 + 0.811 * intensity))
                        # Clamp values
                        r_v = min(255, max(0, r_v))
                        g_v = min(255, max(0, g_v))
                        b_v = min(255, max(0, b_v))
                        pixels[i, j] = (r_v, g_v, b_v, a.getpixel((i, j)))
                    try:
                        if _progress_callback is not None:
                            _progress_callback(int((j + 1) / self.original.height * 100))
                    except Exception:
                        pass
                self.original = sepia
        else:
            # Fall back to regular filter for unsupported parameterized filters
            self.apply_filter(filter_name)
        
        # After modifying the original image, reapply adjustments
        self.apply_adjustments()

    def apply_filter_to_region(self, filter_name: str, box: tuple[int, int, int, int]) -> None:
        """Apply the specified filter only within the given bounding box on the original image.

        :param filter_name: one of the supported filter names (grayscale, invert, blur, etc.).
        :param box: tuple (left, upper, right, lower) specifying region in image coordinates.
        """
        # Ensure region is within bounds
        left, upper, right, lower = box
        left = max(0, left)
        upper = max(0, upper)
        right = min(self.original.width, right)
        lower = min(self.original.height, lower)
        if right <= left or lower <= upper:
            return
        # Crop region from original
        region = self.original.crop((left, upper, right, lower)).copy()
        # Run heavy region filter in background to keep UI responsive
        # Only show a progress dialog if a Tk root exists (avoid creating a new root)
        prog = None
        parent = getattr(tk, '_default_root', None)
        if parent is not None:
            prog = ProgressDialog(parent, title="Applying filter to region", initial_text=f"Applying {filter_name} to region...")
        result_q = queue.Queue()

        def worker():
            try:
                temp_layer = Layer(region, "temp")
                # register a local progress callback to update dialog
                def cb(pct):
                    try:
                        if prog is not None:
                            prog.set(pct)
                    except Exception:
                        pass
                register_progress_callback(cb)
                temp_layer.apply_filter(filter_name)
                register_progress_callback(None)
                result_q.put((True, temp_layer.original))
            except Exception as e:
                register_progress_callback(None)
                result_q.put((False, e))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        # poll until done (only if running inside a Tk app)
        def poll():
            try:
                done, payload = result_q.get_nowait()
            except queue.Empty:
                if getattr(self, 'after', None):
                    self.after(80, poll)
                return
            if prog is not None:
                try:
                    prog.close()
                except Exception:
                    pass
            if not done:
                messagebox.showerror("Filter error", str(payload))
                return
            filtered = payload
            # Paste filtered region back into original
            self.original.paste(filtered, (left, upper))
            # Reapply adjustments to update working copy
            self.apply_adjustments()

        if getattr(self, 'after', None):
            try:
                self.after(80, poll)
            except Exception:
                t.join()
                done, payload = result_q.get()
                if done:
                    self.original.paste(payload, (left, upper))
                    self.apply_adjustments()
        else:
            # No Tk event loop available, block until finished
            t.join()
            done, payload = result_q.get()
            if done:
                self.original.paste(payload, (left, upper))
                self.apply_adjustments()


class TextLayer(Layer):
    """An editable text layer stored as metadata and rendered on demand.

    The TextLayer keeps text metadata (string, font_spec, size, color,
    position and effects) so the layer remains editable. The raster
    `original` image is the full canvas size; calling `render_text()`
    rasterizes the text into `original` and then `apply_adjustments()`
    updates the working image.
    """
    def __init__(self, canvas_size: Tuple[int, int], text: str = "", name: str = "Text",
                 font_spec: Optional[str] = None, font_size: int = 32, color: str = "#000000",
                 position: Tuple[int, int] = (0, 0), effects: Optional[dict] = None):
        img = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        super().__init__(img, name)
        self.text = text
        self.font_spec = font_spec
        self.font_size = int(font_size)
        self.color = color
        self.position = position
        # effects example: {'outline': (color, thickness), 'shadow': (dx,dy,opacity,color), 'style': 'calligraphy'/'comic'}
        self.effects = effects or {}

    def render_text(self) -> None:
        """Rasterize the stored text into self.original and update working image."""
        # Reset original to empty canvas
        self.original = Image.new("RGBA", self.original.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(self.original)
        font = load_font(self.font_spec, max(6, int(self.font_size)))
        x, y = int(self.position[0]), int(self.position[1])
        text = str(self.text or "")

        # Shadow
        shadow = self.effects.get('shadow')
        if shadow:
            try:
                sdx, sdy, sopacity, scol = shadow
                tmp = Image.new('RGBA', self.original.size, (0, 0, 0, 0))
                td = ImageDraw.Draw(tmp)
                td.text((x + int(sdx), y + int(sdy)), text, font=font, fill=scol)
                # apply opacity to alpha channel
                alpha = tmp.split()[3].point(lambda p: int(p * float(sopacity)))
                tmp.putalpha(alpha)
                self.original = Image.alpha_composite(self.original, tmp)
            except Exception:
                pass

        # Outline (stroke) by drawing offsets
        outline = self.effects.get('outline')
        if outline:
            try:
                ocolor, owidth = outline
                owidth = int(owidth)
                for dx in range(-owidth, owidth + 1):
                    for dy in range(-owidth, owidth + 1):
                        if dx == 0 and dy == 0:
                            continue
                        draw.text((x + dx, y + dy), text, font=font, fill=ocolor)
            except Exception:
                pass

        # Main text
        try:
            draw.text((x, y), text, font=font, fill=self.color)
        except Exception:
            draw.text((x, y), text, font=ImageFont.load_default(), fill=self.color)

        # Optional styles
        style = self.effects.get('style')
        if style == 'calligraphy':
            try:
                self.original = self.original.filter(ImageFilter.GaussianBlur(radius=0.3)).convert('RGBA')
            except Exception:
                pass
        elif style == 'comic':
            try:
                enhancer = ImageEnhance.Contrast(self.original)
                self.original = enhancer.enhance(1.3)
                self.original = ImageOps.posterize(self.original.convert('RGB'), bits=5).convert('RGBA')
            except Exception:
                pass

        # After rendering, apply adjustments to update working copy
        self.apply_adjustments()


class ImageEditor(QtWidgets.QMainWindow):
    """Main application window for the image editor."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMAGINE IMAGE EDITOR")
        app = QtWidgets.QApplication.instance()
        screen = app.primaryScreen() if app is not None else None
        if screen is not None:
            geometry = screen.availableGeometry()
            win_w = int(geometry.width() * 0.9)
            win_h = int(geometry.height() * 0.9)
        else:
            win_w = 1200
            win_h = 800
        self.resize(win_w, win_h)
        if app is not None:
            try:
                app.setStyle("Fusion")
            except Exception:
                pass
        self._central_widget = QtWidgets.QWidget(self)
        self._central_layout = QtWidgets.QVBoxLayout(self._central_widget)
        self._central_layout.setContentsMargins(0, 0, 0, 0)
        self._central_layout.setSpacing(0)
        self._central_widget.setStyleSheet("background-color: #fafafa;")
        self.setCentralWidget(self._central_widget)
        status_bar = self.statusBar()
        if status_bar is not None:
            status_bar.showMessage("Ready")

        # List of layers (bottom to top) - now supports multiple layer types
        self.layers: list[Union[Layer, VectorLayer, AdjustmentLayer]] = []
        self.current_layer_index: int | None = None
        # History of document states for undo/redo
        self.history: list[dict] = []
        # index pointing to current position in history; -1 means no state saved yet
        self.history_index: int = -1
        # maximum number of states to keep in history (increased to 100)
        self.max_history: int = 100
        # list of history descriptions for history panel
        self.history_descriptions: list[str] = []
        # Theme state: True for dark mode, False for light mode
        self.dark_mode: bool = self._load_theme_preference()
        # Autosave interval in milliseconds (None means disabled)
        self.autosave_interval_ms: int | None = None
        self.autosave_after_id: str | None = None
        # Performance + workflow helpers
        self.performance_metrics = PerformanceMetrics()
        self.status_var = _QtStringVar("Ready")
        self.session_log: deque[str] = deque(maxlen=400)
        self._autosave_dir = ensure_directory(Path(os.path.expanduser("~")) / ".image_editor_drafts" / "autosave")
        self._autosave_lock = threading.Lock()

        # Tool variables
        # current tool can be "brush", "text", "eraser", "vector_rectangle", "vector_circle", etc. or None
        self.current_tool = None
        self.primary_color = "#ff0000"
        self.secondary_color = "#0000ff"
        self.active_color_slot = "primary"
        self.brush_color = self.primary_color
        self.brush_size = 5
        self.brush_texture = "solid"
        self.gradient_settings = {
            "type": "linear",
            "stops": [(0.0, self.primary_color), (1.0, self.secondary_color)],
            "snap": True,
            "base_angle": 0.0,
        }
        self.pattern_settings = {
            "type": "checker",
            "size": 32,
            "colors": ["#ffffff", "#c0c0c0"],
        }
        self.stamp_image: Optional[Image.Image] = None
        self._stamp_cache: dict[int, Image.Image] = {}
        self._pattern_cache: dict[Tuple[int, str], Image.Image] = {}
        self._smart_eraser_color: Optional[Tuple[int, int, int]] = None
        self._last_composite_image: Optional[Image.Image] = None

        # Advanced brush engine
        self.brush_engine = BrushEngine()
        self.brush_settings = BrushSettings()
        
        # Macro recording
        self.macro_recorder = MacroRecorder()
        
        # Vector tool state
        self.vector_tool_state = {
            "drawing": False,
            "start_point": None,
            "current_objects": []
        }

        # Zoom factor for canvas preview. 1.0 means no scaling.
        self.zoom: float = 1.0

        # Flag to avoid saving multiple history states during a continuous stroke
        self._history_saved_for_stroke = False

        # For drawing operations
        self._drag_prev = None
        # Variables for filter region selection
        self._filter_region_start = None
        self._filter_rect_id = None

        # Cached selection mask for selection-aware tools
        self._last_selection_mask: Optional[Image.Image] = None

        self.extensions_dir = ensure_directory(Path(__file__).resolve().parent / "extensions")
        self.loaded_extensions: dict[str, types.ModuleType] = {}

        # Directory for saving and loading drafts (local storage)
        # Use a folder in the user's home directory for persistence across sessions
        self.draft_dir = os.path.join(os.path.expanduser("~"), ".image_editor_drafts")
        # Create the directory if it does not exist
        try:
            os.makedirs(self.draft_dir, exist_ok=True)
        except Exception:
            pass

        # Set up GUI components
        try:
            self._create_widgets()
        except Exception:
            # TODO: Rebuild widget creation using Qt equivalents.
            pass
        self._load_extensions()

        # Toast container (for non-blocking notifications)
        self._toast_container = None
        self._active_toasts: list[QtWidgets.QWidget] = []

        # Bind command palette
        def _trigger_command_palette():
            handler = getattr(self, "_open_command_palette", None)
            if handler is not None:
                handler(None)

        self._command_palette_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+P"), self)
        self._command_palette_shortcut.activated.connect(_trigger_command_palette)

    # ------------------------------------------------------------------
    # GUI construction
    # ------------------------------------------------------------------
    def _create_widgets(self):
        # Menu bar
        self._create_menus()

        # Define a soft colour palette for the UI
        # Define a soft pastel palette to produce a gentle, modern feel
        bg_panel = "#fafafa"        # panel backgrounds
        bg_toolbar = "#f3f3f3"      # toolbar backgrounds
        btn_bg = "#e6e6e6"         # button backgrounds
        btn_fg = "#333333"         # button text colour (dark grey)
        slider_bg = "#f0f0f0"      # slider background
        slider_fg = "#333333"       # slider text and ticks
        label_bg = bg_panel
        label_fg = "#333333"        # label text dark for contrast

        # Left frame for layer list and controls
        left_frame = tk.Frame(self, width=260, bg=bg_panel)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Button to add new blank layer
        new_layer_btn = tk.Button(left_frame, text="New Layer", command=self._new_layer, bg=btn_bg, fg=btn_fg, relief=tk.RAISED, bd=1)
        new_layer_btn.pack(padx=8, pady=6, fill=tk.X)
        try:
            add_tooltip(new_layer_btn, "Create a new empty layer above the current one")
        except Exception:
            pass

        # Listbox to show layers
        self.layer_listbox = tk.Listbox(
            left_frame,
            selectmode=tk.EXTENDED,
            font=("Arial", 11),
            bg="#ffffff",
            fg="#333333",
            activestyle='none',
            highlightthickness=1,
            borderwidth=1,
        )
        self.layer_listbox.pack(padx=8, pady=6, fill=tk.BOTH, expand=True)
        try:
            add_tooltip(self.layer_listbox, "Layers: drag to reorder, double-click to rename")
        except Exception:
            pass
        self.layer_listbox.bind('<<ListboxSelect>>', self._on_layer_select)
        # Bind double click to rename layer
        self.layer_listbox.bind('<Double-Button-1>', self._rename_layer)
        # Bind drag events for drag-and-drop reordering
        self.layer_listbox.bind('<Button-1>', self._on_layer_drag_start)
        self.layer_listbox.bind('<B1-Motion>', self._on_layer_drag_motion)
        # Context menu for layers (right‑click)
        self.layer_menu = tk.Menu(self, tearoff=0)
        self.layer_menu.add_command(label="Duplicate", command=self._duplicate_layer)
        self.layer_menu.add_command(label="Delete", command=self._delete_layer)
        self.layer_menu.add_command(label="Move Up", command=lambda: self._move_layer(-1))
        self.layer_menu.add_command(label="Move Down", command=lambda: self._move_layer(1))
        self.layer_menu.add_command(label="Toggle Visibility", command=self._toggle_visibility)
        self.layer_menu.add_separator()
        self.layer_menu.add_command(label="Feather Mask", command=self._feather_mask)
        self.layer_menu.add_command(label="Invert Mask", command=self._invert_mask)
        self.layer_menu.add_separator()
        self.layer_menu.add_command(label="Group Selected Layers", command=self._group_selected_layers)
        # Bind right click on layer list
        self.layer_listbox.bind("<Button-3>", self._show_layer_context_menu)

        # Slider for transparency
        alpha_label = ttk.Label(left_frame, text="Opacity", font=("Arial", 10, "bold"))
        alpha_label.pack(padx=5, pady=(10, 0))
        try:
            add_tooltip(alpha_label, "Adjust layer transparency (0 = fully transparent, 1 = fully opaque)")
        except Exception:
            pass
        self.alpha_slider = tk.Scale(
            left_frame,
            from_=0,
            to=1,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            command=self._on_alpha_change,
            bg=slider_bg,
            fg=slider_fg,
            highlightthickness=0,
        )
        self.alpha_slider.pack(padx=8, pady=4, fill=tk.X)
        self.alpha_slider.bind("<ButtonRelease-1>", lambda e: self._reset_history_flag())
        try:
            add_tooltip(self.alpha_slider, "Drag to adjust layer opacity")
        except Exception:
            pass

        # Slider for brightness
        brightness_label = ttk.Label(left_frame, text="Brightness", font=("Arial", 10, "bold"))
        brightness_label.pack(padx=5, pady=(10, 0))
        try:
            add_tooltip(brightness_label, "Adjust layer brightness (0.1 = very dark, 2.0 = very bright)")
        except Exception:
            pass
        self.brightness_slider = tk.Scale(
            left_frame,
            from_=0.1,
            to=2,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            command=self._on_brightness_change,
            bg=slider_bg,
            fg=slider_fg,
            highlightthickness=0,
        )
        self.brightness_slider.set(1.0)
        self.brightness_slider.pack(padx=8, pady=4, fill=tk.X)
        self.brightness_slider.bind("<ButtonRelease-1>", lambda e: self._reset_history_flag())
        try:
            add_tooltip(self.brightness_slider, "Drag to adjust layer brightness")
        except Exception:
            pass

        # Slider for contrast
        contrast_label = ttk.Label(left_frame, text="Contrast", font=("Arial", 10, "bold"))
        contrast_label.pack(padx=5, pady=(10, 0))
        try:
            add_tooltip(contrast_label, "Adjust layer contrast (0.1 = low contrast, 2.0 = high contrast)")
        except Exception:
            pass
        self.contrast_slider = tk.Scale(
            left_frame,
            from_=0.1,
            to=2,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            command=self._on_contrast_change,
            bg=slider_bg,
            fg=slider_fg,
            highlightthickness=0,
        )
        self.contrast_slider.set(1.0)
        self.contrast_slider.pack(padx=8, pady=4, fill=tk.X)
        self.contrast_slider.bind("<ButtonRelease-1>", lambda e: self._reset_history_flag())
        try:
            add_tooltip(self.contrast_slider, "Drag to adjust layer contrast")
        except Exception:
            pass
        

        # Slider for colour (saturation)
        color_label = ttk.Label(left_frame, text="Color", font=("Arial", 10, "bold"))
        color_label.pack(padx=5, pady=(10, 0))
        try:
            add_tooltip(color_label, "Adjust layer color saturation (0.1 = desaturated, 2.0 = highly saturated)")
        except Exception:
            pass
        self.color_slider = tk.Scale(
            left_frame,
            from_=0.1,
            to=2,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            command=self._on_color_change,
            bg=slider_bg,
            fg=slider_fg,
            highlightthickness=0,
        )
        self.color_slider.set(1.0)
        self.color_slider.pack(padx=8, pady=4, fill=tk.X)
        self.color_slider.bind("<ButtonRelease-1>", lambda e: self._reset_history_flag())
        try:
            add_tooltip(self.color_slider, "Drag to adjust layer color saturation")
        except Exception:
            pass

        # Slider for gamma (exposure)
        gamma_label = ttk.Label(left_frame, text="Gamma", font=("Arial", 10, "bold"))
        gamma_label.pack(padx=5, pady=(10, 0))
        try:
            add_tooltip(gamma_label, "Adjust layer gamma/exposure (0.2 = very dark, 3.0 = very bright)")
        except Exception:
            pass
        self.gamma_slider = tk.Scale(
            left_frame,
            from_=0.2,
            to=3.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            command=self._on_gamma_change,
            bg=slider_bg,
            fg=slider_fg,
            highlightthickness=0,
        )
        self.gamma_slider.set(1.0)
        self.gamma_slider.pack(padx=8, pady=4, fill=tk.X)
        self.gamma_slider.bind("<ButtonRelease-1>", lambda e: self._reset_history_flag())
        try:
            add_tooltip(self.gamma_slider, "Drag to adjust layer gamma/exposure")
        except Exception:
            pass

        # Primary / secondary colour palette with eyedropper feedback
        palette_frame = tk.LabelFrame(left_frame, text="Palette", bg=bg_toolbar, fg=label_fg)
        palette_frame.pack(padx=5, pady=(8, 6), fill=tk.X)

        if not hasattr(self, 'tool_buttons'):
            self.tool_buttons = {}

        self.active_color_var = tk.StringVar(value=self.active_color_slot)
        self.color1_value = tk.StringVar()
        self.color2_value = tk.StringVar()

        def set_active(slot: str) -> None:
            self.active_color_slot = slot
            self.active_color_var.set(slot)
            self.brush_color = self.primary_color if slot == "primary" else self.secondary_color
            self._update_color_palette()

        c1_row = tk.Frame(palette_frame, bg=bg_toolbar)
        c1_row.pack(fill=tk.X, pady=2)
        self.color1_radio = tk.Radiobutton(
            c1_row,
            text="Color 1",
            variable=self.active_color_var,
            value="primary",
            command=lambda: set_active("primary"),
            bg=bg_toolbar,
            fg=label_fg,
            selectcolor=self.primary_color,
        )
        self.color1_radio.pack(side=tk.LEFT)
        self.color1_swatch = tk.Label(c1_row, width=4, relief=tk.SUNKEN)
        self.color1_swatch.pack(side=tk.LEFT, padx=4)
        tk.Label(c1_row, textvariable=self.color1_value, bg=bg_toolbar, fg=label_fg).pack(side=tk.LEFT, padx=2)

        c2_row = tk.Frame(palette_frame, bg=bg_toolbar)
        c2_row.pack(fill=tk.X, pady=2)
        self.color2_radio = tk.Radiobutton(
            c2_row,
            text="Color 2",
            variable=self.active_color_var,
            value="secondary",
            command=lambda: set_active("secondary"),
            bg=bg_toolbar,
            fg=label_fg,
            selectcolor=self.secondary_color,
        )
        self.color2_radio.pack(side=tk.LEFT)
        self.color2_swatch = tk.Label(c2_row, width=4, relief=tk.SUNKEN)
        self.color2_swatch.pack(side=tk.LEFT, padx=4)
        tk.Label(c2_row, textvariable=self.color2_value, bg=bg_toolbar, fg=label_fg).pack(side=tk.LEFT, padx=2)

        swap_btn = tk.Button(palette_frame, text="Swap", command=self._swap_colors, bg=btn_bg, fg=btn_fg, bd=1)
        swap_btn.pack(padx=4, pady=(4, 0), fill=tk.X)
        eyedrop_btn = tk.Button(palette_frame, text="Eyedropper Tool", command=self._select_eyedropper, bg=btn_bg, fg=btn_fg, bd=1)
        eyedrop_btn.pack(padx=4, pady=4, fill=tk.X)
        self._eyedropper_button = eyedrop_btn

        # Buttons for tool selection
        tools_frame = tk.Frame(left_frame, bg=bg_toolbar)
        tools_frame.pack(padx=5, pady=(10, 5), fill=tk.X)
        # Create tool buttons and store them for highlighting
        self.tool_buttons = {}
        if hasattr(self, '_eyedropper_button'):
            self.tool_buttons['eyedropper'] = self._eyedropper_button
        brush_btn = tk.Button(tools_frame, text="Brush", command=self._select_brush, bg=btn_bg, fg=btn_fg, relief=tk.RAISED, bd=1, font=("Arial", 9))
        brush_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.tool_buttons['brush'] = brush_btn
        try:
            add_tooltip(brush_btn, "Paint with brush tool - click and drag to draw")
        except Exception:
            pass
        eraser_btn = tk.Button(tools_frame, text="Eraser", command=self._select_eraser, bg=btn_bg, fg=btn_fg, relief=tk.RAISED, bd=1, font=("Arial", 9))
        eraser_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.tool_buttons['eraser'] = eraser_btn
        try:
            add_tooltip(eraser_btn, "Erase pixels - click and drag to remove content")
        except Exception:
            pass
        stamp_btn = tk.Button(tools_frame, text="Stamp", command=self._select_stamp_tool, bg=btn_bg, fg=btn_fg, relief=tk.RAISED, bd=1, font=("Arial", 9))
        stamp_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.tool_buttons['stamp'] = stamp_btn
        try:
            add_tooltip(stamp_btn, "Custom stamp tool - configure text or image stamp and click to apply")
        except Exception:
            pass
        pattern_btn = tk.Button(tools_frame, text="Pattern", command=self._select_pattern_brush, bg=btn_bg, fg=btn_fg, relief=tk.RAISED, bd=1, font=("Arial", 9))
        pattern_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.tool_buttons['pattern'] = pattern_btn
        try:
            add_tooltip(pattern_btn, "Pattern brush - paint using tiling textures like checkerboard or stripes")
        except Exception:
            pass
        smart_eraser_btn = tk.Button(tools_frame, text="Smart Eraser", command=self._select_smart_eraser, bg=btn_bg, fg=btn_fg, relief=tk.RAISED, bd=1, font=("Arial", 9))
        smart_eraser_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.tool_buttons['smart_eraser'] = smart_eraser_btn
        try:
            add_tooltip(smart_eraser_btn, "Smart eraser - remove colours similar to the sampled pixel")
        except Exception:
            pass
        move_btn = tk.Button(tools_frame, text="Move", command=self._select_move, bg=btn_bg, fg=btn_fg, relief=tk.RAISED, bd=1, font=("Arial", 9))
        move_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.tool_buttons['move'] = move_btn
        try:
            add_tooltip(move_btn, "Move layer position - click and drag to reposition")
        except Exception:
            pass
        mask_btn = tk.Button(tools_frame, text="Mask", command=self._select_mask, bg=btn_bg, fg=btn_fg, relief=tk.RAISED, bd=1, font=("Arial", 9))
        mask_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.tool_buttons['mask'] = mask_btn
        try:
            add_tooltip(mask_btn, "Create layer mask - paint to hide/show parts of layer")
        except Exception:
            pass
        crop_btn = tk.Button(tools_frame, text="Crop", command=self._select_crop, bg=btn_bg, fg=btn_fg, relief=tk.RAISED, bd=1, font=("Arial", 9))
        crop_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.tool_buttons['crop'] = crop_btn
        try:
            add_tooltip(crop_btn, "Crop image - drag to select area, release to crop")
        except Exception:
            pass
        select_btn = tk.Button(tools_frame, text="Select", command=self._select_select_tool, bg=btn_bg, fg=btn_fg, relief=tk.RAISED, bd=1, font=("Arial", 9))
        select_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.tool_buttons['select'] = select_btn
        try:
            add_tooltip(select_btn, "Rectangular selection - drag to select area")
        except Exception:
            pass
        text_btn = tk.Button(tools_frame, text="Text", command=self._select_text_tool, bg=btn_bg, fg=btn_fg, relief=tk.RAISED, bd=1, font=("Arial", 9))
        text_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.tool_buttons['text'] = text_btn
        try:
            add_tooltip(text_btn, "Add text - click to place text on canvas")
        except Exception:
            pass
        # Button for applying filter to a region
        region_btn = tk.Button(tools_frame, text="Filter Region", command=self._select_filter_region, bg=btn_bg, fg=btn_fg, relief=tk.RAISED, bd=1, font=("Arial", 9))
        region_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.tool_buttons['filter_region'] = region_btn
        try:
            add_tooltip(region_btn, "Apply filter to selected region - drag to select area")
        except Exception:
            pass
        # Heal, Dodge, Burn and Extract tools for retouching and selection
        heal_btn = tk.Button(tools_frame, text="Heal", command=self._select_heal, bg=btn_bg, fg=btn_fg, relief=tk.RAISED, bd=1, font=("Arial", 9))
        heal_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.tool_buttons['heal'] = heal_btn
        try:
            add_tooltip(heal_btn, "Heal tool - remove blemishes by sampling nearby pixels")
        except Exception:
            pass
        dodge_btn = tk.Button(tools_frame, text="Dodge", command=self._select_dodge, bg=btn_bg, fg=btn_fg, relief=tk.RAISED, bd=1, font=("Arial", 9))
        dodge_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.tool_buttons['dodge'] = dodge_btn
        try:
            add_tooltip(dodge_btn, "Dodge tool - lighten areas by painting over them")
        except Exception:
            pass
        burn_btn = tk.Button(tools_frame, text="Burn", command=self._select_burn, bg=btn_bg, fg=btn_fg, relief=tk.RAISED, bd=1, font=("Arial", 9))
        burn_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.tool_buttons['burn'] = burn_btn
        try:
            add_tooltip(burn_btn, "Burn tool - darken areas by painting over them")
        except Exception:
            pass
        extract_btn = tk.Button(tools_frame, text="Extract", command=self._select_extract, bg=btn_bg, fg=btn_fg, relief=tk.RAISED, bd=1, font=("Arial", 9))
        extract_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.tool_buttons['extract'] = extract_btn
        try:
            add_tooltip(extract_btn, "Extract tool - remove background by painting over it")
        except Exception:
            pass
        # Magic wand tool for selecting contiguous areas
        magic_btn = tk.Button(tools_frame, text="Magic Wand", command=self._select_magic_wand, bg=btn_bg, fg=btn_fg, relief=tk.RAISED, bd=1, font=("Arial", 9))
        magic_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.tool_buttons['magicwand'] = magic_btn
        try:
            add_tooltip(magic_btn, "Magic wand - select similar colored areas with one click")
        except Exception:
            pass
        # Quick Select tool selects similar colours/texture while dragging
        qsel_btn = tk.Button(tools_frame, text="Quick Select", command=self._select_quick_select, bg=btn_bg, fg=btn_fg, relief=tk.RAISED, bd=1, font=("Arial", 9))
        qsel_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.tool_buttons['quickselect'] = qsel_btn
        try:
            add_tooltip(qsel_btn, "Quick select - drag to select similar areas automatically")
        except Exception:
            pass
        # Gradient Fill tool for creating smooth color gradients
        gradient_btn = tk.Button(tools_frame, text="Gradient", command=self._select_gradient, bg=btn_bg, fg=btn_fg, relief=tk.RAISED, bd=1, font=("Arial", 9))
        gradient_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.tool_buttons['gradient'] = gradient_btn
        try:
            add_tooltip(gradient_btn, "Gradient tool - drag to create smooth color transitions")
        except Exception:
            pass

        # Shape tools frame
        shapes_frame = tk.Frame(left_frame, bg=bg_toolbar)
        shapes_frame.pack(padx=5, pady=(5, 5), fill=tk.X)
        shapes_label = ttk.Label(shapes_frame, text="Shapes", font=("Arial", 10, "bold"))
        shapes_label.pack(anchor=tk.W)
        try:
            add_tooltip(shapes_label, "Shape drawing tools for creating geometric objects")
        except Exception:
            pass
        
        # Shape tool buttons
        rect_btn = tk.Button(shapes_frame, text="Rectangle", command=self._select_rectangle, bg=btn_bg, fg=btn_fg, relief=tk.RAISED, bd=1, font=("Arial", 9))
        rect_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.tool_buttons['rectangle'] = rect_btn
        try:
            add_tooltip(rect_btn, "Rectangle tool - drag to draw rectangular shapes")
        except Exception:
            pass
        
        circle_btn = tk.Button(shapes_frame, text="Circle", command=self._select_circle, bg=btn_bg, fg=btn_fg, relief=tk.RAISED, bd=1, font=("Arial", 9))
        circle_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.tool_buttons['circle'] = circle_btn
        try:
            add_tooltip(circle_btn, "Circle tool - drag to draw circular shapes")
        except Exception:
            pass
        
        arrow_btn = tk.Button(shapes_frame, text="Arrow", command=self._select_arrow, bg=btn_bg, fg=btn_fg, relief=tk.RAISED, bd=1, font=("Arial", 9))
        arrow_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.tool_buttons['arrow'] = arrow_btn
        try:
            add_tooltip(arrow_btn, "Arrow tool - drag to draw arrow shapes")
        except Exception:
            pass
        
        polygon_btn = tk.Button(shapes_frame, text="Polygon", command=self._select_polygon, bg=btn_bg, fg=btn_fg, relief=tk.RAISED, bd=1, font=("Arial", 9))
        polygon_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.tool_buttons['polygon'] = polygon_btn
        try:
            add_tooltip(polygon_btn, "Polygon tool - click points to create multi-sided shapes")
        except Exception:
            pass

        # Canvas for displaying composite image; use a neutral mid-grey to be easy on the eyes
        self.canvas = tk.Canvas(self, bg="#cdcdcd", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        # Also allow Ctrl/Shift + LeftClick to perform click-to-zoom without replacing existing handler
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_click_zoom, add=True)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Double-1>", self._on_canvas_double_click)
        try:
            add_tooltip(self.canvas, "Main canvas - use tools to draw, edit, and manipulate images")
        except Exception:
            pass
        self._update_color_palette()
        # Attempt to register drag-and-drop for files.  If tkdnd is available, users can
        # drag image files directly onto the canvas to add them as new layers.  This is
        # wrapped in a try/except so it fails silently if tkdnd is not installed.
        try:
            # Check for tkdnd support
            self.tk.call('tkdnd::version')
            # Register the canvas to accept files
            self.canvas.drop_target_register('DND_Files')
            self.canvas.dnd_bind('<<Drop>>', self._on_drop)
        except Exception:
            pass
        # Status bar at bottom with soft colours
        status_frame = tk.Frame(self, bg="#e6e6e6")
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_label = tk.Label(status_frame, textvariable=self.status_var, bg="#e6e6e6", fg="#333333", anchor=tk.W)
        self.status_label.pack(fill=tk.X)
        # Update status on mouse motion
        self.canvas.bind("<Motion>", self._update_status)

        # Bind Ctrl+MouseWheel for zooming in and out.  When the user holds the
        # control key and scrolls the mouse wheel, the zoom factor is
        # adjusted and the composite view is redrawn.  Note: on some
        # platforms you may need to use '<Control-MouseWheel>' or bind
        # separately for Linux/OSX.
        self.canvas.bind("<Control-MouseWheel>", self._on_zoom)

        # Keyboard shortcuts for common operations (Photoshop‑like)
        # Undo/Redo
        self.bind_all("<Control-z>", lambda e: self._undo())
        self.bind_all("<Control-y>", lambda e: self._redo())
        # Duplicate layer (Ctrl+J)
        self.bind_all("<Control-j>", lambda e: self._duplicate_layer())
        # New layer (Ctrl+N)
        self.bind_all("<Control-n>", lambda e: self._new_layer())
        # Merge visible (Ctrl+E)
        self.bind_all("<Control-e>", lambda e: self._merge_visible_layers())
        # Save (Ctrl+S)
        self.bind_all("<Control-s>", lambda e: self._save_image())
        # Delete current layer (Delete key)
        self.bind_all("<Delete>", lambda e: self._delete_layer())
        # Selection nudge bindings (when select tool active and a selection exists)
        self.bind_all("<Left>", lambda e: self._nudge_selection_event(e, -1, 0))
        self.bind_all("<Right>", lambda e: self._nudge_selection_event(e, 1, 0))
        self.bind_all("<Up>", lambda e: self._nudge_selection_event(e, 0, -1))
        self.bind_all("<Down>", lambda e: self._nudge_selection_event(e, 0, 1))
        self.bind_all("<Shift-Left>", lambda e: self._nudge_selection_event(e, -10, 0))
        self.bind_all("<Shift-Right>", lambda e: self._nudge_selection_event(e, 10, 0))
        self.bind_all("<Shift-Up>", lambda e: self._nudge_selection_event(e, 0, -10))
        self.bind_all("<Shift-Down>", lambda e: self._nudge_selection_event(e, 0, 10))

    def _create_menus(self):
        # File menu
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image...", command=self._open_image)
        file_menu.add_command(label="Save As...", command=self._save_image)
        file_menu.add_command(label="Quick Export Variants…", command=self._quick_export)
        autosave_menu = tk.Menu(file_menu, tearoff=0)
        for label, seconds in (("Off", 0), ("30 seconds", 30), ("2 minutes", 120), ("5 minutes", 300)):
            autosave_menu.add_command(label=label, command=lambda s=seconds: self._set_autosave_interval(s))
        file_menu.add_cascade(label="Autosave", menu=autosave_menu)
        file_menu.add_command(label="View Session Log", command=self._show_session_log)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Undo", command=self._undo)
        edit_menu.add_command(label="Redo", command=self._redo)
        edit_menu.add_separator()
        edit_menu.add_command(label="Delete Layer", command=self._delete_layer)
        edit_menu.add_command(label="Duplicate Layer", command=self._duplicate_layer)
        edit_menu.add_separator()
        edit_menu.add_command(label="Move Layer Up", command=lambda: self._move_layer(-1))
        edit_menu.add_command(label="Move Layer Down", command=lambda: self._move_layer(1))
        edit_menu.add_separator()
        edit_menu.add_command(label="Merge Visible", command=self._merge_visible_layers)
        edit_menu.add_separator()
        edit_menu.add_command(label="Brush Settings", command=self._show_brush_settings)
        menubar.add_cascade(label="Edit", menu=edit_menu)

        # Filter menu
        filter_menu = tk.Menu(menubar, tearoff=0)
        filter_menu.add_command(label="Grayscale", command=lambda: self._preview_and_apply_filter("grayscale"))
        filter_menu.add_command(label="Invert", command=lambda: self._preview_and_apply_filter("invert"))
        filter_menu.add_command(label="Blur", command=lambda: self._preview_and_apply_filter("blur"))
        filter_menu.add_command(label="Sharpen", command=lambda: self._preview_and_apply_filter("sharpen"))
        filter_menu.add_separator()
        filter_menu.add_command(label="Emboss", command=lambda: self._preview_and_apply_filter("emboss"))
        filter_menu.add_command(label="Edge Enhance", command=lambda: self._preview_and_apply_filter("edge"))
        filter_menu.add_command(label="Contour", command=lambda: self._preview_and_apply_filter("contour"))
        filter_menu.add_command(label="Detail", command=lambda: self._preview_and_apply_filter("detail"))
        filter_menu.add_command(label="Smooth", command=lambda: self._preview_and_apply_filter("smooth"))
        filter_menu.add_command(label="Mirror Horizontal", command=lambda: self._preview_and_apply_filter("mirror_horizontal"))
        filter_menu.add_command(label="Mirror Vertical", command=lambda: self._preview_and_apply_filter("mirror_vertical"))
        filter_menu.add_command(label="Liquify (Swirl)", command=lambda: self._preview_and_apply_filter("liquify"))
        liquify_menu = tk.Menu(filter_menu, tearoff=0)
        for label, mode in [("Push", "push"), ("Pull", "pull"), ("Bloat", "bloat"), ("Pucker", "pucker")]:
            liquify_menu.add_command(label=label, command=lambda m=mode: self._apply_liquify_dialog(m))
        filter_menu.add_cascade(label="Liquify Tools", menu=liquify_menu)
        filter_menu.add_separator()
        filter_menu.add_command(label="Sepia", command=lambda: self._preview_and_apply_filter("sepia"))
        filter_menu.add_command(label="Skin Smooth", command=lambda: self._preview_and_apply_filter("skin smooth"))
        # Creative colour presets
        filter_menu.add_command(label="Warm Tone", command=lambda: self._preview_and_apply_filter("warm"))
        filter_menu.add_command(label="Cool Tone", command=lambda: self._preview_and_apply_filter("cool"))
        filter_menu.add_command(label="Vintage", command=lambda: self._preview_and_apply_filter("vintage"))
        filter_menu.add_separator()
        filter_menu.add_command(label="Anime Style", command=lambda: self._preview_and_apply_filter("anime"))
        filter_menu.add_command(label="Oil Painting", command=lambda: self._preview_and_apply_filter("oil"))
        filter_menu.add_command(label="Cyberpunk Tone", command=lambda: self._preview_and_apply_filter("cyberpunk"))
        filter_menu.add_command(label="Portrait Optimize", command=lambda: self._preview_and_apply_filter("portrait optimize"))
        filter_menu.add_command(label="Replace Background", command=self._replace_background)
        # Poster edges (custom
        filter_menu.add_command(label="Poster Edges", command=lambda: self._preview_and_apply_filter("poster_edges"))
        # Additional creative filters
        filter_menu.add_command(label="Posterize", command=lambda: self._preview_and_apply_filter("posterize"))
        filter_menu.add_command(label="Solarize", command=lambda: self._preview_and_apply_filter("solarize"))
        filter_menu.add_separator()
        filter_menu.add_command(label="Adjustable Blur", command=self._adjustable_blur)
        filter_menu.add_command(label="Adjustable Sharpen", command=self._adjustable_sharpen)
        filter_menu.add_command(label="Adjustable Sepia", command=self._adjustable_sepia)
        menubar.add_cascade(label="Filters", menu=filter_menu)

        # Canva inspired quick presets
        canva_menu = tk.Menu(menubar, tearoff=0)
        for label, preset in [
            ("Duotone", "duotone"),
            ("Glow", "glow"),
            ("Cinematic", "cinematic"),
            ("Neon", "neon"),
            ("Film Grain", "film_grain"),
            ("HDR Boost", "hdr"),
            ("Retro Fade", "retro"),
            ("Summer Warmth", "summer"),
            ("Winter Chill", "winter"),
            ("Vignette", "vignette"),
            ("Background Blur", "background_blur"),
        ]:
            canva_menu.add_command(label=label, command=lambda p=preset: self._apply_canva_preset(p))
        menubar.add_cascade(label="Canva", menu=canva_menu)

        # Auto menu for automatic enhancements
        auto_menu = tk.Menu(menubar, tearoff=0)
        auto_menu.add_command(label="Auto Enhance", command=self._auto_enhance_layer)
        auto_menu.add_command(label="One Click Beauty", command=self._one_click_beauty)
        auto_menu.add_command(label="Remove Background", command=self._remove_background)
        menubar.add_cascade(label="Auto", menu=auto_menu)

        # Layer menu
        layer_menu = tk.Menu(menubar, tearoff=0)
        layer_menu.add_command(label="New Layer", command=self._new_layer)
        layer_menu.add_command(label="New Vector Layer", command=self._new_vector_layer)
        layer_menu.add_command(label="New Adjustment Layer", command=self._new_adjustment_layer)
        layer_menu.add_separator()
        layer_menu.add_command(label="Delete Layer", command=self._delete_layer)
        layer_menu.add_command(label="Duplicate Layer", command=self._duplicate_layer)
        layer_menu.add_separator()
        layer_menu.add_command(label="Move Layer Up", command=lambda: self._move_layer(-1))
        layer_menu.add_command(label="Move Layer Down", command=lambda: self._move_layer(1))
        layer_menu.add_separator()
        layer_menu.add_command(label="Merge Visible", command=self._merge_visible_layers)
        menubar.add_cascade(label="Layer", menu=layer_menu)

        # Vector Tools menu
        vector_menu = tk.Menu(menubar, tearoff=0)
        vector_menu.add_command(label="Rectangle", command=self._select_vector_rectangle)
        vector_menu.add_command(label="Circle", command=self._select_vector_circle)
        vector_menu.add_command(label="Ellipse", command=self._select_vector_ellipse)
        vector_menu.add_command(label="Line", command=self._select_vector_line)
        vector_menu.add_command(label="Text", command=self._select_vector_text)
        menubar.add_cascade(label="Vector Tools", menu=vector_menu)

        # Macro menu
        macro_menu = tk.Menu(menubar, tearoff=0)
        macro_menu.add_command(label="Start Recording", command=self._start_macro_recording)
