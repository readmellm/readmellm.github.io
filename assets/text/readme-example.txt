You will be provided with a question, rules, and related context, all delimited with XML tags. For the context, you will be specifically provided functions and examples. Please answer the question using the related context in mind.

<rules>
Rule number 1: When you’re unsure about something, ask the user what information you need
Rule number 2: Reuse the library’s functions and code when applicable
Rule number 3: Consider library dependencies when generating code solutions
</rules>

<context>

<context_description>
The context will be for the Supervision Library. The Supervision library creates tools to enable developers to complete their Computer Vision tasks. The context I am giving you will be functions and examples related to detection, classification, and utilities from the Supervision Library. The context is organized into different numbered sections in order using XML tags. Within each section, there is a context description for that section, a code snippet, and use case examples.
</context_description>

<context_1>

<context_1_description>
The sv.Detections class in the Supervision library standardizes results from various object detection and segmentation models into a consistent format. This class simplifies data manipulation and filtering, providing a uniform API for integration with Supervision trackers, annotators, and tools.
</context_1_description>

<context_1_code_snippet>
@dataclass
class Detections:
    xyxy: np.ndarray
    mask: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    class_id: Optional[np.ndarray] = None
    tracker_id: Optional[np.ndarray] = None
    data: Dict[str, Union[np.ndarray, List]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
    
    def __len__(self):
        
    def __iter__(self) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray], Optional[float], Optional[int], Optional[int], Dict[str, Union[np.ndarray, List]]]]:
        
    def __eq__(self, other: Detections):
        
    @classmethod
    def from_yolov5(cls, yolov5_results) -> Detections:
        
    @classmethod
    def from_ultralytics(cls, ultralytics_results) -> Detections:
        
    @classmethod
    def from_yolo_nas(cls, yolo_nas_results) -> Detections:
        
    @classmethod
    def from_tensorflow(cls, tensorflow_results: dict, resolution_wh: tuple) -> Detections:
        
    @classmethod
    def from_deepsparse(cls, deepsparse_results) -> Detections:
        
    @classmethod
    def from_mmdetection(cls, mmdet_results) -> Detections:
        
    @classmethod
    def from_transformers(cls, transformers_results: dict, id2label: Optional[Dict[int, str]] = None) -> Detections:
        
    @classmethod
    def from_detectron2(cls, detectron2_results: Any) -> Detections:
        
    @classmethod
    def from_inference(cls, roboflow_result: Union[dict, Any]) -> Detections:
        
    @classmethod
    def from_sam(cls, sam_result: List[dict]) -> Detections:
        
    @classmethod
    def from_azure_analyze_image(cls, azure_result: dict, class_map: Optional[Dict[int, str]] = None) -> Detections:
        
    @classmethod
    def from_paddledet(cls, paddledet_result) -> Detections:
        
    @classmethod
    def from_lmm(cls, lmm: Union[LMM, str], result: Union[str, dict], **kwargs: Any) -> Detections:
        
    @classmethod
    def from_vlm(cls, vlm: Union[VLM, str], result: Union[str, dict], **kwargs: Any) -> Detections:
        
    @classmethod
    def from_easyocr(cls, easyocr_results: list) -> Detections:
        
    @classmethod
    def from_ncnn(cls, ncnn_results) -> Detections:
        
    @classmethod
    def empty(cls) -> Detections:
        
    def is_empty(self) -> bool:
        
    @classmethod
    def merge(cls, detections_list: List[Detections]) -> Detections:
        
    def get_anchors_coordinates(self, anchor: Position) -> np.ndarray:
        
    def __getitem__(self, index: Union[int, slice, List[int], np.ndarray, str]) -> Union[Detections, List, np.ndarray, None]:
        
    def __setitem__(self, key: str, value: Union[np.ndarray, List]):
        
    @property
    def area(self) -> np.ndarray:
        
    @property
    def box_area(self) -> np.ndarray:
        
    def with_nms(self, threshold: float = 0.5, class_agnostic: bool = False) -> Detections:
        
    def with_nmm(self, threshold: float = 0.5, class_agnostic: bool = False) -> Detections:
        
def merge_inner_detection_object_pair(detections_1: Detections, detections_2: Detections) -> Detections:
    
def merge_inner_detections_objects(detections: List[Detections], threshold=0.5) -> Detections:
    
def validate_fields_both_defined_or_none(detections_1: Detections, detections_2: Detections) -> None:
</context_1_code_snippet>

<context_1_examples>
=== "Inference"

Use [`sv.Detections.from_inference`](/detection/core/#supervision.detection.core.Detections.from_inference) method, which accepts model results from both detection and segmentation models.

        ```python
        import cv2
        import supervision as sv
        from inference import get_model

        model = get_model(model_id="yolov8n-640")
        image = cv2.imread(<SOURCE_IMAGE_PATH>)
        results = model.infer(image)[0]
        detections = sv.Detections.from_inference(results)
        ```

=== "Ultralytics"

Use [`sv.Detections.from_ultralytics`](/detection/core/#supervision.detection.core.Detections.from_ultralytics) method, which accepts model results from both detection and segmentation models.

        ```python
        import cv2
        import supervision as sv
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        image = cv2.imread(<SOURCE_IMAGE_PATH>)
        results = model(image)[0]
        detections = sv.Detections.from_ultralytics(results)
        ```

=== "Transformers"
Use [`sv.Detections.from_transformers`](/detection/core/#supervision.detection.core.Detections.from_transformers) 
 method, which accepts model results from both detection and segmentation models.

        ```python
        import torch
        import supervision as sv
        from PIL import Image
        from transformers import DetrImageProcessor, DetrForObjectDetection

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        image = Image.open(<SOURCE_IMAGE_PATH>)
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        width, height = image.size
        target_size = torch.tensor([[height, width]])
        results = processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_size)[0]
        detections = sv.Detections.from_transformers(
            transformers_results=results,
            id2label=model.config.id2label)
        ```
</context_1_examples>

</context_1>

<context_2>

<context_2_description>
This context section focuses on visual annotation utilities for object detection tasks from the SuperVision library. It provides various annotators to overlay bounding boxes, oriented boxes, masks, polygons, labels, and other graphical elements onto images based on object detection outputs.
</context_2_description>

<context_2_code_snippet>
class BoxAnnotator(BaseAnnotator):
    def __init__(self, color: Union[Color, ColorPalette], thickness: int, color_lookup: ColorLookup):
    def annotate(self, scene: ImageType, detections: Detections, custom_color_lookup: Optional[np.ndarray]) -> ImageType:

class OrientedBoxAnnotator(BaseAnnotator):
    def __init__(self, color: Union[Color, ColorPalette], thickness: int, color_lookup: ColorLookup):
    def annotate(self, scene: ImageType, detections: Detections, custom_color_lookup: Optional[np.ndarray]) -> ImageType:

class MaskAnnotator(BaseAnnotator):
    def __init__(self, color: Union[Color, ColorPalette], opacity: float, color_lookup: ColorLookup):
    def annotate(self, scene: ImageType, detections: Detections, custom_color_lookup: Optional[np.ndarray]) -> ImageType:

class PolygonAnnotator(BaseAnnotator):
    def __init__(self, color: Union[Color, ColorPalette], thickness: int, color_lookup: ColorLookup):
    def annotate(self, scene: ImageType, detections: Detections, custom_color_lookup: Optional[np.ndarray]) -> ImageType:

class ColorAnnotator(BaseAnnotator):
    def __init__(self, color: Union[Color, ColorPalette], opacity: float, color_lookup: ColorLookup):
    def annotate(self, scene: ImageType, detections: Detections, custom_color_lookup: Optional[np.ndarray]) -> ImageType:

class HaloAnnotator(BaseAnnotator):
    def __init__(self, color: Union[Color, ColorPalette], opacity: float, kernel_size: int, color_lookup: ColorLookup):
    def annotate(self, scene: ImageType, detections: Detections, custom_color_lookup: Optional[np.ndarray]) -> ImageType:

class EllipseAnnotator(BaseAnnotator):
    def __init__(self, color: Union[Color, ColorPalette], thickness: int, start_angle: int, end_angle: int, color_lookup: ColorLookup):
    def annotate(self, scene: ImageType, detections: Detections, custom_color_lookup: Optional[np.ndarray]) -> ImageType:

class BoxCornerAnnotator(BaseAnnotator):
    def __init__(self, color: Union[Color, ColorPalette], thickness: int, corner_length: int, color_lookup: ColorLookup):
    def annotate(self, scene: ImageType, detections: Detections, custom_color_lookup: Optional[np.ndarray]) -> ImageType:

class CircleAnnotator(BaseAnnotator):
    def __init__(self, color: Union[Color, ColorPalette], thickness: int, color_lookup: ColorLookup):
    def annotate(self, scene: ImageType, detections: Detections, custom_color_lookup: Optional[np.ndarray]) -> ImageType:

class DotAnnotator(BaseAnnotator):
    def __init__(self, color: Union[Color, ColorPalette], radius: int, position: Position, color_lookup: ColorLookup, outline_thickness: int, outline_color: Union[Color, ColorPalette]):
    def annotate(self, scene: ImageType, detections: Detections, custom_color_lookup: Optional[np.ndarray]) -> ImageType:

class LabelAnnotator(BaseAnnotator):
    def __init__(self, color: Union[Color, ColorPalette], text_color: Union[Color, ColorPalette], text_scale: float, text_thickness: int, text_padding: int, text_position: Position, color_lookup: ColorLookup, border_radius: int, smart_position: bool):
    def annotate(self, scene: ImageType, detections: Detections, labels: Optional[List[str]], custom_color_lookup: Optional[np.ndarray]) -> ImageType:
    def _validate_labels(self, labels: Optional[List[str]], detections: Detections):
    def _get_label_properties(self, detections: Detections, labels: List[str]) -> np.ndarray:
    def _get_labels_text(self, detections: Detections, custom_labels: Optional[List[str]]) -> List[str]:
    def _draw_labels(self, scene: np.ndarray, labels: List[str], label_properties: np.ndarray, detections: Detections, custom_color_lookup: Optional[np.ndarray]) -> None:
    def draw_rounded_rectangle(self, scene: np.ndarray, xyxy: Tuple[int, int, int, int], color: Tuple[int, int, int], border_radius: int) -> np.ndarray:

class RichLabelAnnotator(BaseAnnotator):
    def __init__(self, color: Union[Color, ColorPalette], text_color: Union[Color, ColorPalette], font_path: Optional[str], font_size: int, text_padding: int, text_position: Position, color_lookup: ColorLookup, border_radius: int, smart_position: bool):
    def annotate(self, scene: ImageType, detections: Detections, labels: Optional[List[str]], custom_color_lookup: Optional[np.ndarray]) -> ImageType:
    def _validate_labels(self, labels: Optional[List[str]], detections: Detections):
    def _get_label_properties(self, draw, detections: Detections, labels: List[str]) -> np.ndarray:
    def _get_labels_text(self, detections: Detections, custom_labels: Optional[List[str]]) -> List[str]:
    def _draw_labels(self, draw, labels: List[str], label_properties: np.ndarray, detections: Detections, custom_color_lookup: Optional[np.ndarray]) -> None:
    def _load_font(self, font_size: int, font_path: Optional[str]):

class IconAnnotator(BaseAnnotator):
    def __init__(self, icon_resolution_wh: Tuple[int, int] = (64, 64), icon_position: Position = Position.TOP_CENTER, offset_xy: Tuple[int, int] = (0, 0)):
    @ensure_cv2_image_for_annotation
    def annotate(self, scene: ImageType, detections: Detections, icon_path: Union[str, List[str]]) -> ImageType:
    @lru_cache
    def _load_icon(self, icon_path: str) -> np.ndarray:

class BlurAnnotator(BaseAnnotator):
    def __init__(self, kernel_size: int = 15):
    @ensure_cv2_image_for_annotation
    def annotate(self, scene: ImageType, detections: Detections) -> ImageType:

class TraceAnnotator(BaseAnnotator):
    def __init__(self, color: Union[Color, ColorPalette] = ColorPalette.DEFAULT, position: Position = Position.CENTER, trace_length: int = 30, thickness: int = 2, color_lookup: ColorLookup = ColorLookup.CLASS):
    @ensure_cv2_image_for_annotation
    def annotate(self, scene: ImageType, detections: Detections, custom_color_lookup: Optional[np.ndarray] = None) -> ImageType:

class HeatMapAnnotator(BaseAnnotator):
    def __init__(self, position: Position = Position.BOTTOM_CENTER, opacity: float = 0.2, radius: int = 40, kernel_size: int = 25, top_hue: int = 0, low_hue: int = 125):
    @ensure_cv2_image_for_annotation
    def annotate(self, scene: ImageType, detections: Detections) -> ImageType:

class PixelateAnnotator(BaseAnnotator):
    def __init__(self, pixel_size: int = 20):
    @ensure_cv2_image_for_annotation
    def annotate(self, scene: ImageType, detections: Detections) -> ImageType:

class TriangleAnnotator(BaseAnnotator):
    def __init__(self, color: Union[Color, ColorPalette] = ColorPalette.DEFAULT, base: int = 10, height: int = 10, position: Position = Position.TOP_CENTER, color_lookup: ColorLookup = ColorLookup.CLASS, outline_thickness: int = 0, outline_color: Union[Color, ColorPalette] = Color.BLACK):
    @ensure_cv2_image_for_annotation
    def annotate(self, scene: ImageType, detections: Detections, custom_color_lookup: Optional[np.ndarray] = None) -> ImageType:

class RoundBoxAnnotator(BaseAnnotator):
    def __init__(self, color: Union[Color, ColorPalette] = ColorPalette.DEFAULT, thickness: int = 2, color_lookup: ColorLookup = ColorLookup.CLASS, roundness: float = 0.6):
    @ensure_cv2_image_for_annotation
    def annotate(self, scene: ImageType, detections: Detections, custom_color_lookup: Optional[np.ndarray] = None) -> ImageType:

class PercentageBarAnnotator(BaseAnnotator):
    def __init__(self, height: int = 16, width: int = 80, color: Union[Color, ColorPalette] = ColorPalette.DEFAULT, border_color: Color = Color.BLACK, position: Position = Position.TOP_CENTER, color_lookup: ColorLookup = ColorLookup.CLASS, border_thickness: Optional[int] = None):
    @ensure_cv2_image_for_annotation
    def annotate(self, scene: ImageType, detections: Detections, custom_color_lookup: Optional[np.ndarray] = None, custom_values: Optional[np.ndarray] = None) -> ImageType:
    @staticmethod
    def calculate_border_coordinates(anchor_xy: Tuple[int, int], border_wh: Tuple[int, int], position: Position) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    @staticmethod
    def validate_custom_values(custom_values: Optional[Union[np.ndarray, List[float]]], detections: Detections) -> None:

class CropAnnotator(BaseAnnotator):
    def __init__(self, position: Position = Position.TOP_CENTER, scale_factor: float = 2.0, border_color: Union[Color, ColorPalette] = ColorPalette.DEFAULT, border_thickness: int = 2, border_color_lookup: ColorLookup = ColorLookup.CLASS):
    @ensure_cv2_image_for_annotation
    def annotate(self, scene: ImageType, detections: Detections, custom_color_lookup: Optional[np.ndarray] = None) -> ImageType:
    @staticmethod
    def calculate_crop_coordinates(anchor: Tuple[int, int], crop_wh: Tuple[int, int], position: Position) -> Tuple[Tuple[int, int], Tuple[int, int]]:

class BackgroundOverlayAnnotator(BaseAnnotator):
    def __init__(self, color: Color = Color.BLACK, opacity: float = 0.5, force_box: bool = False):
    @ensure_cv2_image_for_annotation
    def annotate(self, scene: ImageType, detections: Detections) -> ImageType:

class ComparisonAnnotator:
    def __init__(self, color_1: Color = Color.RED, color_2: Color = Color.GREEN, color_overlap: Color = Color.BLUE, *, opacity: float = 0.75, label_1: str = "", label_2: str = "", label_overlap: str = "", label_scale: float = 1.0):
    @ensure_cv2_image_for_annotation
    def annotate(self, scene: ImageType, detections_1: Detections, detections_2: Detections) -> ImageType:
    @staticmethod
    def _use_obb(detections_1: Detections, detections_2: Detections) -> bool:
    @staticmethod
    def _use_mask(detections_1: Detections, detections_2: Detections) -> bool:
    @staticmethod
    def _mask_from_xyxy(scene: np.ndarray, detections: Detections) -> np.ndarray:
    @staticmethod
    def _mask_from_obb(scene: np.ndarray, detections: Detections) -> np.ndarray:
    @staticmethod
    def _mask_from_mask(scene: np.ndarray, detections: Detections) -> np.ndarray:
    def _draw_labels(self, scene: np.ndarray) -> None:
</context_2_code_snippet>

<context_2_examples	>
image = ...
detections = sv.Detections(...)

# Box Annotator Example
box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)

# Box Corner Annotator Example
corner_annotator = sv.BoxCornerAnnotator()
annotated_frame = corner_annotator.annotate(scene=image.copy(), detections=detections)

# Color Annotator Example
color_annotator = sv.ColorAnnotator()
annotated_frame = color_annotator.annotate(scene=image.copy(), detections=detections)

# Circle Annotator Example
circle_annotator = sv.CircleAnnotator()
annotated_frame = circle_annotator.annotate(scene=image.copy(), detections=detections)

# Dot Annotator Example
dot_annotator = sv.DotAnnotator()
annotated_frame = dot_annotator.annotate(scene=image.copy(), detections=detections)

# Triangle Annotator Example
triangle_annotator = sv.TriangleAnnotator()
annotated_frame = triangle_annotator.annotate(scene=image.copy(), detections=detections)

# Ellipse Annotator Example
ellipse_annotator = sv.EllipseAnnotator()
annotated_frame = ellipse_annotator.annotate(scene=image.copy(), detections=detections)

# Halo Annotator Example
halo_annotator = sv.HaloAnnotator()
annotated_frame = halo_annotator.annotate(scene=image.copy(), detections=detections)

# Percentage Bar Annotator Example
percentage_bar_annotator = sv.PercentageBarAnnotator()
annotated_frame = percentage_bar_annotator.annotate(scene=image.copy(), detections=detections)

# Mask Annotator Example
mask_annotator = sv.MaskAnnotator()
annotated_frame = mask_annotator.annotate(scene=image.copy(), detections=detections)

# Polygon Annotator Example
polygon_annotator = sv.PolygonAnnotator()
annotated_frame = polygon_annotator.annotate(scene=image.copy(), detections=detections)

# Label Annotator Example
labels = [f"{class_name} {confidence:.2f}" for class_name, confidence in zip(detections['class_name'], detections.confidence)]
label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
annotated_frame = label_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

# Rich Label Annotator Example
labels = [f"{class_name} {confidence:.2f}" for class_name, confidence in zip(detections['class_name'], detections.confidence)]
rich_label_annotator = sv.RichLabelAnnotator(font_path="<TTF_FONT_PATH>", text_position=sv.Position.CENTER)
annotated_frame = rich_label_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

# Icon Annotator Example
icon_paths = ["<ICON_PATH>" for _ in detections]
icon_annotator = sv.IconAnnotator()
annotated_frame = icon_annotator.annotate(scene=image.copy(), detections=detections, icon_path=icon_paths)

# Blur Annotator Example
blur_annotator = sv.BlurAnnotator()
annotated_frame = blur_annotator.annotate(scene=image.copy(), detections=detections)

# Pixelate Annotator Example
pixelate_annotator = sv.PixelateAnnotator()
annotated_frame = pixelate_annotator.annotate(scene=image.copy(), detections=detections)

# Trace Annotator Example
model = YOLO('yolov8x.pt')
trace_annotator = sv.TraceAnnotator()
video_info = sv.VideoInfo.from_video_path(video_path='...')
frames_generator = sv.get_video_frames_generator(source_path='...')
tracker = sv.ByteTrack()

with sv.VideoSink(target_path='...', video_info=video_info) as sink:
    for frame in frames_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)
        annotated_frame = trace_annotator.annotate(scene=frame.copy(), detections=detections)
        sink.write_frame(frame=annotated_frame)

# Heat Map Annotator Example
model = YOLO('yolov8x.pt')
heat_map_annotator = sv.HeatMapAnnotator()
video_info = sv.VideoInfo.from_video_path(video_path='...')
frames_generator = sv.get_video_frames_generator(source_path='...')

with sv.VideoSink(target_path='...', video_info=video_info) as sink:
    for frame in frames_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        annotated_frame = heat_map_annotator.annotate(scene=frame.copy(), detections=detections)
        sink.write_frame(frame=annotated_frame)

# Background Overlay Annotator Example
background_overlay_annotator = sv.BackgroundOverlayAnnotator()
annotated_frame = background_overlay_annotator.annotate(scene=image.copy(), detections=detections)

# Comparison Annotator Example
image = ...
detections_1 = sv.Detections(...)
detections_2 = sv.Detections(...)
comparison_annotator = sv.ComparisonAnnotator()
annotated_frame = comparison_annotator.annotate(scene=image.copy(), detections_1=detections_1, detections_2=detections_2)
</context_2_examples>

</context_2>

<context_3>

<context_3_description>
The Supervision library provides a set of utilities for image preprocessing, overlaying annotations, creating image grids, and saving image outputs.
</context_3_description>

<context_3_code_snippet>
def crop_image(
    image: ImageType,
    xyxy: Union[npt.NDArray[int], List[int], Tuple[int, int, int, int]],
) -> ImageType:

def scale_image(image: ImageType, scale_factor: float) -> ImageType:

def resize_image(
    image: ImageType,
    resolution_wh: Tuple[int, int],
    keep_aspect_ratio: bool = False,
) -> ImageType:

def letterbox_image(
    image: ImageType,
    resolution_wh: Tuple[int, int],
    color: Union[Tuple[int, int, int], Color] = Color.BLACK,
) -> ImageType:

def overlay_image(
    image: npt.NDArray[np.uint8],
    overlay: npt.NDArray[np.uint8],
    anchor: Tuple[int, int],
) -> npt.NDArray[np.uint8]:

class ImageSink:
    def __init__(
        self,
        target_dir_path: str,
        overwrite: bool = False,
        image_name_pattern: str = "image_{:05d}.png",
    ):

    def __enter__(self):

    def save_image(self, image: np.ndarray, image_name: Optional[str] = None):

    def __exit__(self, exc_type, exc_value, exc_traceback):

def create_tiles(
    images: List[ImageType],
    grid_size: Optional[Tuple[Optional[int], Optional[int]]] = None,
    single_tile_size: Optional[Tuple[int, int]] = None,
    tile_scaling: Literal["min", "max", "avg"] = "avg",
    tile_padding_color: Union[Tuple[int, int, int], Color] = Color.from_hex("#D9D9D9"),
    tile_margin: int = 10,
    tile_margin_color: Union[Tuple[int, int, int], Color] = Color.from_hex("#BFBEBD"),
    return_type: Literal["auto", "cv2", "pillow"] = "auto",
    titles: Optional[List[Optional[str]]] = None,
    titles_anchors: Optional[Union[Point, List[Optional[Point]]]] = None,
    titles_color: Union[Tuple[int, int, int], Color] = Color.from_hex("#262523"),
    titles_scale: Optional[float] = None,
    titles_thickness: int = 1,
    titles_padding: int = 10,
    titles_text_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    titles_background_color: Union[Tuple[int, int, int], Color] = Color.from_hex(
        "#D9D9D9"
    ),
    default_title_placement: RelativePosition = "top",
) -> ImageType:

def _negotiate_tiles_format(images: List[ImageType]) -> Literal["cv2", "pillow"]:

def _calculate_aggregated_images_shape(
    images: List[np.ndarray], aggregator: Callable[[List[int]], float]
) -> Tuple[int, int]:

def _aggregate_images_shape(
    images: List[np.ndarray], mode: Literal["min", "max", "avg"]
) -> Tuple[int, int]:

def _establish_grid_size(
    images: List[np.ndarray], grid_size: Optional[Tuple[Optional[int], Optional[int]]]
) -> Tuple[int, int]:

def _negotiate_grid_size(images: List[np.ndarray]) -> Tuple[int, int]:

def _generate_tiles(
    images: List[np.ndarray],
    grid_size: Tuple[int, int],
    single_tile_size: Tuple[int, int],
    tile_padding_color: Tuple[int, int, int],
    tile_margin: int,
    tile_margin_color: Tuple[int, int, int],
    titles: Optional[List[Optional[str]]],
    titles_anchors: List[Optional[Point]],
    titles_color: Tuple[int, int, int],
    titles_scale: Optional[float],
    titles_thickness: int,
    titles_padding: int,
    titles_text_font: int,
    titles_background_color: Tuple[int, int, int],
    default_title_placement: RelativePosition,
) -> np.ndarray:

def _draw_texts(
    images: List[np.ndarray],
    titles: Optional[List[Optional[str]]],
    titles_anchors: List[Optional[Point]],
    titles_color: Tuple[int, int, int],
    titles_scale: Optional[float],
    titles_thickness: int,
    titles_padding: int,
    titles_text_font: int,
    titles_background_color: Tuple[int, int, int],
    default_title_placement: RelativePosition,
) -> List[np.ndarray]:

def _prepare_default_titles_anchors(
    images: List[np.ndarray],
    titles_anchors: List[Optional[Point]],
    default_title_placement: RelativePosition,
) -> List[Point]:

def _merge_tiles_elements(
    tiles_elements: List[List[np.ndarray]],
    grid_size: Tuple[int, int],
    single_tile_size: Tuple[int, int],
    tile_margin: int,
    tile_margin_color: Tuple[int, int, int],
) -> np.ndarray:

def _generate_color_image(
    shape: Tuple[int, int], color: Tuple[int, int, int]
) -> np.ndarray:
</context_3_code_snippet>

<context_3_examples>
# Using sv.crop_image to crop an image based on bounding box coordinates
image = cv2.imread(<SOURCE_IMAGE_PATH>)
image.shape
# (1080, 1920, 3)

xyxy = [200, 400, 600, 800]
cropped_image = sv.crop_image(image=image, xyxy=xyxy)
cropped_image.shape
# (400, 400, 3)

# Using sv.scale_image to scale an image by a given factor
scaled_image = sv.scale_image(image=image, scale_factor=0.5)
scaled_image.shape
# (540, 960, 3)

# Using sv.resize_image to resize an image while optionally maintaining aspect ratio
resized_image = sv.resize_image(
    image=image, resolution_wh=(1000, 1000), keep_aspect_ratio=True
)
resized_image.shape
# (562, 1000, 3)

# Using sv.letterbox_image to resize an image while maintaining aspect ratio and adding padding
letterboxed_image = sv.letterbox_image(image=image, resolution_wh=(1000, 1000))
letterboxed_image.shape
# (1000, 1000, 3)

# Using sv.overlay_image to overlay an image onto another at a specified anchor point
image = cv2.imread(<SOURCE_IMAGE_PATH>)
overlay = np.zeros((400, 400, 3), dtype=np.uint8)
result_image = sv.overlay_image(image=image, overlay=overlay, anchor=(200, 400))

# Using sv.ImageSink to save images in a specified directory
frames_generator = sv.get_video_frames_generator(<SOURCE_VIDEO_PATH>, stride=2)

with sv.ImageSink(target_dir_path=<TARGET_CROPS_DIRECTORY>) as sink:
    for image in frames_generator:
        sink.save_image(image=image)

</context_3_examples>

</context_3>
