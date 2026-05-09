"""
YOLO vision quality inspection pipeline.
Camera -> YOLO detection -> pass/fail decision -> structured inspection result.
SDKs: Ultralytics YOLO, OpenCV, Supervision
"""
import cv2
import time
import uuid
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Generator
from dataclasses import dataclass, field, asdict
from enum import Enum

from ultralytics import YOLO
import supervision as sv


class DefectType(str, Enum):
    SCRATCH = "scratch"
    DENT = "dent"
    CRACK = "crack"
    DISCOLORATION = "discoloration"
    MISSING_COMPONENT = "missing_component"
    MISALIGNMENT = "misalignment"
    FOREIGN_OBJECT = "foreign_object"


class InspectionVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    QUARANTINE = "QUARANTINE"   # uncertain, needs human review


@dataclass
class Defect:
    defect_id: str
    defect_type: DefectType
    confidence: float
    bbox: List[float]           # [x1, y1, x2, y2] normalized
    severity: str               # "minor", "major", "critical"
    area_pct: float             # defect area as % of part area


@dataclass
class InspectionResult:
    result_id: str
    timestamp: float
    part_id: str
    line_id: str
    camera_id: str
    verdict: InspectionVerdict
    defects: List[Defect]
    inference_ms: float
    frame_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_defects(self) -> int:
        return len(self.defects)

    @property
    def has_critical(self) -> bool:
        return any(d.severity == "critical" for d in self.defects)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        d["defects"] = [
            {**asdict(defect), "defect_type": defect.defect_type.value}
            for defect in self.defects
        ]
        return d


# Default severity thresholds
SEVERITY_MAP = {
    DefectType.CRACK: "critical",
    DefectType.MISSING_COMPONENT: "critical",
    DefectType.DENT: "major",
    DefectType.MISALIGNMENT: "major",
    DefectType.SCRATCH: "minor",
    DefectType.DISCOLORATION: "minor",
    DefectType.FOREIGN_OBJECT: "major",
}

# Class name -> DefectType mapping (customize per your model)
CLASS_DEFECT_MAP = {
    "scratch": DefectType.SCRATCH,
    "dent": DefectType.DENT,
    "crack": DefectType.CRACK,
    "discoloration": DefectType.DISCOLORATION,
    "missing": DefectType.MISSING_COMPONENT,
    "misalignment": DefectType.MISALIGNMENT,
    "foreign_object": DefectType.FOREIGN_OBJECT,
}


class QualityInspector:
    """
    Vision-based quality inspection using YOLO.
    Detects defects on manufactured parts and issues pass/fail verdicts.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "cuda",
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        line_id: str = "LINE_A",
        camera_id: str = "CAM_01",
        fail_on_any_defect: bool = False,
        critical_conf_threshold: float = 0.6,
        save_frames: bool = False,
        frames_dir: str = "./inspection_frames",
    ):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.device = device
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.line_id = line_id
        self.camera_id = camera_id
        self.fail_on_any_defect = fail_on_any_defect
        self.critical_conf_threshold = critical_conf_threshold
        self.save_frames = save_frames
        self.frames_dir = Path(frames_dir)
        if save_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.class_names = self.model.names
        self._total_inspected = 0
        self._total_failed = 0

        annotator = sv.BoxAnnotator(thickness=2)
        self._annotator = annotator
        print(f"[Inspector] YOLO loaded: {model_path} | device={device} | line={line_id}")

    def inspect_frame(
        self, frame: np.ndarray, part_id: Optional[str] = None
    ) -> InspectionResult:
        """
        Inspect a single BGR frame. Returns structured InspectionResult.
        This is the entry point — camera to verdict.
        """
        part_id = part_id or f"part_{int(time.time()*1000)}"
        t0 = time.perf_counter()

        results = self.model.predict(
            frame, conf=self.conf, iou=self.iou,
            device=self.device, verbose=False,
        )[0]

        inference_ms = (time.perf_counter() - t0) * 1000
        h, w = frame.shape[:2]
        part_area = h * w

        defects = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.class_names.get(cls_id, f"class_{cls_id}").lower()
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            defect_type = CLASS_DEFECT_MAP.get(cls_name, DefectType.FOREIGN_OBJECT)
            severity = SEVERITY_MAP.get(defect_type, "minor")
            if conf >= self.critical_conf_threshold and severity == "major":
                severity = "critical"

            bbox_norm = [x1/w, y1/h, x2/w, y2/h]
            area_pct = ((x2-x1) * (y2-y1)) / part_area * 100

            defects.append(Defect(
                defect_id=str(uuid.uuid4())[:8],
                defect_type=defect_type,
                confidence=round(conf, 3),
                bbox=bbox_norm,
                severity=severity,
                area_pct=round(area_pct, 2),
            ))

        # Verdict logic
        if not defects:
            verdict = InspectionVerdict.PASS
        elif self.fail_on_any_defect:
            verdict = InspectionVerdict.FAIL
        elif any(d.severity == "critical" for d in defects):
            verdict = InspectionVerdict.FAIL
        elif any(d.severity == "major" for d in defects):
            verdict = InspectionVerdict.QUARANTINE
        else:
            verdict = InspectionVerdict.PASS  # minor defects only

        frame_path = None
        if self.save_frames and verdict != InspectionVerdict.PASS:
            frame_path = str(self.frames_dir / f"{part_id}_{verdict.value}.jpg")
            annotated = self._annotate(frame, results)
            cv2.imwrite(frame_path, annotated)

        self._total_inspected += 1
        if verdict == InspectionVerdict.FAIL:
            self._total_failed += 1

        return InspectionResult(
            result_id=str(uuid.uuid4())[:12],
            timestamp=time.time(),
            part_id=part_id,
            line_id=self.line_id,
            camera_id=self.camera_id,
            verdict=verdict,
            defects=defects,
            inference_ms=round(inference_ms, 2),
            frame_path=frame_path,
        )

    def _annotate(self, frame: np.ndarray, results) -> np.ndarray:
        annotated = frame.copy()
        sv_det = sv.Detections.from_ultralytics(results)
        labels = [
            f"{self.class_names[int(c)]} {conf:.2f}"
            for c, conf in zip(sv_det.class_id, sv_det.confidence)
        ]
        annotated = self._annotator.annotate(annotated, sv_det)
        return annotated

    def inspect_stream(
        self,
        source: Union[str, int] = 0,
        max_frames: Optional[int] = None,
    ) -> Generator[InspectionResult, None, None]:
        """Continuously inspect frames from a camera or video source."""
        cap = cv2.VideoCapture(source)
        frame_count = 0
        part_counter = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                part_counter += 1
                result = self.inspect_frame(frame, part_id=f"part_{part_counter:06d}")
                yield result
                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    break
        finally:
            cap.release()

    def stats(self) -> Dict[str, Any]:
        fail_rate = self._total_failed / max(self._total_inspected, 1)
        return {
            "total_inspected": self._total_inspected,
            "total_failed": self._total_failed,
            "fail_rate": round(fail_rate, 4),
            "pass_rate": round(1 - fail_rate, 4),
            "line_id": self.line_id,
        }
