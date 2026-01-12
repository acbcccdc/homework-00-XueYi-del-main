import os
import traceback
from functools import lru_cache
from typing import Optional

import cv2
import gradio as gr
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection


# MediaPipe FaceMesh -> 68点（近似 dlib 68 点语义顺序）
# 说明：FaceMesh本身是468点；这里选取一组常用映射做“68点展示/对比”。
MP_68_FROM_FACEMESH = [
    # jaw (0-16)
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397,
    # left brow (17-21)
    70, 63, 105, 66, 107,
    # right brow (22-26)
    336, 296, 334, 293, 300,
    # nose (27-35)
    168, 197, 195, 5, 4, 1, 19, 94, 2,
    # left eye (36-41)
    33, 160, 158, 133, 153, 144,
    # right eye (42-47)
    362, 385, 387, 263, 373, 380,
    # mouth outer (48-59)
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    # mouth inner (60-67)
    78, 95, 88, 178, 87, 14, 317, 402,
]


def _rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _draw_points_bgr(image_bgr: np.ndarray, points_xy: np.ndarray, color_bgr=(0, 255, 0)) -> np.ndarray:
    out = image_bgr.copy()
    for (x, y) in points_xy.astype(int):
        cv2.circle(out, (int(x), int(y)), 2, color_bgr, -1)
    return out


def _safe_put_text(image_bgr: np.ndarray, text: str) -> np.ndarray:
    out = image_bgr.copy()
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out


@lru_cache(maxsize=1)
def _get_facemesh() -> mp_face_mesh.FaceMesh:
    return mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _get_facemesh_landmarks_xy(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    h, w = image_bgr.shape[:2]
    results = _get_facemesh().process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark
    pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)
    return pts


def _get_mp68_xy(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    pts = _get_facemesh_landmarks_xy(image_bgr)
    if pts is None:
        return None
    return pts[np.array(MP_68_FROM_FACEMESH, dtype=int)]


def apply_media_pipe_face_detection(image_bgr: np.ndarray) -> np.ndarray:
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return image_bgr
        annotated_image = image_bgr.copy()
        for detection in results.detections:
            mp_drawing.draw_detection(annotated_image, detection)
        return annotated_image


def apply_media_pipe_facemesh(image_bgr: np.ndarray) -> np.ndarray:
    results = _get_facemesh().process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return image_bgr
    annotated_image = image_bgr.copy()
    face_landmarks = results.multi_face_landmarks[0]
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
    )
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
    )
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
    )
    return annotated_image


def _patch_scipy_simps_if_needed() -> None:
    """torchlm 在部分 SciPy 版本下会用到 integrate.simps；这里做兼容补丁。"""
    try:
        import scipy.integrate as integrate
    except Exception:
        return
    try:
        integrate.simps  # noqa: B018
    except AttributeError:
        from scipy.integrate import simpson as _simps

        integrate.simps = _simps


class _TorchLMRuntime:
    def __init__(self):
        self._ready = False
        self._error: Optional[str] = None
        self._model = None

    def ensure(self) -> bool:
        if self._ready:
            return True
        if self._error:
            return False
        try:
            _patch_scipy_simps_if_needed()
            import torchlm

            model = torchlm.models.pipnet_resnet18_10x68x32x256_300w(pretrained=True)
            model.eval()
            bound = torchlm.runtime.bind(model)
            self._model = bound
            self._ready = True
            return True
        except Exception as e:
            self._error = f"TorchLM init failed: {e}"
            return False

    @property
    def error(self) -> Optional[str]:
        return self._error

    def predict_68(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        if not self.ensure():
            return None
        m = self._model
        if m is None:
            return None
        try:
            # 兼容不同 torchlm 版本的接口
            out = None
            if hasattr(m, "predict"):
                out = m.predict(image_bgr)
            elif callable(m):
                out = m(image_bgr)
            elif hasattr(m, "forward"):
                out = m.forward(image_bgr)
            else:
                return None

            # 常见返回： (landmarks, bboxes) 或 landmarks
            landmarks = out[0] if isinstance(out, (tuple, list)) else out
            if landmarks is None:
                return None
            if hasattr(landmarks, "detach"):
                landmarks = landmarks.detach().cpu().numpy()
            landmarks = np.asarray(landmarks)
            # 形状可能是 (N,68,2) 或 (68,2)
            if landmarks.ndim == 3:
                landmarks = landmarks[0]
            if landmarks.shape[0] != 68:
                return None
            return landmarks.astype(np.float32)
        except Exception:
            return None


_torchlm_runtime = _TorchLMRuntime()


def apply_torchlm_landmarks(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    pts68 = _torchlm_runtime.predict_68(image_bgr)
    if pts68 is None:
        return None
    return _draw_points_bgr(image_bgr, pts68, color_bgr=(0, 0, 255))


@lru_cache(maxsize=1)
def _load_badge_bgra() -> Optional[np.ndarray]:
    # 优先用 FaceAnalysisWebApp/model/gdut_badge.png
    badge_path = os.path.join(os.path.dirname(__file__), "model", "gdut_badge.png")
    if os.path.exists(badge_path):
        img = cv2.imread(badge_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            if img.shape[2] == 3:
                a = np.full((img.shape[0], img.shape[1], 1), 255, dtype=img.dtype)
                img = np.concatenate([img, a], axis=2)
            return img

    # 退化：用仓库根目录的 gdut.jpg（无透明通道）
    root_badge = os.path.join(os.path.dirname(__file__), "..", "..", "..", "gdut.jpg")
    root_badge = os.path.normpath(root_badge)
    if os.path.exists(root_badge):
        img = cv2.imread(root_badge, cv2.IMREAD_COLOR)
        if img is not None:
            a = np.full((img.shape[0], img.shape[1], 1), 255, dtype=img.dtype)
            return np.concatenate([img, a], axis=2)
    return None


def _alpha_blend_bgra(dst_bgr: np.ndarray, overlay_bgra: np.ndarray) -> np.ndarray:
    out = dst_bgr.astype(np.float32)
    rgb = overlay_bgra[:, :, :3].astype(np.float32)
    a = (overlay_bgra[:, :, 3:4].astype(np.float32)) / 255.0
    out = rgb * a + out * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def _estimate_head_pose(image_bgr: np.ndarray, facemesh_xy: np.ndarray):
    """按 OpenCV pose 教程的思路，用6个关键点 solvePnP。"""
    h, w = image_bgr.shape[:2]
    # 2D points from MediaPipe (face mesh indices)
    image_points = np.array(
        [
            facemesh_xy[1],   # nose tip
            facemesh_xy[152], # chin
            facemesh_xy[33],  # left eye outer
            facemesh_xy[263], # right eye outer
            facemesh_xy[61],  # mouth left
            facemesh_xy[291], # mouth right
        ],
        dtype="double",
    )

    model_points = np.array(
        [
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0),  # Right mouth corner
        ],
        dtype="double",
    )

    focal_length = w
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))
    success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None
    return rvec, tvec, camera_matrix, dist_coeffs


def overlay_gdut_badge(image_bgr: np.ndarray) -> np.ndarray:
    """AR：把GDUT校徽贴到额头上（随头部姿态做透视变换）。"""
    badge = _load_badge_bgra()
    if badge is None:
        return _safe_put_text(image_bgr, "Missing gdut badge (gdut.jpg or model/gdut_badge.png)")

    try:
        facemesh_xy = _get_facemesh_landmarks_xy(image_bgr)
        if facemesh_xy is None:
            return image_bgr

        pose = _estimate_head_pose(image_bgr, facemesh_xy)
        if pose is None:
            return image_bgr
        rvec, tvec, camera_matrix, dist_coeffs = pose

        # 在“鼻尖坐标系”上定义一个贴图平面矩形（放到额头上方）
        # y为正方向表示向上（相对鼻尖）
        quad_3d = np.array(
            [
                (-160.0, 320.0, -120.0),
                (160.0, 320.0, -120.0),
                (160.0, 120.0, -120.0),
                (-160.0, 120.0, -120.0),
            ],
            dtype="double",
        )
        proj, _ = cv2.projectPoints(quad_3d, rvec, tvec, camera_matrix, dist_coeffs)
        dst = proj.reshape(-1, 2).astype(np.float32)

        bh, bw = badge.shape[:2]
        src = np.array([[0, 0], [bw - 1, 0], [bw - 1, bh - 1], [0, bh - 1]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            badge,
            H,
            (image_bgr.shape[1], image_bgr.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        return _alpha_blend_bgra(image_bgr, warped)
    except Exception:
        traceback.print_exc()
        return image_bgr


class FaceProcessing(object):
    def __init__(self, ui_obj):
        self.name = "Face Image Processing"
        self.description = "Call for Face Image and video Processing"
        self.ui_obj = ui_obj

    def take_webcam_photo(self, image):
        return image

    def take_webcam_video(self, images):
        return images

    def mp_webcam_photo(self, image):
        return image

    def mp_webcam_face_mesh(self, image):
        mesh_image = apply_media_pipe_facemesh(image)
        return mesh_image

    def mp_webcam_face_detection(self, image):
        face_detection_img = apply_media_pipe_face_detection(image)
        return face_detection_img

    def ar_apply_badge(self, image):
        return overlay_gdut_badge(image)

    def models_comparison(self, image):
        """按作业要求输出4窗口：
        1 原图
        2 叠加 MediaPipe 68点
        3 叠加 TorchLM 68点
        4 两模型对应点连线（线长=定位差异）
        """
        if image is None:
            return None, None, None, None
        image_bgr = _rgb_to_bgr(image)

        original = image_bgr

        mp68 = _get_mp68_xy(image_bgr)
        if mp68 is None:
            mp_overlay = _safe_put_text(image_bgr, "No face (MediaPipe)")
        else:
            mp_overlay = _draw_points_bgr(image_bgr, mp68, color_bgr=(0, 255, 0))

        torchlm68 = _torchlm_runtime.predict_68(image_bgr)
        if torchlm68 is None:
            msg = _torchlm_runtime.error or "TorchLM not ready"
            torchlm_overlay = _safe_put_text(image_bgr, msg)
        else:
            torchlm_overlay = _draw_points_bgr(image_bgr, torchlm68, color_bgr=(0, 0, 255))

        diff = image_bgr.copy()
        if mp68 is None or torchlm68 is None:
            diff = _safe_put_text(diff, "Need both MP68 & TorchLM68")
        else:
            for i in range(68):
                p = mp68[i]
                q = torchlm68[i]
                length = float(np.linalg.norm(p - q))
                thickness = max(1, int(length / 8.0))
                cv2.line(diff, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), (255, 0, 0), thickness)

        return (
            _bgr_to_rgb(original),
            _bgr_to_rgb(mp_overlay),
            _bgr_to_rgb(torchlm_overlay),
            _bgr_to_rgb(diff),
        )

    def create_ui(self):
        with self.ui_obj:
            gr.Markdown("Face Analysis with Webcam/Video")
            with gr.Tabs():
                with gr.TabItem("AR demo"):
                    with gr.Row():
                        ar_in = gr.Image(sources=["webcam"], streaming=True, label="Webcam")
                        ar_out = gr.Image(label="GDUT badge (AR)")
                    ar_in.change(
                        lambda img: _bgr_to_rgb(overlay_gdut_badge(_rgb_to_bgr(img))) if img is not None else None,
                        inputs=ar_in,
                        outputs=ar_out,
                    )

                with gr.TabItem("models comparison"):
                    with gr.Row():
                        cmp_in = gr.Image(sources=["webcam"], streaming=True, label="Webcam")
                    with gr.Row():
                        cmp_orig = gr.Image(label="Original")
                        cmp_mp = gr.Image(label="MediaPipe 68")
                    with gr.Row():
                        cmp_torchlm = gr.Image(label="TorchLM 68")
                        cmp_diff = gr.Image(label="MP vs TorchLM (diff lines)")

                    cmp_in.change(
                        self.models_comparison,
                        inputs=cmp_in,
                        outputs=[cmp_orig, cmp_mp, cmp_torchlm, cmp_diff],
                    )

    def launch_ui(self):
        share = os.environ.get("GRADIO_SHARE", "0") == "1"
        self.ui_obj.launch(share=share)


if __name__ == '__main__':
    my_app = gr.Blocks()
    face_ui = FaceProcessing(my_app)
    face_ui.create_ui()
    face_ui.launch_ui()

