import cv2
import numpy as np
import base64
from mnist_model import DigitRecognizer

class CNNEngine:
    """Module nhan dang so viet tay su dung OpenCV + MNIST CNN (Hoàn chỉnh xuất Probabilities)."""

    def __init__(self):
        print("[CNN Engine] Loading MNIST digit recognizer...")
        self.digit_recognizer = DigitRecognizer()
        print("[CNN Engine] MNIST ready!")

    def preprocess_handwritten(self, image):
        max_dimension = 1920
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=41,
            C=8
        )
        
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
        
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
        return binary, image

    def recognize_numbers(self, image_path=None, image_array=None):
        if image_array is not None:
            original = image_array.copy()
        elif image_path is not None:
            original = cv2.imread(image_path)
            if original is None:
                return {"success": False, "error": "Cannot read image.", "results": [], "all_numbers": ""}
        else:
            return {"success": False, "error": "Need image_path or image_array.", "results": [], "all_numbers": ""}

        binary, original = self.preprocess_handwritten(original)
        digit_regions = self._find_digit_contours(binary)

        results = []
        for region_info in digit_regions:
            x, y, w, h = region_info['rect']
            bbox = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]

            digit_img = region_info['image']
            prediction = self.digit_recognizer.predict_digit(digit_img)

            if prediction['confidence'] > 0.75:
                results.append({
                    "text": str(prediction['digit']),
                    "confidence": round(prediction['confidence'], 4),
                    "bbox": bbox,
                    "type": "number",
                    "engine": "MNIST CNN",
                    "probabilities": prediction['probabilities']  # GIỮ LẠI MẢNG TỈ LỆ 10 SỐ
                })

        results.sort(key=lambda r: (r["bbox"][0][1], r["bbox"][0][0]))
        results = self._group_nearby_digits(results)

        annotated = self._draw_results(original, results)
        annotated_base64 = self._image_to_base64(annotated)

        all_numbers = " | ".join([r["text"] for r in results])

        return {
            "success": True,
            "results": results,
            "annotated_image": annotated_base64,
            "all_numbers": all_numbers,
            "total_found": len(results),
            "mode_used": "handwritten (CNN)"
        }

    def _find_digit_contours(self, binary_image):
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digit_regions = []
        img_h, img_w = binary_image.shape[:2]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / max(w, 1)
            area = w * h

            if (0.1 < aspect_ratio < 20.0 and area > 300 and w > 10 and h > 30 and w < img_w * 0.98 and h < img_h * 0.98):
                digit_img = binary_image[y:y+h, x:x+w]
                padded = self._pad_digit(digit_img)
                digit_regions.append({
                    'rect': (x, y, w, h),
                    'image': padded
                })
        return digit_regions

    def _pad_digit(self, digit_img):
        h, w = digit_img.shape[:2]
        pad = max(4, int(max(h, w) * 0.2))
        size = max(h, w) + 2 * pad
        padded = np.zeros((size, size), dtype=np.uint8)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        padded[y_offset:y_offset+h, x_offset:x_offset+w] = digit_img
        return cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)

    def _group_nearby_digits(self, results):
        if len(results) <= 1: 
            if len(results) == 1:
                results[0]["probabilities"] = [results[0]["probabilities"]]
            return results
        grouped = []
        used = set()

        for i, r1 in enumerate(results):
            if i in used: continue
            if len(r1["text"]) > 1:
                grouped.append(r1)
                used.add(i)
                continue

            group = [r1]
            used.add(i)

            for j, r2 in enumerate(results):
                if j in used or len(r2["text"]) > 1: continue

                last = group[-1]
                last_right = max(p[0] for p in last["bbox"])
                last_y_center = sum(p[1] for p in last["bbox"]) / 4

                r2_left = min(p[0] for p in r2["bbox"])
                r2_y_center = sum(p[1] for p in r2["bbox"]) / 4
                r2_height = max(p[1] for p in r2["bbox"]) - min(p[1] for p in r2["bbox"])

                h_distance = r2_left - last_right
                v_distance = abs(r2_y_center - last_y_center)

                if 0 < h_distance < r2_height * 0.4 and v_distance < r2_height * 0.5:
                    group.append(r2)
                    used.add(j)

            if len(group) > 1:
                combined_text = "".join([g["text"] for g in group])
                avg_conf = sum(g["confidence"] for g in group) / len(group)
                all_points = [p for g in group for p in g["bbox"]]
                x_min, y_min = min(p[0] for p in all_points), min(p[1] for p in all_points)
                x_max, y_max = max(p[0] for p in all_points), max(p[1] for p in all_points)

                grouped.append({
                    "text": combined_text,
                    "confidence": round(avg_conf, 4),
                    "bbox": [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                    "type": "number",
                    "engine": group[0].get("engine", "MNIST CNN"),
                    "probabilities": [g["probabilities"] for g in group] # Gộp mảng xác suất thành mảng 2 chiều
                })
            else:
                r1_copy = r1.copy()
                r1_copy["probabilities"] = [r1["probabilities"]]
                grouped.append(r1_copy)
        return grouped

    def _draw_results(self, image, results):
        annotated = image.copy()
        for result in results:
            bbox = result["bbox"]
            text = result["text"]
            confidence = result["confidence"]
            color = (0, 255, 100)

            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

            x_min = min(p[0] for p in bbox)
            y_min = min(p[1] for p in bbox)
            label = f"{text} ({confidence:.0%})"

            font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

            cv2.rectangle(annotated, (x_min, y_min - text_h - 10), (x_min + text_w + 4, y_min - 2), color, -1)
            cv2.putText(annotated, label, (x_min + 2, y_min - 6), font, font_scale, (0, 0, 0), thickness)
        return annotated

    def _image_to_base64(self, image):
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"