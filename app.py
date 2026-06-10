import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from cnn_engine import CNNEngine

# ============================================================
# Cấu hình Flask
# ============================================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB upload
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')

# Đuôi file cho phép
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'}

# Tạo thư mục uploads nếu chưa có
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================
# Khởi tạo OCR Engine (load model 1 lần duy nhất)
# ============================================================
print("=" * 50)
print("  NHAN DANG SO V2 - Starting...")
print("=" * 50)
cnn_engine = CNNEngine()
print("=" * 50)
print("  Server ready!")
print("=" * 50)


def allowed_file(filename):
    """Kiểm tra đuôi file có hợp lệ không."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================
# Routes
# ============================================================

@app.route('/')
def index():
    """Trang chủ - Giao diện upload ảnh."""
    return render_template('index.html')


@app.route('/api/recognize', methods=['POST'])
def recognize():
    """
    API nhận dạng số từ ảnh.

    Request: multipart/form-data với field 'image'
    Response: JSON {success, results, annotated_image, all_numbers, total_found}
    """
    # Kiểm tra file upload
    if 'image' not in request.files:
        return jsonify({
            "success": False,
            "error": "Không tìm thấy file ảnh. Vui lòng upload ảnh.",
            "results": [],
            "all_numbers": ""
        }), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({
            "success": False,
            "error": "Chưa chọn file. Vui lòng chọn ảnh để nhận dạng.",
            "results": [],
            "all_numbers": ""
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Định dạng file không hỗ trợ. Chỉ chấp nhận: {', '.join(ALLOWED_EXTENSIONS)}",
            "results": [],
            "all_numbers": ""
        }), 400

    try:
        # Lưu file tạm
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Chạy CNN
        result = cnn_engine.recognize_numbers(image_path=filepath)

        # Xóa file tạm sau khi xử lý xong
        try:
            os.remove(filepath)
        except OSError:
            pass

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Lỗi khi xử lý ảnh: {str(e)}",
            "results": [],
            "all_numbers": ""
        }), 500


# ============================================================
# Chạy server
# ============================================================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
