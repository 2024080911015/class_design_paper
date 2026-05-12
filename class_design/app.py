from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, session, url_for
import io
import json
import os
import shutil
import zipfile


app = Flask(__name__)
app.secret_key = "key"
app.config["TEMPLATES_AUTO_RELOAD"] = True

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
USER_FILE = os.path.join(BASE_DIR, "user.json")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    with open(USER_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(users):
    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def normalize_upload_path(relative_path):
    relative_path = (relative_path or "").replace("\\", "/").strip("/")
    parts = [part for part in relative_path.split("/") if part and part != "."]
    if not parts or any(part == ".." for part in parts):
        abort(400, "无效的文件路径")
    return "/".join(parts)


def safe_upload_path(relative_path):
    clean_path = normalize_upload_path(relative_path)
    full_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, *clean_path.split("/")))
    if os.path.commonpath([UPLOAD_FOLDER, full_path]) != UPLOAD_FOLDER:
        abort(400, "无效的文件路径")
    return clean_path, full_path


def get_directory_structure(root_dir, base_dir=UPLOAD_FOLDER):
    structure = []
    if not os.path.exists(root_dir):
        return structure

    for item in sorted(os.listdir(root_dir), key=lambda name: name.lower()):
        item_path = os.path.join(root_dir, item)
        relative_path = os.path.relpath(item_path, base_dir).replace("\\", "/")
        stat = os.stat(item_path)
        if os.path.isdir(item_path):
            structure.append(
                {
                    "name": item,
                    "type": "folder",
                    "path": relative_path,
                    "modified": stat.st_mtime,
                    "children": get_directory_structure(item_path, base_dir),
                }
            )
        else:
            structure.append(
                {
                    "name": item,
                    "type": "file",
                    "path": relative_path,
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                }
            )
    return structure


def send_directory_zip(directory, relative_path):
    zip_filename = f"{os.path.basename(directory)}.zip"
    zip_buffer = io.BytesIO()
    parent_dir = os.path.dirname(directory)

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(directory):
            for filename in files:
                file_path = os.path.join(root, filename)
                arcname = os.path.relpath(file_path, parent_dir)
                zip_file.write(file_path, arcname=arcname)

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=zip_filename or f"{relative_path}.zip",
    )


@app.route("/")
def index():
    if "username" not in session:
        return redirect(url_for("register"))
    return render_template("index.html", username=session["username"])


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            return jsonify({"message": "账号和密码不能为空"}), 400

        users = load_users()
        if username in users:
            return jsonify({"message": "用户名已存在"}), 409

        users[username] = password
        save_users(users)
        return jsonify({"message": "注册成功"})

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        users = load_users()
        if users.get(username) == password:
            session["username"] = username
            return jsonify({"message": "登录成功"})
        return jsonify({"message": "用户名或密码错误"}), 401

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))


@app.route("/upload", methods=["POST"])
def receive_file():
    file = request.files.get("upload_file")
    if not file or not file.filename:
        return jsonify({"message": "请选择要上传的文件"}), 400

    clean_path, file_path = safe_upload_path(os.path.basename(file.filename))
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file.save(file_path)
    return jsonify({"message": "文件上传成功", "filename": clean_path, "path": clean_path})


@app.route("/upload_folder", methods=["POST"])
def upload_folder():
    file_list = request.files.getlist("upload_folder")
    saved_files = []

    for file in file_list:
        if not file or not file.filename:
            continue
        clean_path, file_path = safe_upload_path(file.filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)
        saved_files.append(clean_path)

    if not saved_files:
        return jsonify({"message": "请选择要上传的文件夹"}), 400

    return jsonify({"message": "文件夹上传成功", "count": len(saved_files), "files": saved_files})


@app.route("/download/<path:file_path>")
def download_file(file_path):
    clean_path, full_path = safe_upload_path(file_path)
    if not os.path.exists(full_path):
        abort(404)

    if os.path.isdir(full_path):
        return send_directory_zip(full_path, clean_path)

    return send_file(full_path, as_attachment=True, download_name=os.path.basename(full_path))


@app.route("/files")
def list_files():
    return jsonify(get_directory_structure(UPLOAD_FOLDER))


@app.route("/delete/<path:file_path>", methods=["POST"])
def delete_file(file_path):
    clean_path, target_path = safe_upload_path(file_path)
    if not os.path.exists(target_path):
        return jsonify({"message": "文件不存在", "path": clean_path}), 404

    if os.path.isdir(target_path):
        shutil.rmtree(target_path)
        message = "文件夹删除成功"
    else:
        os.remove(target_path)
        message = "文件删除成功"

    return jsonify({"message": message, "path": clean_path})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
