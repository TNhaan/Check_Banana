# 🍅 Phân loại tình trạng quả chuối bằng ResNet50 (Flask + Camera)

Ứng dụng web sử dụng **ResNet50 thông qua Roboflow API** để phân loại **tình trạng quả chuối** từ ảnh người dùng **upload hoặc chụp từ camera**. Kết quả sẽ được **trả về kèm độ tin cậy**, trực quan và dễ quan sát.

---

## 🎯 Tính năng chính

- ✅ Upload ảnh từ máy
- ✅ Chụp ảnh trực tiếp bằng webcam trình duyệt
- ✅ Gửi ảnh đến mô hình ResNet50 được triển khai trên Roboflow
- ✅ Hiển thị kết quả nhận diện với độ tin cậy
- ✅ Giao diện web đơn giản, dễ sử dụng

---

## 🛠 Công nghệ sử dụng

- [Flask](https://flask.palletsprojects.com/)
- [Roboflow Inference SDK](https://github.com/roboflow/inference)
- HTML5, CSS3, JavaScript (Webcam API)

---

## ⚙️ Cài đặt và chạy ứng dụng

### 1. Clone dự án và cài thư viện

```bash
git clone https://github.com/TNhaan/Check_Banana.git
cd  check_banana_main
pip install -r requirements.txt
```
- bước 1:
python -m venv venv
.\venv\Scripts\activate
- bước 2:
pip install -r requirements.txt
- bước 3:
python app.py

