Mô hình text recognition dùng để nhận diện văn bản (https://github.com/pbcquoc/vietocr).
Trong project này, mình cài đặt mô hình VietOCR có tính tổng quát cực tốt, thậm chí có độ chính xác khá cao trên một bộ dataset mới mặc dù mô hình chưa được huấn luyện bao giờ.

Cài Đặt
Để cài đặt các bạn gõ lệnh sau
pip install vietocr

Quick Start
Các bạn tham khảo notebook này để biết cách sử dụng: https://colab.research.google.com/drive/1s35psj_3yPYvSrsN0O9pFHgr1_7Jl-Cq#scrollTo=B95BBXNExipj

DATASET
Mô hình training bằng dữ liệu được sinh ra từ https://github.com/Belval/TextRecognitionDataGenerator

THÔNG SỐ
data_root: thư mục lưu toàn bộ dữ liệu ảnh
train_annotation: đường dẫn đến dữ liệu train
valid_annotation: đường dẫn đến dữ liệu valid
print_every: hiện thị train loss sau n bước
valid_every: hiện thị valid loss sau n bước
iters: số lần lặp để train mô hình
export: xuất file weights
metrics: số lượng mẫu trong valid annotation được dùng để tính toán full_sequence_accuracy
