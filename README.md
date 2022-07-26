Trình tạo dữ liệu cho việc nhận dạng văn bản (https://github.com/Belval/TextRecognitionDataGenerator)

Mục đích ?
Tạo dữ liệu OCR ngẫu nhiên từ dữ liệu text cho trước. 

Cài đặt?
Cài đặt pypi package
pip install trdg
Hoặc có thể clone về và chạy pip install -r requirements.txt

Sử dụng:
Chạy câu lệnh sau:
trdg -c 1000 -w 5 -f 64
Code sẽ sinh ra 1,000 dữ liệu ảnh ngẫu nhiên với chữ ở trên đó. 

Giải thích thư mục:
dicts - chứa từ điển dùng để generated từ ngẫu nhiên.
fonts - chứa font chữ dùng khi generated.
images - thư mục chứa ảnh nền của ảnh để in lên bè mặt
texts - chứa từ có thể lấy ra được.
out- chứa toàn bộ text data đã generated.

Detail file run.py:
-i - Sử dụng text file để generated text
-l - Sử dụng tiếng để generated(en)
-c - Số lượng file
-rs - Sử dụng chuỗi ký tự, đi cùng -let (chỉ bao gồm chữ cái), -num( chỉ bao gồm số) và -sym( chỉ bao gồm biểu tượng)
-w - Chỉ ra số lượng ký tự trong mỗi đoạn mẫu generated
-r - Xác định chuỗi với số lượng từ thay đổi.
-f - Định nghĩa chiều cao của chữ cái( hoặc rộng)
-t - CHọn bao nhiêu nhân để xử lý(CPU)
-e - CHọn dạng extension
-k - Có tạo chữ xiên
-rk - random giá trị skewing
-bl -làm mờ ảnh đầu ra
-rbl - làm mờ ngẫu nhiên giá trị 0 và -bl
-b - CHọn loại nền cho chữ (0: Gaussian Noise, 1: Plain white, 2: Quasicrystal, 3: Image")
-na - CHọn đầu ra được gắn nhãn ntn ( default: [ID].[EXT] + one file labels.txt containing id-to-label mappings)
-d - tạo biến dạng đầu ra (0: None (Default), 1: Sine wave, 2: Cosine wave, 3: Random)
-wd - chọn đầu ra cho kết quả
-al - chọn ví trí của chữ cái trong ảnh.
-ws - Tách ra theo từng chữ