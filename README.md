Huấn luyện craft

Mô tả: Craft detection sẽ khoanh vùng nơi có các kí tự theo nhiều hình dạng khác nhau như hình vuông, hình chữ nhật hay hình cong bao quanh các cụm kí tự.
- Dữ liệu huấn luyện: VinAI
- Bài báo: https://arxiv.org/pdf/1904.01941.pdf
- Code gốc: https://github.com/faustomorales/keras-ocr

Dữ liệu đầu vào gồm:
- Ảnh: Ảnh có đuôi .jpg và có kích cỡ không giới hạn, trong ảnh bao gồm các vùng văn bản để nhận diện
- Nhãn: Nhãn được ghi trong file .txt gồm 9 phần tử bao gồm: 8 phần tử số và phần tử chữ ở cuối được nhận diện sẽ được đánh dấu trong cặp dấu ngoặc kép(""), các phần tử cách nhau bằng dấu cách.