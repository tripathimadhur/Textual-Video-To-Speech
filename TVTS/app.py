import logging
from werkzeug.utils import secure_filename
import imutils
logging.captureWarnings(True)
import os
from gtts import gTTS
from flask import Flask
from flask import render_template
from flask import request
from flask import send_from_directory
from PIL import Image
import cv2
from modules import cap
import pytesseract


app = Flask(__name__, static_url_path="", static_folder="static")
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['mp4'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']



@app.route('/')
def index():
    return render_template('index.html')


# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        print(file.filename)
        # Move the file form the temporal folder to
        # the upload folder we setup
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Redirect the user to the uploaded_file route, which
        # will basicaly show on the browser the uploaded file

        # print '\n\nI am Here\n\n'

        total = cap.frameCapture('uploads/' + file.filename)

        min_match_count = 7
        img1 = cv2.imread('images/1.jpg', 0)
        img1 = cap.binarize(img1)
        img1 = imutils.resize(img1, width=1000)

        for i in range(2, (total)):
            print('Stitching image ' + str(i))
            img2 = cv2.imread('images/' + str(i) + '.jpg', 0)
            img2 = cap.binarize(img2)
            img2 = imutils.resize(img2, width=1000)
            img1 = cap.stitch(img1, img2, min_match_count)

        cv2.imwrite('images/0.jpg', img1)
        cv2.imwrite('static/0.jpg', img1)

        print('\n\nConverting Image to Text')
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        string = pytesseract.image_to_string(Image.open('images/0.jpg'))
        print('\n\nOCR OUTPUT\n\n' + string + '\n\n')

        f = open("static/test.txt", "w")
        f.write(string)
        f.close()

        string = '"{}"'.format(string)
        print('Converting Text to Speech\n\n')
        tts = gTTS(text=string, lang='en')
        tts.save("static/tts.mp3");

        return render_template('index1.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == '__main__':
    app.run(
        port="8000",
        debug=True
    )