import base64
import io
import os
from flask import Flask, render_template, request
from model import transform_image

app = Flask(__name__)

def convert_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    if imagefile.filename == '':
        return render_template('index.html', transformed_image=None)

    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    to_monet = request.form.get('to_monet', False)
    original, transformed_image = transform_image(image_path, to_monet)

    os.remove(image_path)

    img_str = convert_image_to_base64(original)
    trans_img_str = convert_image_to_base64(transformed_image)


    return render_template('index.html', input_image=img_str, transformed_image=trans_img_str, to_monet=to_monet)
    

if __name__ == '__main__':
    app.run(port=3000, debug=True)
