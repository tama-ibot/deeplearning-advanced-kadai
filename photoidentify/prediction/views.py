from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from io import BytesIO
import os
import base64

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html',{'form': form})
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # アップロードされた画像を取得
            img_file = form.cleaned_data['image']

            # content_type を事前に取得
            content_type = img_file.content_type

            # ファイルをBytesIOに変換
            img_file = BytesIO(img_file.read())


            img = load_img(img_file, target_size=(224,224))
            img_array = img_to_array(img)
            img_array = img_array.reshape((1, 224, 224, 3))
            img_array = preprocess_input(img_array)
            
            model_path = os.path.join(settings.BASE_DIR, 'prediction','models','vgg16.h5')
            model = load_model(model_path)
            
            result = model.predict(img_array)

            decoded_result = decode_predictions(result, top=5)[0]
            
            predictions = [{'label': pred[1], 'probability': f"{pred[2] * 100:.2f}%"} for pred in decoded_result]
        
            img_bytes = img_file.getvalue()  # BytesIOからバイト列を取得
            img_data = base64.b64encode(img_bytes).decode('utf-8')
            img_data = f"data:{content_type};base64,{img_data}"

            return render(request, 'home.html', {
                'form': form, 
                'predictions': predictions, 
                'img_data': img_data
            })
        else:
            form = ImageUploadForm()
            return render(request,'home.html',{'form': form})