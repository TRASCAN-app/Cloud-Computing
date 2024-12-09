from fastapi import APIRouter, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Dropout
from tensorflow.keras.activations import swish
from PIL import Image
from io import BytesIO
from google.cloud import firestore, storage
import requests
import tempfile

router = APIRouter()
db = firestore.Client()
storage_client = storage.Client()

class FixedDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)

    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

get_custom_objects().update({'FixedDropout': FixedDropout, 'swish': swish})

class_labels = [
    'ATK', 'alas_kaki', 'alat_makan_plastik', 'botol_detergen', 'botol_kaca', 'botol_plastik', 'box', 'gelas_kertas',
    'gelas_plastik', 'kaleng_spray', 'kertas', 'koran', 'majalah', 'masker', 'minuman_kaleng', 'pakaian', 
    'plastik_bungkus_makanan', 'plastik_tempat_sampah', 'rokok', 'sampah_elektronik', 'sedotan_plastik', 'sisa_makanan', 
    'tas_plastik', 'tempat_kosmetik', 'tempat_makan_aluminium', 'tempat_makan_plastik', 'tempat_makan_styrofoam', 'toples'
]

waste_type_labels = {
    'ATK': 'Anorganik', 'alas_kaki': 'Anorganik', 'alat_makan_plastik': 'Anorganik',
    'botol_detergen': 'Anorganik', 'botol_kaca': 'Anorganik', 'botol_plastik': 'Anorganik',
    'box': 'Anorganik', 'gelas_kertas': 'Anorganik', 'gelas_plastik': 'Anorganik',
    'kaleng_spray': 'B3', 'kertas': 'Anorganik', 'koran': 'Anorganik', 'majalah': 'Anorganik',
    'masker': 'Anorganik', 'minuman_kaleng': 'Anorganik', 'pakaian': 'Anorganik',
    'plastik_bungkus_makanan': 'Anorganik', 'plastik_tempat_sampah': 'Anorganik',
    'rokok': 'B3', 'sampah_elektronik': 'B3', 'sedotan_plastik': 'Anorganik',
    'sisa_makanan': 'Organik', 'tas_plastik': 'Anorganik', 'tempat_kosmetik': 'Anorganik',
    'tempat_makan_aluminium': 'Anorganik', 'tempat_makan_plastik': 'Anorganik',
    'tempat_makan_styrofoam': 'Anorganik', 'toples': 'Anorganik'
}

recyclable_labels = {
    'ATK': False, 'alas_kaki': True, 'alat_makan_plastik': True, 'botol_detergen': True, 
    'botol_kaca': True, 'botol_plastik': True, 'box': True, 'gelas_kertas': False, 'gelas_plastik': True, 
    'kaleng_spray': False, 'kertas': True, 'koran': True, 'majalah': True, 'masker': False, 
    'minuman_kaleng': True, 'pakaian': True, 'plastik_bungkus_makanan': True, 'plastik_tempat_sampah': False, 
    'rokok': False, 'sampah_elektronik': False, 'sedotan_plastik': True, 'sisa_makanan': True, 
    'tas_plastik': True, 'tempat_kosmetik': True, 'tempat_makan_aluminium': True, 
    'tempat_makan_plastik': True, 'tempat_makan_styrofoam': False, 'toples': True
    }


def load_model_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
        temp_file.write(response.content)
        model_path = temp_file.name
    return tf.keras.models.load_model(model_path)

model_url = 'https://storage.googleapis.com/trascan-model/model_EfficientNetB3.h5'
model = load_model_from_url(model_url)

def read_and_process_image(file: UploadFile):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Tipe file gambar invalid. Harap gunakan JPEG/JPG/PNG.")
    image_data = file.file.read()
    img = Image.open(BytesIO(image_data)).resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0)

def upload_image_to_gcs(file: UploadFile):
    bucket_name = 'trascan-images' 
    file_name = f"images/{file.filename}"
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    file.file.seek(0)
    blob.upload_from_file(file.file, content_type=file.content_type)
    blob.make_public()

    return blob.public_url

def save_prediction_to_firestore(data):
    db.collection("predictions").add(data)

MAX_FILE_SIZE = 1 * 1024 * 1024  # 1 MB dalam bytes
@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        file_size = 0
        contents = await file.read()
        file_size = len(contents)
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Ukuran gambar tidak boleh lebih dari 1MB")
        
        # Reset file pointer setelah membaca isinya
        file.file.seek(0)
    
        img_array = read_and_process_image(file)
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_class_label = class_labels[predicted_class_idx]
        waste_type = waste_type_labels.get(predicted_class_label, "unknown")
        is_recyclable = recyclable_labels.get(predicted_class_label, False)
        confidence_score = float(predictions[0][predicted_class_idx])
        image_url = upload_image_to_gcs(file)

        if 0.9 <= confidence_score <= 1.0:
            response_data = {
                "message": "success",
                "result": predicted_class_label,
                "id": int(predicted_class_idx),
                "score": confidence_score,
                "recyclable": is_recyclable,
                "waste_type": waste_type,
                "image_url": image_url 
            }

            save_prediction_to_firestore(response_data)
            return response_data
        
        return {"error": "Score prediksi terlalu rendah, gambar tidak terdeteksi dengan benar!"}
    except HTTPException as e:
        raise e
    except Exception as e:
        error_data = {
            "error": str(e),
            "filename": file.filename
        }
        db.collection("error_logs").add(error_data)
        return {"error": str(e)}
