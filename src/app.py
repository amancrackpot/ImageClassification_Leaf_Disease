from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates
import uvicorn, aiohttp, asyncio
from io import BytesIO, StringIO
import tensorflow as tf
import sys
import PIL
import base64
import pdb
from pathlib import Path
import numpy as np


export_file_name = 'final_model.h5'
classes = ['Cassava Bacterial Blight (CBB)', 'Cassava Mosaic Disease (CMD)', 'Cassava Brown Streak Disease (CBSD)', 'Cassava Green Mottle (CGM)', 'Healthy']
path = Path(__file__).parent

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

templates = Jinja2Templates(directory='src/templates')
app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='src/static'))


fin_model = tf.keras.models.load_model(path/'saved'/export_file_name)
graph = tf.get_default_graph()

def model_predict(img_b):  
    image = PIL.Image.open(BytesIO(img_b))
    image_numpy = np.expand_dims(np.array(image), 0) #add batch dim
    # these 2 will be the only preprocessing steps required
    
    outputs = fin_model(image_numpy, training=False).numpy()
    label = classes[np.argmax(outputs)]
    formatted_outputs = [f"{i*100:.2f}" for i in outputs]	
    
	
	
    pred_probs = (classes, formatted_outputs)

    img_data = base64.b64encode(img_b).decode()

    result = {"class":label, "probs":pred_probs, "image":img_data}
    return result
   


@app.route('/upload', methods=["POST"])
async def upload(request):
    data = await request.form()
    img_b = await (data["file"].read())
    result = model_predict(img_b)

    return templates.TemplateResponse('result.html', {'request' : request, 'result' : result})
	
@app.route("/classify-url", methods=["GET","POST"])
async def classify_url(request):
    data = await request.form()
    url = data["url"]
    img_b = await get_bytes(url)
	
    result = model_predict(img_b)
    return templates.TemplateResponse('result.html', {'request' : request, 'result' : result})
    
@app.route("/")
def form(request):
    return templates.TemplateResponse('index.html', {'request' : request})

if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app = app, host="0.0.0.0", port=8080)
