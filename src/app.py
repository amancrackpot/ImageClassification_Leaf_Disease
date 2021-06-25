from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates
import uvicorn, aiohttp, asyncio
from io import BytesIO, StringIO
from fastai.vision.all import *
import base64
import pdb

export_file_name = 'export.pkl'
classes = ['Normal', 'Covid', 'Viral Pneumonia']
path = Path(__file__).parent.parent

templates = Jinja2Templates(directory='src/templates')
app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='src/static'))

async def setup_learner():
#     await download_file(export_file_url, path/'models'/export_file_name)
    defaults.device = torch.device('cpu')
    learn = load_learner(path/export_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

	
async def model_predict(img_b):
    img = PILImage.create(BytesIO(img_b))
    outputs = learn.predict(img)[2].numpy()
    formatted_outputs = [f"{i*100:.2f}" for i in outputs]
    pred_probs = zip(classes, map(str, formatted_outputs))

    img_bytes  = img.to_bytes_format()
    img_data = base64.b64encode(img_bytes).decode()

    result = {"probs":pred_probs, "image":img_data}
    return result
   


@app.route('/upload', methods=["POST"])
async def upload():
    img_b = await request.files['file'].read()
    result = await model_predict(img_b)

    return templates.TemplateResponse('result.html', result)
	
@app.route("/classify-url", methods=["POST"])
async def classify_url():
    url = request.form["url"]
    response = await requests.get(url)
	
    results = await model_predict(response.content)
    return templates.TemplateResponse('result.html', result)
    

@app.route("/")
def form(request):
    index_html = path/'static'/'index.html'
    return HTMLResponse(index_html.open().read())

if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app = app, host="0.0.0.0", port=8080)
