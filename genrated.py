from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware if needed for external environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up device for model (using GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the GPT-2 model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Define the directory for templates
templates = Jinja2Templates(directory="templates")

# Serve static files (if any CSS/JS files are needed)



class CodeRequest(BaseModel):
    prompt: str
    language: str


@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate-code")
async def generate_code(request: CodeRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_length=300,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7
        )
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_code": generated_code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

