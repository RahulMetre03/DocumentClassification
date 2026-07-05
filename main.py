import os
import io
import json
import uuid
import logging
import requests
import fitz  # pymupdf
import pytesseract
import ollama
import chromadb
from urllib.parse import urlparse, parse_qs
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb import Documents, EmbeddingFunction, Embeddings

import sys
from dotenv import load_dotenv

load_dotenv()

# force=True ensures this takes effect even after uvicorn has already
# configured the root logger (basicConfig is normally a no-op if called twice).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# ---------- CONFIG ----------
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

from google import genai
import json
import re
import time

GEMINI_API = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API)


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Chroma embedding function backed by Gemini."""

    def __call__(self, input: Documents) -> Embeddings:
        logger.info("Embedding %d chunk(s) via Gemini...", len(input))

        try:
            response = client.models.embed_content(
                model="gemini-embedding-2",
                contents=list(input)
            )

            embeddings = [e.values for e in response.embeddings]

            logger.info("Embedding complete.")
            return embeddings

        except Exception as e:
            logger.error("Gemini embedding failed: %r", e)
            raise

chroma = chromadb.CloudClient(
  api_key=os.getenv("CHROMA_API"),
  tenant='4bffc8e2-f490-4ec7-94ec-633c3407ddf3',
  database='dev'
)

collection = chroma.get_or_create_collection(
    name="medical_documents",
    embedding_function=GeminiEmbeddingFunction(),
)

app = FastAPI()


class DownloadRequest(BaseModel):
    url: str


# ---------- PDF EXTRACTION (native text + OCR fallback) ----------
def extract_content(file_path: str) -> str:
    doc = fitz.open(file_path)
    extracted_text = []

    for page in doc:
        text = page.get_text().strip()
        if len(text) > 50:
            extracted_text.append(text)
            continue

        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        extracted_text.append(pytesseract.image_to_string(img))

    doc.close()
    return "\n\n".join(extracted_text)


# ---------- METADATA EXTRACTION (LLM) ----------

def extract_metadata(document_text: str) -> dict:
    prompt = f"""
Extract metadata from the medical document.

Return ONLY valid JSON.

Schema:
{{
    "title": "",
    "document_type": "",
    "specialty": "",
    "disease": [],
    "keywords": [],
    "summary": ""
}}

Document:
{document_text[:12000]}
"""

    last_error = None

    for attempt in range(3):
        try:
            logger.info("Calling Gemini attempt=%d", attempt + 1)

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )

            content = response.text.strip()
            logger.info("Gemini response: %s", content)

            try:
                metadata = json.loads(content)
            except json.JSONDecodeError:
                clean = re.sub(r"```(?:json)?|```", "", content).strip()
                metadata = json.loads(clean)

            # Chroma accepts only primitive metadata values
            for k, v in metadata.items():
                if isinstance(v, list):
                    metadata[k] = ", ".join(map(str, v))

            return metadata

        except Exception as e:
            last_error = e
            logger.warning("Gemini attempt=%d failed: %s", attempt + 1, e)
            time.sleep(5)

    raise RuntimeError("Failed to extract metadata.") from last_error


# ---------- CHUNK + STORE IN CHROMA ----------
def store_document(document_text: str, file_name: str, metadata: dict) -> tuple[str, int]:
    document_id = str(uuid.uuid4())
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(document_text)

    ids = [f"{document_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {"document_id": document_id, "chunk_number": i, "file_name": file_name, **metadata}
        for i in range(len(chunks))
    ]

    collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    return document_id, len(chunks)


# ---------- MAIN ENDPOINT ----------
@app.post("/download-document")
async def download_document(request: DownloadRequest):
    try:
        url = request.url
        logger.info("Received request for url: %s", url)

        # Determine download URL
        if "docs.google.com/document" in url:
            doc_id = url.split("/d/")[1].split("/")[0]
            download_url = f"https://docs.google.com/document/d/{doc_id}/export?format=pdf"

        elif "drive.google.com/file/d/" in url:
            file_id = url.split("/d/")[1].split("/")[0]
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

        elif "drive.google.com/open" in url and "id=" in url:
            file_id = parse_qs(urlparse(url).query)["id"][0]
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

        elif "drive.google.com/uc" in url:
            download_url = url

        else:
            download_url = url

        logger.info("Downloading from: %s", download_url)

        # Download
        response = requests.get(download_url, timeout=60)
        response.raise_for_status()

        logger.info(
            "Download complete | Status=%s | Content-Type=%s | Bytes=%d",
            response.status_code,
            response.headers.get("Content-Type"),
            len(response.content),
        )

        # Validate response
        content_type = response.headers.get("Content-Type", "").lower()

        if "text/html" in content_type:
            raise HTTPException(
                status_code=400,
                detail="Unable to download the Google Drive file. Ensure it is shared publicly ('Anyone with the link') and that the URL points to a downloadable file."
            )

        # Generate filename
        filename = os.path.basename(urlparse(download_url).path)

        if not filename or "." not in filename:
            filename = f"{uuid.uuid4()}.pdf"

        file_path = os.path.join(DOWNLOAD_DIR, filename)

        with open(file_path, "wb") as f:
            f.write(response.content)

        logger.info("Saved file to %s", file_path)

        logger.info("Extracting text...")
        extracted_text = extract_content(file_path)
        logger.info("Extracted %d characters", len(extracted_text))

        logger.info("Extracting metadata via LLM...")
        metadata = extract_metadata(extracted_text)
        logger.info("Metadata: %s", metadata)

        logger.info("Chunking + storing in Chroma...")
        document_id, chunk_count = store_document(extracted_text, filename, metadata)
        logger.info("Stored document_id=%s chunks=%d", document_id, chunk_count)

        return {
            "success": True,
            "file_path": file_path,
            "document_id": document_id,
            "chunks_stored": chunk_count,
            "metadata": metadata,
        }

    except Exception as e:
        logger.exception("Request failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/search-medical-documents")
def search_medical_documents(request: SearchRequest):
    try:
        results = collection.query(
            query_texts=[request.query],
            n_results=request.top_k,
            include=["documents", "metadatas", "distances"]
        )

        documents = []

        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            documents.append({
                "score": round(1 - distance, 4),   # optional similarity score
                "metadata": meta,
                "content": doc,
            })

        return {
            "query": request.query,
            "count": len(documents),
            "results": documents,
        }

    except Exception as e:
        logger.exception("Medical document search failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, log_config=None)
