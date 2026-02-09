"""
Receipt Processing API
Simple API that accepts receipt images and returns date, total, and category.

Install dependencies:
    pip install fastapi uvicorn python-multipart --break-system-packages

Test:
    curl -X POST "http://localhost:8000/process-receipt" -F "file=@receipt.jpg"
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core_processor import get_processor

# Initialize processor once (reuse for all requests)
print("Initializing processor...")
processor = get_processor(use_gpu=True)
print("Initialization complete!")

app = FastAPI(
    title="Receipt Processor API",
    description="Upload a receipt image and get date, total, and category",
    version="1.0.0"
)

# Enable CORS (allows requests from web browsers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
async def root():
    """Welcome endpoint - test if API is running"""
    return {
        "message": "Receipt Processor API is running!",
        "version": "1.0.0",
        "endpoints": {
            "POST /process-receipt": "Upload receipt image to process",
            "GET /health": "Check API health"
        },
        "usage": "Send POST request to /process-receipt with 'file' parameter"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "API is running and ready to process receipts"
    }


@app.post("/process-receipt")
async def process_receipt(file: UploadFile = File(...)):
    """
    Process a receipt image and extract date, total, and category.

    Parameters:
    - file: Receipt image file (JPG, PNG, JPEG)

    Returns:
    - JSON with date, total, and category
    """

    # Validate file type
    allowed_extensions = [".jpg", ".jpeg", ".png"]
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
        )

    # Save uploaded file to temporary location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            # Read and save uploaded file
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Process the receipt
        result = process_receipt_internal(tmp_path)

        # Clean up temporary file
        os.unlink(tmp_path)

        return JSONResponse(content={
            "success": True,
            "data": result,
            "filename": file.filename
        })

    except Exception as e:
        # Clean up temporary file if it exists
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

        raise HTTPException(
            status_code=500,
            detail=f"Error processing receipt: {str(e)}"
        )


def process_receipt_internal(image_path: str) -> dict:
    """
    Internal function to process receipt.
    Uses the core processor.

    Args:
        image_path: Path to receipt image

    Returns:
        Dict with date, total, and category
    """
    return processor.process(image_path)


@app.post("/process-receipt-detailed")
async def process_receipt_detailed(file: UploadFile = File(...)):
    """
    Process receipt with additional debug information.

    Returns date, total, category, plus:
    - OCR confidence
    - Number of text elements found
    - Line items detected
    """

    # Validate file type
    allowed_extensions = [".jpg", ".jpeg", ".png"]
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
        )

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Process with details using core processor
        result = processor.process_with_details(tmp_path)

        # Clean up
        os.unlink(tmp_path)

        return JSONResponse(content={
            "success": True,
            "data": result,
            "filename": file.filename
        })

    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

        raise HTTPException(
            status_code=500,
            detail=f"Error processing receipt: {str(e)}"
        )


# Run the server
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Starting Receipt Processor API")
    print("=" * 60)
    print("\nAPI will be available at:")
    print("  - Local:   http://localhost:8000")
    print("  - Network: http://0.0.0.0:8000")
    print("\nEndpoints:")
    print("  - GET  /              : API info")
    print("  - GET  /health        : Health check")
    print("  - POST /process-receipt : Process receipt (main endpoint)")
    print("\nAPI Documentation:")
    print("  - Swagger UI: http://localhost:8000/docs")
    print("  - ReDoc:      http://localhost:8000/redoc")
    print("\n" + "=" * 60 + "\n")

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all network interfaces
        port=8000,
        log_level="info"
    )