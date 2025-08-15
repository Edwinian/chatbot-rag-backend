import os
import logging

from typing import List
import easyocr
import fitz  # PyMuPDF
import tempfile

# Configure logging
logging.basicConfig(filename="app.log", level=logging.INFO)
logger = logging.getLogger(__name__)


class UtilsService:
    def convert_pdf_to_image(self, pdf_path: str) -> List[str]:
        try:
            doc = fitz.open(pdf_path)
            temp_dir = tempfile.mkdtemp()
            image_paths = []

            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=200)  # Adjust DPI as needed
                img_path = os.path.join(temp_dir, f"page_{i}.png")
                pix.save(img_path)
                image_paths.append(img_path)

            return image_paths
        except Exception as e:
            logger.error(f"PDF to image conversion failed: {str(e)}")
            raise ValueError(f"PDF to image conversion failed: {str(e)}")

    def extract_texts_from_image(
        image_path: str, languages: list[str] = []
    ) -> List[str]:
        try:
            # All supported languages: https://www.jaided.ai/easyocr/
            target_languages = set(["en", "ch_sim", "ch_tra"] + languages)
            reader = easyocr.Reader(target_languages)
            results = reader.readtext(image_path)
            return results
        except Exception as e:
            logger.error(f"Image text extraction failed: {str(e)}")
            raise ValueError(f"Image text extraction failed: {str(e)}")
