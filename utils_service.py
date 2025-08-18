import os
import tempfile
from typing import List

import fitz  # PyMuPDF
import easyocr
from docx2pdf import convert


class UtilsService:
    def convert_docx_to_pdf(self, file_path: str) -> str:
        try:
            temp_dir = tempfile.mkdtemp()
            pdf_path = os.path.join(temp_dir, "converted.pdf")

            # Convert the docx to pdf
            convert(file_path, pdf_path)

            print(f"Successfully converted DOCX to PDF: {pdf_path}")
            return pdf_path
        except Exception as e:
            print(f"Failed to convert DOCX to PDF: {str(e)}")
            raise ValueError(f"DOCX to PDF conversion failed: {str(e)}")

    def convert_pdf_to_image(self, file_path: str) -> List[str]:
        """Existing method from your code"""
        try:
            doc = fitz.open(file_path)
            temp_dir = tempfile.mkdtemp()
            image_paths = []

            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=200)  # Adjust DPI as needed
                img_path = os.path.join(temp_dir, f"page_{i}.png")
                pix.save(img_path)
                image_paths.append(img_path)

            return image_paths
        except Exception as e:
            print(f"PDF to image conversion failed: {str(e)}")
            raise ValueError(f"PDF to image conversion failed: {str(e)}")

    def extract_texts_from_image(self, image_path: str) -> List[str]:

        def _get_results(image_path: str, languages: list[str]) -> List[str]:
            reader = easyocr.Reader(languages)
            # Perform OCR
            results = reader.readtext(image_path)
            return results

        try:
            # All supported languages: https://www.jaided.ai/easyocr/
            # easyocr cannot parse both simplified and traditional Chinese at the same time
            results = _get_results(image_path, ["en", "ch_tra"])
            # results += _get_results(image_path, ["ch_sim"])
            return results
        except Exception as e:
            print(f"Image text extraction failed: {str(e)}")
            raise ValueError(f"Image text extraction failed: {str(e)}")
