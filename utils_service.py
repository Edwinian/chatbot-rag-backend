from typing import List

import easyocr


class UtilsService:
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
            results += _get_results(image_path, ["ch_sim"])
            return results
        except Exception as e:
            print(f"Image text extraction failed: {str(e)}")
            raise ValueError(f"Image text extraction failed: {str(e)}")
