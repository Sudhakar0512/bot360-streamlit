


# pdf_processor.py
from typing import List, Dict
import PyPDF2
import spacy
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os

class EnhancedPDFProcessor:
    def __init__(self, max_workers: int = 6, model: str = "en_core_web_sm"):
        self.max_workers = max_workers
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        try:
            self.nlp = spacy.load(model)
            # Optimize for better performance
            if not "sentencizer" in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")
        except OSError:
            self.logger.info(f"Downloading spacy model: {model}")
            os.system(f"python -m spacy download {model}")
            self.nlp = spacy.load(model)

    def _process_page(self, page: PyPDF2.PageObject, page_num: int) -> Dict[str, str]:
        """Process a single PDF page."""
        try:
            text = page.extract_text()
            return {
                f"page_{page_num + 1}": text
            }
        except Exception as e:
            self.logger.error(f"Error processing page {page_num + 1}: {str(e)}")
            return {f"page_{page_num + 1}": ""}

    def extract_text_from_pdf(self, pdf_file) -> Dict[str, str]:
        """Extract text content from PDF with metadata using parallel processing."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            pages = {}
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_page = {
                    executor.submit(self._process_page, page, i): i 
                    for i, page in enumerate(pdf_reader.pages)
                }
                
                for future in as_completed(future_to_page):
                    page_result = future.result()
                    pages.update(page_result)
            
            return {
                "text": "\n".join(pages.values()),
                "metadata": {
                    "page_count": len(pages),
                    "pages": pages
                }
            }
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and special characters."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        return text.strip()

    def _process_section(self, section: str, respect_sentences: bool, chunk_size: int, overlap: int, section_index: int) -> List[Dict[str, str]]:
        """Process a single section into chunks using spaCy."""
        chunks = []
        doc = self.nlp(section)
        sentences = list(doc.sents) if respect_sentences else [doc]
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_text = sentence.text.strip()
            sentence_words = sentence_text.split()
            sentence_size = len(' '.join(sentence_words))
            
            if current_size + sentence_size > chunk_size:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "chunk_size": len(chunk_text),
                            "word_count": len(current_chunk),
                            "chunk_index": len(chunks),
                            "section_index": section_index,
                            "section_context": section[:100] + "..."
                        }
                    })
                
                if overlap > 0 and current_chunk:
                    overlap_words = current_chunk[-overlap:]
                    current_chunk = overlap_words
                    current_size = len(' '.join(overlap_words))
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.extend(sentence_words)
            current_size += sentence_size + 1
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "chunk_size": len(chunk_text),
                    "word_count": len(current_chunk),
                    "chunk_index": len(chunks),
                    "section_index": section_index,
                    "section_context": section[:100] + "..."
                }
            })
        
        return chunks

    def split_text_into_chunks(
        self, 
        text: str, 
        chunk_size: int = 1000,
        overlap: int = 100,
        respect_sentences: bool = True
    ) -> List[Dict[str, str]]:
        """Hybrid chunking implementation with parallel processing."""
        cleaned_text = self._clean_text(text)
        sections = self._split_by_section(cleaned_text)
        all_chunks = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_section = {
                executor.submit(
                    self._process_section,
                    section,
                    respect_sentences,
                    chunk_size,
                    overlap,
                    idx
                ): idx
                for idx, section in enumerate(sections)
            }
            
            for future in as_completed(future_to_section):
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                except Exception as e:
                    self.logger.error(f"Error processing section: {str(e)}")
                    continue
        
        all_chunks.sort(key=lambda x: (x['metadata']['section_index'], x['metadata']['chunk_index']))
        
        for i, chunk in enumerate(all_chunks):
            chunk['metadata']['chunk_index'] = i
        
        return all_chunks

    def _split_by_section(self, text: str) -> List[str]:
        """Split text by semantic sections."""
        doc = self.nlp(text)
        sections = []
        current_section = []

        for sent in doc.sents:
            if (len(sent.text.strip()) > 0 and 
                (sent.text.strip().endswith(':') or 
                 bool(re.match(r'^[A-Z][a-z]+ \d+\.', sent.text.strip())) or
                 (sent.text.isupper() and len(sent.text.split()) < 4))):
                if current_section:
                    sections.append(' '.join(current_section))
                current_section = [sent.text]
            else:
                current_section.append(sent.text)
        
        if current_section:
            sections.append(' '.join(current_section))
        
        return [s.strip() for s in sections if s.strip()]