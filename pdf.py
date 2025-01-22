import fitz
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os
import re

def clean_text(text: str) -> str:
    if not text:
        return ""
    
    text = re.sub(r'^[\s•●○▪■□→⇒★-]\s*', '', text)
    
    text = re.sub(r'^\d+\.\s+', '', text)
    
    noise_patterns = [
        r'^\s*page\s+\d+\s*$',  # Page numbers
        r'^\s*chapter\s+\d+\s*$',  # Chapter numbers
        r'^\s*section\s+\d+\s*$',  # Section numbers
        r'//',  # Comments
        r'\[[\w\s]+\]',  # Text in brackets
        r'_{2,}',  # Multiple underscores
        r'-{2,}',  # Multiple dashes
        r'={2,}',  # Multiple equals signs
        r'\(\s*\w+\s*\)',  # Single letter/number in parentheses
        r'©',  # Copyright symbol
        r'®',  # Registered trademark
        r'™',  # Trademark
        r'…',  # Ellipsis
    ]
    
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text)
    
    text = re.sub(r'[^\w\s.,;:?!()\-\'\"]+', ' ', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r'^\d+\s*$', '', text)
    
    return text.strip()

@dataclass
class Context:
    heading: str
    text: str
    word_count: int = 0

    def __post_init__(self):
        self.heading = clean_text(self.heading)
        self.text = clean_text(self.text)
        self.word_count = len(self.text.split())

@dataclass
class Section:
    title: str
    content: List[str] = field(default_factory=list)
    subsections: List['Section'] = field(default_factory=list)
    level: int = 0
    parent: Optional['Section'] = None

    def add_subsection(self, section: 'Section'):
        section.parent = self
        self.subsections.append(section)

    def add_content(self, text: str):
        cleaned_text = clean_text(text)
        if cleaned_text:  # Only add if text remains after cleaning
            if self.content and len(self.content[-1]) < 100:
                combined = f"{self.content[-1]} {cleaned_text}"
                if len(combined) < 200:
                    self.content[-1] = combined
                    return
            self.content.append(cleaned_text)

    def merge_with_next(self, next_section: 'Section') -> bool:
        if (len(self.title) < 100 and len(next_section.title) < 100 and
            not self.content and not next_section.content):
            self.title = clean_text(f"{self.title} {next_section.title}")
            self.subsections.extend(next_section.subsections)
            return True
        return False

    def get_full_title(self) -> str:
        titles = []
        current = self
        while current:
            titles.append(clean_text(current.title))
            current = current.parent
        return " > ".join(reversed(titles))


@dataclass
class Chunk:
    contexts: List[Context] = field(default_factory=list)
    total_words: int = 0

    def can_add_context(self, context: Context) -> bool:
        return self.total_words + context.word_count <= 600

    def add_context(self, context: Context):
        self.contexts.append(context)
        self.total_words += context.word_count

class PDFStructureExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.root_sections: List[Section] = []
        self.temp_heading_buffer = []
        self.last_font_properties = None
        self.chunks: List[Chunk] = []

    def _get_font_properties(self, span) -> Tuple[bool, float]:
        is_bold = "bold" in span["font"].lower()
        font_size = span["size"]
        return is_bold, font_size

    def _determine_section_level(self, is_bold: bool, font_size: float) -> int:
        if is_bold and font_size >= 14:
            return 0
        elif is_bold and font_size >= 12:
            return 1
        elif is_bold:
            return 2
        return -1

    def _find_parent_section(self, current_level: int, last_sections: List[Optional[Section]]) -> Optional[Section]:
        for level in range(current_level - 1, -1, -1):
            if level < len(last_sections) and last_sections[level] is not None:
                return last_sections[level]
        return None

    def _process_heading_buffer(self, level: int, last_sections: List[Optional[Section]]):
        if not self.temp_heading_buffer:
            return None

        heading_text = " ".join(self.temp_heading_buffer)
        self.temp_heading_buffer.clear()
        
        new_section = Section(title=heading_text, level=level)
        
        if level == 0:
            if self.root_sections and self.root_sections[-1].merge_with_next(new_section):
                return self.root_sections[-1]
            self.root_sections.append(new_section)
        else:
            parent = self._find_parent_section(level, last_sections)
            if parent:
                parent.add_subsection(new_section)
            else:
                self.root_sections.append(new_section)
        
        return new_section

    def extract_structure(self):
        doc = fitz.open(self.pdf_path)
        last_sections = [None] * 10
        current_section = None

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_instances = page.get_text("dict")

            for block in text_instances["blocks"]:
                if block['type'] == 0:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if not text:
                                continue

                            is_bold, font_size = self._get_font_properties(span)
                            level = self._determine_section_level(is_bold, font_size)

                            if level >= 0:
                                if (self.last_font_properties and 
                                    self.last_font_properties == (is_bold, font_size)):
                                    self.temp_heading_buffer.append(text)
                                else:
                                    if self.temp_heading_buffer:
                                        current_section = self._process_heading_buffer(level, last_sections)
                                    self.temp_heading_buffer = [text]
                                
                                self.last_font_properties = (is_bold, font_size)
                                last_sections[level] = current_section
                                
                                for i in range(level + 1, len(last_sections)):
                                    last_sections[i] = None
                            
                            elif current_section is not None:
                                if self.temp_heading_buffer:
                                    current_section = self._process_heading_buffer(level, last_sections)
                                current_section.add_content(text)
                                self.last_font_properties = None

        if self.temp_heading_buffer:
            self._process_heading_buffer(0, last_sections)

    def create_chunks(self):
        self.chunks = []
        current_chunk = Chunk()
        
        def process_section(section: Section):
            nonlocal current_chunk
            
            if section.content:
                full_text = " ".join(section.content)
                context = Context(
                    heading=section.get_full_title(),
                    text=full_text
                )
                
                if context.word_count < 200:
                    if section.subsections:
                        next_section = section.subsections[0]
                        if next_section.content:
                            next_text = " ".join(next_section.content)
                            context.text = f"{context.text} {next_text}"
                            context.word_count = len(context.text.split())
                            section.subsections = section.subsections[1:]
                
                while context.word_count > 600:
                    words = context.text.split()
                    chunk_words = words[:600]
                    remaining_words = words[600:]
                    
                    new_context = Context(
                        heading=context.heading,
                        text=" ".join(chunk_words)
                    )
                    
                    if not current_chunk.can_add_context(new_context):
                        self.chunks.append(current_chunk)
                        current_chunk = Chunk()
                    
                    current_chunk.add_context(new_context)
                    context.text = " ".join(remaining_words)
                    context.word_count = len(remaining_words)
                
                if context.word_count > 0:
                    if not current_chunk.can_add_context(context):
                        self.chunks.append(current_chunk)
                        current_chunk = Chunk()
                    current_chunk.add_context(context)
            
            for subsection in section.subsections:
                process_section(subsection)
        
        for section in self.root_sections:
            process_section(section)
        
        if current_chunk.contexts:
            self.chunks.append(current_chunk)

    def display_chunks(self, output_file_path: str = None):
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(self.chunks, 1):
                header = f"\nChunk {i} (Total words: {chunk.total_words})"
                separator = "=" * 80
                
                print(header)
                print(separator)
                f.write(header + '\n')
                f.write(separator + '\n')
                
                for context in chunk.contexts:
                    output = [
                        f"\nHeading: {context.heading}",
                        f"Words: {context.word_count}",
                        f"Content: {context.text}",
                        # "-" * 40
                    ]
                    
                    for line in output:
                        print(line)
                        f.write(line + '\n')

def main():
    # pdf_path = "/home/sudhakardev/AppXcess/Test/RagGraph/GraphDB/User Access Management System.pdf"
    pdf_path="/home/sudhakardev/AppXcess/Test/RagGraph/GraphDB/federal_constitution.pdf"
    output_dir = os.path.join(os.path.dirname(pdf_path), "output")
    output_file = os.path.join(output_dir, "chunks.txt")
    
    extractor = PDFStructureExtractor(pdf_path)
    extractor.extract_structure()
    extractor.create_chunks()
    extractor.display_chunks(output_file)
    print(f"\nOutput has been saved to: {output_file}")

if __name__ == "__main__":
    main()