class CitationGenerator:
    def __init__(self, author, title, publisher, year, url=None):
        self.author = author
        self.title = title
        self.publisher = publisher
        self.year = year
        self.url = url

    def generate(self, format_type):
        if format_type == "APA":
            return f"{self.author} ({self.year}). *{self.title}*. {self.publisher}. {self.url or ''}"
        elif format_type == "MLA":
            return f"{self.author}. \"{self.title}.\" {self.publisher}, {self.year}. {self.url or ''}"
        elif format_type == "Chicago":
            return f"{self.author}. *{self.title}*. {self.publisher}, {self.year}. {self.url or ''}"
        else:
            raise ValueError("Unsupported format type")
